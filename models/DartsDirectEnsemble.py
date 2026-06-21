import os
import json
import numpy as np
import pandas as pd
import optuna
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import metrics

from utils import optuna_darts_direct_ensemble_search


def train_direct_ensemble_multistep(data_path, checkpoint_dir, n_out, train_size):
    """
    Полный Direct-пайплайн многошагового прогнозирования ТЭЦ на базе нативного ансамбля бустингов в Darts.
    """
    step_checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_ensemble')
    os.makedirs(step_checkpoint_dir, exist_ok=True)

    model_checkpoint_path = os.path.join(step_checkpoint_dir, 'Qpred_DirectEnsemble_model.pt')
    params_json_path = os.path.join(step_checkpoint_dir, 'ensemble_best_params.json')
    metrics_json_path = os.path.join(step_checkpoint_dir, 'ensemble_metrics.json')

    # 1. Загрузка данных и генерация признаков
    base_data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    base_data['Month'] = base_data['Date'].dt.month
    base_data['Year'] = base_data['Date'].dt.year
    base_data['Month_sin'] = np.sin(2 * np.pi * base_data['Month'] / 12)
    base_data['Month_cos'] = np.cos(2 * np.pi * base_data['Month'] / 12)
    base_data.set_index('Date', inplace=True)
    base_data = base_data.asfreq('D')

    # === ЛОГИКА ВОССТАНОВЛЕНИЯ ИЗ ЧЕКПОИНТА ===
    if os.path.exists(model_checkpoint_path) and os.path.exists(params_json_path) and os.path.exists(metrics_json_path):
        print("--- [Darts Ensemble] Найден сохраненный пайплайн. Загрузка... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
        with open(metrics_json_path, 'r') as f:
            saved_metrics = json.load(f)

        print("\n" + "=" * 60)
        print("=== ИЗВЛЕЧЕННЫЕ РЕЗУЛЬТАТЫ DARTS DIRECT ENSEMBLE ===")
        print("=" * 60)
        for step in range(n_out):
            print(
                f"День {step + 1:<2} -> MAE: {saved_metrics[f'day_{step + 1}_mae']:.4f} | R2: {saved_metrics[f'day_{step + 1}_r2']:.4f}")
        print("=" * 60)

        metrics_flat = []
        for step in range(n_out):
            metrics_flat.extend([saved_metrics[f'day_{step + 1}_mae'], 0.0, 0.0, saved_metrics[f'day_{step + 1}_r2']])
        return np.array(metrics_flat)

    # === ОПТИМИЗАЦИЯ OPTUNA ===
    print("\nЗапуск Optuna для подбора окон сглаживания ансамбля...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(lambda trial: optuna_darts_direct_ensemble_search(trial, base_data, train_size, n_out), n_trials=100)

    best_params = study.best_params
    with open(params_json_path, 'w') as f:
        json.dump(best_params, f)

    best_t_smooth = best_params['t_smooth_window']
    best_q_smooth = best_params['q_smooth_window']
    best_lags = best_params['lags']

    print(f"[УСПЕХ] Окна подобраны: T — {best_t_smooth} дн, Q — {best_q_smooth} дн. Lags: {best_lags}")

    # 2. ПОДГОТОВКА ПРИЗНАКОВ С ОПТИМАЛЬНЫМ СГЛАЖИВАНИЕМ
    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=best_q_smooth, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=best_t_smooth, min_periods=1).mean()

    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    future_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3']
    future_covariates = TimeSeries.from_dataframe(data, value_cols=future_cov_cols, freq='D')

    past_cov_cols = ['Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=past_cov_cols, freq='D')

    split_date = data.index[train_size]
    train_target, _ = target_series.split_before(split_date)

    lgb = LGBMRegressor(max_depth=6, learning_rate=0.03, n_estimators=250, random_state=42, n_jobs=-1, verbose=-1)
    xgb = XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=250, gamma=2.0, tree_method='hist',
                       random_state=42, n_jobs=-1, verbosity=0)
    cbr = CatBoostRegressor(depth=5, learning_rate=0.03, iterations=250, l2_leaf_reg=3.0, random_seed=42, verbose=0)

    final_voting_ensemble = VotingRegressor(estimators=[('lgb', lgb), ('xgb', xgb), ('cbr', cbr)])

    # Передаем готовый ансамбль в параметр `model`
    model_ensemble = RegressionModel(
        model=final_voting_ensemble,
        lags=best_lags,
        lags_future_covariates=(best_lags, n_out),
        lags_past_covariates=best_lags,
        output_chunk_length=n_out
    )

    print("Запуск финального Direct-обучения мета-ансамбля бустингов...")
    model_ensemble.fit(
        series=train_target,
        past_covariates=past_covariates,
        future_covariates=future_covariates
    )

    model_ensemble.save(model_checkpoint_path)
    print(f"[SAVE SUCCESS] Веса Direct-ансамбля сохранены в: {model_checkpoint_path}")

    # 4. СКОЛЬЗЯЩИЙ ИНФЕРЕНС НА ТЕСТЕ С ОГРАНИЧЕНИЕМ ПО ВРЕМЕНИ ВПЕРЕД
    max_test_idx = len(data) - n_out
    y_true_list, y_pred_list = [], []

    for idx in range(train_size, max_test_idx):
        current_time_node = target_series.time_index[idx]

        actual_slice = target_series.slice_n_points_after(current_time_node, n_out)
        if len(actual_slice) < n_out:
            continue

        history_target = target_series.slice_n_points_before(current_time_node, best_lags)

        # Модель делает чистый Direct прогноз силами трех внутренних алгоритмов
        pred = model_ensemble.predict(
            n=n_out,
            series=history_target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            verbose=False
        )

        y_true_list.append(actual_slice.values().flatten())
        y_pred_list.append(pred.values().flatten())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # 5. РАСЧЕТ И КЭШИРОВАНИЕ ЧЕСТНЫХ МЕТРИК
    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ГОРИЗОНТА ДЛЯ DARTS DIRECT ENSEMBLE ===")
    print("=" * 50)

    final_metrics_list = []
    metrics_to_cache = {}

    for step in range(n_out):
        step_true, step_pred = y_true[:, step], y_pred[:, step]
        valid_idx = ~np.isnan(step_true) & ~np.isnan(step_pred)
        step_true, step_pred = step_true[valid_idx], step_pred[valid_idx]

        step_mae = metrics.mean_absolute_error(step_true, step_pred)
        step_mse = metrics.mean_squared_error(step_true, step_pred)
        step_mape = metrics.mean_absolute_percentage_error(step_true, step_pred) * 100
        step_r2 = metrics.r2_score(step_true, step_pred)

        print(f"День {step + 1:<2} -> Ансамбль MAE: {step_mae:<8.4f} | R2: {step_r2:<8.4f}")
        final_metrics_list.append([step_mae, step_mse, step_mape, step_r2])

        metrics_to_cache[f"day_{step + 1}_mae"] = float(step_mae)
        metrics_to_cache[f"day_{step + 1}_mse"] = float(step_mse)
        metrics_to_cache[f"day_{step + 1}_mape"] = float(step_mape)
        metrics_to_cache[f"day_{step + 1}_r2"] = float(step_r2)

    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_cache, f)

    print("=" * 50)
    return np.hstack(final_metrics_list)
