import os
import json
import pickle
import numpy as np
import pandas as pd
import optuna
from mlforecast.lag_transforms import RollingMean
from coreforecast.rolling import rolling_mean
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from lightgbm import LGBMRegressor
from sklearn import metrics

from utils import optuna_nixtla_lgb_search


def train_nixtla_lgb_multistep(data_path, checkpoint_dir, n_out, train_size):
    """
    Полный пайплайн Direct-прогнозирования ТЭЦ на фреймворке Nixtla MLForecast
    БЕЗ использования рекурсии (каждый шаг горизонта прогнозируется своей моделью).
    """
    nixtla_dir = os.path.join(checkpoint_dir, 'nixtla_lgbm')
    os.makedirs(nixtla_dir, exist_ok=True)

    model_checkpoint_path = os.path.join(nixtla_dir, 'nixtla_direct_pipeline.pkl')
    params_json_path = os.path.join(nixtla_dir, 'nixtla_best_params.json')
    metrics_json_path = os.path.join(nixtla_dir, 'nixtla_metrics.json')

    base_data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    base_data['Month'] = base_data['Date'].dt.month
    base_data['Month_sin'] = np.sin(2 * np.pi * base_data['Month'] / 12)
    base_data['Month_cos'] = np.cos(2 * np.pi * base_data['Month'] / 12)

    df_nixtla = base_data.rename(columns={'Date': 'ds', 'TEC_Q_Aver': 'y'})
    df_nixtla['unique_id'] = 'TEC_1'

    for lag in range(1, 8):
        df_nixtla[f'T_lag_{lag}'] = df_nixtla['T'].shift(lag)

    df_nixtla = df_nixtla.sort_values('ds').reset_index(drop=True)

    # Разделение на выборки для Optuna
    df_train_opt = df_nixtla.iloc[:train_size].copy()
    df_val_opt = df_nixtla.iloc[train_size:train_size + n_out].copy()

    # === ЛОГИКА ВОССТАНОВЛЕНИЯ ИЗ ЧЕКПОИНТА ===
    if os.path.exists(model_checkpoint_path) and os.path.exists(params_json_path) and os.path.exists(metrics_json_path):
        print("--- [Nixtla] Найден сохраненный пайплайн. Мгновенная загрузка... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
        with open(metrics_json_path, 'r') as f:
            saved_metrics = json.load(f)

        print("\n" + "=" * 60)
        print("=== ИЗВЛЕЧЕННЫЕ РЕЗУЛЬТАТЫ NIXTLA MLFORECAST (DIRECT) ===")
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
    print("\nЗапуск поиска параметров в Nixtla...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optuna_nixtla_lgb_search(
        trial, df_train_opt, df_val_opt, n_out
    ), n_trials=100)

    best_params = study.best_params
    with open(params_json_path, 'w') as f:
        json.dump(best_params, f)

    print(
        f"[NIXTLA SUCCESS] Подбор окон завершен. Лаги: {best_params['lags_count']}, Глубина: {best_params['max_depth']}")

    # 2. ПОСТРОЕНИЕ ФИНАЛЬНОГО ОБУЧЕНИЯ (DIRECT)
    # Сглаживание признаков делаем ОДИН раз для всего датафрейма во избежание багов
    df_nixtla['T_rolling'] = df_nixtla['T'].rolling(window=best_params['t_smooth'], min_periods=1).mean()

    df_final_train = df_nixtla.iloc[:train_size].copy()

    final_lgb = LGBMRegressor(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['lr'],
        n_estimators=best_params['n_estimators'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Включаем стратегию Direct через target_transforms.
    # Библиотека обучит n_out независимых LightGBM моделей.
    fcst = MLForecast(
        models=[final_lgb],
        freq='D',
        lags=list(range(1, best_params['lags_count'] + 1)),
        lag_transforms={}  # Отключаем неэффективный RollingMean по таргету для Direct
    )

    lag_t_cols = [f'T_lag_{lag}' for lag in range(1, 8)]
    keep_cols = ['unique_id', 'ds', 'y', 'T', 'T_rolling', 'Month_sin', 'Month_cos'] + lag_t_cols

    df_final_train_clean = df_final_train[keep_cols]

    print("Финальное обучение ансамбля Direct-моделей в Nixtla...")
    fcst.fit(
        df_final_train_clean,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        max_horizon=n_out,  # <-- ИМЕННО ЭТА СТРОКА ОТКЛЮЧАЕТ РЕКУРСИЮ И ВКЛЮЧАЕТ DIRECT В MLFORECAST!
        static_features=[]
    )

    # СОХРАНЕНИЕ ПАЙПЛАЙНА
    with open(model_checkpoint_path, 'wb') as f:
        pickle.dump(fcst, f)
    print(f"[SAVE SUCCESS] Полный Direct-пайплайн сохранен в: {model_checkpoint_path}")

    future_cols = [c for c in keep_cols if c != 'y']

    # 3. СКОЛЬЗЯЩИЙ ИНФЕРЕНС НА ТЕСТЕ
    max_test_idx = len(df_nixtla) - n_out + 1
    y_true_list, y_pred_list = [], []

    for idx in range(train_size, max_test_idx):
        # Вырезаем чистую историю строго до текущей точки инференса idx
        full_history = df_nixtla.iloc[:idx].copy()
        history_df_clean = full_history[keep_cols]

        # Будущие экзогенные фичи (прогноз погоды и календарь) на n_out шагов вперед
        future_features_df = df_nixtla.iloc[idx:idx + n_out].copy()
        actual_target_values = future_features_df['y'].values

        if len(actual_target_values) < n_out:
            continue

        # Формируем чистый датафрейм будущего без целевого расхода 'y'
        X_df_future_clean = future_features_df[future_cols].reset_index(drop=True)

        # Честный Direct-прогноз без авторегрессионного зацикливания
        res = fcst.predict(
            h=n_out,
            new_df=history_df_clean,  # Передаем историю для построения лагов таргета
            X_df=X_df_future_clean  # Передаем экзогенные факторы на горизонт вперед
        )

        y_true_list.append(actual_target_values)
        y_pred_list.append(res['LGBMRegressor'].values)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # 4. СБОР И КЭШИРОВАНИЕ МЕТРИК
    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ГОРИЗОНТА ДЛЯ NIXTLA MLFORECAST (DIRECT) ===")
    print("=" * 50)

    metrics_list = []
    metrics_to_cache = {}

    for step in range(n_out):
        step_true, step_pred = y_true[:, step], y_pred[:, step]
        valid_idx = ~np.isnan(step_true) & ~np.isnan(step_pred)
        step_true, step_pred = step_true[valid_idx], step_pred[valid_idx]

        step_mae = metrics.mean_absolute_error(step_true, step_pred)
        step_mse = metrics.mean_squared_error(step_true, step_pred)
        step_mape = metrics.mean_absolute_percentage_error(step_true, step_pred) * 100
        step_r2 = metrics.r2_score(step_true, step_pred)

        print(f"День {step + 1:<2} -> MAE: {step_mae:<8.4f} | R2: {step_r2:<8.4f}")
        metrics_list.append([step_mae, step_mse, step_mape, step_r2])

        metrics_to_cache[f"day_{step + 1}_mae"] = float(step_mae)
        metrics_to_cache[f"day_{step + 1}_mse"] = float(step_mse)
        metrics_to_cache[f"day_{step + 1}_mape"] = float(step_mape)
        metrics_to_cache[f"day_{step + 1}_r2"] = float(step_r2)

    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_cache, f)

    print("=" * 50)
    return np.hstack(metrics_list)
