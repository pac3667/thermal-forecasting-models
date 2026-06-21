import os
import json
import numpy as np
import pandas as pd
import optuna
from darts import TimeSeries
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn import metrics

from utils import optuna_darts_tsmixer_search


def train_tsmixer_darts_multistep(data_path, checkpoint_dir, n_out, train_size):
    """
    Полный Direct-пайплайн многошагового прогнозирования ТЭЦ на базе архитектуры TSMixer.
    """
    step_checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_tsmixer')
    os.makedirs(step_checkpoint_dir, exist_ok=True)

    model_checkpoint_path = os.path.join(step_checkpoint_dir, 'Qpred_tsmixer_direct_model.pt')
    params_json_path = os.path.join(step_checkpoint_dir, 'tsmixer_best_params.json')
    metrics_json_path = os.path.join(step_checkpoint_dir, 'tsmixer_metrics.json')

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
        print("--- [Darts TSMixer] Найден сохраненный пайплайн. Загрузка... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
        with open(metrics_json_path, 'r') as f:
            saved_metrics = json.load(f)

        print("\n" + "=" * 60)
        print("=== ИЗВЛЕЧЕННЫЕ РЕЗУЛЬТАТЫ DARTS TSMIXER DIRECT ===")
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
    print("\nЗапуск Optuna для подбора гиперпараметров TSMixer...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optuna_darts_tsmixer_search(trial, base_data, train_size, n_out), n_trials=100)

    best_params = study.best_params
    with open(params_json_path, 'w') as f:
        json.dump(best_params, f)

    best_t_smooth = best_params['t_smooth_window']
    best_q_smooth = best_params['q_smooth_window']
    input_chunk = best_params['input_chunk_length']

    print(f"[УСПЕХ] Параметры подобраны. Lags: {input_chunk}, ff_size: {best_params['ff_size']}")

    # 2. ПОДГОТОВКА ПРИЗНАКОВ С ОПТИМАЛЬНЫМ СГЛАЖИВАНИЕМ
    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=best_q_smooth, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=best_t_smooth, min_periods=1).mean()

    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    all_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=all_cov_cols, freq='D')

    split_date = data.index[train_size]
    train_target, _ = target_series.split_before(split_date)

    early_stopper = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.02, mode="min")

    # 3. ИНИЦИАЛИЗАЦИЯ ФИНАЛЬНОЙ МОДЕЛИ TSMIXER
    model_tsmixer = TSMixerModel(
        input_chunk_length=input_chunk,
        output_chunk_length=n_out,
        num_blocks=2,  # 2 блока для финального обучения
        ff_size=best_params['ff_size'],
        dropout=0.3,
        n_epochs=120,
        batch_size=32,
        optimizer_kwargs={"lr": best_params['lr']},
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto"
        },
        random_state=42
    )

    print("Запуск финального обучения Direct-модели TSMixer...")
    model_tsmixer.fit(series=train_target, past_covariates=past_covariates)

    model_tsmixer.save(model_checkpoint_path)
    print(f"[SAVE SUCCESS] Веса TSMixer сохранены в: {model_checkpoint_path}")

    # 4. СКОЛЬЗЯЩИЙ ИНФЕРЕНС НА ТЕСТЕ С ОГРАНИЧЕНИЕМ ПО ВРЕМЕНИ ВПЕРЕД
    max_test_idx = len(data) - n_out
    y_true_list, y_pred_list = [], []

    for idx in range(train_size, max_test_idx):
        current_time_node = target_series.time_index[idx]

        actual_slice = target_series.slice_n_points_after(current_time_node, n_out)
        if len(actual_slice) < n_out:
            continue

        history_target = target_series.slice_n_points_before(current_time_node, input_chunk)

        pred = model_tsmixer.predict(
            n=n_out,
            series=history_target,
            past_covariates=past_covariates,
            verbose=False
        )

        y_true_list.append(actual_slice.values().flatten())
        y_pred_list.append(pred.values().flatten())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # 5. РАСЧЕТ И КЭШИРОВАНИЕ МЕТРИК
    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ГОРИЗОНТА ДЛЯ DARTS TSMIXER DIRECT ===")
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

        print(f"День {step + 1:<2} -> MAE: {step_mae:<8.4f} | R2: {step_r2:<8.4f}")
        final_metrics_list.append([step_mae, step_mse, step_mape, step_r2])

        metrics_to_cache[f"day_{step + 1}_mae"] = float(step_mae)
        metrics_to_cache[f"day_{step + 1}_mse"] = float(step_mse)
        metrics_to_cache[f"day_{step + 1}_mape"] = float(step_mape)
        metrics_to_cache[f"day_{step + 1}_r2"] = float(step_r2)

    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_cache, f)

    print("=" * 50)
    return np.hstack(final_metrics_list)