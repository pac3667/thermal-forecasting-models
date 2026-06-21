import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from sklearn import metrics
from pytorch_lightning.callbacks import EarlyStopping
# Возвращаем оригинальную функцию поиска, которая подбирает параметры ДЛЯ ВСЕГО ГОРИЗОНТА СРАЗУ
from utils import optuna_darts_tide_search
import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)


def train_tide_darts_multistep(data_path, checkpoint_dir, n_out, train_size):
    torch.set_float32_matmul_precision('high')

    # Для MIMO схемы нам нужна всего ОДНА папка и один чекпоинт на весь горизонт
    mimo_checkpoint_dir = os.path.join(checkpoint_dir, 'mimo_tide')
    os.makedirs(mimo_checkpoint_dir, exist_ok=True)

    model_checkpoint_path = os.path.join(mimo_checkpoint_dir, 'Qpred_TiDE_MIMO_model.pt')
    params_json_path = os.path.join(mimo_checkpoint_dir, 'Qpred_TiDE_MIMO_best_params.json')
    metrics_json_path = os.path.join(mimo_checkpoint_dir, 'Qpred_TiDE_MIMO_metrics.json')

    # Базовая загрузка (делается один раз)
    base_data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    base_data['Month'] = base_data['Date'].dt.month
    base_data['Year'] = base_data['Date'].dt.year
    base_data['Month_sin'] = np.sin(2 * np.pi * base_data['Month'] / 12)
    base_data['Month_cos'] = np.cos(2 * np.pi * base_data['Month'] / 12)
    base_data.set_index('Date', inplace=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # === ЛОГИКА ЗАГРУЗКИ ГОТОВОЙ MIMO-МОДЕЛИ ===
    if os.path.exists(model_checkpoint_path) and os.path.exists(params_json_path) and os.path.exists(metrics_json_path):
        print(f"--- [TiDE MIMO] Найден сохраненный чекпоинт всего горизонта. Загрузка... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
        with open(metrics_json_path, 'r') as f:
            saved_metrics = json.load(f)

        print(f"[LOAD SUCCESS] Окна: T — {best_params['t_smooth_window']} дн, Q — {best_params['q_smooth_window']} дн")

        # Выводим сохраненный отчет по дням
        print("\n" + "=" * 60)
        print("=== ИЗВЛЕЧЕННЫЕ РЕЗУЛЬТАТЫ МОНОЛИТНОЙ MIMO TiDE ===")
        print("=" * 60)
        for step in range(n_out):
            print(
                f"День {step + 1:<2} -> MAE: {saved_metrics[f'day_{step + 1}_mae']:.4f} | R2: {saved_metrics[f'day_{step + 1}_r2']:.4f}")
        print("=" * 60)

        metrics_flat = []
        for step in range(n_out):
            metrics_flat.extend([
                saved_metrics[f'day_{step + 1}_mae'], saved_metrics[f'day_{step + 1}_mse'],
                saved_metrics[f'day_{step + 1}_mape'], saved_metrics[f'day_{step + 1}_r2']
            ])
        return np.array(metrics_flat)

    print("\nЗапуск Optuna для оптимизации глобальной MIMO-архитектуры Google TiDE...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optuna_darts_tide_search(
        trial, base_data, train_size, n_out
    ), n_trials=100)

    best_params = study.best_params
    with open(params_json_path, 'w') as f:
        json.dump(best_params, f)

    best_t_smooth = best_params['t_smooth_window']
    best_q_smooth = best_params['q_smooth_window']
    print(f"[TiDE OPTUNA SUCCESS] Лучшие глобальные окна: T — {best_t_smooth} дн, Q — {best_q_smooth} дн")

    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=best_q_smooth).mean()
    data['T_rolling_3'] = data['T'].rolling(window=best_t_smooth).mean()
    data_cleaned = data.dropna()

    split_date = data_cleaned.index[train_size]

    target_raw_series = TimeSeries.from_dataframe(data_cleaned, value_cols='TEC_Q_Aver', freq='D')

    # В ковариаты идут только текущие и прошлые фичи. Встроенный в Darts TiDE сам
    # спроецирует их в будущее внутри архитектуры нейросети.
    covariate_cols = ['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling']
    covariates_raw_series = TimeSeries.from_dataframe(data_cleaned, value_cols=covariate_cols, freq='D')

    # Разделение до нормализации (Жесткая защита от Data Leakage)
    train_target_raw, _ = target_raw_series.split_before(split_date)
    train_cov_raw, _ = covariates_raw_series.split_before(split_date)

    # Нормализация
    scaler_target = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target_raw)
    target_scaled = scaler_target.transform(target_raw_series)

    scaler_covariates = Scaler()
    _ = scaler_covariates.fit(train_cov_raw)
    covariates_scaled = scaler_covariates.transform(covariates_raw_series)

    early_stopper = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.0001, mode="min")

    model_tide_mimo = TiDEModel(
        input_chunk_length=best_params['input_chunk_length'],
        output_chunk_length=n_out,
        num_encoder_layers=best_params['num_layers'],
        num_decoder_layers=best_params['num_layers'],
        decoder_output_dim=16,
        hidden_size=best_params['hidden_size'],
        dropout=best_params['dropout'],
        batch_size=best_params['batch_size'],
        n_epochs=120,
        optimizer_kwargs={"lr": best_params['lr']},
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [early_stopper],
            "logger": False
        }
    )

    print("Обучение единой MIMO-модели Google TiDE на весь горизонт...")
    model_tide_mimo.fit(series=train_target_scaled, past_covariates=covariates_scaled)

    # СОХРАНЕНИЕ ЕДИНОЙ МОДЕЛИ
    max_test_idx = len(data_cleaned) - n_out + 1
    y_true_list, y_pred_list = [], []

    for idx in range(train_size, max_test_idx):
        current_time_node = target_raw_series.time_index[idx]
        actual_slice = target_raw_series.slice_n_points_after(current_time_node, n_out)

        if len(actual_slice) < n_out:
            continue

        history_target_scaled = target_scaled.slice_n_points_before(
            current_time_node,
            best_params['input_chunk_length']
        )

        pred_scaled = model_tide_mimo.predict(
            n=n_out,
            series=history_target_scaled,
            past_covariates=covariates_scaled,
            verbose=False
        )
        pred_real = scaler_target.inverse_transform(pred_scaled)

        y_true_list.append(actual_slice.values().flatten())
        y_pred_list.append(pred_real.values().flatten())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

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

        metrics_list.append([step_mae, step_mse, step_mape, step_r2])

        metrics_to_cache[f"day_{step + 1}_mae"] = float(step_mae)
        metrics_to_cache[f"day_{step + 1}_mse"] = float(step_mse)
        metrics_to_cache[f"day_{step + 1}_mape"] = float(step_mape)
        metrics_to_cache[f"day_{step + 1}_r2"] = float(step_r2)

    with open(metrics_json_path, 'w') as f:json.dump(metrics_to_cache, f)

    return np.hstack(metrics_list)
