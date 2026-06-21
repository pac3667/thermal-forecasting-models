import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
from darts import TimeSeries
from darts.models import DLinearModel
from darts.dataprocessing.transformers import Scaler
from sklearn import metrics
from pytorch_lightning.callbacks import EarlyStopping
from utils import optuna_darts_dlm_single_step_search
import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)


def train_dlm_direct_multistep(data_path, checkpoint_dir, n_out, train_size):
    torch.set_float32_matmul_precision('high')

    test_predictions = {step: [] for step in range(n_out)}
    test_ground_truth = {step: [] for step in range(n_out)}
    final_metrics_list = []

    step_checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_dlm')
    os.makedirs(step_checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(step_checkpoint_dir, 'Qpred_DLM_LAG_model_step_')

    base_data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    base_data['Month'] = base_data['Date'].dt.month
    base_data['Year'] = base_data['Date'].dt.year
    base_data['Month_sin'] = np.sin(2 * np.pi * base_data['Month'] / 12)
    base_data['Month_cos'] = np.cos(2 * np.pi * base_data['Month'] / 12)
    base_data.set_index('Date', inplace=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for step_idx in range(n_out):
        step_num = step_idx + 1
        print(f"\n" + "=" * 60)
        print(f" ПОДГОТОВКА МОДЕЛИ DLM ДЛЯ ШАГА {step_num} ИЗ {n_out}")
        print("=" * 60)

        current_checkpoint = f"{checkpoint_base}{step_idx}.pt"
        params_json_path = f"{checkpoint_base}{step_idx}_params.json"
        metrics_json_path = f"{checkpoint_base}{step_idx}_metrics.json"

        data = base_data.copy()

        if os.path.exists(current_checkpoint) and os.path.exists(params_json_path) and os.path.exists(
                metrics_json_path):
            print(f"--- Шаг {step_num}: Найден сохраненный чекпоинт и параметры. Загрузка... ---")
            with open(params_json_path, 'r') as f:
                best_params = json.load(f)
            with open(metrics_json_path, 'r') as f:
                saved_metrics = json.load(f)

            print(f"[LOAD SUCCESS] Окна декомпозиции: T — {best_params['t_smooth_window']} дн, Q — {best_params['q_smooth_window']} дн. История: {best_params['input_chunk_length']} дн.")
            print(f"[LOAD SUCCESS] Извлеченные метрики -> MAE: {saved_metrics['mae']:.4f} | R2: {saved_metrics['r2']:.4f}")

            final_metrics_list.append([saved_metrics['mae'], saved_metrics['mse'], saved_metrics['mape'], saved_metrics['r2']])
            continue

        print(f"Запуск Optuna с ранней остановкой (Pruning) для шага {step_num}...")

        pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=10)
        study = optuna.create_study(direction='minimize', pruner=pruner)

        study.optimize(lambda trial: optuna_darts_dlm_single_step_search(trial, data, train_size, step_num=step_num), n_trials=100)

        best_params = study.best_params
        with open(params_json_path, 'w') as f: json.dump(best_params, f)

        best_t_smooth = best_params['t_smooth_window']
        best_q_smooth = best_params['q_smooth_window']
        in_len = best_params['input_chunk_length']
        print(f"[УСПЕХ ШАГ {step_num}] Окна: T — {best_t_smooth} дн, Q — {best_q_smooth} дн. История: {in_len} дн.")

        data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=best_q_smooth).mean()
        data['T_rolling_3'] = data['T'].rolling(window=best_t_smooth).mean()

        data[f'T_lag{step_num}'] = data['T'].shift(-step_num)
        data[f'T_mean{step_num}'] = data['T_rolling_3'].shift(-step_num)
        data_with_lag = data.dropna()

        split_date = data_with_lag.index[train_size]

        target_raw_series = TimeSeries.from_dataframe(data_with_lag, value_cols='TEC_Q_Aver', freq='D')

        covariate_cols = [
            'T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling',
            f'T_mean{step_num}', f'T_lag{step_num}'
        ]
        covariates_raw_series = TimeSeries.from_dataframe(data_with_lag, value_cols=covariate_cols, freq='D')

        train_target_raw, _ = target_raw_series.split_before(split_date)
        train_cov_raw, _ = covariates_raw_series.split_before(split_date)

        # Индивидуальное масштабирование
        scaler_target = Scaler()
        train_target_scaled = scaler_target.fit_transform(train_target_raw)
        target_scaled = scaler_target.transform(target_raw_series)

        scaler_covariates = Scaler()
        _ = scaler_covariates.fit(train_cov_raw)
        covariates_scaled = scaler_covariates.transform(covariates_raw_series)

        early_stopper = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.0001, mode="min")

        # Инициализируем модель шага. Выход равен строго 1 точке
        model_dlm_step = DLinearModel(
            input_chunk_length=in_len,
            output_chunk_length=1,
            kernel_size=best_params['kernel_size'],
            shared_weights=False,
            const_init=True,
            batch_size=best_params['batch_size'],
            n_epochs=120,
            optimizer_kwargs={"lr": best_params['lr']},
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "callbacks": [early_stopper]
            }
        )

        print(f"Запуск финального обучения DLinear для шага {step_num}...")
        model_dlm_step.fit(series=train_target_scaled, past_covariates=covariates_scaled)

        model_dlm_step.save(current_checkpoint)
        print(f"[SAVE SUCCESS] Веса модели DLM сохранены в: {current_checkpoint}")

        max_test_idx = len(data_with_lag) - n_out + 1

        for idx in range(train_size, max_test_idx):
            current_time_node = target_raw_series.time_index[idx]

            actual_slice = target_raw_series.slice_n_points_after(current_time_node, step_num)
            if len(actual_slice) < step_num:
                continue
            step_true_val = actual_slice.values().flatten()[-1]

            history_target_scaled = target_scaled.slice_n_points_before(current_time_node, in_len)

            pred_scaled = model_dlm_step.predict(
                n=1,
                series=history_target_scaled,
                past_covariates=covariates_scaled,
                verbose=False
            )
            pred_real = scaler_target.inverse_transform(pred_scaled)
            step_pred_val = pred_real.values().flatten()[0]

            test_ground_truth[step_idx].append(step_true_val)
            test_predictions[step_idx].append(step_pred_val)

        step_true = np.array(test_ground_truth[step_idx])
        step_pred = np.array(test_predictions[step_idx])

        valid_idx = ~np.isnan(step_true) & ~np.isnan(step_pred)
        step_true, step_pred = step_true[valid_idx], step_pred[valid_idx]

        step_mae = metrics.mean_absolute_error(step_true, step_pred)
        step_mse = metrics.mean_squared_error(step_true, step_pred)
        step_mape = metrics.mean_absolute_percentage_error(step_true, step_pred) * 100
        step_r2 = metrics.r2_score(step_true, step_pred)

        step_metrics = {"mae": float(step_mae), "mse": float(step_mse), "mape": float(step_mape), "r2": float(step_r2)}
        with open(metrics_json_path, 'w') as f:json.dump(step_metrics, f)

        final_metrics_list.append([step_mae, step_mse, step_mape, step_r2])

    return np.hstack(final_metrics_list)
