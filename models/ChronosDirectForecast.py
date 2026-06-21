import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
from darts import TimeSeries
from darts.models import Chronos2Model  # или ChronosModel в зависимости от версии darts
from darts.utils.likelihood_models import QuantileRegression
from sklearn import metrics
import logging
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)

from utils import optuna_darts_chronos_search


def train_chronos_darts_multistep(data_path, checkpoint_dir, n_out, train_size):
    torch.set_float32_matmul_precision('high')

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_chronos')
    os.makedirs(checkpoint_dir, exist_ok=True)
    params_json_path = os.path.join(checkpoint_dir, 'Qpred_Chronos_best_params.json')

    # Загрузка
    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    target_raw_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    split_date = data.index[train_size]

    # === БЛОК OPTUNA ОПТИМИЗАЦИИ С РАЗОГРЕВОМ ПРУНЕРА ===
    if os.path.exists(params_json_path):
        print(f"--- Найдена конфигурация Chronos. Загрузка параметров... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
    else:
        print("\nЗапуск Optuna для оптимизации параметров инференса Amazon Chronos...")

        # Глушим логи самой Optuna для чистоты консоли
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Настраиваем Медианный Прунер с разогревом на 15 триалов
        pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=10)
        study = optuna.create_study(direction='minimize', pruner=pruner)

        study.optimize(lambda trial: optuna_darts_chronos_search(
            trial, target_raw_series, split_date, n_out
        ), n_trials=30)  # Для Chronos 30 триалов обычно достаточно, так как модель тяжелая

        best_params = study.best_params
        with open(params_json_path, 'w') as f:
            json.dump(best_params, f)

    in_len = best_params['input_chunk_length']
    print(f"\n[CHRONOS SUCCESS] Настройки: Окно истории: {in_len}, Сэмплов: {best_params['num_samples']}")

    # Инициализируем финальную модель
    model_chronos = Chronos2Model(
        input_chunk_length=in_len,
        output_chunk_length=n_out,
        likelihood=QuantileRegression(),
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False
        }
    )

    # Если вы используете предобученные веса zero-shot, fit обучается мгновенно на метаданных ряда
    model_chronos.fit(series=target_raw_series)

    # Контролируемый пошаговый инференс без утечек
    max_test_idx = len(data) - n_out + 1
    y_true_list, y_pred_list = [], []

    for idx in range(train_size, max_test_idx):
        current_time_node = target_raw_series.time_index[idx]

        actual_slice = target_raw_series.slice_n_points_after(current_time_node, n_out)
        if len(actual_slice) < n_out:
            continue

        # ИСПРАВЛЕНО: Вырезаем скользящую историю СТРОГО фиксированной длины in_len ДО текущего узла
        history_target = target_raw_series.slice_n_points_before(current_time_node, in_len)

        # Вызываем предикт с оптимальным числом сэмплов для медианы
        pred = model_chronos.predict(
            n=n_out,
            series=history_target,
            num_samples=best_params['num_samples'],
            verbose=False
        )

        y_true_list.append(actual_slice.values().flatten())
        y_pred_list.append(pred.values().flatten())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ГОРИЗОНТА ДЛЯ AMAZON CHRONOS (ОПТИМИЗИРОВАН) ===")
    print("=" * 50)

    metrics_list = []
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

    print("=" * 50)
    return np.hstack(metrics_list)
