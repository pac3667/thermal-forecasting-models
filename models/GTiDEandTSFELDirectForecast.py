import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
import tsfel
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from sklearn import metrics
from pytorch_lightning.callbacks import EarlyStopping
from utils import optuna_darts_tide_search


def train_tide_and_tsfeel_darts_multistep(data_path, checkpoint_dir, n_out, train_size):
    torch.set_float32_matmul_precision('high')

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_tide_tsfel_clean')
    os.makedirs(checkpoint_dir, exist_ok=True)
    params_json_path = os.path.join(checkpoint_dir, 'Qpred_TiDE_and_TSFEL_Direct_best_params.json')

    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)

    data.sort_index(inplace=True)

    print("[TSFEL] Извлечение скрытых статистических признаков ряда...")
    cfg = tsfel.get_features_by_domain('statistical')

    tsfel_features_q = tsfel.time_series_features_extractor(cfg, data['TEC_Q_Aver'], window_size=7, overlap=0.86,
                                                            verbose=0)
    tsfel_features_t = tsfel.time_series_features_extractor(cfg, data['T'], window_size=7, overlap=0.86, verbose=0)

    tsfel_features_q.columns = [f"Q_tsfel_{str(col)}" for col in tsfel_features_q.columns]
    tsfel_features_t.columns = [f"T_tsfel_{str(col)}" for col in tsfel_features_t.columns]

    tsfel_features_q.index = data.index[len(data) - len(tsfel_features_q):]
    tsfel_features_t.index = data.index[len(data) - len(tsfel_features_t):]

    print("\n" + "=" * 60)
    print("[TSFEL] АНАЛИЗ И ОЧИСТКА АВТОМАТИЧЕСКИХ ПРИЗНАКОВ")
    print("=" * 60)

    corr_matrix_q = tsfel_features_q.corr().abs()
    upper_q = corr_matrix_q.where(np.triu(np.ones(corr_matrix_q.shape), k=1).astype(bool))
    to_drop_q = [column for column in upper_q.columns if any(upper_q[column] > 0.95)]
    tsfel_features_q.drop(columns=to_drop_q, inplace=True)

    corr_matrix_t = tsfel_features_t.corr().abs()
    upper_t = corr_matrix_t.where(np.triu(np.ones(corr_matrix_t.shape), k=1).astype(bool))
    to_drop_t = [column for column in upper_t.columns if any(upper_t[column] > 0.95)]
    tsfel_features_t.drop(columns=to_drop_t, inplace=True)

    print(f"--- [Успех] Избыточные фичи удалены. Исключено: Q — {len(to_drop_q)}, T — {len(to_drop_t)} ---")
    print(f"--- Оставлено уникальных фичей TSFEL: Q — {tsfel_features_q.shape[1]}, T — {tsfel_features_t.shape[1]} ---")

    tsfel_features_q.index = data.index[len(data) - len(tsfel_features_q):]
    tsfel_features_t.index = data.index[len(data) - len(tsfel_features_t):]

    temp_analysis_df = pd.concat([data['TEC_Q_Aver'], tsfel_features_q, tsfel_features_t], axis=1).dropna()

    print("\nСПИСОК ВЫБРАННЫХ ПРИЗНАКОВ TSFEL И ИХ ЗНАЧИМОСТЬ ДЛЯ ТЭЦ:")
    print("-" * 75)
    print(f"{'Название извлеченного признака':<42} | {'Связь с расходом':<16} | {'Значимость':<20}")
    print("-" * 75)

    for idx in range(1, temp_analysis_df.shape[1]):
        col_name = temp_analysis_df.columns[idx]

        feature_vector = temp_analysis_df.iloc[:, idx]

        target_corr = temp_analysis_df.iloc[:, 0].corr(feature_vector)

        if np.isnan(target_corr):
            target_corr = 0.0

        abs_corr = abs(target_corr)
        if abs_corr >= 0.7:
            reason = "🔥 Критическая"
        elif 0.4 <= abs_corr < 0.7:
            reason = "⚡ Высокая (Инерция)"
        else:
            reason = "📉 Слабая (Микро-шум)"

        short_name = col_name if len(col_name) <= 42 else f"{col_name[:39]}..."
        print(f"{short_name:<42} | {target_corr:<16.4f} | {reason:<20}")

    print("-" * 75)
    print("============================================================\n")

    data_clean = pd.concat([data, tsfel_features_q, tsfel_features_t], axis=1)

    data_clean.dropna(inplace=True)

    data_clean.index = pd.to_datetime(data_clean.index)
    data_clean.sort_index(inplace=True)

    print(f"[INFO] Датасет успешно собран и очищен от NaN. Итоговый размер: {data_clean.shape}")

    # 5. Переводим в TimeSeries Darts
    target_raw_series = TimeSeries.from_dataframe(data_clean, value_cols='TEC_Q_Aver', freq='D')

    covariate_cols = ['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos']
    covariate_cols += [col for col in data_clean.columns if 'tsfel_' in col]

    covariates_raw_series = TimeSeries.from_dataframe(data_clean, value_cols=covariate_cols, freq='D')

    # Нормализация
    scaler_target = Scaler()
    scaler_covariates = Scaler()
    target_scaled = scaler_target.fit_transform(target_raw_series)
    covariates_scaled = scaler_covariates.fit_transform(covariates_raw_series)

    split_date = data_clean.index[train_size]
    train_target_scaled, _ = target_scaled.split_before(split_date)

    # === БЛОК OPTUNA ОПТИМИЗАЦИИ ===
    if os.path.exists(params_json_path):
        print(f"--- Загрузка чистой конфигурации TiDE... ---")
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
    else:
        print("\nЗапуск Optuna для оптимизации чистой архитектуры Google TiDE...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_darts_tide_and_tsfel_search(
            trial, train_target_scaled, covariates_scaled, target_scaled, scaler_target, target_raw_series, split_date,
            n_out
        ), n_trials=100)

        best_params = study.best_params
        with open(params_json_path, 'w') as f:
            json.dump(best_params, f)

    print(f"\n[TiDE CLEAN SUCCESS] Параметры: {best_params}")

    early_stopper = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.0001, mode="min")

    model_tide = TiDEModel(
        input_chunk_length=14,  # Окно истории, из которого TiDE сам извлечет фичи!
        output_chunk_length=n_out,  # Наш итоговый горизонт 14 дней
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
            "callbacks": [early_stopper]
        }
    )

    model_tide.fit(series=train_target_scaled, future_covariates=covariates_scaled)

    print("Генерация многошаговых прогнозов Google TiDE + TSFEL на тестовом периоде...")
    pred_series_scaled = model_tide.historical_forecasts(
        series=target_scaled,
        future_covariates=covariates_scaled, # ИСПРАВЛЕНО: future вместо past
        start=split_date,
        forecast_horizon=n_out,
        stride=1,
        overlap_end=False, # Жесткая защита от краевых NaN на конце файла
        retrain=False,
        verbose=False
    )


    # Денормализуем прогноз к реальному масштабу Гкал ТЭЦ
    pred_series = scaler_target.inverse_transform(pred_series_scaled)
    actual_test_slice = target_raw_series.slice_intersect(pred_series)

    # === ИСПРАВЛЕННЫЙ БЕЗОПАСНЫЙ СРЕЗ МАТРИЦ ===
    # Извлекаем чистые NumPy массивы напрямую через метод .values()
    # и явно разворачиваем их в 2D форму (строки — дни теста, столбцы — n_out шагов горизонта)
    pred_series = scaler_target.inverse_transform(pred_series_scaled)
    actual_test_slice = target_raw_series.slice_intersect(pred_series)

    # 1. Извлекаем чистые одномерные массивы (убираем оси через .flatten)
    raw_true = actual_test_slice.values().flatten()
    raw_pred = pred_series.values().flatten()

    # 2. Если Darts выдал 1 столбец вместо 14, мы преобразуем скользящую «колбасу»
    # в честную двумерную матрицу горизонта методом скользящего окна
    X_true_list, X_pred_list = [], []

    # Нарезаем ряды на 14 дней вперед для каждой доступной точки
    for step in range(n_out):
        X_true_list.append(raw_true[step: len(raw_true) - n_out + step])
        X_pred_list.append(raw_pred[step: len(raw_pred) - n_out + step])

    y_true = np.column_stack(X_true_list)
    y_pred = np.column_stack(X_pred_list)

    print(f"[INFO] Сборка матриц успешно завершена. Итоговый размер: {y_true.shape}")

    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ГОРИЗОНТА ДЛЯ GOOGLE TiDE + TSFEL ===")
    print("=" * 50)

    metrics_list = []
    for step in range(n_out):
        step_true = y_true[:, step]
        step_pred = y_pred[:, step]

        # Фильтруем возможные случайные пропуски на краях TSFEL окон
        valid_idx = ~np.isnan(step_true) & ~np.isnan(step_pred)
        step_true = step_true[valid_idx]
        step_pred = step_pred[valid_idx]

        if len(step_true) == 0:
            print(f"День {step + 1:<2} -> Данные на тесте отсутствуют (срез NaN)")
            metrics_list.append([0.0, 0.0, 0.0, 0.0])
            continue

        step_mae = metrics.mean_absolute_error(step_true, step_pred)
        step_mse = metrics.mean_squared_error(step_true, step_pred)
        step_mape = metrics.mean_absolute_percentage_error(step_true, step_pred) * 100
        step_r2 = metrics.r2_score(step_true, step_pred)

        print(f"День {step + 1:<2} -> MAE: {step_mae:<8.4f} | R2: {step_r2:<8.4f}")
        metrics_list.append([step_mae, step_mse, step_mape, step_r2])

    print("=" * 50)
    return np.hstack(metrics_list)