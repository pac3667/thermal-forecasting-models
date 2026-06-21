import os
import json
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from utils import optuna_single_step_search


def train_cbr_direct_multistep(data_path, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []

    # Корректно настраиваем директорию для весов бустинга
    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_cbr')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, 'Qpred_CBR_LAG_model_step_')

    # Первичная загрузка данных и генерация общих признаков календаря
    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data.set_index('Date', inplace=True)

    # Запускаем цикл по каждому шагу прогнозирования
    for i in range(0, n_out, 1):
        step_num = i + 1
        print(f"\n" + "=" * 50)
        print(f"=== Настройка и обучение CatBoost: Шаг {step_num} из {n_out} ===")
        print("=" * 50)

        current_checkpoint = f"{checkpoint_base}{i}.cbr"
        params_json_path = f"{checkpoint_base}{i}_params.json"

        # Проверяем, есть ли уже готовый чекпоинт на диске
        if os.path.exists(current_checkpoint) and os.path.exists(params_json_path):
            print(f"--- Шаг {step_num}: Найден чекпоинт. Загрузка параметров из JSON... ---")
            with open(params_json_path, 'r') as f:
                best_params = json.load(f)

            best_window = best_params['window_size']
            print(f"[LOADED] Восстановлено окно лагов: {best_window}")

            # Строим датасет под сохраненное окно истории
            df_final = pd.DataFrame({
                'T': data['T'], 'TEC_Q_Aver': data['TEC_Q_Aver'], 'Year': data['Year'],
                'Month_sin': data['Month_sin'], 'Month_cos': data['Month_cos'],
                'T_mean': data['T'].rolling(window=best_window).mean().shift(-step_num),
                'T_lag': data['T'].shift(-step_num),
                'Q_rolling': data['TEC_Q_Aver'].rolling(window=best_window).mean(),
                'Target_Q': data['TEC_Q_Aver'].shift(-step_num)
            }).dropna()

            x = df_final[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
            y = df_final[['Target_Q']].values

            X_train, X_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            # Просто загружаем готовую обученную модель с диска
            model = CatBoostRegressor()
            model.load_model(current_checkpoint)
            print(f"--- Шаг {step_num}: Модель успешно загружена. Пропускаем обучение. ---")

        else:
            print(f"--- Шаг {step_num}: Чекпоинт не найден. Запуск подбора Optuna... ---")

            # 1. Запускаем индивидуальный подбор Optuna под текущий шаг
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: optuna_single_step_search(trial, data, step_idx=i, train_size=train_size),
                           n_trials=50)

            best_params = study.best_params
            best_window = best_params['window_size']
            print(f"[OPTUNA SUCCESS] Лучший размер окна лагов: {best_window}")

            # Сохраняем лучшие параметры в JSON-файл
            with open(params_json_path, 'w') as f:
                json.dump(best_params, f)

            # 2. Строим финальный набор признаков под выбранное лучшее окно
            t_lag = data['T'].shift(-step_num)
            t_mean = data['T'].rolling(window=best_window).mean().shift(-step_num)
            q_lag = data['TEC_Q_Aver'].shift(-step_num)
            q_rolling = data['TEC_Q_Aver'].rolling(window=best_window).mean()

            df_final = pd.DataFrame({
                'T': data['T'],
                'TEC_Q_Aver': data['TEC_Q_Aver'],
                'Year': data['Year'],
                'Month_sin': data['Month_sin'],
                'Month_cos': data['Month_cos'],
                'T_mean': t_mean,
                'T_lag': t_lag,
                'Q_rolling': q_rolling,
                'Target_Q': q_lag
            }).dropna()

            x = df_final[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
            y = df_final[['Target_Q']].values

            X_train, X_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Масштабирование
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            model = CatBoostRegressor(iterations=2000,
                                      depth=best_params['depth'],
                                      learning_rate=best_params['lr'],
                                      loss_function='MAE',
                                      task_type='GPU',  # <-- Активируем CUDA вычисления
                                      verbose=0)

            model.fit(X_train_scaled, y_train_scaled,
                      eval_set=(X_test_scaled, y_test_scaled),
                      early_stopping_rounds=250,
                      use_best_model=True)

            # Сразу сохраняем обученную модель в файл .cbr
            model.save_model(current_checkpoint)
            print(f"[SAVED] Модель шага {step_num} сохранена в {current_checkpoint}")

        # Секция прогнозирования и расчета метрик (выполняется как при обучении, так и при загрузке)
        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))

        # Сбор метрик по текущему шагу
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat) * 100
        R2_test = metrics.r2_score(y_test, yhat)

        print(f"Результат шага {step_num} -> MAE: {MAE_test:.4f} | R2: {R2_test:.4f}")

        results_list.append(yhat)
        metrics_list.append([MAE_test, MSE_test, MAPE_test, R2_test])

    return np.hstack(metrics_list)