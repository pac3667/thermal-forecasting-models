import os
import json
import joblib
import pandas as pd
import numpy as np
import optuna
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from utils import optuna_rf_step_search


def train_rfr_direct_multistep(data_path, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []

    # Настраиваем уникальную директорию для чекпоинтов случайного леса
    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_rfr')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, 'Qpred_RFR_LAG_model_step_')

    # Загрузка данных и создание базового календаря
    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data.set_index('Date', inplace=True)

    # Цикл по шагам прогнозирования вперед
    for i in range(0, n_out, 1):
        step_num = i + 1
        print(f"\n" + "=" * 50)
        print(f"=== Настройка и обучение Random Forest: Шаг {step_num} из {n_out} ===")
        print("=" * 50)

        current_checkpoint = f"{checkpoint_base}{i}.joblib"
        params_json_path = f"{checkpoint_base}{i}_params.json"

        # Проверяем наличие готового сохраненного состояния на диске
        if os.path.exists(current_checkpoint) and os.path.exists(params_json_path):
            print(f"--- Шаг {step_num}: Найден чекпоинт. Восстановление параметров... ---")
            with open(params_json_path, 'r') as f:
                best_params = json.load(f)

            best_window = best_params['window_size']
            print(
                f"[LOADED] Окно лагов: {best_window}, Деревьев: {best_params['n_estimators']}, Глубина: {best_params['max_depth']}")

            # Формируем признаки под восстановленное окно истории
            df_final = pd.DataFrame({
                'T': data['T'], 'TEC_Q_Aver': data['TEC_Q_Aver'], 'Year': data['Year'],
                'Month_sin': data['Month_sin'], 'Month_cos': data['Month_cos'],
                'T_mean': data['T'].rolling(window=best_window).mean().shift(-step_num),
                'T_lag': data['T'].shift(-step_num),
                'Q_rolling': data['TEC_Q_Aver'].rolling(window=best_window).mean(),
                'Target_Q': data['TEC_Q_Aver'].shift(-step_num)
            }).dropna()

            x = df_final[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
            y = df_final[['Target_Q']].values.ravel()

            X_train, X_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Загружаем бинарный слепок обученной модели
            model = joblib.load(current_checkpoint)
            print(f"--- Шаг {step_num}: Модель успешно загружена. Обучение пропущено. ---")

        else:
            print(f"--- Шаг {step_num}: Чекпоинт не найден. Запуск подбора Optuna... ---")

            # 1. Запуск Optuna подбора для текущего шага
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: optuna_rf_step_search(trial, data, step_idx=i, train_size=train_size),
                           n_trials=50)

            best_params = study.best_params
            best_window = best_params['window_size']
            print(
                f"[RF OPTUNA SUCCESS] Лучшее окно: {best_window}, Деревьев: {best_params['n_estimators']}, Глубина: {best_params['max_depth']}")

            # Сохраняем словарь подобранных гиперпараметров в JSON
            with open(params_json_path, 'w') as f:
                json.dump(best_params, f)

            # 2. Формируем финальные признаки на основе лучшего окна
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
            y = df_final[['Target_Q']].values.ravel()

            X_train, X_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 3. Финальное обучение модели RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Сохраняем объект сериализованной модели на диск
            joblib.dump(model, current_checkpoint)
            print(f"[SAVED] Модель леса шага {step_num} сохранена в {current_checkpoint}")

        # Прогноз (общая зона для веток создания и загрузки модели)
        yhat = model.predict(X_test_scaled).reshape(-1, 1)

        # Расчет метрик качества
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat) * 100
        R2_test = metrics.r2_score(y_test, yhat)

        print(f"Результат RF шага {step_num} -> MAE: {MAE_test:.4f} | R2: {R2_test:.4f}")

        results_list.append(yhat)
        metrics_list.append([MAE_test, MSE_test, MAPE_test, R2_test])

    return np.hstack(metrics_list)