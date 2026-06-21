import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras import backend as K

from utils import optuna_mlp_step_search

def train_mlp_direct_multistep(data_path, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_mlp')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, 'Qpred_MLP_LAG_model_step_')

    # Загрузка данных и создание базового календаря
    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data.set_index('Date', inplace=True)

    def build_model(input_dim, dense_units):
        model = Sequential([
            Dense(dense_units, activation='relu', input_shape=(input_dim,)),
            Dense(int(dense_units / 2), activation='relu'),
            Dense(1, dtype='float32')
        ])
        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape'])
        return model

    # Цикл по шагам прогнозирования вперед
    for i in range(0, n_out, 1):
        step_num = i + 1
        print(f"\n" + "=" * 50)
        print(f"=== Настройка и обучение MLP: Шаг {step_num} из {n_out} ===")
        print("=" * 50)

        current_checkpoint = f"{checkpoint_base}{i}.keras"
        params_json_path = f"{checkpoint_base}{i}_params.json"

        # Проверяем наличие чекпоинта
        if os.path.exists(current_checkpoint) and os.path.exists(params_json_path):
            print(f"--- Шаг {step_num}: Найден чекпоинт. Загрузка параметров... ---")
            with open(params_json_path, 'r') as f:
                best_params = json.load(f)

            best_window = best_params['window_size']
            dense_units = best_params['n_units_dense']

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

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(x[:train_size])
            X_test_scaled = scaler.transform(x[train_size:])

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y[:train_size])
            y_test_scaled = scaler_y.transform(y[train_size:])
            y_test = y[train_size:]

            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
        else:
            print(f"--- Шаг {step_num}: Чекпоинт не найден. Запуск подбора Optuna... ---")

            # Запуск подбора
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: optuna_mlp_step_search(trial, data, step_idx=i, train_size=train_size),
                           n_trials=100)

            best_params = study.best_params
            best_window = best_params['window_size']
            dense_units = best_params['n_units_dense']
            print(f"[MLP STEP {step_num}] Лучшее окно: {best_window}, Нейронов в слое: {dense_units}")

            with open(params_json_path, 'w') as f:
                json.dump(best_params, f)

            # Формируем финальные признаки на основе лучшего окна
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

            model = build_model(X_train_scaled.shape[1], dense_units)
            current_epochs = 1000

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        current_batch_size = 128

        # Подготовка tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_scaled))
        train_dataset = (train_dataset
                         .shuffle(buffer_size=len(X_train_scaled))
                         .batch(current_batch_size)
                         .prefetch(tf.data.AUTOTUNE))

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test_scaled))
        val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

        # Финальное обучение модели
        model.fit(train_dataset,
                  epochs=current_epochs,
                  validation_data=val_dataset,
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=0,
                  shuffle=False)

        # Прогноз
        yhat_scaled = model.predict(val_dataset, verbose=0)
        yhat = scaler_y.inverse_transform(yhat_scaled)

        # Расчет метрик
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat) * 100
        R2_test = metrics.r2_score(y_test, yhat)

        print(f"Результат MLP шага {step_num} -> MAE: {MAE_test:.4f} | R2: {R2_test:.4f}")

        results_list.append(yhat)
        metrics_list.append([MAE_test, MSE_test, MAPE_test, R2_test])

    return np.hstack(metrics_list)