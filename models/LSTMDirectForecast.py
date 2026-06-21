import json
import os

import optuna
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Dense, LSTM
from keras.models import load_model
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from utils import create_lstm_windows_for_step, optuna_lstm_step_search


def train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep_lstm')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, 'Qpred_LSTM_LAG_model_step_')

    data = pd.read_csv(data_path, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data.set_index('Date', inplace=True)

    def build_model(input_shape, lstm_units):
        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=input_shape))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, dtype='float32'))
        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape'])
        return model

    for i in range(0, n_out, 1):
        step_num = i + 1
        print(f"\n" + "=" * 50)
        print(f"=== Настройка и обучение LSTM: Шаг {step_num} из {n_out} ===")
        print("=" * 50)

        current_checkpoint = f"{checkpoint_base}{i}.keras"
        params_json_path = f"{checkpoint_base}{i}_params.json"

        if os.path.exists(current_checkpoint) and os.path.exists(params_json_path):
            print(f"--- Шаг {step_num}: Найден чекпоинт. Загрузка параметров... ---")
            with open(params_json_path, 'r') as f:
                best_params = json.load(f)

            best_window = best_params['window_size']
            lstm_units = best_params['n_units_lstm']

            X_win, y_win = create_lstm_windows_for_step(data, step_num, best_window)
            X_train, X_test = X_win[:train_size], X_win[train_size:]
            y_train, y_test = y_win[:train_size], y_win[train_size:]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
        else:
            print(f"--- Шаг {step_num}: Чекпоинт не найден. Запуск подбора Optuna... ---")

            pruner = optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=10)
            study = optuna.create_study(direction='minimize', pruner=pruner)
            study.optimize(lambda trial: optuna_lstm_step_search(trial, data, step_idx=i, train_size=train_size),
                           n_trials=100)

            best_params = study.best_params
            best_window = best_params['window_size']
            lstm_units = best_params['n_units_lstm']
            print(f"[УСПЕХ] Лучшее окно истории: {best_window} дней, LSTM нейронов: {lstm_units}")

            with open(params_json_path, 'w') as f: json.dump(best_params, f)

            X_win, y_win = create_lstm_windows_for_step(data, step_num, best_window)
            X_train, X_test = X_win[:train_size], X_win[train_size:]
            y_train, y_test = y_win[:train_size], y_win[train_size:]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]), lstm_units)
            current_epochs = 1000

            if i > 0:
                prev_params_path = f"{checkpoint_base}{i - 1}_params.json"
                if os.path.exists(prev_params_path):
                    with open(prev_params_path, 'r') as f:
                        prev_params = json.load(f)
                    if prev_params['window_size'] == best_window and prev_params['n_units_lstm'] == lstm_units:
                        prev_model_path = f"{checkpoint_base}{i - 1}.keras"
                        if os.path.exists(prev_model_path):
                            model.load_weights(prev_model_path)
                            print(f"--> Успешно унаследованы веса от шага {i}")

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        current_batch_size = 128

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_scaled))
        train_dataset = (train_dataset
                         .shuffle(buffer_size=len(X_train_scaled))
                         .batch(current_batch_size)
                         .prefetch(tf.data.AUTOTUNE))

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test_scaled))
        val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

        model.fit(train_dataset,
                  epochs=current_epochs,
                  validation_data=val_dataset,
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=0,
                  shuffle=False)

        yhat_scaled = model.predict(val_dataset, verbose=0)
        yhat = scaler_y.inverse_transform(yhat_scaled)
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)*100
        R2_test = metrics.r2_score(y_test, yhat)

        print(f"Финальный результат шага {step_num} -> MAE: {MAE_test:.4f} | R2: {R2_test:.4f}")

        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])
    return np.hstack(metrics_list)