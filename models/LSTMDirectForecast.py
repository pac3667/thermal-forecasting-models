import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Dense, LSTM
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


def train_lstm_direct_multistep(data, checkpoint_dir, n_out, train_size):
    results_list = []
    metrics_list = []
    checkpoint_dir = checkpoint_dir + 'multistep'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = checkpoint_dir + '/Qpred_LSTM_LAG_model_step_'
    data = pd.read_csv(data, delimiter=';', parse_dates=['Date'], dayfirst=True)
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Q_rolling_3'] = data['TEC_Q_Aver'].rolling(window=3).mean()
    data.set_index('Date', inplace=True)

    df = DataFrame(data)
    df['T_rolling_3'] = df['T'].rolling(window=3).mean()

    cols, names ,agg = list(), list(), list()
    for i in range(1, n_out + 1):
        cols.append(df[['T']].shift(-i))
        cols.append(df[['T_rolling_3']].shift(-i))
        cols.append(df[['TEC_Q_Aver']].shift(-i))
        names += [f'T_lag{i}', f'T_mean{i}', f'Q_lag{i}']

    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    data_with_lag = pd.concat([data, agg], axis=1)
    data_with_lag.dropna(inplace=True)
    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(100, input_shape=input_shape))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])
        return model

    for i in range(0, n_out, 1):
        print(f"\n=== Step training {i + 1} ===")

        x = data_with_lag[['T','TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', f'T_mean{i+1}', f'T_lag{i+1}', 'Q_rolling_3']].values
        y = data_with_lag.loc[:, [f'Q_lag{i+1}']].values
        X_train, X_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

        current_checkpoint = checkpoint_base + str(i) + '.keras'

        if os.path.exists(current_checkpoint):
            print(f"--- Step {i + 1}: File found. Loading and RESUMING training... ---")
            from keras.models import load_model
            model = load_model(current_checkpoint)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
        else:
            print(f"--- Step {i + 1}: File not found. Starting NEW training... ---")
            current_epochs = 1000

            if i > 0:
                prev_model_path = checkpoint_base + str(i - 1) + '.keras'
                if os.path.exists(prev_model_path):
                    model.load_weights(prev_model_path)
                    print(f"Weights initialized from step {i}")

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        model.fit(X_train_scaled, y_train_scaled,
                  epochs=current_epochs,
                  batch_size=32,
                  validation_data=(X_test_scaled, y_test_scaled),
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=2,
                  shuffle=False)

        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled)
        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)*100
        R2_test = metrics.r2_score(y_test, yhat)
        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])
    return np.hstack(metrics_list)