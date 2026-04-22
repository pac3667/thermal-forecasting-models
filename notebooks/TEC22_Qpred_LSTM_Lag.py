import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

data = pd.read_csv(r'C:\Users\guryanov\PycharmProjects\TEC_Qpred\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['Q_rolling_3'] = data['TEC_Q_Aver'].rolling(window=3).mean()
data.set_index('Date', inplace=True)

n_out = 14

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

checkpoint_base = r'C:\Users\guryanov\PycharmProjects\TEC_Qpred\checkpoint\TEC22_LSTM_LAG_model_step_'

metrics_list = []
results_list = np.array(())
for i in range(0, n_out, 1):
    print(f"\n=== Обучение шага {i + 1} ===")

    x = data_with_lag[['T','TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', f'T_mean{i+1}', f'T_lag{i+1}', 'Q_rolling_3']].values
    y = data_with_lag.loc[:, [f'Q_lag{i+1}']].values
    X_train, X_test = x[:3258], x[3258:]
    y_train, y_test = y[:3258], y[3258:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

    if i > 0:
        prev_model_path = checkpoint_base + str(i - 1) + '.keras'
        try:
            model.load_weights(prev_model_path)
            print(f"Веса загружены из шага {i}")
        except:
            print("Предыдущие веса не найдены, учимся с нуля")

    # Настраиваем сохранение текущего шага
    current_checkpoint = checkpoint_base + str(i) + '.keras'
    checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    # model.load_weights(r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Npred_LSTM.keras')

    print(X_train_scaled)
    print(y_train_scaled)

    model.fit(X_train_scaled, y_train_scaled,
              epochs=1500,
              batch_size=32,
              validation_data=(X_test_scaled, y_test_scaled),
              callbacks=[early_stop_callback, checkpoint_callback],
              verbose=2)
    yhat_scaled = model.predict(X_test_scaled)
    yhat = scaler_y.inverse_transform(yhat_scaled)
    print('yhat_scaled', yhat_scaled)
    print('yhat', yhat)
    #yhat= model.predict(X_test_scaled)

    MAE_test = metrics.mean_absolute_error(y_test, yhat)
    MSE_test = metrics.mean_squared_error(y_test, yhat)
    MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)
    R2_test = metrics.r2_score(y_test, yhat)
    print(f"MAE_test: {MAE_test:.2f}")
    print(f"MSE_test: {MSE_test:.2f}")
    print(f"MAPE_test: {MAPE_test:.2f}")
    print(f"R2_test: {R2_test:.2f}")
    metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])

    if i == 0:
        results_list = yhat
    else:
        results_list = np.hstack((results_list, yhat))

    # plot history
