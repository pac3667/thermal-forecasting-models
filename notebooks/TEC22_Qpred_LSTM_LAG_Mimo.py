import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
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

t_lags = [f'T_lag{i}' for i in range(1, n_out + 1)]
t_means = [f'T_mean{i}' for i in range(1, n_out + 1)]
q_targets = [f'Q_lag{i}' for i in range(1, n_out + 1)]

features = ['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'Q_rolling_3']

x = data_with_lag[features].values
y = data_with_lag[q_targets].values

n = 3258
X_train, X_test = x[:n], x[n:]
y_train, y_test = y[:n], y[n:]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                    min_delta=0.0001)

model = Sequential()
model.add(LSTM(150, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(LSTM(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_out, activation='relu'))

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])


history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=2500,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    verbose=2,
    shuffle=False,
    callbacks=[early_stop_callback]
)

yhat_scaled = model.predict(X_test_scaled)
yhat = scaler_y.inverse_transform(yhat_scaled)

print(f"Общая MAE по всем 14 шагам: {mean_absolute_error(y_test, yhat):.2f}")

step_metrics = []

for i in range(n_out):
    mae = mean_absolute_error(y_test[:, i], yhat[:, i])
    r2 = r2_score(y_test[:, i], yhat[:, i])
    step_metrics.append(mae)
    print(f"Шаг {i+1}: MAE = {mae:.2f}, R2 = {r2:.3f}")

# Визуализация роста ошибки
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_out + 1), step_metrics, color='skyblue', edgecolor='navy')
plt.plot(range(1, n_out + 1), step_metrics, color='red', marker='o') # Линия тренда ошибки

plt.title('Рост ошибки прогноза в зависимости от горизонта (шага)')
plt.xlabel('День прогноза (t + n)')
plt.ylabel('Средняя абсолютная ошибка (MAE)')
plt.xticks(range(1, n_out + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()