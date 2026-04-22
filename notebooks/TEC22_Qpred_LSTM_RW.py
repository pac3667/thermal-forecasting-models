import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras.src.layers import Dense, LSTM, Normalization
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tukeyplot import results
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
from tensorflow.keras.callbacks import ModelCheckpoint

data = pd.read_csv(r'C:\Users\Andrey\PyCharmMiscProject\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

data.set_index('Date', inplace=True)

x = data.loc[:, ['T', 'Year', 'Month_sin', 'Month_cos']]
x = x.values
y = data.loc[:, ['TEC_Q_Aver']]
y = y.values
print(y)

n=3258

X_train = x[:n]
X_test = x[n: ]
y_train = y[:n]
y_test = y[n: ]

def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        # Берем окно признаков (например, за 24 часа)
        X.append(x_data[i : i + window_size])
        # Целевое значение — следующее значение после окна
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

window_size = 7
X_train_win, y_train_win = create_windows(X_train_scaled, y_train_scaled, window_size)
X_test_win, y_test_win = create_windows(X_test_scaled, y_test_scaled, window_size)

checkpoint_filepath = r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Qpred_LSTM_RW.keras'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True, min_delta=0.0001)

model = Sequential()
model.add(LSTM(50, input_shape=(X_train_win.shape[1], X_train_win.shape[2])))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

#model.load_weights(r'C:\Users\Andrey\PyCharmMiscProject\checkpoint\TEC22_Qpred_LSTM.keras')

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'r2_score'])

# fit network
history = model.fit(X_train_win, y_train_win, epochs=2500, batch_size=32, validation_data=(X_test_win, y_test_win), verbose=2, shuffle=False, callbacks = [early_stop_callback, model_checkpoint_callback])

y_test_pred_scaled = model.predict(X_test_win)
yhat = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_win)

# plot history
'''plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.grid()
plt.legend()
plt.show()'''

MAE_test = metrics.mean_absolute_error(y_test_actual, yhat)
MSE_test = metrics.mean_squared_error(y_test_actual, yhat)
MAPE_test = metrics.mean_absolute_percentage_error(y_test_actual, yhat)
R2_test = metrics.r2_score(y_test_actual, yhat)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")


plt.figure(figsize=(20, 8))
test_dates = data.index[n + window_size:] # Даты для тестового окна
plt.plot(test_dates, y_test_actual, label='Факт')
plt.plot(test_dates, yhat, label='Прогноз LSTM (Window=24)')
plt.legend()
plt.grid(True)
plt.show()


'''def evaluate_window_size(w_size, X_tr_scaled, y_tr, X_te_scaled, y_te):
    # Создаем окна
    X_train_w, y_train_w = create_windows(X_tr_scaled, y_tr, w_size)
    X_test_w, y_test_w = create_windows(X_te_scaled, y_te, w_size)

    # Строим модель
    model = Sequential([
        LSTM(50, input_shape=(X_train_w.shape[1], X_train_w.shape[2])),
        Dense(25),
        Dense(1)
    ])
    model.compile(loss='mae', optimizer='adam')

    # Обучаем (меньше эпох для теста)
    model.fit(X_train_w, y_train_w, epochs=500, batch_size=64, verbose=0, shuffle=False, callbacks = [early_stop_callback])

    # Оценка
    preds = model.predict(X_test_w)
    mae = metrics.mean_absolute_error(y_test_w, preds)
    return mae


# Список окон для анализа: 6ч, 12ч, 24ч (сутки), 48ч (двое суток), 168ч (неделя)
window_sizes = [6, 12, 24, 48, 72]
mae_results = []

for w in window_sizes:
    mae = evaluate_window_size(w, X_train_scaled, y_train, X_test_scaled, y_test)
    mae_results.append(mae)
    print(f"Window: {w} hours | MAE: {mae:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(window_sizes, mae_results, marker='o', linestyle='--', color='b')
plt.title('Влияние размера окна на ошибку модели (MAE)')
plt.xlabel('Размер окна (часы)')
plt.ylabel('MAE на тестовой выборке')
plt.grid(True)
plt.show()'''