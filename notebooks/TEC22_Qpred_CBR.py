import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

tf.random.set_seed(42)

data = pd.read_csv(r'C:\Users\guryanov\PycharmProjects\TEC_Qpred\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)

data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

data.set_index('Date', inplace=True)

x = data.loc[:, ['T', 'Year', 'Month_sin', 'Month_cos']]
x = x.values
y = data.loc[:, ['TEC_Q_Aver']]
y = y.values

n=3258

X_train = x[:n]
X_test = x[n: ]
y_train = y[:n]
y_test = y[n: ]

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

model = CatBoostRegressor(iterations=2000,
                          depth=6,
                          learning_rate=0.05,
                          loss_function='MAE',
                          verbose=2,
                          random_seed=42,
                          use_best_model=True)

model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), early_stopping_rounds=50)

params = model.get_all_params()
print(params.get('random_seed'))

y_test_pred = model.predict(X_test_scaled).reshape(-1, 1)
print(y_test_pred)
yhat = scaler_y.inverse_transform(y_test_pred)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# plot history
'''plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.grid()
plt.legend()
plt.show()'''

MAE_test = metrics.mean_absolute_error(y_test, yhat)
MSE_test = metrics.mean_squared_error(y_test, yhat)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)
R2_test = metrics.r2_score(y_test, yhat)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")


'''plt.figure(figsize=(20, 8))
test_dates = data.index[n + window_size:] # Даты для тестового окна
plt.plot(test_dates, y_test_actual, label='Факт')
plt.plot(test_dates, yhat, label='Прогноз LSTM (Window=24)')
plt.legend()
plt.grid(True)
plt.show()

learning_rates = [0.01, 0.05, 0.1, 0.2]
depths = [4, 6, 8, 10]

results = []

# Цикл перебора
for d in depths:
    row = []
    for lr in learning_rates:
        test_model = CatBoostRegressor(iterations=1000, # 1000 достаточно для теста
                                      depth=d,
                                      learning_rate=lr,
                                      loss_function='MAE',
                                      verbose=0,
                                      early_stopping_rounds=50)
        test_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test))
        preds = test_model.predict(X_test_scaled)
        mae = metrics.mean_absolute_error(y_test, preds)
        row.append(mae)
    results.append(row)

# Визуализация через Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(results, annot=True, fmt=".2f",
            xticklabels=learning_rates,
            yticklabels=depths,
            cmap='YlGnBu')
plt.xlabel('Learning Rate')
plt.ylabel('Depth')
plt.title('Влияние параметров на MAE (чем меньше значение, тем лучше)')
plt.show()'''