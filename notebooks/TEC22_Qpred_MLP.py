import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from keras import models, layers
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Flatten, Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn import datasets, linear_model, metrics, model_selection
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(1)

mpl.rcParams['figure.figsize'] = (25, 10)

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

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(Flatten(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))

model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae', metrics=['mse'])

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                    min_delta=0.1)

history_MNN = model.fit(X_train_scaled,
                    y_train,
                    epochs=1000,
                    batch_size=30,
                    validation_split=0.2,
                    validation_batch_size=30,
                    callbacks=early_stop_callback,
                    verbose="auto")

y_test_pred = model.predict(X_test_scaled)

MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")