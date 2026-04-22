import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model, metrics, model_selection

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

params = {
    "n_estimators": 35,
    "max_features": 3,
    "random_state": 1,
}

model_RFR = RandomForestRegressor(**params).fit(X_train_scaled,y_train)

y_test_pred = model_RFR.predict(X_test_scaled)
MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

scores_train =[]
scores_test =[]
calc_range = 250
for k in range(1, calc_range):
    print(k)
    rfc = RandomForestRegressor(n_estimators=k, max_features = 3, random_state=1, min_weight_fraction_leaf=0.0)
    rfc.fit(X_train_scaled, y_train)
    y_pred_test = rfc.predict(X_test_scaled)
    scores_test.append(mean_absolute_error(y_test, y_pred_test))
    y_pred_train = rfc.predict(X_train_scaled)
    scores_train.append(mean_absolute_error(y_train, y_pred_train))
plt.plot(range(1, calc_range), scores_test, label="Test Set MSE")
plt.plot(range(1, calc_range), scores_train, label="Train Set MSE")
plt.xlabel('Value of n_estimators for RandomForestRegressor')
plt.ylabel('Testing Accuracy')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()