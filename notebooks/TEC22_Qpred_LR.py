import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

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

model_LR = linear_model.LinearRegression(tol=.0000001)
model_LR.fit(X_train_scaled, y_train)
y_test_pred = model_LR.predict(X_test_scaled)

MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

poly = PolynomialFeatures(3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
model_LR.fit(X_train_poly, y_train)

X_test_poly = poly.transform(X_test_scaled)
y_test_poly_pred = model_LR.predict(X_test_poly)

MAE_test = metrics.mean_absolute_error(y_test, y_test_poly_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_poly_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_poly_pred)
R2_test = metrics.r2_score(y_test, y_test_poly_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")


scores_train =[]
scores_test =[]
calc_range = 5
for k in range(1, calc_range):
    print(k)
    poly = PolynomialFeatures(k, include_bias=False)
    poly_df = poly.fit_transform(X_train_scaled)
    model_LR.fit(poly_df, y_train)
    X_new_poly_test = poly.transform(X_test_scaled)
    y_pred_test = model_LR.predict(X_new_poly_test)
    scores_test.append(mean_absolute_error(y_test, y_pred_test))
    X_new_poly_train = poly.transform(X_train_scaled)
    y_pred_train = model_LR.predict(X_new_poly_train)
    scores_train.append(mean_absolute_error(y_train, y_pred_train))
plt.plot(range(1, calc_range), scores_test, label="Test Set MSE")
plt.plot(range(1, calc_range), scores_train, label="Train Set MSE")
plt.xlabel('Value of n_estimators for LinearRegression')
plt.ylabel('Testing Accuracy')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


'''with open("model_TEC14_Qpred_LRwP.pkl", 'wb') as file:
    pickle.dump(model_LR, file)

plt.scatter(X_test.iloc[:, 2:3], y_test, color='g', linewidth = 2, label='Train Data')
plt.scatter(X_test.iloc[:, 2:3], y_test_pred, color='y', linewidth = 2, label='Linear Model Approximation')
plt.scatter(X_test.iloc[:, 2:3], y_test_poly_pred, color='b', linewidth = 2, label='Linear Model with Polynomial Features Model Approximation')
plt.title('Сопоставление результатов предсказания с фактом',  fontsize=18)
plt.xlabel('Температура окружающего воздуха, Ц',  fontsize=16)
plt.ylabel('Тепловая нагрузка, Гкал/ч',  fontsize=16)
plt.grid(True)
plt.legend()
###plt.show()

plt.plot(data.iloc[:, 0:1][2838:], y_test, color='g', linewidth=2, label='Фактические данные')
plt.plot(data.iloc[:, 0:1][2838:], y_test_poly_pred, color='r', linewidth=2, label='Линейная полиномиальная модель')
plt.plot(data.iloc[:, 0:1][2838:], y_test_pred, color='b', linewidth=2, label='Линейная модель')
plt.title('Сопоставление результатов предсказания с фактом', fontsize=18)
plt.xlabel('Дата', fontsize=16)
plt.ylabel('Тепловая нагрузка, Гкал/ч', fontsize=16)
plt.grid(True)
plt.legend()
###plt.show()

y_result = np.hstack([y_test, y_test_pred, y_test_poly_pred])
df = pd.DataFrame(y_result, columns=['Qfact', 'Qlinear', 'QlinearWithPoly'])
df.to_excel('Linear Model Approximation.xlsx')'''