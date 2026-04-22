import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from keras import models, layers
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Flatten, Dense
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn import datasets, linear_model, metrics, model_selection
import seaborn as sns
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(1)

mpl.rcParams['figure.figsize'] = (25, 10)

def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        # Берем окно признаков (например, за 24 часа)
        X.append(x_data[i : i + window_size])
        # Целевое значение — следующее значение после окна
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)

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
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 2. Создаем окна (теперь X_train_win будет иметь форму [samples, 7, 4])
window_size = 7
X_train_win, y_train_win = create_windows(X_train_scaled, y_train_scaled, window_size)
X_test_win, y_test_win = create_windows(X_test_scaled, y_test_scaled, window_size)

model = Sequential()
model.add(Flatten(input_shape=(X_train_win.shape[1], X_train_win.shape[2])))
model.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mape', 'r2_score'])

early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                    min_delta=0.0001)

history_MNN = model.fit(X_train_win,
                    y_train_win,
                    epochs=1000,
                    batch_size=30,
                    validation_split=0.2,
                    validation_batch_size=30,
                    callbacks=early_stop_callback,
                    verbose="auto")

y_test_pred_scaled = model.predict(X_test_win)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_win)

MAE_test = metrics.mean_absolute_error(y_test_actual, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test_actual, y_test_pred)
MAPE_test = metrics.mean_absolute_percentage_error(y_test_actual, y_test_pred)
R2_test = metrics.r2_score(y_test_actual, y_test_pred)
print(f"MAE_test: {MAE_test:.2f}")
print(f"MSE_test: {MSE_test:.2f}")
print(f"MAPE_test: {MAPE_test:.2f}")
print(f"R2_test: {R2_test:.2f}")

feature_names = ['T', 'Year', 'Month_sin', 'Month_cos']
cols = []
for i in range(window_size, 0, -1):
    for f in feature_names:
        cols.append(f"{f}_lag_{i}")

# 2. Превращаем X_train_win и y_train_win в один DataFrame
df_corr = pd.DataFrame(X_train_win.reshape(X_train_win.shape[0], -1), columns=cols)
df_corr['TARGET'] = y_train_win

# 3. Считаем корреляцию всех признаков с таргетом
correlations = df_corr.corr()['TARGET'].sort_values(ascending=False)

# 4. Визуализируем только корреляцию признаков с целевой переменной
plt.figure(figsize=(10, 12))
sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Корреляция лагов в окне с целевым значением (Q)")
plt.show()