import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    data.set_index('Date', inplace=True)

    X = data[['T', 'Year', 'Month_sin', 'Month_cos']].values
    y = data[['TEC_Q_Aver']].values

    n = 3258
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    # 2. Масштабирование
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.fit_transform(y_test)
    return X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index

def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        X.append(x_data[i : i + window_size])
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)