import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn import metrics
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


def train_cbr_direct_multistep(data, n_out, train_size):
    results_list = []
    metrics_list = []
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

    for i in range(0, n_out, 1):
        print(f"\n=== Step training {i + 1} ===")

        x = data_with_lag[['T','TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', f'T_mean{i+1}', f'T_lag{i+1}', 'Q_rolling_3']].values
        y = data_with_lag.loc[:, [f'Q_lag{i+1}']].values
        X_train, X_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        model = CatBoostRegressor(iterations=2000,
                                  depth=6,
                                  learning_rate=0.05,
                                  loss_function='MAE',
                                  verbose=0)

        model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), early_stopping_rounds=250, use_best_model=True)

        yhat_scaled = model.predict(X_test_scaled)
        yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))

        MAE_test = metrics.mean_absolute_error(y_test, yhat)
        MSE_test = metrics.mean_squared_error(y_test, yhat)
        MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)*100
        R2_test = metrics.r2_score(y_test, yhat)

        results_list.append(yhat)
        metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])
    return np.hstack(metrics_list)

