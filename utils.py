import logging
import warnings
import pandas as pd
import optuna
import numpy as np
import tensorflow as tf
import keras.backend as K
import torch
from darts.models import DLinearModel, TiDEModel, Chronos2Model, XGBModel
from darts.utils.likelihood_models import QuantileRegression
from optuna_integration import PyTorchLightningPruningCallback
from sklearn.preprocessing import MinMaxScaler
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras import mixed_precision
from darts.metrics import mae as darts_mae
from darts import TimeSeries
from darts.models import RegressionModel, LightGBMModel
from darts.dataprocessing.transformers import Scaler
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from mlforecast.lag_transforms import RollingMean
from darts import TimeSeries
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn import metrics
from darts.models import TSMixerModel
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import metrics

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)


def prepare_data(filepath, test_start_index):
    data = pd.read_csv(filepath, delimiter=';', parse_dates=['Date'], dayfirst=True)

    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    first_work_day = data[data['TEC_N_Aver'] > 0].index[0]

    data.set_index('Date', inplace=True)

    X = data[['T', 'Year', 'Month_sin', 'Month_cos']].values
    y = data[['TEC_Q_Aver']].values

    n = test_start_index
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    # 2. Масштабирование
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)
    return X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, data.index
def create_windows(x_data, y_data, window_size=14):
    X, y = [], []
    for i in range(len(x_data) - window_size):
        X.append(x_data[i : i + window_size])
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)
def create_lstm_windows_for_step(data_raw, step_num, window_size):
    """
    Генерирует трехмерную структуру (samples, window_size, features),
    куда включены скользящие средние для учета накопленного эффекта теплопотерь.
    """
    # Таргет, сдвинутый на нужный шаг вперед
    q_lag = data_raw['TEC_Q_Aver'].shift(-step_num)

    # Считаем rolling-признаки ДО упаковки в окна, чтобы они были доступны на каждом шаге
    # Окно сглаживания берем равным window_size (или фиксированным, например 3, но лучше динамическим)
    t_mean_dynamic = data_raw['T'].rolling(window=window_size, min_periods=1).mean()
    q_rolling_dynamic = data_raw['TEC_Q_Aver'].rolling(window=window_size, min_periods=1).mean()

    # Собираем расширенный матричный массив признаков текущего момента
    # Теперь здесь 7 признаков вместо прежних 5
    extended_features = np.column_stack([
        data_raw['T'].values,
        data_raw['TEC_Q_Aver'].values,
        data_raw['Year'].values,
        data_raw['Month_sin'].values,
        data_raw['Month_cos'].values,
        t_mean_dynamic.values,  # Передаем среднюю температуру за период
        q_rolling_dynamic.values  # Передаем средний расход за период
    ])

    X, y = [], []
    # Запас на окно истории и шаг прогноза вперед
    for i in range(len(data_raw) - window_size - step_num + 1):
        X.append(extended_features[i: i + window_size])
        y.append(q_lag.iloc[i + window_size - 1])

    return np.array(X), np.array(y).reshape(-1, 1)
def optuna_lstm_search(trial, x_train, y_train, x_test, y_test, scaler_y, input_shape,
                       y_train_s_combined, y_test_s_combined, loss_type='custom'):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_units_lstm = trial.suggest_int('n_units_lstm', 20, 150) if trial else 50
    n_units_dense = trial.suggest_int('n_units_dense', 10, 50) if trial else 25
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) if trial else 0.001

    model = Sequential([
        LSTM(n_units_lstm, input_shape=input_shape, activation='relu'),
        Dense(n_units_dense, activation='relu'),
        Dense(1, dtype='float32')
    ])
    optimizer = Adam(learning_rate=lr)

    model_loss = 'mae'
    model.compile(optimizer=optimizer, loss=model_loss)

    current_batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_s_combined))
    train_dataset = (train_dataset
                     .shuffle(buffer_size=len(x_train))
                     .batch(current_batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_s_combined))
    val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        verbose=0,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    )

    pred_lstm_s = model.predict(val_dataset, verbose=0)
    y_pred_unscaled = scaler_y.inverse_transform(pred_lstm_s)
    y_test_unscaled = scaler_y.inverse_transform(y_test)

    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)
    return mae
def optuna_cbr_search(trial, x_train, y_train, x_test, y_test, scaler_y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.1, log=True),
        "loss_function": "MAE",
        "verbose": 0,
        "early_stopping_rounds": 50
    }

    pruning_callback = CatBoostPruningCallback(trial, "MAE")

    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, callbacks=[pruning_callback])

    preds = model.predict(x_test)
    y_pred_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)

    return mae
def optuna_rfr_search(trial, x_train, y_train, x_test, y_test, scaler_y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train.ravel())

    preds = model.predict(x_test)
    yhat = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mae = metrics.mean_absolute_error(y_true, yhat)

    return mae
def optuna_mlp_search(trial, x_train, y_train, x_test, y_test, scaler_y, y_train_s_combined, y_test_s_combined):


    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = Sequential()
    for i in range(n_layers):
        units = trial.suggest_int(f'units_l{i}', 16, 128)
        model.add(Dense(units, activation='relu'))

    model.add(Dense(1, dtype='float32'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')

    current_batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_s_combined))
    train_dataset = (train_dataset
                     .shuffle(buffer_size=len(x_train))
                     .batch(current_batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_s_combined))
    val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        verbose=0,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    )

    preds = model.predict(val_dataset, verbose=0)

    y_pred_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae = metrics.mean_absolute_error(y_pred_unscaled, y_test_unscaled)

    return mae
def optuna_lstm_with_window_search(trial, X_train_s, y_train_s, X_test_s, y_test_s):
    window_size = trial.suggest_int('window_size', 3, 21)
    n_units_lstm = trial.suggest_int('n_units_lstm', 20, 150)
    n_units_dense = trial.suggest_int('n_units_dense', 10, 50)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    X_train_win, y_train_win = create_windows(X_train_s, y_train_s, window_size)
    X_test_win, y_test_win = create_windows(X_test_s, y_test_s, window_size)

    input_shape = (X_train_win.shape[1], X_train_win.shape[2])

    model = Sequential([
        LSTM(n_units_lstm, input_shape=input_shape, activation='relu'),
        Dense(n_units_dense, activation='relu'),
        Dense(y_train_win.shape[1], dtype='float32')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')

    model.fit(
        X_train_win, y_train_win,
        validation_data=(X_test_win, y_test_win),
        epochs=15,
        batch_size=64,
        verbose=0,
        shuffle=False,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
    )

    yhat_s = model.predict(X_test_win, verbose=0)
    mae = metrics.mean_absolute_error(y_test_win, yhat_s)
    K.clear_session()
    return mae
def optuna_catboost_window_search(trial, X_train_s, y_train_s, X_test_s, y_test_s, scaler_y):
    # Подбираем размер окна в том же диапазоне, что и для LSTM
    window_size = trial.suggest_int('window_size', 3, 21)

    # Опционально: можно также слегка подбирать глубину дерева под размер окна
    depth = trial.suggest_int('cb_depth', 4, 8)
    l2_leaf_reg = trial.suggest_float('cb_l2', 1, 10)

    # 1. Создаем окна для текущей итерации
    X_train_win, y_train_win = create_windows(X_train_s, y_train_s, window_size)
    X_test_win, y_test_win = create_windows(X_test_s, y_test_s, window_size)

    # 2. Выравниваем в плоский массив для CatBoost
    X_train_flat = X_train_win.reshape(X_train_win.shape[0], -1)
    X_test_flat = X_test_win.reshape(X_test_win.shape[0], -1)

    # 3. Инициализируем модель с динамическими параметрами
    # (Используем встроенный get_catboost или создаем быстрый аналог для поиска)
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=500,  # Для экономии времени в Optuna ставим поменьше
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function='MAE',
        verbose=0
    )

    # 4. Обучаем модель
    model.fit(
        X_train_flat, y_train_win,
        eval_set=(X_test_flat, y_test_win),
        early_stopping_rounds=50,
        verbose=0
    )

    # 5. Делаем прогноз и возвращаем к реальному масштабу цен
    y_pred_s = model.predict(X_test_flat).reshape(-1, 1)
    y_pred_real = scaler_y.inverse_transform(y_pred_s)

    y_test_real = scaler_y.inverse_transform(y_test_win.reshape(-1, 1))

    # 6. Считаем MAE на реальных данных
    mae = metrics.mean_absolute_error(y_test_real, y_pred_real)
    return mae
def optuna_single_step_search(trial, data_raw, step_idx, train_size):
    window_size = trial.suggest_int('window_size', 3, 21)
    depth = trial.suggest_int('depth', 4, 8)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    step_num = step_idx + 1

    t_lag = data_raw['T'].shift(-step_num)
    t_mean = data_raw['T'].rolling(window=window_size).mean().shift(-step_num)
    q_lag = data_raw['TEC_Q_Aver'].shift(-step_num)
    q_rolling = data_raw['TEC_Q_Aver'].rolling(window=window_size).mean()

    df_step = pd.DataFrame({
        'T': data_raw['T'],
        'TEC_Q_Aver': data_raw['TEC_Q_Aver'],
        'Year': data_raw['Year'],
        'Month_sin': data_raw['Month_sin'],
        'Month_cos': data_raw['Month_cos'],
        'T_mean': t_mean,
        'T_lag': t_lag,
        'Q_rolling': q_rolling,
        'Target_Q': q_lag
    }).dropna()

    if len(df_step) <= train_size:
        return float('inf')  # Защита, если окно съело слишком много данных

    # Выделяем признаки и таргет
    x = df_step[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
    y = df_step[['Target_Q']].values

    # Разделение выборки
    X_train, X_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_test) == 0:
        return float('inf')

    # Нормализация
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Быстрая модель-оценщик для ускорения работы Optuna
    model = CatBoostRegressor(
        iterations=300,
        depth=depth,
        learning_rate=lr,
        loss_function='MAE',
        verbose=0
    )
    model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), early_stopping_rounds=30,
              use_best_model=True)

    yhat_scaled = model.predict(X_test_scaled)
    yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))

    mae = metrics.mean_absolute_error(y_test, yhat)
    return mae
def optuna_lstm_step_search(trial, data_raw, step_idx, train_size):
    # Оставляем подбор окна истории
    window_size = trial.suggest_int('window_size', 3, 21)
    n_units_lstm = trial.suggest_int('n_units_lstm', 32, 128)

    # Подбираем более консервативный learning rate для стабильности градиентов
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

    step_num = step_idx + 1

    # Генерируем расширенные 3D массивы
    X_win, y_win = create_lstm_windows_for_step(data_raw, step_num, window_size)

    if len(X_win) <= train_size:
        return float('inf')

    X_train, X_test = X_win[:train_size], X_win[train_size:]
    y_train, y_test = y_win[:train_size], y_win[train_size:]

    if len(X_test) == 0: return float('inf')

    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = Sequential([
        LSTM(n_units_lstm, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, dtype='float32')
    ])

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=30,
        batch_size=128,
        verbose=0,
        shuffle=False
    )

    yhat_scaled = model.predict(X_test_scaled, verbose=0)
    yhat = scaler_y.inverse_transform(yhat_scaled)

    mae = metrics.mean_absolute_error(y_test, yhat)
    K.clear_session()
    return mae
def optuna_rf_step_search(trial, data_raw, step_idx, train_size):
    # 1. Подбираем размер скользящего окна истории
    window_size = trial.suggest_int('window_size', 3, 21)

    # 2. Подбираем гиперпараметры случайного леса для защиты от переобучения
    n_estimators = trial.suggest_int('n_estimators', 50, 200, step=50)
    max_depth = trial.suggest_int('max_depth', 4, 12)

    step_num = step_idx + 1

    # Формируем признаки на основе текущего window_size
    t_lag = data_raw['T'].shift(-step_num)
    t_mean = data_raw['T'].rolling(window=window_size).mean().shift(-step_num)
    q_lag = data_raw['TEC_Q_Aver'].shift(-step_num)
    q_rolling = data_raw['TEC_Q_Aver'].rolling(window=window_size).mean()

    # Собираем DataFrame для очистки от краевых NaN
    df_step = pd.DataFrame({
        'T': data_raw['T'],
        'TEC_Q_Aver': data_raw['TEC_Q_Aver'],
        'Year': data_raw['Year'],
        'Month_sin': data_raw['Month_sin'],
        'Month_cos': data_raw['Month_cos'],
        'T_mean': t_mean,
        'T_lag': t_lag,
        'Q_rolling': q_rolling,
        'Target_Q': q_lag
    }).dropna()

    if len(df_step) <= train_size:
        return float('inf')

    x = df_step[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
    y = df_step[['Target_Q']].values.ravel()  # Спрямляем таргет для sklearn

    X_train, X_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_test) == 0:
        return float('inf')

    # Масштабирование
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Инициализация леса с параметрами из текущего триала
    # n_jobs=-1 ускорит подбор, задействуя все ядра процессора
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Делаем прогноз (масштабировать y не нужно, РФР не чувствителен к шкале таргета)
    yhat = model.predict(X_test_scaled)

    # Считаем MAE
    mae = metrics.mean_absolute_error(y_test, yhat)
    return mae
def optuna_mlp_step_search(trial, data_raw, step_idx, train_size):
    # Подбираем размер скользящего окна истории
    window_size = trial.suggest_int('window_size', 3, 21)

    # Подбираем гиперпараметры архитектуры нейросети
    n_units_dense = trial.suggest_int('n_units_dense', 32, 256)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

    step_num = step_idx + 1

    # Формируем признаки на основе текущего window_size
    t_lag = data_raw['T'].shift(-step_num)
    t_mean = data_raw['T'].rolling(window=window_size).mean().shift(-step_num)
    q_lag = data_raw['TEC_Q_Aver'].shift(-step_num)
    q_rolling = data_raw['TEC_Q_Aver'].rolling(window=window_size).mean()

    # Собираем DataFrame для очистки от краевых NaN
    df_step = pd.DataFrame({
        'T': data_raw['T'],
        'TEC_Q_Aver': data_raw['TEC_Q_Aver'],
        'Year': data_raw['Year'],
        'Month_sin': data_raw['Month_sin'],
        'Month_cos': data_raw['Month_cos'],
        'T_mean': t_mean,
        'T_lag': t_lag,
        'Q_rolling': q_rolling,
        'Target_Q': q_lag
    }).dropna()

    if len(df_step) <= train_size:
        return float('inf')

    x = df_step[['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_mean', 'T_lag', 'Q_rolling']].values
    y = df_step[['Target_Q']].values

    X_train, X_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_test) == 0:
        return float('inf')

    # Масштабирование признаков и таргета
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Строим полносвязную архитектуру MLP
    model = Sequential([
        Dense(n_units_dense, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(int(n_units_dense / 2), activation='relu'),
        Dense(1, dtype='float32')
    ])

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    # Быстрое обучение внутри Optuna
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=32,
        batch_size=128,
        verbose=0,
        shuffle=False
    )

    yhat_scaled = model.predict(X_test_scaled, verbose=0)
    yhat = scaler_y.inverse_transform(yhat_scaled)

    # Считаем MAE на оригинальной шкале данных
    mae = metrics.mean_absolute_error(y_test, yhat)
    K.clear_session()
    return mae
def optuna_darts_dlm_single_step_search(trial, data_raw, train_size, step_num):
    """
    Оптимизация гиперпараметров DLinear и окон сглаживания под ОДИН конкретный шаг (step_num).
    """
    # Разгоняем тензорные ядра вашей RTX 5080
    torch.set_float32_matmul_precision('high')

    # 1. ПОДБИРАЕМ ПАРАМЕТРЫ СГЛАЖИВАНИЯ ФИЧЕЙ ДЛЯ ДЕКОМПОЗИЦИИ
    t_smooth_window = trial.suggest_int('t_smooth_window', 2, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 2, 14)

    # Подбираем гиперпараметры DLinear
    input_chunk_length = trial.suggest_int('input_chunk_length', 5, 21)

    # Корректный расчет нечетного размера ядра свертки для декомпозиции тренда
    max_kernel = min(25, input_chunk_length)
    if max_kernel % 2 == 0:
        max_kernel -= 1
    kernel_size = 3 if max_kernel < 3 else trial.suggest_int('kernel_size', 3, max_kernel, step=2)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # 2. Динамически пересчитываем признаки под текущий триал
    df_step = data_raw.copy()
    df_step['Q_rolling'] = df_step['TEC_Q_Aver'].rolling(window=q_smooth_window).mean()
    df_step['T_rolling_3'] = df_step['T'].rolling(window=t_smooth_window).mean()

    # Генерируем сдвиги будущего СТРОГО под текущий изолированный шаг
    df_step[f'T_lag{step_num}'] = df_step['T'].shift(-step_num)
    df_step[f'T_mean{step_num}'] = df_step['T_rolling_3'].shift(-step_num)

    df_final = df_step.dropna()

    if len(df_final) <= train_size:
        return float('inf')

    # Определение точки разделения train/test до сборки TimeSeries
    split_date = df_final.index[train_size]

    # Собираем список колонок ковариат для текущего шага
    covariate_cols = [
        'T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling',
        f'T_mean{step_num}', f'T_lag{step_num}'
    ]

    # Создаем TimeSeries
    target_raw_series = TimeSeries.from_dataframe(df_final, value_cols='TEC_Q_Aver', freq='D')
    covariates_raw_series = TimeSeries.from_dataframe(df_final, value_cols=covariate_cols, freq='D')

    # Разделяем сырые ряды ДО масштабирования (Жесткий барьер против Data Leakage)
    train_target_raw, _ = target_raw_series.split_before(split_date)
    train_cov_raw, _ = covariates_raw_series.split_before(split_date)

    # Нормализация без утечек данных: fit только на train, transform на всё
    scaler_target = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target_raw)
    target_scaled = scaler_target.transform(target_raw_series)

    scaler_covariates = Scaler()
    _ = scaler_covariates.fit(train_cov_raw)
    covariates_scaled = scaler_covariates.transform(covariates_raw_series)

    optuna_pruner = PyTorchLightningPruningCallback(trial, monitor="train_loss")

    # Инициализируем DLinearModel под ОДИН шаг на выходе
    model = DLinearModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        kernel_size=kernel_size,
        shared_weights=False,
        const_init=True,
        batch_size=batch_size,
        n_epochs=15,  # Прунер начнет оценивать модель в процессе этих 15 эпох
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "callbacks": [optuna_pruner] # Передаем колбэк в Lightning Trainer через Darts
        }
    )

    # Обучаем модель на тренировочном сете
    try:
        model.fit(series=train_target_scaled, past_covariates=covariates_scaled, verbose=False)
    except optuna.TrialPruned:
        raise optuna.TrialPruned()  # Пробрасываем исключение вверх для Optuna
    except Exception as e:
        return float('inf')  # На случай непредвиденных ошибок (например, взрыв градиентов)

    # Генерируем скользящий валидационный прогноз (одношаговый)
    pred_val_scaled = model.historical_forecasts(
        series=target_scaled,
        past_covariates=covariates_scaled,
        start=split_date,
        forecast_horizon=1,  # Прогнозируем на 1 шаг вперед
        stride=1,
        retrain=False,
        verbose=False
    )

    # Денормализуем предсказания в реальный физический масштаб (расход ТЭЦ)
    pred_val_real = scaler_target.inverse_transform(pred_val_scaled)

    # Защита индексов: сдвигаем реальный таргет назад на step_num,
    # чтобы сопоставить прогноз "на этот день" с его реальным фактом
    actual_shifted_series = target_raw_series.shift(-step_num)
    actual_test_slice = actual_shifted_series.slice_intersect(pred_val_real)
    pred_val_real = pred_val_real.slice_intersect(actual_test_slice)

    # Считаем чистый физический MAE для оптимизации Optuna
    mae = darts_mae(actual_test_slice, pred_val_real)

    # Очистка сессии (в PyTorch Lightning/Darts K.clear_session не нужен,
    # но если вы используете бэкенды очистки памяти, можно оставить)
    return mae
def optuna_darts_lgb_multistep_search(trial, base_data, train_size, n_out):
    """
    Оптимизация гиперпараметров Optuna для Direct-модели Darts (БЕЗ рекурсии).
    Гарантирует полное отсутствие утечек данных между Train и Val выборками.
    """
    # 1. Пространство поиска параметров сглаживания и лагов
    t_smooth_window = trial.suggest_int('t_smooth_window', 1, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 1, 14)
    lags = trial.suggest_int('lags', 2, 21)

    # 2. Пространство поиска параметров LightGBM
    max_depth = trial.suggest_int('max_depth', 3, 12)
    lr = trial.suggest_float('lr', 1e-3, 3e-1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)

    # Жесткие границы индексов для валидации
    val_start_idx = train_size
    val_end_idx = train_size + n_out

    if val_end_idx > len(base_data):
        raise ValueError("Размер датасета слишком мал для указанных train_size и n_out.")

    # === ИСПРАВЛЕНИЕ УТЕЧКИ 1: Расчет rolling строго внутри своих выборок ===
    data = base_data.copy()

    # Для тренировочной выборки считаем окна только на ее основе
    train_df = data.iloc[:val_start_idx].copy()
    train_df['Q_rolling'] = train_df['TEC_Q_Aver'].rolling(window=q_smooth_window, min_periods=1).mean()
    train_df['T_rolling_3'] = train_df['T'].rolling(window=t_smooth_window, min_periods=1).mean()

    # Для валидационной выборки (история + горизонт) считаем окна изолированно
    val_history_and_horizon = data.iloc[:val_end_idx].copy()
    val_history_and_horizon['Q_rolling'] = val_history_and_horizon['TEC_Q_Aver'].rolling(window=q_smooth_window,
                                                                                         min_periods=1).mean()
    val_history_and_horizon['T_rolling_3'] = val_history_and_horizon['T'].rolling(window=t_smooth_window,
                                                                                  min_periods=1).mean()

    # Склеиваем обратно, чтобы построить непрерывные TimeSeries для Darts
    data.iloc[:val_start_idx] = train_df
    data.iloc[val_start_idx:val_end_idx] = val_history_and_horizon.iloc[val_start_idx:val_end_idx]

    # Создание Darts-последовательностей (ограничиваем val_end_idx, чтобы не брать лишнее)
    active_data = data.iloc[:val_end_idx].reset_index(drop=True)

    target_series = TimeSeries.from_dataframe(active_data, value_cols='TEC_Q_Aver', freq='D')

    future_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3']
    future_covariates = TimeSeries.from_dataframe(active_data, value_cols=future_cov_cols, freq='D')

    past_cov_cols = ['Q_rolling']
    past_covariates = TimeSeries.from_dataframe(active_data, value_cols=past_cov_cols, freq='D')

    # === ИСПРАВЛЕНИЕ УТЕЧКИ 2: Корректное разбиение таргета ===
    # split_before по умолчанию включает переданный индекс в левую часть. Убираем это через inclusive=False.
    split_date_train = target_series.time_index[val_start_idx]
    train_target, _ = target_series.split_before(split_date_train, inclusive=False)

    # Инициализация Честной Direct-модели Darts (без рекурсии)
    model = LightGBMModel(
        lags=lags,
        lags_future_covariates=(lags, n_out),
        lags_past_covariates=lags,
        output_chunk_length=n_out,  # Стратегия Direct: n_out независимых моделей под каждый шаг
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Обучение
    model.fit(
        series=train_target,
        past_covariates=past_covariates,
        future_covariates=future_covariates
    )

    # === ИСПРАВЛЕНИЕ УТЕЧКИ 3: Точка инференса и срезы без заглядывания в будущее ===
    # Исторический узел должен заканчиваться строго на последнем дне обучения
    last_train_node = target_series.time_index[val_start_idx - 1]

    # Берем историю таргета строго ДО начала валидации включительно
    history_target = target_series.slice_n_points_before(last_train_node, lags)

    # Истинные значения — это строго n_out шагов, начиная со следующей точки после last_train_node
    actual_val_slice = target_series.slice_n_points_after(target_series.time_index[val_start_idx], n_out)

    # Выполняем Direct-прогноз на n_out шагов вперед
    pred_val = model.predict(
        n=n_out,
        series=history_target,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        verbose=False
    )

    y_true = actual_val_slice.values().flatten()
    y_pred = pred_val.values().flatten()

    # Метрика валидации
    val_mae = metrics.mean_absolute_error(y_true, y_pred)

    # Репорт в Optuna для прунинга
    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae
def optuna_darts_tide_search(trial, data_raw, train_size, n_out):
    torch.set_float32_matmul_precision('high')

    # Глушим системный спам PyTorch Lightning в консоли триалов
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)

    # 1. ПОДБИРАЕМ ПАРАМЕТРЫ СГЛАЖИВАНИЯ И ГЛУБИНУ ИСТОРИИ
    t_smooth_window = trial.suggest_int('t_smooth_window', 2, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 2, 14)

    # Выносим глубину истории в Optuna
    input_chunk_length = trial.suggest_int('input_chunk_length', 7, 30)

    # Подбираем архитектурные гиперпараметры TiDE
    num_layers = trial.suggest_int('num_layers', 1, 2)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    # 2. Динамически пересчитываем базовые признаки под текущий триал
    df_step = data_raw.copy()
    df_step['Q_rolling'] = df_step['TEC_Q_Aver'].rolling(window=q_smooth_window).mean()
    df_step['T_rolling_3'] = df_step['T'].rolling(window=t_smooth_window).mean()
    df_final = df_step.dropna()

    if len(df_final) <= train_size:
        return float('inf')

    split_date = df_final.index[train_size]

    # Собираем базовые ковариаты (без будущих ручных сдвигов!)
    covariate_cols = ['T', 'TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling']

    # Создаем TimeSeries
    target_raw_series = TimeSeries.from_dataframe(df_final, value_cols='TEC_Q_Aver', freq='D')
    covariates_raw_series = TimeSeries.from_dataframe(df_final, value_cols=covariate_cols, freq='D')

    # Разделяем ДО масштабирования (Защита от утечки данных!)
    train_target_raw, _ = target_raw_series.split_before(split_date)
    train_cov_raw, _ = covariates_raw_series.split_before(split_date)

    # Корректный Scaler без подглядывания в будущее
    scaler_target = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target_raw)
    target_scaled = scaler_target.transform(target_raw_series)

    scaler_covariates = Scaler()
    _ = scaler_covariates.fit(train_cov_raw)
    covariates_scaled = scaler_covariates.transform(covariates_raw_series)

    # Инициализируем TiDE в честном MIMO-режиме (output_chunk_length = n_out)
    model = TiDEModel(
        input_chunk_length=input_chunk_length,  # Передаем подобранный параметр
        output_chunk_length=n_out,  # Прогноз всего вектора сразу
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        decoder_output_dim=16,
        hidden_size=hidden_size,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=15,  # 15 быстрых эпох для оценки на GPU внутри триала
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False
        }
    )

    model.fit(series=train_target_scaled, past_covariates=covariates_scaled, verbose=False)

    # Честный MIMO валидационный прогноз без рекурсии
    pred_val_scaled = model.historical_forecasts(
        series=target_scaled,
        past_covariates=covariates_scaled,
        start=split_date,
        forecast_horizon=n_out,  # СТРОГО равен output_chunk_length
        stride=1,
        retrain=False,
        verbose=False
    )

    # Денормализуем и считаем честный физический MAЕ на всем горизонте
    pred_val_real = scaler_target.inverse_transform(pred_val_scaled)
    actual_test_slice = target_raw_series.slice_intersect(pred_val_real)
    pred_val_real = pred_val_real.slice_intersect(actual_test_slice)

    mae = darts_mae(actual_test_slice, pred_val_real)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mae
def optuna_darts_chronos_search(trial, target_raw_series, split_date, n_out):
    torch.set_float32_matmul_precision('high')

    # Глушим системные логи PyTorch Lightning, чтобы не спамить в консоль
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=UserWarning)

    # 1. Подбираем параметры инференса
    input_chunk_length = trial.suggest_int('input_chunk_length', 7, 30)
    num_samples = trial.suggest_int('num_samples', 20, 100, step=10)

    # Инициализируем модель как вероятностную
    model = Chronos2Model(
        input_chunk_length=input_chunk_length,
        output_chunk_length=n_out,
        likelihood=QuantileRegression(),
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False
        }
    )

    # Фаза "обучения" метаданных
    model.fit(series=target_raw_series)

    # 2. Пошаговый инференс для реализации прунинга
    # Вместо долгого historical_forecasts на всей выборке сразу,
    # мы будем генерировать прогнозы батчами или проверять промежуточный скор
    try:
        pred_val = model.historical_forecasts(
            series=target_raw_series,
            start=split_date,
            forecast_horizon=n_out,
            stride=1,
            retrain=False,
            num_samples=num_samples,
            verbose=False
        )

        actual_test_slice = target_raw_series.slice_intersect(pred_val)
        mae = darts_mae(actual_test_slice, pred_val)

        # Передаем финальный шаг в Optuna для сопоставления с MedianPruner
        trial.report(mae, step=1)

        if trial.should_prune():
            raise optuna.TrialPruned()

    except optuna.TrialPruned:
        raise optuna.TrialPruned()
    except Exception as e:
        return float('inf')

    # Очистка кэша CUDA, так как языковые модели Chronos сильно забивают память GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mae
def optuna_darts_tide_and_tsfel_search(trial, train_target_scaled, covariates_scaled, target_scaled, scaler_target,
                             target_raw_series, split_date, n_out):
    # Активируем тензорные ядра вашей RTX 5080
    torch.set_float32_matmul_precision('high')

    # Пространство гиперпараметров для перебора
    num_layers = trial.suggest_int('num_layers', 1, 2)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    # Строим чистую модель TiDE без внешних костылей
    model = TiDEModel(
        input_chunk_length=14,  # Фиксированное окно истории (пусть ИИ сам ищет тренды!)
        output_chunk_length=n_out,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        decoder_output_dim=16,
        hidden_size=hidden_size,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=15,  # 15 эпох для экспресс-оценки триала на GPU
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False
        }
    )

    model.fit(series=train_target_scaled, future_covariates=covariates_scaled, verbose=False)

    pred_val_scaled = model.historical_forecasts(
        series=target_scaled,
        future_covariates=covariates_scaled,
        start=split_date,
        forecast_horizon=n_out,
        stride=1,
        retrain=False,
        verbose=False
    )

    # Денормализуем прогноз обратно к реальной шкале Гкал ТЭЦ
    pred_val_real = scaler_target.inverse_transform(pred_val_scaled)

    # === НАДЁЖНАЯ ЗАЩИТА ОТ NaN ДЛЯ OPTUNA ===
    # 1. Извлекаем пересекающиеся временные куски факта и прогноза
    actual_test_slice = target_raw_series.slice_intersect(pred_val_real)

    # 2. Если из-за сдвигов TSFEL массивы оказались пустыми, штрафуем триал
    if len(actual_test_slice) == 0 or len(pred_val_real) == 0:
        K.clear_session()
        return float('inf')

    # 3. Переводим в чистые NumPy векторы для безопасного расчета MAE
    y_true = actual_test_slice.values().flatten()
    y_pred = pred_val_real.values().flatten()

    # Исключаем любые случайные NaN, возникшие на стыке краев выборки TSFEL
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]

    # Если после очистки NaN данных не осталось — штрафуем триал
    if len(y_true_clean) == 0:
        K.clear_session()
        return float('inf')

    # Считаем MAE средствами sklearn, полностью застрахованными от NaN
    mae = metrics.mean_absolute_error(y_true_clean, y_pred_clean)

    K.clear_session()
    return mae
def optuna_nixtla_lgb_search(trial, df_train, df_val, n_out):
    """
    Честная оптимизация гиперпараметров для Direct-модели Nixtla.
    Полностью устранены утечки через скользящие окна, добавлен исторический контекст погоды.
    """
    # 1. Расширенное пространство гиперпараметров под Direct-стратегию
    lags_count = trial.suggest_int('lags_count', 7, 30)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    lr = trial.suggest_float('lr', 1e-3, 2e-1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    t_smooth = trial.suggest_int('t_smooth', 2, 14)

    # Динамически формируем список колонок с учетом 7 лагов погоды
    lag_t_cols = [f'T_lag_{lag}' for lag in range(1, 8)]
    keep_cols = ['unique_id', 'ds', 'y', 'T', 'T_rolling', 'Month_sin', 'Month_cos'] + lag_t_cols
    future_cols = [c for c in keep_cols if c != 'y']

    # === ИСПРАВЛЕНИЕ УТЕЧКИ И ДОБАВЛЕНИЕ ЛАГОВ ПОГОДЫ ===
    # Склеиваем датасеты, чтобы окна и лаги на валидации корректно рассчитывались из тренировочного прошлого
    full_df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Расчет скользящего среднего по температуре
    full_df['T_rolling'] = full_df['T'].rolling(window=t_smooth, min_periods=1).mean()

    # Ручная генерация исторических лагов температуры (устраняем отставание от Darts/LGB)
    for lag in range(1, 8):
        full_df[f'T_lag_{lag}'] = full_df['T'].shift(lag)

    # Разрезаем датасет обратно строго по границам выборок
    df_train_final = full_df.iloc[:len(df_train)][keep_cols].copy()

    # Для будущего (валидации) берем строки валидации с уже посчитанными фичами
    df_val_built = full_df.iloc[len(df_train):len(df_train) + n_out].copy()
    X_df_future = df_val_built[future_cols].reset_index(drop=True)

    # 2. Инициализация LightGBM
    lgb_model = LGBMRegressor(
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 3. Сборка пайплайна Nixtla
    # Отключаем lag_transforms (RollingMean по 'y' неэффективен для чистых Direct-моделей)
    fcst = MLForecast(
        models=[lgb_model],
        freq='D',
        lags=list(range(1, lags_count + 1)),
        lag_transforms={}
    )

    # Обучение Direct-ансамбля
    fcst.fit(
        df_train_final,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        static_features=[],
        max_horizon=n_out
    )

    # Direct-инференс
    predictions = fcst.predict(
        h=n_out,
        new_df=df_train_final,  # Передаем историю для построения лагов таргета
        X_df=X_df_future
    )

    # Считаем MAE по всему валидационному горизонту
    y_true = df_val['y'].iloc[:n_out].values
    y_pred = predictions['LGBMRegressor'].values

    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_idx):
        return float('inf')

    mae = mean_absolute_error(y_true[valid_idx], y_pred[valid_idx])
    return mae
def optuna_darts_nhits_search(trial, base_data, train_size, n_out):
    """
    Оптимизация параметров NHiTS. Все экзогенные фичи передаются в past_covariates.
    """
    t_smooth_window = trial.suggest_int('t_smooth_window', 1, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 1, 14)
    input_chunk_length = trial.suggest_int('input_chunk_length', 7, 28)
    layer_widths = trial.suggest_categorical('layer_widths', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=q_smooth_window, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=t_smooth_window, min_periods=1).mean()

    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    # Объединяем ВСЕ фичи (включая погоду и календарь) в один блок past_covariates
    all_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=all_cov_cols, freq='D')

    val_start_idx = train_size
    val_end_idx = train_size + n_out

    if val_end_idx > len(data):
        raise ValueError("Размер датасета слишком мал.")

    split_date_train = data.index[val_start_idx]
    train_target, _ = target_series.split_before(split_date_train)

    optuna_early_stopper = EarlyStopping(monitor="train_loss", patience=5, min_delta=0.05, mode="min")

    model = NHiTSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=n_out,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=layer_widths,
        dropout=0.2,
        n_epochs=60,
        batch_size=32,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "callbacks": [optuna_early_stopper],
            "accelerator": "auto",
            "enable_progress_bar": False,
            "enable_model_summary": False
        },
        random_state=42
    )

    # Передаем только past_covariates
    model.fit(
        series=train_target,
        past_covariates=past_covariates
    )

    current_time_node = target_series.time_index[val_start_idx]
    history_target = target_series.slice_n_points_before(current_time_node, input_chunk_length)
    actual_val_slice = target_series.slice_n_points_after(current_time_node, n_out)

    # При предсказании также убираем аргумент future_covariates
    pred_val = model.predict(
        n=n_out,
        series=history_target,
        past_covariates=past_covariates,
        verbose=False
    )

    y_true = actual_val_slice.values().flatten()
    y_pred = pred_val.values().flatten()

    val_mae = metrics.mean_absolute_error(y_true, y_pred)

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae
def optuna_darts_tsmixer_search(trial, base_data, train_size, n_out):
    """
    Оптимизация параметров TSMixer для маленького датасета.
    """
    t_smooth_window = trial.suggest_int('t_smooth_window', 1, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 1, 14)
    input_chunk_length = trial.suggest_int('input_chunk_length', 7, 30)

    # Сжимаем скрытые слои, чтобы модель не переобучалась на 3000 строках
    ff_size = trial.suggest_categorical('ff_size', [32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=q_smooth_window, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=t_smooth_window, min_periods=1).mean()

    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    all_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3', 'Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=all_cov_cols, freq='D')

    val_start_idx = train_size
    val_end_idx = train_size + n_out

    if val_end_idx > len(data):
        raise ValueError("Размер датасета слишком мал.")

    split_date_train = data.index[val_start_idx]
    train_target, _ = target_series.split_before(split_date_train)

    # Ранний стоп для экономии времени в Optuna
    optuna_early_stopper = EarlyStopping(monitor="train_loss", patience=5, min_delta=0.05, mode="min")

    model = TSMixerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=n_out,
        num_blocks=1,  # Всего 1 блок смешивания для защиты от оверфиттинга
        ff_size=ff_size,  # Размер скрытого полносвязного слоя
        dropout=0.3,  # Повышенный дропаут для регуляризации
        n_epochs=50,
        batch_size=32,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "callbacks": [optuna_early_stopper],
            "accelerator": "auto",
            "enable_progress_bar": False,
            "enable_model_summary": False
        },
        random_state=42
    )

    model.fit(series=train_target, past_covariates=past_covariates)

    current_time_node = target_series.time_index[val_start_idx]
    history_target = target_series.slice_n_points_before(current_time_node, input_chunk_length)
    actual_val_slice = target_series.slice_n_points_after(current_time_node, n_out)

    pred_val = model.predict(n=n_out, series=history_target, past_covariates=past_covariates, verbose=False)

    y_true = actual_val_slice.values().flatten()
    y_pred = pred_val.values().flatten()

    val_mae = metrics.mean_absolute_error(y_true, y_pred)

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae
def optuna_darts_xgb_multistep_search(trial, base_data, train_size, n_out):
    """
    Функция оптимизации гиперпараметров Optuna для Direct-модели XGBoost в Darts.
    """
    # 1. Параметры сглаживания и лагов
    t_smooth_window = trial.suggest_int('t_smooth_window', 1, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 1, 14)
    lags = trial.suggest_int('lags', 2, 21)

    # 2. Параметры структуры деревьев XGBoost
    max_depth = trial.suggest_int('max_depth', 3, 10)
    lr = trial.suggest_float('lr', 1e-3, 3e-1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 50, 400)

    # Ключевой параметр XGBoost для защиты от оверфиттинга на мелких данных
    gamma = trial.suggest_float('gamma', 0.0, 5.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

    # Подготовка признаков
    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=q_smooth_window, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=t_smooth_window, min_periods=1).mean()

    # Ряды Darts
    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    future_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3']
    future_covariates = TimeSeries.from_dataframe(data, value_cols=future_cov_cols, freq='D')

    past_cov_cols = ['Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=past_cov_cols, freq='D')

    val_start_idx = train_size
    val_end_idx = train_size + n_out

    if val_end_idx > len(data):
        raise ValueError("Размер датасета слишком мал.")

    split_date_train = data.index[val_start_idx]
    train_target, _ = target_series.split_before(split_date_train)

    # Инициализация XGBoost
    model = XGBModel(
        lags=lags,
        lags_future_covariates=(lags, n_out),
        lags_past_covariates=lags,
        output_chunk_length=n_out,  # Автоматический Direct-ансамбль
        max_depth=max_depth,
        learning_rate=lr,
        n_estimators=n_estimators,
        gamma=gamma,  # Регуляризация сплитов
        min_child_weight=min_child_weight,  # Минимальный вес в листьях
        tree_method='hist',  # Быстрый гистограммный метод
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(
        series=train_target,
        past_covariates=past_covariates,
        future_covariates=future_covariates
    )

    current_time_node = target_series.time_index[val_start_idx]
    history_target = target_series.slice_n_points_before(current_time_node, lags)
    actual_val_slice = target_series.slice_n_points_after(current_time_node, n_out)

    pred_val = model.predict(
        n=n_out,
        series=history_target,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        verbose=False
    )

    y_true = actual_val_slice.values().flatten()
    y_pred = pred_val.values().flatten()

    val_mae = metrics.mean_absolute_error(y_true, y_pred)

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae
def optuna_darts_direct_ensemble_search(trial, base_data, train_size, n_out):
    """
    Подбор лагов и окон сглаживания для единого Direct-ансамбля бустингов.
    """
    t_smooth_window = trial.suggest_int('t_smooth_window', 1, 14)
    q_smooth_window = trial.suggest_int('q_smooth_window', 1, 14)
    lags = trial.suggest_int('lags', 2, 21)

    data = base_data.copy()
    data['Q_rolling'] = data['TEC_Q_Aver'].rolling(window=q_smooth_window, min_periods=1).mean()
    data['T_rolling_3'] = data['T'].rolling(window=t_smooth_window, min_periods=1).mean()

    target_series = TimeSeries.from_dataframe(data, value_cols='TEC_Q_Aver', freq='D')

    future_cov_cols = ['T', 'Year', 'Month_sin', 'Month_cos', 'T_rolling_3']
    future_covariates = TimeSeries.from_dataframe(data, value_cols=future_cov_cols, freq='D')

    past_cov_cols = ['Q_rolling']
    past_covariates = TimeSeries.from_dataframe(data, value_cols=past_cov_cols, freq='D')

    val_start_idx = train_size
    val_end_idx = train_size + n_out

    if val_end_idx > len(data):
        raise ValueError("Размер датасета слишком мал.")

    split_date_train = data.index[val_start_idx]
    train_target, _ = target_series.split_before(split_date_train)

    lgb = LGBMRegressor(max_depth=6, learning_rate=0.05, n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
    xgb = XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=150, tree_method='hist', random_state=42,
                       n_jobs=-1, verbosity=0)
    cbr = CatBoostRegressor(depth=5, learning_rate=0.05, iterations=150, random_seed=42, verbose=0)

    # Объединяем их в ансамбль голосования sklearn
    voting_ensemble = VotingRegressor(estimators=[('lgb', lgb), ('xgb', xgb), ('cbr', cbr)])

    # Передаем ансамбль напрямую в параметр `model`
    model = RegressionModel(
        model=voting_ensemble,
        lags=lags,
        lags_future_covariates=(lags, n_out),
        lags_past_covariates=lags,
        output_chunk_length=n_out
    )

    model.fit(series=train_target, past_covariates=past_covariates, future_covariates=future_covariates)

    current_time_node = target_series.time_index[val_start_idx]
    history_target = target_series.slice_n_points_before(current_time_node, lags)
    actual_val_slice = target_series.slice_n_points_after(current_time_node, n_out)

    pred_val = model.predict(n=n_out, series=history_target, past_covariates=past_covariates,
                             future_covariates=future_covariates, verbose=False)

    y_true = actual_val_slice.values().flatten()
    y_pred = pred_val.values().flatten()

    val_mae = metrics.mean_absolute_error(y_true, y_pred)

    trial.report(val_mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_mae