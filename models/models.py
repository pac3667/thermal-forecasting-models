from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

def get_catboost():
    return CatBoostRegressor(
        iterations=2000,
        depth=6,
        learning_rate=0.05,
        loss_function='MAE',
        verbose=0,
        random_seed=42,
        use_best_model=True,
    )

def get_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def get_linear():
    return LinearRegression()

def get_simple_mlp(input_shape):
    model = Sequential([
            Dense(128, activation='relu',input_shape=input_shape),
            Dense(1)
        ])
    model.compile(optimizer='adam', loss='mae')
    return model

def get_mlp(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
        ])
    model.compile(optimizer='adam', loss='mae')
    return model

def get_rfr():
    params = {
        "n_estimators": 35,
        "max_features": 3,
        "random_state": 1
    }
    return RandomForestRegressor(**params)