import json
import os
import tensorflow as tf
import optuna

from catboost import CatBoostRegressor
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.src.layers import LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras import mixed_precision

from utils import optuna_cbr_search, optuna_rfr_search, optuna_mlp_search, optuna_lstm_search


def get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optuna_cbr_search(trial, X_train_s, y_train_s,
                                               X_test_s, y_test_s, scaler_y),
                   n_trials=50)

    best_params = study.best_params

    model = CatBoostRegressor(iterations=2000,
                              depth=best_params['depth'],
                              learning_rate=best_params['lr'],
                              loss_function='MAE',
                              verbose=0)
    return model
def get_lstm_with_window(input_shape, output_dim, checkpoint_dir, calc_goal, best_params):
    checkpoint_filepath = f"{checkpoint_dir}{calc_goal}_LSTM.keras"
    params_filepath = f"{checkpoint_dir}{calc_goal}_LSTM_params.json"

    # Если мы хотим обучить модель С НУЛЯ на лучших параметрах:
    print("Creating fresh model using best parameters from Optuna...")
    with open(params_filepath, 'w') as f:
        json.dump(best_params, f)

    current_epochs = 1000
    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=50, verbose=1, restore_best_weights=True, min_delta=0.0001
    )

    model = Sequential([
        LSTM(best_params['n_units_lstm'], input_shape=input_shape, activation='relu'),
        Dense(best_params['n_units_dense'], activation='relu'),
        Dense(output_dim, dtype='float32')
    ])

    optimizer = Adam(learning_rate=best_params['lr'], clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mae')

    return model, early_stop_callback, current_epochs, checkpoint_filepath
def get_lstm(input_shape, X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y,
             y_train_s_combined, y_test_s_combined, checkpoint_dir, calc_goal):

    checkpoint_filepath = f"{checkpoint_dir}{calc_goal}_LSTM.keras"
    params_filepath = f"{checkpoint_dir}{calc_goal}_LSTM_params.json"

    model_loss = 'mae'

    if os.path.exists(checkpoint_filepath) and os.path.exists(params_filepath):
        print(f"Loading model and best parameters")
        model = load_model(checkpoint_filepath)
        model.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print(f"Starting hyperparameter optimization")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_lstm_search(trial, X_train_s, y_train_s,
                                                        X_test_s, y_test_s, scaler_y, input_shape,
                                                        y_train_s_combined, y_test_s_combined, loss_type='mae'),
                       n_trials=50)
        best_params = study.best_params

        with open(params_filepath, 'w') as f:
            json.dump(best_params, f)

        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

        model = Sequential([
            LSTM(best_params['n_units_lstm'], input_shape=input_shape, activation='relu'),
            Dense(best_params['n_units_dense']),
            Dense(1, dtype='float32')
        ])
        optimizer = Adam(learning_rate=best_params['lr'])
        model.compile(optimizer=optimizer, loss=model_loss)
        policy_name = mixed_precision.global_policy().name
        is_mixed = "mixed_float16" in policy_name
        print(f"--- Аппаратный Loss Scale для float16 активен: {is_mixed} (Политика: {policy_name}) ---")

    return model, early_stop_callback, current_epochs, checkpoint_filepath
def get_linear():
    return LinearRegression()
def get_mlp(input_shape, X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, y_train_s_combined, y_test_s_combined, checkpoint_filepath, params_filepath):
    if os.path.exists(checkpoint_filepath) and os.path.exists(params_filepath):
        print("Loading model and best parameters...")
        model = load_model(checkpoint_filepath)

        model.optimizer.learning_rate.assign(1e-4)

        current_epochs = 50
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                            min_delta=0.0001)
    else:
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_mlp_search(trial, X_train_s, y_train_s,
                                                   X_test_s, y_test_s, scaler_y, y_train_s_combined, y_test_s_combined), n_trials=50)
        best_params = study.best_params

        with open(params_filepath, 'w') as f:json.dump(best_params, f)

        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

        model = Sequential()
        for i in range(best_params['n_layers']):
            model.add(Dense(best_params[f'units_l{i}'], activation='relu'))

        model.add(Dense(1, dtype='float32'))

        optimizer = Adam(learning_rate=best_params['lr'])
        model.compile(optimizer=optimizer, loss='mae')
    return model, early_stop_callback, current_epochs
def get_rfr(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optuna_rfr_search(trial, X_train_s, y_train_s, X_test_s, y_test_s, scaler_y),
                   n_trials=50)

    best_params = study.best_params
    params = {
        "n_estimators": best_params['n_estimators'],
        "max_depth": best_params['max_depth'],
        "min_samples_split": best_params['min_samples_split'],
        "min_samples_leaf": best_params['min_samples_leaf'],
        "max_features": best_params['max_features'],
        "n_jobs": -1
    }
    return RandomForestRegressor(**params)