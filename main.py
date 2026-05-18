import os

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from models.CBRDirectForecast import train_cbr_direct_multistep
from models.LSTMDirectForecast import train_lstm_direct_multistep
from models.models import (
    get_catboost, get_linear, get_lstm,
    get_mlp, get_rfr, get_simple_mlp
)
from utils import create_windows, prepare_data

tf.random.set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/TEC14_Data.csv', help='Путь к файлу данных') #data/TEC22_Data.csv data/TEC14_Data.csv
    parser.add_argument('--start', type=int, default=2840, help='Индекс начала тестовых данных')     #3258 2840
    parser.add_argument('--forecast_window', type=int, default=14, help='окно прогноза')
    args = parser.parse_args()

    data_path = args.file
    test_start_index = args.start
    n_out = args.forecast_window

    model_name = os.path.basename(data_path).split('.')[0]
    checkpoint_dir = f'checkpoint/{model_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Prepare Data
    X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, dates = prepare_data(data_path, test_start_index=test_start_index)
    os.makedirs(checkpoint_dir+'multistep', exist_ok=True)
    results = {}
    # ---  LSTM ---
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    checkpoint_filepath = checkpoint_dir+'Qpred_LSTM.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    model_lstm = get_lstm((1, X_train_s.shape[1]))

    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_lstm = load_model(checkpoint_filepath)
        model_lstm.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000

    model_lstm.fit(
        X_train_lstm, y_train_s,
        epochs=current_epochs,
        batch_size=30,
        validation_data=(X_test_lstm, y_test_s),
        callbacks=[early_stop_callback, model_checkpoint_callback],
        verbose="auto",
        shuffle=False
    )

    pred_lstm_s = model_lstm.predict(X_test_lstm)
    results['LSTM'] = scaler_y.inverse_transform(pred_lstm_s)

    # ---  Linear Regression ---
    model_lr = get_linear()
    model_lr.fit(X_train_s, y_train)
    results['Linear'] = model_lr.predict(X_test_s)

    # ---  poly Regression ---
    poly = PolynomialFeatures(2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    model_poly = get_linear()
    model_poly.fit(X_train_poly, y_train)
    results['Polynomial (d=2)'] = model_poly.predict(X_test_poly)

    # ---  Gradient Boosting ---
    model_cb = get_catboost()
    model_cb.fit(X_train_s, y_train_s, eval_set=(X_test_s, y_test_s), early_stopping_rounds=250, use_best_model=True)
    y_test_pred = model_cb.predict(X_test_s).reshape(-1, 1)
    results['Boosting'] = scaler_y.inverse_transform(y_test_pred)

    # ---  simple MLP ---
    model_smlp = get_simple_mlp((X_train_s.shape[1],))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    checkpoint_filepath = checkpoint_dir+'Qpred_sMLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_smlp = load_model(checkpoint_filepath)
        model_smlp.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000

    model_smlp.fit(X_train_s ,
                y_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(X_test_s, y_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose="auto",
                shuffle=False)
    y_test_pred_scaled = model_smlp.predict(X_test_s)
    results['simple_MLP'] = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---  MLP ---
    model_mlp = get_mlp((X_train_s.shape[1],))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    checkpoint_filepath = checkpoint_dir+'Qpred_MLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_mlp = load_model(checkpoint_filepath)
        model_mlp.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000

    model_mlp.fit(X_train_s ,
                y_train_s,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(X_test_s, y_test_s),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose="auto",
                shuffle=False)

    y_test_pred_scaled = model_mlp.predict(X_test_s)
    results['MLP'] = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---  Random Forest Regression---
    model_rfr = get_rfr()
    model_rfr.fit(X_train_s, y_train_s)
    y_test_pred = model_rfr.predict(X_test_s).reshape(-1, 1)
    results['RandomForest'] = scaler_y.inverse_transform(y_test_pred)

    # ---  Gradient booster with window ---
    window_size=7
    X_train_win, y_train_win = create_windows(X_train_s, y_train_s, window_size)
    X_test_win, y_test_win = create_windows(X_test_s, y_test_s, window_size)

    X_train_flat = X_train_win.reshape(X_train_win.shape[0], -1)
    X_test_flat = X_test_win.reshape(X_test_win.shape[0], -1)

    model_cbw = get_catboost()
    model_cbw.fit(X_train_flat, y_train_win, eval_set=(X_test_flat, y_test_win), early_stopping_rounds=250, use_best_model=True)
    y_test_pred = model_cbw.predict(X_test_flat).reshape(-1, 1)
    results['Boosting_with_Window'] = scaler_y.inverse_transform(y_test_pred)

    # ---  LSTM with window ---
    window_size = 7
    X_train_win, y_train_win = create_windows(X_train_s, y_train_s, window_size)
    X_test_win, y_test_win = create_windows(X_test_s, y_test_s, window_size)

    checkpoint_filepath = checkpoint_dir+'Qpred_LSTM_with_RW.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    model_lstmrw = get_lstm((window_size, X_train_s.shape[1]))

    if os.path.exists(checkpoint_filepath):
        print("Loading model for further training...")
        model_lstmrw = load_model(checkpoint_filepath)
        model_lstmrw.optimizer.learning_rate.assign(1e-4)
        current_epochs = 50
    else:
        print("Checkpoint not found, starting training...")
        current_epochs = 1000
    model_lstmrw.fit(X_train_win, y_train_win,
                epochs=current_epochs,
                batch_size=30,
                validation_data=(X_test_win, y_test_win),
                validation_batch_size=30,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose="auto",
                shuffle=False)

    pred_lstm_s = model_lstmrw.predict(X_test_win)
    results['LSTM_with_Window'] = scaler_y.inverse_transform(pred_lstm_s)

    # ---  LSTM with window direct forecast---
    lstm_multi_results = train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    lstm_multi_results = np.array(lstm_multi_results).reshape(-1, 4)
    # --- Boosting with window direct forecast---
    cbr_multi_results = train_cbr_direct_multistep(data_path, n_out, test_start_index)
    cbr_multi_results = np.array(cbr_multi_results).reshape(-1, 4)

    # 3. Model Evaluation & Comparison
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON REPORT")
    print("=" * 50)

    report = []
    min_len = min([len(p) for p in results.values()])
    for name, pred in results.items():
        y_pred_final = pred[-min_len:]
        y_true_final = y_test[-min_len:]

        mae = metrics.mean_absolute_error(y_true_final, y_pred_final)
        rmse = np.sqrt(metrics.mean_squared_error(y_true_final, y_pred_final))
        mape = metrics.mean_absolute_percentage_error(y_true_final, y_pred_final)
        r2 = metrics.r2_score(y_true_final, y_pred_final)

        report.append({
            'Model Name': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape * 100,
            'R2 Score': r2
        })

    df_report = pd.DataFrame(report)
    df_report = df_report.sort_values(by='MAE').reset_index(drop=True)

    print(df_report.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    df_report.to_csv('model_evaluation_report.csv', index=False)
    print("\n[INFO] Report saved to 'model_evaluation_report.csv'")

    min_len = min([len(p) for p in results.values()])
    export_df = pd.DataFrame({'Actual_Q': y_test[-min_len:].flatten()})

    for name, pred in results.items():
        export_df[name] = pred[-min_len:].flatten()

    export_df.to_excel('final_predictions_comparison.xlsx', index=False)
    print("[INFO] Predictions exported to 'final_predictions_comparison.xlsx'")

    print("\n" + "=" * 60)
    print("HORIZON ANALYSIS (LSTM vs CBR)")
    print("-" * 60)
    print(f"{'Day':<5} | {'LSTM MAE':<12} | {'LSTM MSE':<12} | {'LSTM MAPE':<12} | {'LSTM R2':<12} | {'CBR MAE':<12} | {'CBR MSE':<12} | {'CBR MAPE':<12} | {'CBR R2':<12}")

    for i in range(lstm_multi_results.shape[0]):
        print(f"{i + 1:<5} | {lstm_multi_results[i][0]:<12.4f} | {lstm_multi_results[i][1]:<12.4f} | {lstm_multi_results[i][2]:<12.4f} | {lstm_multi_results[i][3]:<12.4f} | {cbr_multi_results[i][0]:<12.4f} | {cbr_multi_results[i][1]:<12.4f} | {cbr_multi_results[i][2]:<12.4f} | {cbr_multi_results[i][3]:<12.4f}")

if __name__ == "__main__":
    main()