import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.src.saving import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import PolynomialFeatures

from models.CBRDirectForecast import train_cbr_direct_multistep
from utils import prepare_data, create_windows
from models.models import get_catboost, get_lstm, get_linear, get_simple_mlp, get_rfr, get_mlp
from models.LSTMDirectForecast import train_lstm_direct_multistep
from sklearn import metrics
import os

tf.random.set_seed(42)

def main():
    # 1. Prepare Data
    X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, dates = prepare_data('data/TEC22_Data.csv')
    os.makedirs('checkpoint/multistep', exist_ok=True)
    results = {}
    # ---  LSTM ---
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    checkpoint_filepath = 'checkpoint/Qpred_LSTM.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    model_lstm = get_lstm((1, X_train_s.shape[1]))
    if os.path.exists(checkpoint_filepath):
        print("Load Model...")
        model_lstm = load_model(checkpoint_filepath)
    else:
        print("Checkpoint not found, starting training...")
        model_lstm.fit(X_train_lstm, y_train_s,
                    epochs=1000,
                    batch_size=30,
                    validation_data=(X_test_lstm, y_test_s),
                    validation_batch_size=30,
                    callbacks=[early_stop_callback,model_checkpoint_callback],
                    verbose="auto",
                    shuffle=False)

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
    model_mlp = get_simple_mlp((X_train_s.shape[1],))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    checkpoint_filepath = 'checkpoint/Qpred_sMLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    if os.path.exists(checkpoint_filepath):
        print("Load Model...")
        model = load_model(checkpoint_filepath)
    else:
        print("Checkpoint not found, starting training...")
        model_mlp.fit(X_train_s ,
                    y_train_s,
                    epochs=1000,
                    batch_size=30,
                    validation_data=(X_test_s, y_test_s),
                    validation_batch_size=30,
                    callbacks=[early_stop_callback,model_checkpoint_callback],
                    verbose="auto",
                    shuffle=False)
    y_test_pred_scaled = model_mlp.predict(X_test_s)
    results['simple_MLP'] = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---  MLP ---
    model_mlp = get_mlp((X_train_s.shape[1],))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    checkpoint_filepath = 'checkpoint/Qpred_MLP.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    if os.path.exists(checkpoint_filepath):
        print("Load Model...")
        model = load_model(checkpoint_filepath)
    else:
        print("Checkpoint not found, starting training...")
        model_mlp.fit(X_train_s ,
                    y_train_s,
                    epochs=2500,
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

    checkpoint_filepath = 'checkpoint/Qpred_LSTM_with_RW.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True,
                                        min_delta=0.0001)

    model_lstmrw = get_lstm((window_size, X_train_s.shape[1]))
    if os.path.exists(checkpoint_filepath):
        print("Load Model...")
        model_lstmrw = load_model(checkpoint_filepath)
    else:
        print("Checkpoint not found, starting training...")
        model_lstmrw.fit(X_train_win, y_train_win,
                    epochs=1000,
                    batch_size=30,
                    validation_data=(X_test_win, y_test_win),
                    validation_batch_size=30,
                    callbacks=[early_stop_callback,model_checkpoint_callback],
                    verbose="auto",
                    shuffle=False)

    pred_lstm_s = model_lstmrw.predict(X_test_win)
    results['LSTM_with_Window'] = scaler_y.inverse_transform(pred_lstm_s)

    # ---  LSTM with window direct forecast---
    lstm_multi_results = train_lstm_direct_multistep('data/TEC22_Data.csv', n_out=14)

    # --- Boosting with window direct forecast---
    cbr_multi_results = train_cbr_direct_multistep('data/TEC22_Data.csv', n_out=14)

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
    print(f"{'Day':<5} | {'LSTM MAE':<12} | {'CBR MAE':<12}")

    for i in range(lstm_multi_results.shape[0]):
        print(f"{i + 1:<5} | {lstm_multi_results[i]:<12.4f} | {cbr_multi_results[i]:<12.4f}")

if __name__ == "__main__":
    main()