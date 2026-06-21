import os
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.python.platform import build_info as tf_build_info
from models.CBRDirectForecast import train_cbr_direct_multistep
from models.ChronosDirectForecast import train_chronos_darts_multistep
from models.DLMDirectForecast import train_dlm_direct_multistep
from models.DartsDirectEnsemble import train_direct_ensemble_multistep
from models.GTiDEDirectFarecast import train_tide_darts_multistep
from models.LSTMDirectForecast import train_lstm_direct_multistep
from models.LightGBMDirectForecast import train_lgbm_darts_multistep
from models.MLPDirectForecast import train_mlp_direct_multistep
from models.NHITSDirectForescast import train_nhits_direct_multistep
from models.NixtlaLGBDirectForecast import train_nixtla_lgb_multistep
from models.RFRDirectForecast import train_rfr_direct_multistep
from models.GTiDEandTSFELDirectForecast import train_tide_and_tsfeel_darts_multistep
from models.TSMixerDirectForecast import train_tsmixer_darts_multistep
from models.XGBoostDirectForecast import train_xgb_darts_direct_multistep
from models.models import (
    get_catboost, get_linear, get_lstm,
    get_mlp, get_rfr, get_lstm_with_window
)

from utils import create_windows, prepare_data, optuna_lstm_with_window_search, optuna_catboost_window_search
from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.random.set_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/TEC22_Data.csv', help='Путь к файлу данных') #data/TEC22_Data.csv data/TEC14_Data.csv
    parser.add_argument('--start', type=int, default=3258, help='Индекс начала тестовых данных')     #3258 2840
    parser.add_argument('--forecast_window', type=int, default=14, help='окно прогноза')
    args = parser.parse_args()

    calc_goal = 'TEC'

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
    print('LSTM_Stat_Model')
    x_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    x_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))


    model_lstm, early_stop_callback, current_epochs, checkpoint_filepath = get_lstm(
        (1, X_train_s.shape[1]), x_train_lstm, x_test_lstm, y_train_s, y_test_s, y_train, y_test,
        scaler_y, y_train_s, y_test_s, checkpoint_dir, calc_goal)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False,
        monitor='val_loss', mode='min', save_best_only=True, verbose=0
    )

    model_lstm.fit(
        x_train_lstm, y_train_s,
        epochs=current_epochs,
        batch_size=128,
        validation_data=(x_test_lstm, y_test_s),
        callbacks=[early_stop_callback, model_checkpoint_callback],
        verbose=0,
        shuffle=False
    )

    yhat_s = model_lstm.predict(x_test_lstm)
    yhat = scaler_y.inverse_transform(yhat_s)

    results['LSTM'] = yhat

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
    print('CBR_Stat_Model')
    model_cb = get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_cb.fit(X_train_s, y_train_s, eval_set=(X_test_s, y_test_s), early_stopping_rounds=250, use_best_model=True)
    y_test_pred = model_cb.predict(X_test_s).reshape(-1, 1)
    results['Boosting'] = scaler_y.inverse_transform(y_test_pred)

    # ---  MLP ---
    print('MLP_Stat_Model')

    checkpoint_filepath = checkpoint_dir + calc_goal + '_MLP.keras'
    params_filepath = checkpoint_dir + calc_goal + '_MLP_params.json'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    model_mlp, early_stop_callback, current_epochs = get_mlp((X_train_s.shape[1],), X_train_s, X_test_s, y_train_s,
                                                             y_test_s, y_train, y_test, scaler_y, y_train_s,
                                                             y_test_s, checkpoint_filepath, params_filepath)

    model_mlp.fit(X_train_s,
                  y_train_s,
                  epochs=current_epochs,
                  batch_size=128,
                  validation_data=(X_test_s, y_test_s),
                  validation_batch_size=128,
                  callbacks=[early_stop_callback, model_checkpoint_callback],
                  verbose=0,
                  shuffle=False)

    y_test_pred_scaled = model_mlp.predict(X_test_s)
    results['MLP'] = scaler_y.inverse_transform(y_test_pred_scaled)

    # ---  Random Forest Regression---
    print('RFR_Stat_Model')
    model_rfr = get_rfr(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y)
    model_rfr.fit(X_train_s, y_train_s)
    y_test_pred = model_rfr.predict(X_test_s).reshape(-1, 1)
    results['RandomForest'] = scaler_y.inverse_transform(y_test_pred)

    # ---  LSTM with window direct forecast---
    lstm_multi_results = train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    lstm_multi_results = np.array(lstm_multi_results).reshape(-1, 4)
    # --- Boosting with window direct forecast---
    cbr_multi_results = train_cbr_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    cbr_multi_results = np.array(cbr_multi_results).reshape(-1, 4)
    # --- Random Forest Regression with window direct forecast---
    rfr_multi_results = train_rfr_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    rfr_multi_results = np.array(rfr_multi_results).reshape(-1, 4)
    # --- MLP with window direct forecast---
    mlp_multi_results = train_mlp_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    mlp_multi_results = np.array(mlp_multi_results).reshape(-1, 4)
    # --- DLinearModel with window direct forecast---
    dlm_multi_results = train_dlm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    dlm_multi_results = np.array(dlm_multi_results).reshape(-1, 4)
    # --- DLinearModel with window direct forecast---
    lgb_multi_results = train_lgbm_darts_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    lgb_multi_results = np.array(lgb_multi_results).reshape(-1, 4)
    # --- Google TiDE with window direct forecast---
    tide_multi_results = train_tide_darts_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    tide_multi_results = np.array(tide_multi_results).reshape(-1, 4)
    # --- nixtla_lgb with window direct forecast---
    nixtla_lgb = train_nixtla_lgb_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    nixtla_lgb = np.array(nixtla_lgb).reshape(-1, 4)
    # --- NHITS with window direct forecast---
    nhits = train_nhits_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    nhits = np.array(nhits).reshape(-1, 4)
    # --- TSMixer with window direct forecast---
    tsmixer = train_tsmixer_darts_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    tsmixer = np.array(tsmixer).reshape(-1, 4)
    # --- XGB with window direct forecast---
    xgb = train_xgb_darts_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    xgb = np.array(xgb).reshape(-1, 4)
    # --- Darts ensemble with window direct forecast---
    dartsensemble = train_direct_ensemble_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    dartsensemble = np.array(dartsensemble).reshape(-1, 4)
    # --- Chronos with window direct forecast---
    #chronos_multi_results = train_chronos_darts_multistep(data_path, checkpoint_dir, n_out, test_start_index)
    #chronos_multi_results = np.array(chronos_multi_results).reshape(-1, 4)

    # 3. Model Evaluation & Comparison
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON REPORT")
    print("=" * 50)

    report = []
    for name, pred in results.items():
        lost_days = len(y_test) - len(pred)

        y_true_final = y_test[lost_days:]
        y_pred_final = pred

        assert len(y_true_final) == len(y_pred_final), f"Рассинхрон длины в модели {name}!"

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

    lstm_metrics = lstm_multi_results.reshape(-1, 4)
    cbr_metrics = cbr_multi_results.reshape(-1, 4)
    rf_metrics = rfr_multi_results.reshape(-1, 4)
    mlp_metrics = mlp_multi_results.reshape(-1, 4)
    dlm_metrics = dlm_multi_results.reshape(-1, 4)
    lgb_metrics = lgb_multi_results.reshape(-1, 4)
    tide_metrics = tide_multi_results.reshape(-1, 4)
    nixtla_lgb = nixtla_lgb.reshape(-1, 4)
    nhits = nhits.reshape(-1, 4)
    tsmixer = tsmixer.reshape(-1, 4)
    xgb = xgb.reshape(-1, 4)
    dartsensemble = dartsensemble.reshape(-1, 4)
    #chronos_metrics = chronos_multi_results.reshape(-1, 4)

    print("\n" + "=" * 175)
    print("FINAL HORIZON COMPARISON REPORT")
    print("-" * 175)

    header = (f"{'Day':<4} | "
              f"{'LSTM MAE':<8} || "
              f"{'CBR MAE':<8} || "
              f"{'RFR MAE':<8} || "
              f"{'LGB MAE':<8} || "
              f"{'MLP MAE':<8} || "
              f"{'TiDE MAE':<8} || "
              f"{'DLM MAE':<8} || "
              f"{'nixtla_lgb MAE':<8} || "
              f"{'NHITS MAE':<8} || "
              f"{'TSMixer MAE':<8} || "
              f"{'XGB MAE':<8} || "
              f"{'Darts ensemble':<8} || "
              #f"{'CRN MAE':<8} | {'CRN R2':<6}"
              )
    print(header)
    print("-" * 175)

    for i in range(lstm_metrics.shape[0]):
        print(f"{i + 1:<4} | "
              f"{lstm_metrics[i][0]:<8.4f} || "
              f"{cbr_metrics[i][0]:<8.4f} || "
              f"{rf_metrics[i][0]:<8.4f} || "
              f"{lgb_metrics[i][0]:<8.4f} || "
              f"{mlp_metrics[i][0]:<8.4f} || "
              f"{tide_metrics[i][0]:<8.4f} || "
              f"{dlm_metrics[i][0]:<8.4f} || "
              f"{nixtla_lgb[i][0]:<8.4f} || "
              f"{nhits[i][0]:<8.4f} || "
              f"{tsmixer[i][0]:<8.4f} || "
              f"{xgb[i][0]:<8.4f} || "
              f"{dartsensemble[i][0]:<8.4f} || "
              #f"{chronos_metrics[i][0]:<8.4f} | {chronos_metrics[i][3]:<6.4f}"
              )

    print("=" * 175)
if __name__ == "__main__":
    main()