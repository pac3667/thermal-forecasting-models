import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import datasets, linear_model, metrics, model_selection, __all__, ensemble
import seaborn as sns

data = pd.read_csv(r'C:\Users\guryanov\PycharmProjects\TEC_Qpred\data\TEC22_Data.csv', delimiter=';', parse_dates=['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['Q_rolling_3'] = data['TEC_Q_Aver'].rolling(window=3).mean()
data.set_index('Date', inplace=True)

n_out = 14

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

metrics_list = []
results_list = np.array(())
for i in range(0, n_out, 1):
    print(f"\n=== Обучение шага {i + 1} ===")

    x = data_with_lag[['T','TEC_Q_Aver', 'Year', 'Month_sin', 'Month_cos', f'T_mean{i+1}', f'T_lag{i+1}', 'Q_rolling_3']].values
    y = data_with_lag.loc[:, [f'Q_lag{i+1}']].values
    X_train, X_test = x[:3258], x[3258:]
    y_train, y_test = y[:3258], y[3258:]

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

    model.fit(X_train_scaled, y_train_scaled, eval_set=(X_test_scaled, y_test_scaled), early_stopping_rounds=50)

    y_test_pred = model.predict(X_test_scaled).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(y_test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_scaled)

    MAE_test = metrics.mean_absolute_error(y_test, yhat)
    MSE_test = metrics.mean_squared_error(y_test, yhat)
    MAPE_test = metrics.mean_absolute_percentage_error(y_test, yhat)
    R2_test = metrics.r2_score(y_test, yhat)
    print(f"MAE_test: {MAE_test:.2f}")
    print(f"MSE_test: {MSE_test:.2f}")
    print(f"MAPE_test: {MAPE_test:.2f}")
    print(f"R2_test: {R2_test:.2f}")
    metrics_list.append([MAE_test,MSE_test,MAPE_test,R2_test])

    if i == 0:
        results_list = yhat
    else:
        results_list = np.hstack((results_list, yhat))

    # plot history

metrics_np = np.array(metrics_list)

sns.set_style("whitegrid")

# Увеличиваем базовый шрифт на треть (был 16, стал ~22)
plt.rcParams.update({'font.size': 22})

fig, ax1 = plt.subplots(figsize=(20, 11)) # Немного увеличили размер окна для крупных шрифтов

days = np.arange(1, n_out + 1)
mae_values = metrics_np[:, 0]
r2_values = metrics_np[:, 3]

# График MAE (Bars)
color_mae = '#3498db'
ax1.set_xlabel('Forecast Horizon (Days)', fontsize=28, fontweight='bold', labelpad=20)
ax1.set_ylabel('MAE (Mean Absolute Error)', color=color_mae, fontsize=28, fontweight='bold', labelpad=20)
bars = ax1.bar(days, mae_values, color=color_mae, alpha=0.7, label='MAE', edgecolor='black', linewidth=1.5)

# Настройка делений (Ticks)
ax1.tick_params(axis='y', labelcolor=color_mae, labelsize=22)
ax1.tick_params(axis='x', labelsize=22)

# Вторая ось для R2 (Line)
ax2 = ax1.twinx()
color_r2 = '#e74c3c'
ax2.set_ylabel('R2 Score (Accuracy)', color=color_r2, fontsize=28, fontweight='bold', labelpad=20)
ax2.plot(days, r2_values, color=color_r2, marker='o', markersize=14, linewidth=6, label='R2 Score')
ax2.tick_params(axis='y', labelcolor=color_r2, labelsize=22)

# Лимиты осей
ax1.set_ylim(0, max(mae_values) * 1.3) # Динамический запас сверху
ax2.set_ylim(min(r2_values) - 0.55, 1.1)

# Заголовок (увеличен на треть)
plt.title('Prediction Quality per Forecast Day\n(Step-by-Step Gradient booster Training)',
          fontsize=32, fontweight='bold', pad=40)

plt.xticks(days)

# Легенда (крупнее и ниже)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=24, shadow=True)

# Подписи MAE внутри столбцов (увеличены)
for bar in bars:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        yval * 0.85,
        f'{yval:.1f}',
        ha='center', va='top',
        fontsize=18, fontweight='bold', color='white'
    )

# Подписи R2 над точками (увеличены)
for x, y in zip(days, r2_values):
    ax2.text(
        x, y + 0.02,
        f'{y:.2f}',
        ha='center', va='bottom',
        fontsize=18, fontweight='bold', color=color_r2,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
    )

plt.tight_layout()
plt.show()