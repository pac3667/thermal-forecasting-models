markdown
# Thermal Energy Consumption Forecasting (TEC)

This project provides a comprehensive machine learning pipeline for predicting thermal energy consumption (Gcal/h) based on meteorological data and historical lags. It implements various strategies from classical regression to deep learning and multi-step direct forecasting.

##  Features

- **Multiple Architectures**: Linear Regression, Polynomial Features, Random Forest, CatBoost, MLP, and LSTM.
- **Direct Multi-step Forecasting**: Predicts a 14-day horizon using a chain of independent models.
- **Time-Series Engineering**: Implementation of sin/cos month encoding, rolling averages, and historical lags.
- **Production-Ready**: Automatic checkpoint saving/loading and standardized evaluation metrics.

##  Project Structure

```text
├── data/                           # Source CSV datasets
├── checkpoint/                     # Saved model weights (.keras files)
├── src/                            # Utility scripts
│   ├── models.py                   # Model architectures (LSTM, MLP, CatBoost)
│   ├── CBRDirectForecast.py        # Direct multi-step logic (CatBoost)
│   └── LSTMDirectForecast.py       # Direct multi-step logic (LSTM)
├── main.py                         # Main entry point for training and evaluation
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

##  Model Evaluation Strategy

The project compares models on a fixed test set using the following metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R2 Score** (Coefficient of Determination)

### Multi-step Horizon Analysis
One of the key features is the comparison between **LSTM** and **CatBoost** over a 14-day forecasting horizon, analyzing how prediction error increases as the horizon expands.

##  Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com
   cd TEC_Qpred
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   python main.py
   ```

##  Results Preview

The system generates a consolidated report in the console and exports detailed predictions to `final_predictions_comparison.xlsx`. 


| Model | MAE | R2 Score |
| :--- | :--- | :--- |
| **CatBoost (Windowed)** | ~44.4 | 0.94 |
| **LSTM (Direct 1d)** | ~45.6 | 0.94 |
| **Linear Regression** | ~59.4 | 0.91 |

##  Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** (Deep Learning)
- **CatBoost** (Gradient Boosting)
- **Scikit-Learn** (Classical ML & Preprocessing)
- **Pandas/NumPy** (Data Manipulation)
- **Matplotlib** (Visualization)