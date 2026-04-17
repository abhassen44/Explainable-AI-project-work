# ⚡ CHEFS: Smart Electricity AI Project

## Overview
This project predicts household electricity consumption using deep learning, providing highly accurate next-hour forecasting. To ensure trust and transparency in the AI's predictions, the project integrates two leading Explainable AI (XAI) frameworks — **SHAP** and **LIME** — wrapped in an interactive, real-time Streamlit dashboard.

## Objectives
1. **Forecast:** Predict the specific next-hour global active power (kW) using historical sensor data and time features.
2. **Explain:** Break down the AI's prediction into explicitly interpretable feature contributions.
3. **Visualize:** Provide an intuitive, real-time UI for monitoring trends, analyzing errors, and viewing XAI metrics side-by-side.

---

## Architecture & Algorithm

- **Model:** Feedforward Neural Network (PyTorch)
- **Architecture:** 3 linear layers with ReLU activations (Input [10 features] → 128 → 64 → Output [1])
- **Loss & Optimizer:** Mean Squared Error (MSE), Adam optimizer (lr=0.001)

### Why this algorithm?
Neural networks are highly effective for modeling complex, non-linear relationships in time-series data. By combining raw power measurements with engineered temporal features (hour, weekday, month), the MLP successfully identifies consumption patterns to forecast future usage efficiently.

---

## Dataset & Preprocessing

**Dataset:** UCI Household Power Consumption dataset (~2 Million records). [(Dataset Link)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set/data)

**Pipeline:**
1. **Cleaning:** Handles missing values via median imputation.
2. **Resampling:** Aggregates minute-by-minute data into **Hourly** averages to smooth noise.
3. **Feature Engineering:** Extracts `hour`, `weekday`, and `month` from the datetime index.
4. **Target:** Future shifting — the model predicts `Global_active_power` for `t+1`.
5. **Scaling:** Uses `MinMaxScaler` for both inputs (10 features) and the target output.

---

## Project Structure

```text
SMART-ELECTRICITY-PROJECT/
│
├── train.py                 # Data processing, PyTorch model training, and global metric generation
├── dashboard.py             # Streamlit dashboard featuring live simulation and XAI graphs
├── evaluate.py              # Generates standalone offline regression metrics and plots
│
├── chefs_model.pth          # Saved PyTorch model state_dict
├── x_scaler.pkl             # MinMaxScaler for inputs
├── y_scaler.pkl             # MinMaxScaler for outputs
├── simulation_data.csv      # Exported test subset (latest 200 hrs) specifically for dashboard use
│
└── household_power_consumption.txt  # Raw dataset file (must be downloaded)
```

---

## Features & Usage

### 1. Training the Model
Run the primary training script to process the data, train the PyTorch MLP, and save the necessary models/scalers and artifact `.png` generated graphs.
```bash
python train.py
```

### 2. Live Evaluator Dashboard
Launch the interactive Streamlit dashboard. The dashboard pulls from `simulation_data.csv` to mimic a live real-time stream.
```bash
streamlit run dashboard.py
```
**Dashboard Features:**
- **Time Navigation Slider:** Scrub through 200 hours of simulated testing data.
- **Trend Charts:** Interactive Plotly graphs comparing actual vs predicted power.
- **Error Tracking:** Real-time percentage error and KPI metric monitoring.

### 3. Explainable AI (XAI)
The dashboard features dedicated modules for AI transparency:
- **SHAP (DeepExplainer):** Uses game theory to distribute the predicted value amongst the 10 input features. Displays exactly how many kW each feature added or subtracted from the baseline.
- **LIME (Tabular Regression):** Creates a local proxy linear model to weight the importance of features for that specific hourly prediction.

---

## Requirements

- Python 3.9+
- PyTorch
- Streamlit
- SHAP & lime
- Plotly, Matplotlib, Seaborn
- Pandas, Scikit-Learn
*(See `requirements.txt` for exact versions)*

