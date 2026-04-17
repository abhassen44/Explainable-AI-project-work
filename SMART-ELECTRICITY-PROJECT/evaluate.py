"""
evaluate.py — Smart Electricity Project: Generate & Save All Key Metric Plots (PNG)

Outputs saved to the project folder:
  1. actual_vs_predicted.png       — Scatter plot + perfect-fit line
  2. prediction_trend.png          — Time-series: Actual vs Predicted (first 200 hours)
  3. residual_analysis.png         — Residual histogram + residual over time
  4. regression_metrics_summary.png — RMSE, MAE, R², MAPE in a visual card
  5. feature_correlation.png       — Feature correlation heatmap
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 1.  LOAD ARTIFACTS
# ------------------------------------------------------------------
print("1/6  Loading model & data …")

data = pd.read_csv(os.path.join(SAVE_DIR, "simulation_data.csv"))
feature_names = data.drop("Actual_kW", axis=1).columns.tolist()

x_scaler = joblib.load(os.path.join(SAVE_DIR, 'x_scaler.pkl'))
y_scaler = joblib.load(os.path.join(SAVE_DIR, 'y_scaler.pkl'))

input_dim = len(feature_names)
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "chefs_model.pth"),
                                  map_location=torch.device('cpu')))
model.eval()

# ------------------------------------------------------------------
# 2.  GENERATE PREDICTIONS
# ------------------------------------------------------------------
print("2/6  Running predictions …")

X_raw   = data.drop("Actual_kW", axis=1).values
actual  = data["Actual_kW"].values

X_tensor = torch.tensor(X_raw, dtype=torch.float32)
with torch.no_grad():
    preds_scaled = model(X_tensor).numpy().flatten()

# Inverse-transform predictions back to kW
data_min = y_scaler.data_min_[0]
data_max = y_scaler.data_max_[0]
preds = preds_scaled * (data_max - data_min) + data_min

# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
rmse  = np.sqrt(mean_squared_error(actual, preds))
mae   = mean_absolute_error(actual, preds)
r2    = r2_score(actual, preds)
mape  = np.mean(np.abs((actual - preds) / np.where(actual == 0, 1, actual))) * 100
residuals = actual - preds

print(f"   RMSE  : {rmse:.4f} kW")
print(f"   MAE   : {mae:.4f} kW")
print(f"   R²    : {r2:.4f}")
print(f"   MAPE  : {mape:.2f}%")

# ------------------------------------------------------------------
# PLOT  1 — Actual vs Predicted Scatter
# ------------------------------------------------------------------
print("3/6  Actual vs Predicted Scatter …")

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(actual, preds, alpha=0.5, s=20, c='#3498db', edgecolors='none')
lims = [min(actual.min(), preds.min()), max(actual.max(), preds.max())]
ax.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Power (kW)', fontsize=12)
ax.set_ylabel('Predicted Power (kW)', fontsize=12)
ax.set_title(f'Actual vs Predicted  (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'actual_vs_predicted.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT  2 — Prediction Trend  (Time-series overlay)
# ------------------------------------------------------------------
print("4/6  Prediction Trend …")

fig, ax = plt.subplots(figsize=(14, 5))
hours = np.arange(len(actual))
ax.plot(hours, actual, label='Actual', color='#2c3e50', lw=1.5, alpha=0.7)
ax.plot(hours, preds,  label='Predicted', color='#e74c3c', lw=1.5, ls='--')
ax.fill_between(hours, actual, preds, alpha=0.15, color='#e74c3c')
ax.set_xlabel('Hour Index', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title(f'Prediction Trend — RMSE: {rmse:.3f} kW | MAE: {mae:.3f} kW',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'prediction_trend.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT  3 — Residual Analysis (histogram + over time)
# ------------------------------------------------------------------
print("5/6  Residual Analysis …")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(residuals, kde=True, ax=ax1, color='#34495e', bins=30)
ax1.axvline(0, color='red', linestyle='--', lw=1.5)
ax1.set_xlabel('Residual (kW)', fontsize=12)
ax1.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Residuals over time
ax2.scatter(hours, residuals, s=15, alpha=0.6, c=['#2ecc71' if r >= 0 else '#e74c3c' for r in residuals],
            edgecolors='none')
ax2.axhline(0, color='black', lw=1)
ax2.axhline(np.mean(residuals), color='blue', lw=1.5, ls='--', label=f'Mean = {np.mean(residuals):.4f}')
ax2.set_xlabel('Hour Index', fontsize=12)
ax2.set_ylabel('Residual (kW)', fontsize=12)
ax2.set_title('Residuals Over Time', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'residual_analysis.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT  4 — Regression Metrics Summary Card
# ------------------------------------------------------------------
print("6/6  Metrics Summary Card …")

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

metrics = [
    ('RMSE',  f'{rmse:.4f} kW',  '#e74c3c'),
    ('MAE',   f'{mae:.4f} kW',   '#f39c12'),
    ('R²',    f'{r2:.4f}',       '#2ecc71'),
    ('MAPE',  f'{mape:.2f}%',    '#3498db'),
]

for i, (name, value, color) in enumerate(metrics):
    x_pos = 0.125 + i * 0.25
    # Card background
    rect = plt.Rectangle((x_pos - 0.09, 0.15), 0.18, 0.7,
                          facecolor=color, alpha=0.15, edgecolor=color,
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    # Value
    ax.text(x_pos, 0.55, value, transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='center', va='center', color=color)
    # Label
    ax.text(x_pos, 0.25, name, transform=ax.transAxes,
            fontsize=14, ha='center', va='center', color='#333333')

ax.set_title('Regression Metrics Summary — CHEFS Energy Model',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'regression_metrics_summary.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT  5 — Feature Correlation Heatmap
# ------------------------------------------------------------------
print("   Bonus: Feature Correlation Heatmap …")

corr = data.drop("Actual_kW", axis=1).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'feature_correlation.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
print(f"\n✅ ALL PLOTS SAVED to: {SAVE_DIR}")
print("   → actual_vs_predicted.png")
print("   → prediction_trend.png")
print("   → residual_analysis.png")
print("   → regression_metrics_summary.png")
print("   → feature_correlation.png")
