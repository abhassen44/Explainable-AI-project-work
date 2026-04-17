import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import seaborn as sns
import joblib  # For saving scalers

# ==========================================
# 0. CONFIGURATION & STYLE
# ==========================================
# Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Hardware: {device}")

plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DATA PIPELINE (Optimized)
# ==========================================
def load_and_preprocess():
    print("\n[1/4] Reading and Cleaning Dataset...")
    df = pd.read_csv("household_power_consumption.txt", sep=";", 
                     low_memory=False, na_values=["nan", "?"])
    
    # Fast datetime conversion
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.drop(columns=["Date", "Time"]).set_index("dt")
    
    # Numeric conversion & Imputation
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    
    # Resample to Hourly (Smoothing)
    df = df.resample("1h").mean()
    
    # Feature Engineering
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    
    # Target: Predict NEXT hour
    df["target"] = df["Global_active_power"].shift(-1)
    df.dropna(inplace=True)
    
    return df

df = load_and_preprocess()
feature_names = df.drop("target", axis=1).columns.tolist()

# Scaling
print("[2/4] Scaling and Splitting...")
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_raw = df.drop("target", axis=1).values
y_raw = df["target"].values.reshape(-1, 1)

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# Time-Series Split (80/20)
split = int(0.8 * len(X_scaled))

# Convert to Tensors (move to device)
X_train = torch.FloatTensor(X_scaled[:split]).to(device)
X_test = torch.FloatTensor(X_scaled[split:]).to(device)
y_train = torch.FloatTensor(y_scaled[:split]).to(device)
y_test = torch.FloatTensor(y_scaled[split:]).to(device)

# ==========================================
# 2. MODEL & TRAINING
# ==========================================
print("[3/4] Training Neural Network...")

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")

# ==========================================
# 3. EVALUATION & VISUALIZATION
# ==========================================
print("\n[4/4] Generating Visualizations...")

model.eval()
with torch.no_grad():
    preds_scaled = model(X_test).cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()
    
# Inverse scaling for real-world units (kW)
preds = y_scaler.inverse_transform(preds_scaled)
actual = y_scaler.inverse_transform(y_test_cpu)

# Calculate Metrics
rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)

# --- FIGURE 1: Global Performance Dashboard ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2)

# Plot A: Forecast Fidelity
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(actual[:150], label="Actual (kW)", color='#2c3e50', alpha=0.5, lw=2)
ax1.plot(preds[:150], label="Predicted (kW)", color='#e74c3c', ls='--', lw=2)
ax1.set_title(f"Forecast Fidelity (RMSE: {rmse:.3f} kW | MAE: {mae:.3f} kW)", fontsize=14, loc='left', fontweight='bold')
ax1.legend()

# Plot B: Global Importance (SHAP Beeswarm)
# Move model to CPU for SHAP to avoid errors
model.cpu()
X_train_cpu = X_train.cpu()
X_test_cpu = X_test.cpu()

# Use independent masker for speed
masker = shap.maskers.Independent(X_train_cpu[:100].numpy(), max_samples=100)
explainer = shap.Explainer(
    lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(), 
    masker
)
shap_values = explainer(X_test_cpu[:200].numpy()) # Calculate for 200 samples
shap_values.feature_names = feature_names

ax2 = fig.add_subplot(gs[1, 0])
plt.sca(ax2)
shap.plots.beeswarm(shap_values, show=False)
ax2.set_title("Global Feature Impact (SHAP Beeswarm)", fontsize=14, loc='left')

# Plot C: Error Distribution
ax3 = fig.add_subplot(gs[1, 1])
residuals = actual - preds
sns.histplot(residuals, kde=True, ax=ax3, color='#34495e')
ax3.axvline(0, color='red', linestyle='--')
ax3.set_title("Error Distribution (Residuals)", fontsize=14, loc='left')
ax3.set_xlabel("Error (kW)")

plt.tight_layout()
plt.show()

# --- FIGURE 2: Local Explanation (SHAP Waterfall) ---
print("   Generating SHAP Waterfall Plot for Single Sample...")
plt.figure() # Create new figure
shap.plots.waterfall(shap_values[0], show=True) # Plot the first sample

# --- FIGURE 3: LIME Local Explanation ---
print("   Generating LIME Explanation for Single Sample...")

# Create prediction function for LIME
def predict_fn(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return preds.flatten()

# Initialize LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_cpu.numpy(),
    feature_names=feature_names,
    mode='regression',
    verbose=False
)

# Explain first test sample
sample_idx = 0
lime_exp = lime_explainer.explain_instance(
    X_test_cpu[sample_idx].numpy(),
    predict_fn,
    num_features=len(feature_names)
)

# Plot LIME explanation
fig_lime, ax_lime = plt.subplots(figsize=(10, 6))
lime_weights = dict(lime_exp.as_list())
sorted_features = sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)
labels = [f[0] for f in sorted_features]
values = [f[1] for f in sorted_features]
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

ax_lime.barh(range(len(labels)), values, color=colors)
ax_lime.set_yticks(range(len(labels)))
ax_lime.set_yticklabels(labels, fontsize=9)
ax_lime.set_xlabel("Feature Weight (Local Linear Model)")
ax_lime.set_title("LIME Local Explanation (Sample #1)", fontsize=14, fontweight='bold')
ax_lime.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

# Add value labels
for i, v in enumerate(values):
    ax_lime.text(v, i, f" {v:.4f}", va='center', fontsize=8)

plt.tight_layout()
plt.show()

print(f"   LIME R2 Score: {lime_exp.score:.4f}")

# ==========================================
# 4. SAVE ARTIFACTS (For Dashboard/Demo)
# ==========================================
print("\n Saving Project Artifacts...")

# 1. Save Model
torch.save(model.state_dict(), "chefs_model.pth")
print("    Model saved: 'chefs_model.pth'")

# 2. Save Scalers (Essential for the dashboard)
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
print("    Scalers saved: 'x_scaler.pkl', 'y_scaler.pkl'")

# 3. Save Simulation Data (Last 200 hours for live demo)
sim_df = pd.DataFrame(X_test_cpu.numpy()[-200:], columns=feature_names)
sim_df['Actual_kW'] = actual[-200:]
sim_df.to_csv("simulation_data.csv", index=False)
print("   Test Data saved: 'simulation_data.csv'")

print("\n DONE! You are ready to run 'dashboard.py'.")