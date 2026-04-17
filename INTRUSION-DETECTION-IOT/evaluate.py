"""
evaluate.py — DNN Evaluation (paper-accurate config)

Paper: "Explainable AI for Intrusion Detection in IoT Networks"

Pipeline mirrors train_main.py:
  - 5-class mapping (DoS, Normal, Probe, R2L, U2R)
  - 36 features (input_dim=36)
  - Dropout 0.01, Adam(lr=0.001, weight_decay=0.0001)
  - 20 epochs

Outputs saved to 'dnn results/' folder:
   1. confusion_matrix.png
   2. classification_report.png
   3. roc_auc_curve.png
   4. accuracy_loss_summary.png
   5. class_distribution.png
   6. feature_importance.png
   7. precision_recall_curve.png
   8. score_summary.png
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    f1_score, precision_score, recall_score,
    r2_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, hamming_loss,
    precision_recall_curve as sk_pr_curve, average_precision_score
)
from tensorflow.keras.callbacks import EarlyStopping
from model_dnn import build_dnn_model

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH  = 'KDDTrain+.txt'
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dnn results')
EPOCHS     = 20        # paper spec
BATCH_SIZE = 32

COL_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# Drop 5 highly-correlated features → keeps 36 features
DROP_COLS = [
    'srv_serror_rate', 'srv_rerror_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_srv_rerror_rate'
]

# NSL-KDD attack → 5-class category mapping (paper spec)
ATTACK_MAP = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'xlock': 'R2L',
    'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}

CLASS_NAMES = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

# ------------------------------------------------------------------
# 0.  Create output directory
# ------------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1.  LOAD & PREPROCESS
# ------------------------------------------------------------------
print("1/9  Loading & preprocessing (5-class, 36 features) …")
df = pd.read_csv(DATA_PATH, names=COL_NAMES)
df = df.drop('difficulty', axis=1)

# Map to 5 categories
df['label'] = df['label'].map(ATTACK_MAP).fillna('Normal')

encoder_dict = {}
for col in ["protocol_type", "service", "flag", "label"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoder_dict[col] = le

df = df.drop(columns=DROP_COLS)

X = df.drop('label', axis=1)
y = df['label']
feature_names = X.columns.tolist()
class_names   = CLASS_NAMES

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

num_classes = len(np.unique(y))   # 5
print(f"   Features: {len(feature_names)} | Classes: {num_classes} → {class_names}")

# ------------------------------------------------------------------
# 2.  TRAIN MODEL  +  CAPTURE HISTORY
# ------------------------------------------------------------------
print("2/9  Training DNN (20 epochs) to capture metrics history …")
model = build_dnn_model(input_dim=X_train.shape[1], num_classes=num_classes)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

# Predictions
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_prob = model.predict(X_test, verbose=0)

print(f"   Test Accuracy : {acc*100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")

# ------------------------------------------------------------------
# PLOT 1 — Confusion Matrix
# ------------------------------------------------------------------
print("3/9  Confusion Matrix …")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix — DNN on NSL-KDD (5-class)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 2 — Classification Report (Precision / Recall / F1)
# ------------------------------------------------------------------
print("4/9  Classification Report …")

unique_labels = sorted(np.unique(np.concatenate([np.unique(y_test), np.unique(y_pred)])))
used_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]

report = classification_report(
    y_test, y_pred,
    labels=unique_labels,
    target_names=used_class_names,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).T

rows_to_keep = used_class_names + ['macro avg', 'weighted avg']
report_df = report_df.loc[
    [r for r in rows_to_keep if r in report_df.index],
    ['precision', 'recall', 'f1-score']
]

fig, ax = plt.subplots(figsize=(10, 6))
report_df.plot(kind='barh', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_xlabel('Score', fontsize=12)
ax.set_title('Classification Report — Precision / Recall / F1-Score (DNN)',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.legend(loc='lower right')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'classification_report.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 3 — ROC-AUC Curve (One-vs-Rest)
# ------------------------------------------------------------------
print("5/9  ROC-AUC Curves …")

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for idx, i in enumerate(unique_labels):
    y_true_bin = (np.array(y_test) == i).astype(int)
    if i < y_prob.shape[1]:
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
    else:
        continue
    roc_auc = auc(fpr, tpr)
    label_name = class_names[i] if i < len(class_names) else f'Class {i}'
    ax.plot(fpr, tpr, color=colors[idx], lw=2,
            label=f'{label_name}  (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC-AUC Curves — One-vs-Rest (DNN)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'roc_auc_curve.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 4 — Training Accuracy & Loss History
# ------------------------------------------------------------------
print("6/9  Accuracy & Loss History …")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],    label='Train Accuracy', lw=2, color='#2ecc71')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy',  lw=2, color='#3498db', ls='--')
ax1.set_title('Model Accuracy', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history.history['loss'],     label='Train Loss', lw=2, color='#e74c3c')
ax2.plot(history.history['val_loss'], label='Val Loss',   lw=2, color='#f39c12', ls='--')
ax2.set_title('Model Loss', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'accuracy_loss_summary.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 5 — Class Distribution
# ------------------------------------------------------------------
print("7/9  Class Distribution …")

class_counts = pd.Series(y).map(lambda x: class_names[x]).value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(class_counts.index, class_counts.values,
              color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6'])
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('NSL-KDD — 5-Class Distribution', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%d', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'class_distribution.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 6 — Permutation-based Feature Importance (top 15)
# ------------------------------------------------------------------
print("8/9  Feature Importance (permutation-based) …")

baseline_acc = accuracy_score(y_test, y_pred)
importances = []

for i, fname in enumerate(feature_names):
    X_test_perm = X_test.copy()
    np.random.seed(42)
    X_test_perm[:, i] = np.random.permutation(X_test_perm[:, i])
    y_perm_pred = np.argmax(model.predict(X_test_perm, verbose=0), axis=1)
    perm_acc = accuracy_score(y_test, y_perm_pred)
    importances.append(baseline_acc - perm_acc)

imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], color='#2980b9')
ax.set_xlabel('Accuracy Drop (higher = more important)', fontsize=12)
ax.set_title('Top 15 Feature Importance — Permutation (DNN)',
             fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%.4f', fontsize=8, padding=3)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 7 — Precision-Recall Curves (One-vs-Rest)
# ------------------------------------------------------------------
print("   Bonus: Precision-Recall Curves …")

y_test_bin = label_binarize(np.array(y_test), classes=list(range(num_classes)))

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for idx, i in enumerate(unique_labels):
    if i < y_prob.shape[1]:
        prec, rec, _ = sk_pr_curve(y_test_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
        label_name = class_names[i] if i < len(class_names) else f'Class {i}'
        ax.plot(rec, prec, color=colors[idx], lw=2,
                label=f'{label_name}  (AP = {ap:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — One-vs-Rest (DNN)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'precision_recall_curve.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 8 — Comprehensive Score Summary Table
# ------------------------------------------------------------------
print("9/9  Score Summary Table …")

y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)

metrics = {
    'Accuracy':            accuracy_score(y_test_arr, y_pred_arr),
    'F1 Score (macro)':    f1_score(y_test_arr, y_pred_arr, average='macro', zero_division=0),
    'F1 Score (weighted)': f1_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0),
    'Precision (macro)':   precision_score(y_test_arr, y_pred_arr, average='macro', zero_division=0),
    'Precision (weighted)':precision_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0),
    'Recall (macro)':      recall_score(y_test_arr, y_pred_arr, average='macro', zero_division=0),
    'Recall (weighted)':   recall_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0),
    'R² Score':            r2_score(y_test_arr, y_pred_arr),
    'Matthews Corr Coeff': matthews_corrcoef(y_test_arr, y_pred_arr),
    "Cohen's Kappa":       cohen_kappa_score(y_test_arr, y_pred_arr),
    'Log Loss':            log_loss(y_test_arr, y_prob, labels=list(range(num_classes))),
    'Hamming Loss':        hamming_loss(y_test_arr, y_pred_arr),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')
ax.set_title('Comprehensive Score Summary — DNN (Paper Config)',
             fontsize=14, fontweight='bold', pad=20)

table = ax.table(
    cellText=[[m, f'{v:.4f}'] for m, v in metrics.items()],
    colLabels=['Metric', 'Value'],
    cellLoc='center', loc='center',
    colWidths=[0.55, 0.25]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

for j in range(2):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

for i in range(1, len(metrics) + 1):
    for j in range(2):
        cell = table[i, j]
        cell.set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'score_summary.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
print(f"\n✅ ALL PLOTS SAVED to: {SAVE_DIR}")
print("   → confusion_matrix.png")
print("   → classification_report.png")
print("   → roc_auc_curve.png")
print("   → accuracy_loss_summary.png")
print("   → class_distribution.png")
print("   → feature_importance.png")
print("   → precision_recall_curve.png")
print("   → score_summary.png")
print("\n📊 Score Highlights:")
for name, val in metrics.items():
    print(f"   {name:.<30s} {val:.4f}")
