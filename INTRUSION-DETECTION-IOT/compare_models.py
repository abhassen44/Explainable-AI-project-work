"""
compare_models.py — Side-by-side comparison of DNN vs 2D-CNN on NSL-KDD
                     (paper-accurate config)

Paper: "Explainable AI for Intrusion Detection in IoT Networks"

Config:
  - 5-class mapping (DoS, Normal, Probe, R2L, U2R)
  - 36 features after Pearson Correlation feature selection
  - Dropout 0.01, Adam(lr=0.001, weight_decay=0.0001)
  - 20 epochs

Outputs saved to 'comparison/' folder:
   1. accuracy_comparison.png
   2. metric_comparison_table.png
   3. confusion_matrix_comparison.png
   4. roc_auc_comparison.png
   5. training_history_comparison.png
   6. classification_report_comparison.png
   7. precision_recall_comparison.png
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
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
from model_2dcnn import build_2dcnn_model

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH  = 'KDDTrain+.txt'
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison')
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

# Drop 5 correlated features → 36 remain (paper spec)
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


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """Compute all scalar metrics for a model."""
    return {
        'Accuracy':            accuracy_score(y_true, y_pred),
        'F1 (macro)':          f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 (weighted)':       f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Precision (macro)':   precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision (weighted)':precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall (macro)':      recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (weighted)':   recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'R² Score':            r2_score(y_true, y_pred),
        'MCC':                 matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa":       cohen_kappa_score(y_true, y_pred),
        'Log Loss':            log_loss(y_true, y_prob, labels=list(range(num_classes))),
        'Hamming Loss':        hamming_loss(y_true, y_pred),
    }


# ------------------------------------------------------------------
# 0.  Setup
# ------------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1.  LOAD & PREPROCESS (5-class, 36 features)
# ------------------------------------------------------------------
print("1/5  Loading & preprocessing (5-class, 36 features) …")
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

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

num_classes = len(np.unique(y))   # 5
y_test_arr  = np.array(y_test)

# Prepare 2D reshaped data (36 → 6×6×1)
X_train_2d = X_train.reshape(-1, 6, 6, 1)
X_test_2d  = X_test.reshape(-1, 6, 6, 1)

print(f"   Features: {X_train.shape[1]} | Classes: {num_classes} → {CLASS_NAMES}")

# NOTE: Create a FRESH EarlyStopping per model — reusing the same instance
# carries over internal state (best loss, wait counter) from DNN → CNN,
# which caused the 2D-CNN to stop after only ~4 epochs.
def make_early_stop():
    return EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ------------------------------------------------------------------
# 2.  TRAIN BOTH MODELS
# ------------------------------------------------------------------
print("2/5  Training DNN (20 epochs) …")
dnn = build_dnn_model(input_dim=X_train.shape[1], num_classes=num_classes)
hist_dnn = dnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, callbacks=[make_early_stop()], verbose=0)

dnn_pred = np.argmax(dnn.predict(X_test, verbose=0), axis=1)
dnn_prob = dnn.predict(X_test, verbose=0)
dnn_loss, dnn_acc = dnn.evaluate(X_test, y_test, verbose=0)
print(f"   DNN  — Accuracy: {dnn_acc*100:.2f}%")

print("3/5  Training 2D-CNN (20 epochs) …")
cnn2d = build_2dcnn_model(input_shape=(6, 6, 1), num_classes=num_classes)
hist_cnn = cnn2d.fit(X_train_2d, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.1, callbacks=[make_early_stop()], verbose=0)

cnn_pred = np.argmax(cnn2d.predict(X_test_2d, verbose=0), axis=1)
cnn_prob = cnn2d.predict(X_test_2d, verbose=0)
cnn_loss, cnn_acc = cnn2d.evaluate(X_test_2d, y_test, verbose=0)
print(f"   2D-CNN — Accuracy: {cnn_acc*100:.2f}%")

# Compute all metrics
dnn_metrics = compute_metrics(y_test_arr, dnn_pred, dnn_prob, num_classes)
cnn_metrics = compute_metrics(y_test_arr, cnn_pred, cnn_prob, num_classes)

unique_labels = sorted(np.unique(np.concatenate([np.unique(y_test),
                                                  np.unique(dnn_pred),
                                                  np.unique(cnn_pred)])))

# ------------------------------------------------------------------
# PLOT 1 — Accuracy Comparison Bar Chart
# ------------------------------------------------------------------
print("4/5  Generating comparison plots …")

fig, ax = plt.subplots(figsize=(8, 5))
models = ['DNN', '2D-CNN']
accs   = [dnn_metrics['Accuracy'], cnn_metrics['Accuracy']]
colors = ['#3498db', '#e74c3c']
bars   = ax.bar(models, accs, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.bar_label(bars, fmt='%.4f', fontsize=12, fontweight='bold', padding=5)
ax.set_ylim(min(accs) - 0.02, 1.005)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Model Accuracy Comparison (Paper Config)', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'accuracy_comparison.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 2 — Full Metrics Comparison Table
# ------------------------------------------------------------------
compare_df = pd.DataFrame({
    'Metric': list(dnn_metrics.keys()),
    'DNN':    [f'{v:.4f}' for v in dnn_metrics.values()],
    '2D-CNN': [f'{v:.4f}' for v in cnn_metrics.values()],
})

# Determine winner per row
winners = []
for m in dnn_metrics:
    d, c = dnn_metrics[m], cnn_metrics[m]
    if m in ('Log Loss', 'Hamming Loss'):
        winners.append('DNN' if d <= c else '2D-CNN')
    else:
        winners.append('DNN' if d >= c else '2D-CNN')
compare_df['Better'] = winners

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.set_title('Comprehensive Metric Comparison — DNN vs 2D-CNN (Paper Config)',
             fontsize=15, fontweight='bold', pad=20)

cell_text = compare_df.values.tolist()
table = ax.table(
    cellText=cell_text,
    colLabels=compare_df.columns.tolist(),
    cellLoc='center', loc='center',
    colWidths=[0.35, 0.17, 0.17, 0.17]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.55)

for j in range(4):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

for i in range(1, len(cell_text) + 1):
    bg = '#ecf0f1' if i % 2 == 0 else '#ffffff'
    for j in range(4):
        table[i, j].set_facecolor(bg)
    winner = cell_text[i - 1][3]
    win_col = 1 if winner == 'DNN' else 2
    table[i, win_col].set_text_props(fontweight='bold', color='#27ae60')
    table[i, 3].set_text_props(fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'metric_comparison_table.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 3 — Side-by-side Confusion Matrices
# ------------------------------------------------------------------
cm_dnn = confusion_matrix(y_test, dnn_pred)
cm_cnn = confusion_matrix(y_test, cnn_pred)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm_dnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
ax1.set_title('DNN — Confusion Matrix', fontsize=13, fontweight='bold')

sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
ax2.set_title('2D-CNN — Confusion Matrix', fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison (5-class)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'confusion_matrix_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 4 — Overlaid ROC-AUC Curves (macro-averaged)
# ------------------------------------------------------------------
y_test_bin = label_binarize(y_test_arr, classes=list(range(num_classes)))

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, probs, color, ls in [('DNN', dnn_prob, '#3498db', '-'),
                                       ('2D-CNN', cnn_prob, '#e74c3c', '--')]:
    fpr_all, tpr_all = [], []
    for i in unique_labels:
        if i < probs.shape[1]:
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)

    all_fpr = np.unique(np.concatenate(fpr_all))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_i, tpr_i in zip(fpr_all, tpr_all):
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(fpr_all)
    macro_auc = auc(all_fpr, mean_tpr)

    ax.plot(all_fpr, mean_tpr, color=color, lw=2.5, ls=ls,
            label=f'{model_name}  (macro AUC = {macro_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC-AUC Comparison — Macro Average', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'roc_auc_comparison.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 5 — Training History Comparison (Accuracy & Loss)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(hist_dnn.history['accuracy'],     label='Train', lw=2, color='#2ecc71')
axes[0, 0].plot(hist_dnn.history['val_accuracy'],  label='Val',   lw=2, color='#3498db', ls='--')
axes[0, 0].set_title('DNN — Accuracy', fontweight='bold')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Accuracy')

axes[0, 1].plot(hist_cnn.history['accuracy'],     label='Train', lw=2, color='#2ecc71')
axes[0, 1].plot(hist_cnn.history['val_accuracy'],  label='Val',   lw=2, color='#e74c3c', ls='--')
axes[0, 1].set_title('2D-CNN — Accuracy', fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')

axes[1, 0].plot(hist_dnn.history['loss'],     label='Train', lw=2, color='#e74c3c')
axes[1, 0].plot(hist_dnn.history['val_loss'], label='Val',   lw=2, color='#f39c12', ls='--')
axes[1, 0].set_title('DNN — Loss', fontweight='bold')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')

axes[1, 1].plot(hist_cnn.history['loss'],     label='Train', lw=2, color='#e74c3c')
axes[1, 1].plot(hist_cnn.history['val_loss'], label='Val',   lw=2, color='#f39c12', ls='--')
axes[1, 1].set_title('2D-CNN — Loss', fontweight='bold')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')

plt.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'training_history_comparison.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 6 — Per-class F1-Score Grouped Bar Chart
# ------------------------------------------------------------------
used_class_names = [CLASS_NAMES[i] for i in unique_labels if i < len(CLASS_NAMES)]

dnn_report = classification_report(y_test, dnn_pred, labels=unique_labels,
                                   target_names=used_class_names,
                                   output_dict=True, zero_division=0)
cnn_report = classification_report(y_test, cnn_pred, labels=unique_labels,
                                   target_names=used_class_names,
                                   output_dict=True, zero_division=0)

f1_dnn = [dnn_report[c]['f1-score'] for c in used_class_names]
f1_cnn = [cnn_report[c]['f1-score'] for c in used_class_names]

x = np.arange(len(used_class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width/2, f1_dnn, width, label='DNN', color='#3498db', edgecolor='white')
b2 = ax.bar(x + width/2, f1_cnn, width, label='2D-CNN', color='#e74c3c', edgecolor='white')
ax.bar_label(b1, fmt='%.2f', fontsize=9, padding=2)
ax.bar_label(b2, fmt='%.2f', fontsize=9, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(used_class_names, fontsize=11)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class F1-Score Comparison — DNN vs 2D-CNN',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'classification_report_comparison.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# PLOT 7 — Overlaid Precision-Recall Curves (macro-averaged)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, probs, color, ls in [('DNN', dnn_prob, '#3498db', '-'),
                                       ('2D-CNN', cnn_prob, '#e74c3c', '--')]:
    ap_scores = []
    for i in unique_labels:
        if i < probs.shape[1]:
            p, r, _ = sk_pr_curve(y_test_bin[:, i], probs[:, i])
            ap_scores.append(average_precision_score(y_test_bin[:, i], probs[:, i]))
            ax.plot(r, p, color=color, alpha=0.15, lw=0.8)

    mean_ap = np.mean(ap_scores)
    ax.plot([], [], color=color, lw=2.5, ls=ls,
            label=f'{model_name}  (mean AP = {mean_ap:.4f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Comparison — DNN vs 2D-CNN',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'precision_recall_comparison.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("\n5/5  Done!")
print(f"\n✅ ALL COMPARISON PLOTS SAVED to: {SAVE_DIR}")
print("   → accuracy_comparison.png")
print("   → metric_comparison_table.png")
print("   → confusion_matrix_comparison.png")
print("   → roc_auc_comparison.png")
print("   → training_history_comparison.png")
print("   → classification_report_comparison.png")
print("   → precision_recall_comparison.png")

print("\n📊 Quick Comparison:")
print(f"   {'Metric':<25s} {'DNN':>10s} {'2D-CNN':>10s} {'Better':>10s}")
print(f"   {'─' * 55}")
for m in dnn_metrics:
    d, c = dnn_metrics[m], cnn_metrics[m]
    w = winners[list(dnn_metrics.keys()).index(m)]
    print(f"   {m:<25s} {d:>10.4f} {c:>10.4f} {w:>10s}")
