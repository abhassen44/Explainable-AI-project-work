"""
train_main.py — Standalone training script for the DNN model
on the NSL-KDD dataset (paper-accurate config).

Paper: "Explainable AI for Intrusion Detection in IoT Networks"

Pipeline
────────
1. Load KDDTrain+.txt
2. Map attack names → 5 categories (DoS, Normal, Probe, R2L, U2R)
3. Encode categoricals, drop correlated features → 36 features
4. Scale with MinMaxScaler
5. Train DNN for 20 epochs with Adam(lr=0.001, weight_decay=0.0001)
6. Save model + preprocessors

Outputs:
  - dnn_model.keras
  - preprocessors.pkl
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from model_dnn import build_dnn_model

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH               = 'KDDTrain+.txt'
MODEL_SAVE_PATH         = 'dnn_model.keras'
PREPROCESSOR_SAVE_PATH  = 'preprocessors.pkl'
EPOCHS                  = 20        # paper spec
BATCH_SIZE              = 32

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

# Drop 5 highly-correlated features → keeps 36 features (paper spec)
DROP_COLS = [
    'srv_serror_rate', 'srv_rerror_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_srv_rerror_rate'
]

# NSL-KDD attack → 5-class category mapping (paper spec)
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'xlock': 'R2L',
    'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}

CLASS_NAMES = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']


def main():
    # ---- 1. Load ----
    print("1/4  Loading data …")
    df = pd.read_csv(DATA_PATH, names=COL_NAMES)
    df = df.drop('difficulty', axis=1)

    # ---- 2. Map to 5 categories ----
    print("2/4  Preprocessing (5-class mapping, 36 features) …")
    df['label'] = df['label'].map(ATTACK_MAP).fillna('Normal')

    # Encode categoricals
    encoder_dict = {}
    for col in ["protocol_type", "service", "flag", "label"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le

    # Drop correlated features → 36 remain
    df = df.drop(columns=DROP_COLS)

    X = df.drop('label', axis=1)
    y = df['label']
    feature_names = X.columns.tolist()
    num_features  = len(feature_names)

    print(f"   Features kept : {num_features}  (expected 36)")
    assert num_features == 36, f"Expected 36 features, got {num_features}"

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(y))   # should be 5
    print(f"   Classes        : {num_classes}  → {CLASS_NAMES}")

    # ---- 3. Train ----
    print("3/4  Training DNN (20 epochs) …")
    model = build_dnn_model(input_dim=num_features, num_classes=num_classes)

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy : {acc*100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")

    # ---- 4. Save ----
    print("4/4  Saving model & preprocessors …")
    model.save(MODEL_SAVE_PATH)

    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'encoders': encoder_dict,
            'scaler': scaler,
            'feature_names': feature_names,
            'drop_cols': DROP_COLS,
            'attack_map': ATTACK_MAP,
            'class_names': CLASS_NAMES,
            'background_data': X_train[:200]
        }, f)

    print(f"\n✅  Done!  Saved → {MODEL_SAVE_PATH}, {PREPROCESSOR_SAVE_PATH}")


if __name__ == "__main__":
    main()