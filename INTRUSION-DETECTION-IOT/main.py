import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf

# Import the 3 models
from model_dnn import build_dnn_model
from model_1dcnn import build_1dcnn_model
from model_2dcnn import build_2dcnn_model

# --- CONFIGURATION ---
DATA_PATH = 'KDDTrain+.txt'  # Replace with your actual CSV path
EPOCHS = 30
BATCH_SIZE = 32

# Standard KDD column headers
COL_NAMES = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"]

def load_and_preprocess_data(path):
    print("Loading Data...")
    try:
        df = pd.read_csv(path, names=COL_NAMES)
    except FileNotFoundError:
        # Fallback if file not found, creating dummy data for demonstration
        print("Warning: Dataset not found. Generating dummy data for testing code flow.")
        df = pd.DataFrame(np.random.rand(1000, 42), columns=COL_NAMES)
        df['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], 1000)
        df['service'] = np.random.choice(['http', 'ftp', 'smtp'], 1000)
        df['flag'] = np.random.choice(['SF', 'S0', 'REJ'], 1000)
        df['label'] = np.random.choice(['normal', 'neptune', 'satan', 'ipsweep', 'portsweep'], 1000)

    # 1. Encoding Categorical Features (separate encoder per column)
    print("Encoding Data...")
    encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Separate Features and Target
    X = df.drop(['label', 'difficulty'], axis=1, errors='ignore')
    y = df['label']
    
    # 2. Normalization (MinMax)
    print("Normalizing Data...")
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

def correlation_feature_selection(X, threshold=0.95):
    print("Running Correlation-based Feature Reduction...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    print(f"Features reduced from {X.shape[1]} to {X_reduced.shape[1]}")
    print(f"Dropped: {to_drop}")
    return X_reduced

def reshape_for_2d_cnn(X):
    # Calculate the nearest square size
    num_features = X.shape[1]
    side = math.ceil(math.sqrt(num_features))
    padding = (side * side) - num_features
    
    # Pad with zeros
    X_padded = np.pad(X, ((0, 0), (0, padding)), 'constant')
    
    # Reshape to (Batch, Side, Side, Channels)
    X_reshaped = X_padded.reshape(-1, side, side, 1)
    return X_reshaped

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

def main():
    # 1. Pipeline execution
    X, y = load_and_preprocess_data(DATA_PATH)
    X = correlation_feature_selection(X, threshold=0.95)
    
    num_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    results = []

    # --- MODEL 1: DNN ---
    print("\nTraining DNN...")
    start_time = time.time()
    dnn = build_dnn_model(X_train.shape[1], num_classes)
    dnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
            validation_split=0.1, callbacks=[early_stop], verbose=1)
    dnn_time = time.time() - start_time
    loss, acc = dnn.evaluate(X_test, y_test, verbose=0)
    results.append({'Model': 'DNN', 'Accuracy': acc, 'Time': dnn_time})
    print(f"DNN Accuracy: {acc:.4f}, Time: {dnn_time:.2f}s")

    # --- MODEL 2: 1D-CNN ---
    print("\nTraining 1D-CNN...")
    # Reshape: (Batch, Features, 1)
    X_train_1d = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_1d = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    start_time = time.time()
    cnn1d = build_1dcnn_model(X_train_1d.shape[1], 1, num_classes)
    cnn1d.fit(X_train_1d, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_split=0.1, callbacks=[early_stop], verbose=1)
    cnn1d_time = time.time() - start_time
    loss, acc = cnn1d.evaluate(X_test_1d, y_test, verbose=0)
    results.append({'Model': '1D-CNN', 'Accuracy': acc, 'Time': cnn1d_time})
    print(f"1D-CNN Accuracy: {acc:.4f}, Time: {cnn1d_time:.2f}s")

    # --- MODEL 3: 2D-CNN ---
    print("\nTraining 2D-CNN...")
    X_train_2d = reshape_for_2d_cnn(X_train.values)
    X_test_2d = reshape_for_2d_cnn(X_test.values)
    input_shape = (X_train_2d.shape[1], X_train_2d.shape[2], 1)
    
    start_time = time.time()
    cnn2d = build_2dcnn_model(input_shape, num_classes)
    cnn2d.fit(X_train_2d, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_split=0.1, callbacks=[early_stop], verbose=1)
    cnn2d_time = time.time() - start_time
    loss, acc = cnn2d.evaluate(X_test_2d, y_test, verbose=0)
    results.append({'Model': '2D-CNN', 'Accuracy': acc, 'Time': cnn2d_time})
    print(f"2D-CNN Accuracy: {acc:.4f}, Time: {cnn2d_time:.2f}s")

    # --- FINAL COMPARISON ---
    print("\n--- FINAL RESULTS ---")
    res_df = pd.DataFrame(results)
    print(res_df)
    
    # Simple Plot
    res_df.plot(x='Model', y='Accuracy', kind='bar', legend=False)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
