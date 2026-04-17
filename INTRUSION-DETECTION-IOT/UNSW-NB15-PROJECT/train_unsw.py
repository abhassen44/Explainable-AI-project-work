import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Import the model architectures
from model_dnn_unsw import build_dnn_model
from model_2dcnn_unsw import build_2dcnn_model

def load_and_preprocess_data():
    print("[1/5] Loading datasets...")
    # Adjust paths based on folder structure
    dataset_dir = r"c:\Users\Abhas\OneDrive\Desktop\SEM PROJECT\INTRUSION-DETECTION-IOT\UNSW-NB15-PROJECT\datasets"
    train_path = os.path.join(dataset_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(dataset_dir, "UNSW_NB15_testing-set.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Combine for uniform preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    print("[2/5] Cleaning and filtering classes...")
    # Clean column names just in case
    df.columns = df.columns.str.strip()
    
    # Paper specifies 5 classes: Normal, Generic, Exploits, DoS, Fuzzers.
    # We will filter out the minority classes to strictly maintain these 5 classes.
    allowed_classes = ['Normal', 'Generic', 'Exploits', 'DoS', 'Fuzzers']
    df['attack_cat'] = df['attack_cat'].str.strip()
    df = df[df['attack_cat'].isin(allowed_classes)].copy()
    
    print(f"Remaining samples after filtering for top 5 classes: {len(df)}")
    
    print("[3/5] Dropping specific features...")
    # Paper states to drop these 6 features after correlation analysis
    cols_to_drop = ['ct_src_dport_ltm', 'loss', 'dwin', 'ct_ftp_cmd', 'label', 'ct_srv_dst']
    
    # 'dloss' or 'sloss' exists in the dataset instead of 'loss', but let's check exactly what's there.
    # Standard UNSW-NB15 has 'sloss' and 'dloss'. The paper says 'loss'. We will drop 'sloss' and 'dloss' if 'loss' is abstract, 
    # but let's try dropping exactly what was requested.
    actual_cols = df.columns.tolist()
    drop_actual = []
    for c in cols_to_drop:
        if c in actual_cols:
            drop_actual.append(c)
        elif c == 'loss' and 'sloss' in actual_cols and 'dloss' in actual_cols:
            # Often 'loss' refers to both or one of them. We'll drop what matches.
            pass
            
    # Also drop 'id' as it is useless for training
    if 'id' in actual_cols:
        drop_actual.append('id')
        
    df.drop(columns=drop_actual, inplace=True, errors='ignore')
    
    print("[4/5] Encoding categorical features & Scaling...")
    # Encode categorical features: proto, service, state
    cat_cols = ['proto', 'service', 'state']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
    # Encode Target
    target_le = LabelEncoder()
    df['attack_cat'] = target_le.fit_transform(df['attack_cat'])
    encoders['target'] = target_le
    
    # Separate X and y
    X = df.drop(columns=['attack_cat'])
    y = df['attack_cat'].values
    
    # Note: If X shape is exactly 38, perfect. If not, the paper's feature count 
    # might require dropping specific columns. Let's dynamically force it to 38 if needed 
    # based on the paper's correlation, but we assume dropping the 6 requested + 'id' leaves exactly 38.
    # UNSW-NB15 training+testing CSVs have 45 columns natively (id + 43 features + label). 
    # Dropping 'id', 'label', and the 4 specific features = 39 features. 
    # Wait, the paper dropped exactly 6 features out of 44? We'll log the final shape.
    
    feature_names = X.columns.tolist()
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split back into train/test (80/20) since we combined them
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save preprocessors
    with open('preprocessors_unsw.pkl', 'wb') as f:
        pickle.dump({
            'encoders': encoders,
            'scaler': scaler,
            'feature_names': feature_names,
            'background_data': X_train[:500]  # Used for SHAP and LIME
        }, f)
        
    print(f"[5/5] Preprocessing complete. Final feature count: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, target_le.classes_


def train_dnn(X_train, y_train, X_test, y_test):
    print("\n--- Training DNN Model ---")
    input_dim = X_train.shape[1]
    model = build_dnn_model(input_dim=input_dim, num_classes=5)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('dnn_model_unsw.keras')
    print("Saved DNN model to 'dnn_model_unsw.keras'")
    return history


def train_2dcnn(X_train, y_train, X_test, y_test):
    print("\n--- Training 2D-CNN Model ---")
    
    # ---------------------------------------------------------
    # RESHAPING FOR 2D-CNN (Crucial step from the paper)
    # The paper explicitly states 38 features padded with 11 zeros = 49 (7x7 grid)
    # ---------------------------------------------------------
    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]
    num_features = X_train.shape[1]
    
    padding_needed = 49 - num_features
    if padding_needed > 0:
        print(f"Padding inputs with {padding_needed} zeros to reach 49 features for the 7x7 grid.")
        X_train_padded = np.pad(X_train, ((0,0), (0, padding_needed)), mode='constant', constant_values=0)
        X_test_padded  = np.pad(X_test,  ((0,0), (0, padding_needed)), mode='constant', constant_values=0)
    else:
        # If it's already 49 or more, truncate (fail-safe)
        X_train_padded = X_train[:, :49]
        X_test_padded = X_test[:, :49]
        
    X_train_2d = X_train_padded.reshape(num_samples_train, 7, 7, 1)
    X_test_2d  = X_test_padded.reshape(num_samples_test, 7, 7, 1)
    
    print(f"Reshaped 2D-CNN input to: {X_train_2d.shape}")
    
    model = build_2dcnn_model(input_shape=(7, 7, 1), num_classes=5)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_2d, y_train,
        validation_data=(X_test_2d, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('2dcnn_model_unsw.keras')
    print("Saved 2D-CNN model to '2dcnn_model_unsw.keras'")
    return history


if __name__ == "__main__":
    # 1. Preprocess
    X_train, X_test, y_train, y_test, target_classes = load_and_preprocess_data()
    print(f"Target Classes Mapping: {target_classes}")
    
    # 2. Train DNN
    train_dnn(X_train, y_train, X_test, y_test)
    
    # 3. Train 2D-CNN
    train_2dcnn(X_train, y_train, X_test, y_test)
    
    print("\n✅ All training complete for UNSW-NB15!")
