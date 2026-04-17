"""
train_cnn_lstm.py — CNN-LSTM Training Pipeline
================================================
Trains the CNN-LSTM model for binary sentiment classification.

Pipeline:
  1. Set random seeds for reproducibility
  2. Configure GPU
  3. Load & preprocess text via preprocess.py (DL mode)
  4. Build CNN-LSTM architecture
  5. Train with EarlyStopping & ReduceLROnPlateau
  6. Evaluate on test set
  7. Save model + artifacts (including the fitted tokenizer)

Usage:
  python train_cnn_lstm.py
"""

import os
import sys
import pickle
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocess import (
    load_data,
    preprocess_for_dl,
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    MODELS_DIR,
)
from model_cnn_lstm import build_cnn_lstm

# ─── Configuration ───────────────────────────────────────────
EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def set_seeds(seed=RANDOM_SEED):
    """Sets random seeds for reproducible training runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seeds set to {seed}")

def configure_gpu():
    """Configures GPU memory growth to prevent TensorFlow from allocating all VRAM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} GPU(s) with memory growth.")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPU detected. Training will use CPU.")

def main():
    logger.info("=" * 60)
    logger.info("  CNN-LSTM Training Pipeline Started")
    logger.info("=" * 60)

    # 1. Setup Environment
    set_seeds()
    configure_gpu()

    # 2. Load data
    logger.info("Loading data...")
    X_train_raw, X_test_raw, y_train, y_test = load_data()

    # 3. Preprocess for DL
    logger.info("Preprocessing data for DL...")
    X_train_pad, X_test_pad, tokenizer = preprocess_for_dl(
        X_train_raw, X_test_raw
    )

    # 4. Determine vocabulary size
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    logger.info(f"Vocabulary size: {vocab_size}")

    # 5. Build model
    logger.info("Building CNN-LSTM model...")
    model = build_cnn_lstm(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
    )
    model.summary(print_fn=logger.info)

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 7. Train
    logger.info("Starting training...")
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred_prob = model.predict(X_test_pad, batch_size=BATCH_SIZE)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Using print here specifically to maintain the readable grid formatting of the report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # 9. Save model and artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save Keras Model
    model_path = os.path.join(MODELS_DIR, "cnn_lstm_model.keras")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = os.path.join(MODELS_DIR, "cnn_lstm_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    logger.info(f"Training history saved to {history_path}")

    # Save the actual Tokenizer object (CRITICAL FOR INFERENCE)
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Fitted tokenizer saved to {tokenizer_path}")

    # Save preprocessor metadata
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessors.pkl")
    preprocessor_info = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "vocab_size": vocab_size,
    }
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor_info, f)
    logger.info(f"Preprocessor info saved to {preprocessor_path}")

    logger.info("=" * 60)
    logger.info("  CNN-LSTM Training Complete!")
    logger.info("=" * 60)

    return model, history, tokenizer

if __name__ == "__main__":
    main()