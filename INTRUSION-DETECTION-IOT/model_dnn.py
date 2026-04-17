"""
model_dnn.py — Deep Neural Network for IoT Intrusion Detection (NSL-KDD)

Architecture follows the paper:
  "Explainable AI for Intrusion Detection in IoT Networks: A Deep Learning Based Approach"

Input : (36,)        — 36 features after Pearson Correlation feature selection
Output: 5 neurons    — DoS, Normal, Probe, R2L, U2R (Softmax)

Layer stack
──────────────────────────────────────────────────────────
  Dense   64 neurons  ReLU
  Dropout 0.01
  ──────────────────────────────────────────────────────
  Dense   64 neurons  ReLU
  Dropout 0.01
  ──────────────────────────────────────────────────────
  Dense   64 neurons  ReLU
  Dropout 0.01
  ──────────────────────────────────────────────────────
  Dense   5  Softmax
──────────────────────────────────────────────────────────

Optimizer : Adam  (lr = 0.001, weight_decay = 0.0001)
Loss      : sparse_categorical_crossentropy
Epochs    : 20   (set in the training script)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_dnn_model(input_dim=36, num_classes=5):
    """
    Builds the DNN model as specified in the paper.

    Parameters
    ----------
    input_dim : int, default 36
        Number of input features (after Pearson Correlation selection).
    num_classes : int, default 5
        Number of output classes (DoS, Normal, Probe, R2L, U2R).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras Sequential model.
    """
    model = Sequential([
        # --- Hidden Layer 1 ---
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.01),

        # --- Hidden Layer 2 ---
        Dense(64, activation='relu'),
        Dropout(0.01),

        # --- Hidden Layer 3 ---
        Dense(64, activation='relu'),
        Dropout(0.01),

        # --- Output Layer ---
        Dense(num_classes, activation='softmax')
    ])

    # Adam with learning rate 0.001 and weight decay 0.0001 (paper spec)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        weight_decay=0.0001
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model