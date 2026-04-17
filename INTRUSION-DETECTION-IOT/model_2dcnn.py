"""
model_2dcnn.py — 2D-CNN for IoT Intrusion Detection (NSL-KDD)

Architecture follows the paper:
  "Explainable AI for Intrusion Detection in IoT Networks: A Deep Learning Based Approach"

Input : (6, 6, 1)   — 36 features reshaped into a 6×6 single-channel image
Output: 5 neurons    — DoS, Normal, Probe, R2L, U2R (Softmax)

Layer stack
──────────────────────────────────────────────────────────
  Conv2D  64 filters  (3×3)  ReLU   padding='same'
  MaxPool2D           (2×2)         padding='same'
  Dropout             0.01
  ──────────────────────────────────────────────────────
  Conv2D  32 filters  (3×3)  ReLU   padding='same'
  MaxPool2D           (2×2)         padding='same'
  Dropout             0.01
  ──────────────────────────────────────────────────────
  Conv2D  32 filters  (3×3)  ReLU   padding='same'
  MaxPool2D           (2×2)         padding='same'
  Dropout             0.01
  ──────────────────────────────────────────────────────
  Flatten
  Dense   5  Softmax
──────────────────────────────────────────────────────────

Optimizer : Adam  (lr = 0.001, weight_decay = 0.0001)
Loss      : sparse_categorical_crossentropy
Epochs    : 20   (set in the training script)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_2dcnn_model(input_shape=(6, 6, 1), num_classes=5):
    """
    Builds the 2D-CNN model as specified in the paper.

    Parameters
    ----------
    input_shape : tuple, default (6, 6, 1)
        Shape of a single sample (height, width, channels).
        36 features are reshaped into a 6×6 single-channel grid.
    num_classes : int, default 5
        Number of output classes (DoS, Normal, Probe, R2L, U2R).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras Sequential model.
    """
    model = Sequential([
        # --- Block 1 ---
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.01),

        # --- Block 2 ---
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.01),

        # --- Block 3 ---
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.01),

        # --- Classifier ---
        Flatten(),
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