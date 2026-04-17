"""
model_cnn_lstm.py — CNN-LSTM Hybrid Architecture
==================================================
Implements the CNN-LSTM model for binary sentiment classification
as described in the paper. Architecture:

  Input → Embedding → Conv1D → MaxPool → LSTM → Dense → Dropout → Output

Target: Accuracy ≈ 87.29%, Precision ≈ 88.63%
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam


def build_cnn_lstm(
    vocab_size,
    embed_dim=128,
    max_len=300,
    num_filters=128,
    kernel_size=5,
    lstm_units=128,
    dense_units=64,
    dropout_rate=0.5,
    learning_rate=0.001,
):
    """
    Build the CNN-LSTM hybrid model.

    Architecture:
        Embedding(vocab_size, embed_dim)     → Trainable word embeddings
        SpatialDropout1D(0.2)                → Regularize embeddings
        Conv1D(128, 5, relu)                 → Local feature extraction
        MaxPooling1D(2)                      → Downsample
        LSTM(128, dropout=0.2)               → Sequential pattern learning
        Dense(64, relu)                      → Non-linear transformation
        Dropout(0.5)                         → Regularization
        Dense(1, sigmoid)                    → Binary classification

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential(
        [
            # Embedding Layer — Trainable
            Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                input_length=max_len,
                name="embedding",
            ),
            # Spatial dropout on embeddings
            SpatialDropout1D(0.2, name="spatial_dropout"),
            # CNN Layer — Extract local patterns (n-grams)
            Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                name="conv1d",
            ),
            # MaxPooling — Reduce dimensionality
            MaxPooling1D(pool_size=2, name="maxpool1d"),
            # LSTM Layer — Capture sequential dependencies
            LSTM(
                lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                name="lstm",
            ),
            # Dense Layer
            Dense(dense_units, activation="relu", name="dense_hidden"),
            # Dropout
            Dropout(dropout_rate, name="dropout"),
            # Output Layer — Binary classification
            Dense(1, activation="sigmoid", name="output"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick test — build and print summary
    model = build_cnn_lstm(vocab_size=50000)
    model.summary()
