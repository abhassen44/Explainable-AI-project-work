import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_2dcnn_model(input_shape=(7, 7, 1), num_classes=5):
    """
    Builds the 2D-CNN model for UNSW-NB15 as specified in the paper.
    
    Architecture:
      Input: (7, 7, 1) - from 38 features + 11 padding zeros
      Conv2D (64 filters, 3x3) -> MaxPool (2x2)
      Conv2D (32 filters, 3x3) -> MaxPool (2x2)
      Conv2D (32 filters, 3x3) -> MaxPool (2x2)
      Flatten
      Dense 5 (Softmax)
    """
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),
        
        # Block 2
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        
        # Block 3
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        
        # Classifier
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

    # Adam optimizer with parameters from the paper
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
