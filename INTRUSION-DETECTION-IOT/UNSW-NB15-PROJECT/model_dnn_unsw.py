import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_dnn_model(input_dim=38, num_classes=5):
    """
    Builds the DNN model for UNSW-NB15 as specified in the paper.
    
    Architecture:
      Input (38)
      Dense 64 (ReLU)
      Dense 64 (ReLU)
      Dense 64 (ReLU)
      Dense 5 (Softmax)
    """
    model = Sequential([
        # Hidden Layer 1
        Dense(64, activation='relu', input_shape=(input_dim,)),
        
        # Hidden Layer 2
        Dense(64, activation='relu'),
        
        # Hidden Layer 3
        Dense(64, activation='relu'),
        
        # Output Layer
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
