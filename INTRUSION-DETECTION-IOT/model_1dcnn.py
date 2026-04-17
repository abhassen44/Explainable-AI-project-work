import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_1dcnn_model(input_steps, features, num_classes):
    """
    Builds the 1D-CNN model.
    Architecture:
    - Conv1D (64 filters, kernel=3, ReLU) → MaxPooling1D → Dropout(0.3)
    - Conv1D (32 filters, kernel=3, ReLU) → MaxPooling1D → Dropout(0.3)
    - Conv1D (32 filters, kernel=3, ReLU) → MaxPooling1D → Dropout(0.3)
    - Flatten → Output Dense (num_classes, Softmax)
    
    Note: 'padding=same' is used to handle small input dimensions
    without the dimensionality vanishing before the final layers.
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(input_steps, features)),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.3),
        
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model