import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np
from preprocess import load_data

# Load dataset
X_train, y_train = load_data('../dataset/train/')
X_test, y_test = load_data('../dataset/test/')

print("X_train shape:", X_train.shape)  # Debugging: Ensure correct shape

# Build CNN Model with explicit Input layer
model = Sequential([
    Input(shape=(128, 128, 3)),  # Define input shape
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
# Save the model in the recommended format
model.save('../models/leaf_disease_model.keras')

