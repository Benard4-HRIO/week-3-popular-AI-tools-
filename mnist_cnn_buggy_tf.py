# mnist_cnn_buggy_tf.py
"""
Buggy TensorFlow MNIST CNN
This version contains several intentional errors:
- Missing channel dimension
- Missing Flatten layer before Dense
- Wrong loss function for integer labels
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow version:", tf.__version__)

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ❌ Missing normalization and channel dimension
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# 2. Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # ❌ Will break because shape is (28,28)
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    # ❌ Missing Flatten
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# ❌ Wrong loss for integer labels
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ❌ Will cause shape errors during training
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)
