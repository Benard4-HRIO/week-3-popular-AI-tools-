# mnist_cnn_tf.py
"""
MNIST CNN with TensorFlow/Keras
- Loads MNIST dataset
- Builds CNN model
- Trains and evaluates accuracy
- Visualizes predictions
- ✅ Saves model for Streamlit app deployment
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("✅ TensorFlow version:", tf.__version__)

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ✅ Add channel dimension (for CNN input)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Split validation set
x_val = x_train[-6000:]
y_val = y_train[-6000:]
x_train = x_train[:-6000]
y_train = y_train[:-6000]

# 2. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val)
)

# 5. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# 6. Visualize sample predictions
preds = model.predict(x_test[:5])
pred_labels = np.argmax(preds, axis=1)

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"True: {y_test[i]}\nPred: {pred_labels[i]}")
    plt.axis('off')

plt.suptitle(f"MNIST Predictions (Test Accuracy: {test_acc:.2%})")
plt.show()

# 7. ✅ Save trained model
model.save("mnist_cnn_model.h5")
print("✅ Model saved successfully as mnist_cnn_model.h5")
