# mnist_cnn_tf_fixed.py
"""
Fixed version of the buggy TensorFlow MNIST CNN
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Expand dims to add channel dimension (28,28,1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# 2. Build model (add Flatten layer)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                     # ✅ Added this
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 3. Compile (fixed loss)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # ✅ Fixed this
    metrics=["accuracy"]
)

# 4. Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.1,
    batch_size=128,
    verbose=2
)

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# 6. Predict some samples
import numpy as np
import matplotlib.pyplot as plt

predictions = model.predict(x_test[:5])
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].numpy().squeeze(), cmap="gray")
    plt.title(f"Pred: {np.argmax(predictions[i])}, True: {y_test[i]}")
    plt.axis("off")
plt.show()
