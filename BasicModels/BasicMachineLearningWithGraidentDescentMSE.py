import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() # load dataset and split into training and testing sets

plt.figure(figsize=(10, 2))
for i in range(5):
  plt.subplot(1, 5, i+1)
  plt.imshow(train_images[i])
  plt.title(f"Label: {train_labels[i]}")
plt.tight_layout()
plt.show()


train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

train_images = tf.reshape(train_images, [-1, 784])
test_images = tf.reshape(test_images, [-1, 784])

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=['accuracy']
)

pred_labels = model(train_images)
loss = loss_function(train_labels, pred_labels)
print(f"Loss on training data: {loss.numpy()}")

model.fit(
    train_images,
    train_labels, 
    epochs=10,
    batch_size=32
)


model.fit(
    train_images,
    train_labels, 
    epochs=50,
    batch_size=1000
)


predictions = model.predict(test_images)
predicted_class = tf.argmax(predictions[0]).numpy()
actual_class = tf.argmax(test_labels[0]).numpy()
print(f"Predicted Class: {predicted_class}, Actual Class: {actual_class}")

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%") 
