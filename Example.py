import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 1) Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 2) Visualize some training images
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3) Normalize and reshape images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add a channel dimension (grayscale = 1 channel)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# 4) Convert labels to one-hot for 'categorical_crossentropy'
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# 5) Build a CNN model with 'same' padding to maintain spatial dimensions
model = tf.keras.models.Sequential([
    # Convolutional layer 1 with stride (2,2) and 'same' padding
    tf.keras.layers.Conv2D(
        32, 
        (3, 3), 
        strides=(1, 1),  # Non-overlapping stride
        activation='relu', 
        padding='same',  # Use 'same' padding
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2)),  # Non-overlapping pooling
    
    # Convolutional layer 2 with stride (2,2) and 'same' padding
    tf.keras.layers.Conv2D(
        64, 
        (2, 2), 
        strides=(1, 1),  # Non-overlapping stride
        activation='relu',
        padding='same'  # Use 'same' padding
    ),
    tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2)),  # Non-overlapping pooling
    
    # Flatten for Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer (10 classes)
    tf.keras.layers.Dense(10, activation='softmax')
])

# 6) Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7) Print model summary to verify layer outputs
model.summary()

# 8) Train (fit) the model
model.fit(
    train_images,
    train_labels, 
    epochs=5,
    batch_size=32
)

# Continue training for more epochs with a different batch size
model.fit(
    train_images,
    train_labels, 
    epochs=100,
    batch_size=1000
)

# 9) Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 10) Make a prediction on a single example from the test set
predictions = model.predict(test_images)
predicted_class = tf.argmax(predictions[0]).numpy()
actual_class = tf.argmax(test_labels[0]).numpy()
print(f"Predicted Class: {predicted_class}, Actual Class: {actual_class}")

##############################################################################
# IMPORTANT: We need to ensure the model has been called before creating the 
#            'activation_model'. The above training or prediction calls do that.
##############################################################################

# Identify layers to visualize: (conv and pooling layers)
layer_outputs = [
    layer.output for layer in model.layers
    if 'conv2d' in layer.name or 'average_pooling2d' in layer.name
]

# Build the activation model AFTER the model has been called
activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

# Let's pick the first test image to visualize
sample_image = test_images[:1]  # shape = (1, 28, 28, 1)

# Get the activations for our sample image
activations = activation_model.predict(sample_image)

# Prepare layer names (for labeling the plots)
layer_names = [
    layer.name for layer in model.layers
    if 'conv2d' in layer.name or 'average_pooling2d' in layer.name
]

def plot_layer_activations(activation, title='', cols=8):
    """
    Plots feature maps of shape (height, width, channels).
    """
    channels = activation.shape[-1]  # number of filters/feature maps
    rows = (channels + cols - 1) // cols
    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    for i in range(channels):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(activation[..., i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot each layer's feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    plot_layer_activations(layer_activation[0], title=layer_name)
