import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE


# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Preprocess the data
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.SGD(learning_rate=0.1)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the model with curriculum learning and capture history
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
        batch_size=500
        )

history = model.fit(
    train_images,
    train_labels, 
    epochs=1000,
    batch_size=1000
)

'''
End of trainng Models

Now Generating and Saving Plots to Files

'''
save_dir = "C:\\Users\\james\\Downloads\\MathematicsOfMachineLearningProject\\Graphs\\BasicModels"

# Save the model architecture visualization
save_path_model_png = os.path.join(save_dir, "model_architecture.png")

try:
    # Debug: Check file path
    print(f"Saving model architecture to: {save_path_model_png}")

    # Attempt to save the model architecture
    keras.utils.plot_model(model, to_file=save_path_model_png, show_shapes=True, show_layer_names=True)
    print(f"Model architecture plot saved to {save_path_model_png}")
except ImportError as e:
    print("Graphviz or Pydot is not installed. Install it using `pip install graphviz pydot` and install Graphviz binary.")
    print(f"Error message: {e}")
except Exception as e:
    print(f"Error saving model architecture plot: {e}")
    print(f"Error type: {type(e)}")

# Plot training history and save the figure
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the training history plot
save_path_history = os.path.join(save_dir, "training_history.pdf")
try:
    plt.savefig(save_path_history)
    print(f"Training history plot saved to {save_path_history}")
    plt.show()
except Exception as e:
    print(f"Error saving training history plot: {e}")
plt.close()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Define class names for better readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Select ten random indices
indices = np.random.choice(len(test_images), 10, replace=False)

# Plot the images with true and predicted labels
plt.figure(figsize=(10, 5))
plt.suptitle('True vs Predicted Labels for Random Images', fontsize=16)
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i+1)
    img = test_images[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    true_label = np.argmax(test_labels[idx])
    pred_label = np.argmax(model.predict(np.expand_dims(test_images[idx], axis=0)))
    if true_label == pred_label:
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nCorrect"
    else:
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nIncorrect"
    plt.title(title, fontsize=10)
    plt.axis('off')

# Save the image grid plot
save_path_images = os.path.join(save_dir, "image_grid.pdf")
try:
    plt.savefig(save_path_images)
    print(f"Image grid plot saved to {save_path_images}")
    plt.show()
except Exception as e:
    print(f"Error saving image grid plot: {e}")
plt.close()

save_path_neurons = os.path.join(save_dir, "neurons_visualization.pdf")

try:
    # Iterate through each layer in the model
    plt.figure(figsize=(12, 8))
    layer_idx = 1
    for layer in model.layers:
        # Check if the layer has weights (Dense layers)
        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            weights, biases = layer.get_weights()

            # Plot the weights as a heatmap
            plt.subplot(2, len(model.layers) // 2 + 1, layer_idx)
            plt.imshow(weights, aspect='auto', cmap='seismic')
            plt.colorbar()
            plt.title(f"Layer {layer_idx}: {layer.name}\nWeights")
            plt.xlabel("Neurons")
            plt.ylabel("Inputs")
            layer_idx += 1

    # Save the neurons visualization plot
    plt.suptitle("Visualization of Neurons' Weights in ANN", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path_neurons, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Neuron visualization plot saved to {save_path_neurons}")
except Exception as e:
    print(f"Error visualizing neurons: {e}")
plt.close()

