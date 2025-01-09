"""
Model.py

~~~~~~~~

This is a module that implements a stochastic gradient descent learning algorithm for a feedforward neural network.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import itertools
from collections import Counter
import os
import pickle
import json


class Network(object):

    def __init__(self, sizes: List[int]) -> None:
        "The input 'sizes' is a list of the desired size of each layer at each level "

        # The number of layers corresponds to the number of items in the sizes list
        self.num_layers = len(sizes)
        # Stores the values of the List
        self.sizes = sizes
        # Randomly assigns biases for every layer into a vector of dimensions (number of neurons in layer) x 1 except the first layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Randomly assigns the weights for every layer into matrices with size (number of neurons in next layer) x (number of neurons in current layer)
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None) -> None:
        """Train the neural network using mini-batch stochastic gradient descent."""
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j + 1}: {accuracy} / {n_test} ({(accuracy / n_test) * 100:.2f}%)")
            else:
                print(f"Epoch {j + 1} complete")

    def update_mini_batch(self, mini_batch, eta) -> None:
        """Update the network's weights and biases by applying gradient descent using backpropagation."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                       for w, nw in zip(self.weights, nabla_w)
                       ]
        self.biases = [b - (eta / len(mini_batch)) * nb
                      for b, nb in zip(self.biases, nabla_b)
                      ]

    def backprop(self, x, y) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return a tuple representing the gradient for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer

        # Forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Iterate over the layers in reverse order
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: List) -> int:
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                       for x, y in test_data
                       ]
        return sum(int(x == y) for (x, y) in test_results)

    def get_predictions(self, test_data: List) -> Tuple[List[int], List[int]]:
        """
        Return lists of predictions and true labels for the given test data.
        """
        predictions = []
        true_labels = []
        for x, y in test_data:
            output = self.feedforward(x)
            predicted_label = np.argmax(output)
            predictions.append(predicted_label)
            if isinstance(y, (int, np.integer)):
                true_labels.append(int(y))  # Ensure it's a native Python int
            else:
                # If y is one-hot encoded
                true_labels.append(np.argmax(y))
        return predictions, true_labels

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the vector of partial derivatives (dC_x / d a) for the output activations."""
        return output_activations - y


def sigmoid(z) -> float:
    """The sigmoid function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z) -> float:
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def load_data_sklearn():
    print("Fetching MNIST data from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    # Normalize the input data to [0, 1]
    X = X / 255.0
    X = [x.reshape(784, 1) for x in X]

    # Convert labels to one-hot encoding for training
    training_labels_one_hot = [vectorized_result(label) for label in y]

    # Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [training_labels_one_hot[i] for i in train_index], [y[i] for i in test_index]

    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    return training_data, test_data


def vectorized_result(j):
    """Convert a digit (0...9) into a one-hot encoded vector."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
        cm_display = cm_normalized
    else:
        print('Confusion Matrix, without normalization')
        cm_display = cm

    print(cm_display)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_display, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.

    # Add text annotations.
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        plt.text(j, i, format(cm_display[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_display[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # Remove plt.show() to prevent displaying the plot
    # plt.show()


# Helper function to visualize some predictions
def visualize_predictions(test_data, predictions, num_samples=10):
    """
    Visualize a number of test images along with their predicted and true labels.
    """
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(len(test_data), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, true_label = test_data[idx]
        predicted_label = predictions[idx]

        # Reshape the image for display
        image_reshaped = image.reshape(28, 28)

        plt.subplot(5, 2, i + 1)
        plt.imshow(image_reshaped, cmap='gray')
        plt.title(f"True: {true_label} | Predicted: {predicted_label}")
        plt.axis('off')

    plt.tight_layout()
    # Remove plt.show() to prevent displaying the plot
    # plt.show()


def check_label_distribution(test_data):
    label_counts = Counter(y for _, y in test_data)
    print("Label distribution in test data:")
    for label in sorted(label_counts.keys()):
        print(f"Label {label}: {label_counts[label]} samples")


def save_model(model, filename):
    """Serialize and save the model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename):
    """Load and deserialize the model from a file."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model


def save_metrics(metrics, filename):
    """Save metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename}")


def main():
    # Define the base directory where all files will be saved
    base_dir = r'C:\Users\james\Downloads\MathematicsOfMachineLearningProject\Graphs\SelfMadeModelGraphs'

    # Ensure the base directory exists; if not, create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")

    # Define file paths
    model_filename = os.path.join(base_dir, 'trained_network.pkl')
    metrics_filename = os.path.join(base_dir, 'training_metrics.json')
    confusion_matrix_path = os.path.join(base_dir, 'confusion_matrix.png')
    classification_report_path = os.path.join(base_dir, 'classification_report.txt')
    predictions_plot_path = os.path.join(base_dir, 'predictions_visualization.png')

    # Load data using scikit-learn
    training_data, test_data = load_data_sklearn()
    print(f"Training data: {len(training_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Check label distribution
    check_label_distribution(test_data)

    # Initialize the network
    net = Network([784, 128, 32, 16, 10])

    load_choice = 'y'

    # Training parameters
    epochs = 3
    mini_batch_size = 10
    learning_rate = 3.0

    # Train the network using SGD if not loading an existing model
    if not os.path.exists(model_filename) or load_choice.lower() != 'y':
        print(f"Training the network for {epochs} epochs...")
        net.SGD(training_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=learning_rate, test_data=test_data)
        # Save the trained model
        save_model(net, model_filename)
    else:
        # Load the existing model
        net = load_model(model_filename)

    # Final Evaluation
    print("\nFinal Evaluation on Test Data:")
    final_accuracy = net.evaluate(test_data)
    accuracy_percentage = (final_accuracy / len(test_data)) * 100
    print(f"Final Test Accuracy: {final_accuracy} / {len(test_data)} ({accuracy_percentage:.2f}%)")

    # Prepare metrics to save
    metrics = {
        'final_accuracy': final_accuracy,
        'total_test_samples': len(test_data),
        'accuracy_percentage': accuracy_percentage
    }

    # Detailed Metrics
    print("\nCalculating Detailed Metrics...")
    predictions, true_labels = net.get_predictions(test_data)

    # Check unique labels
    unique_true = set(true_labels)
    unique_pred = set(predictions)
    print(f"Unique true labels: {unique_true}")
    print(f"Unique predicted labels: {unique_pred}")

    # Confusion Matrix with all classes
    cm = confusion_matrix(true_labels, predictions, labels=range(10))
    print("\nConfusion Matrix:")
    print(cm)

    # Add confusion matrix to metrics
    metrics['confusion_matrix'] = cm.tolist()  # Convert numpy array to list for JSON serialization

    # Classification Report with zero_division set to 0
    cr = classification_report(true_labels, predictions, zero_division=0)
    print("\nClassification Report:")
    print(cr)

    # Add classification report to metrics
    metrics['classification_report'] = cr

    # Save metrics
    save_metrics(metrics, metrics_filename)

    # Optionally, save the classification report to a text file
    with open(classification_report_path, 'w') as f:
        f.write(cr)
    print(f"Classification report saved as '{classification_report_path}'")

    # Plot Confusion Matrix
    try:
        plot_confusion_matrix(cm, classes=[str(i) for i in range(10)],
                              title='Confusion Matrix', normalize=True)  # You can set normalize=False if preferred
        plt.savefig(confusion_matrix_path)
        print(f"Confusion matrix plot saved as '{confusion_matrix_path}'")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Failed to save confusion matrix plot: {e}")

    # Visualize Some Predictions
    try:
        visualize_predictions(test_data, predictions, num_samples=10)
        plt.savefig(predictions_plot_path)
        print(f"Predictions visualization saved as '{predictions_plot_path}'")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Failed to save predictions visualization: {e}")

    # Optionally, show the plots (uncomment if you want to display the plots)
    # plt.show()


if __name__ == '__main__':
    main()