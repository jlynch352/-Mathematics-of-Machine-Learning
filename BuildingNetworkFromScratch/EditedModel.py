"""
Model.py

~~~~~~~

This module implements a stochastic gradient descent learning algorithm for a feedforward neural network
using softmax activation in the output layer for multi-class classification on the MNIST dataset.

"""

import random
import numpy as np
from typing import List, Tuple, Optional
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import itertools
import sys


class Network(object):

    def __init__(self, sizes: List[int]) -> None:
        "The input 'sizes' is a list of the desired size of each layer at each level "

        # The number of layers corresponds to the number of items in the sizes list
        self.num_layers = len(sizes)
        # Stores the sizes of each layer
        self.sizes = sizes
        # Initialize biases for each layer except the first (input) layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights with He initialization for better convergence
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Return the output of the network if 'a' is input.
        Uses sigmoid activation for hidden layers and softmax for the output layer.
        """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        # Apply softmax to the output layer
        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w, a) + b
        a = softmax(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None) -> None:
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        # If test data is provided, get its length
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        # Perform gradient descent
        for j in range(epochs):
            # Shuffle training data
            random.shuffle(training_data)
            # Create mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            # Update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Evaluate performance after each epoch
            if test_data:
                evaluation = self.evaluate(test_data)
                average = evaluation / n_test
                print("Epoch {0}: {1} / {2} -> ({3:.2%})".format(
                    j + 1,
                    evaluation,
                    n_test,
                    average
                ))
            else:
                print("Epoch {0} complete".format(j + 1))

    def update_mini_batch(self, mini_batch, eta) -> None:
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.
        """
        # Initialize gradients for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Iterate over each (input, target) pair in the mini-batch
        for x, y in mini_batch:
            # Perform backpropagation to get gradients for this training example
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Accumulate the gradients
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases using the average gradients
        self.weights = [w - (eta / len(mini_batch)) * nw
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                      for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation to compute the gradient of the cost function with respect to weights and biases.
        """
        # Initialize gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]  # List to store all activations layer by layer
        zs = []  # List to store all z vectors layer by layer

        # Forward pass for hidden layers
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Forward pass for output layer with softmax
        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # Backward pass

        # Compute delta for output layer
        delta = activations[-1] - y  # Cross-entropy loss derivative with softmax
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data: List) -> int:
        """Returns the number of test inputs in which the neural network got the correct result."""
        # Generates a list of tuples containing (predicted, actual)
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        # Counts the number of correct predictions
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
            if isinstance(y, int):
                true_labels.append(y)
            else:
                # If y is one-hot encoded
                true_labels.append(np.argmax(y))
        return predictions, true_labels


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    The softmax function applies to the output layer to handle multi-class classification.
    It converts raw scores into probabilities that sum to 1.
    """
    # Subtract the max for numerical stability
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / e_z.sum(axis=0, keepdims=True)


def load_data_sklearn():
    """Load MNIST data using scikit-learn's fetch_openml with stratified splitting."""
    print("Fetching MNIST data from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    # Normalize the input data to [0, 1]
    X = X / 255.0
    X = [x.reshape(784, 1) for x in X]

    # Stratified splitting to ensure all classes are represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, stratify=y, random_state=42
    )

    # Convert labels to one-hot encoding for training
    training_labels_one_hot = [vectorized_result(label) for label in y_train]

    # Create training and test datasets
    training_data = list(zip(X_train, training_labels_one_hot))
    test_data = list(zip(X_test, y_test))

    return training_data, test_data


def vectorized_result(j):
    """Convert a digit (0...9) into a one-hot encoded vector."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def check_label_distribution(data, dataset_name="Dataset"):
    """Check and print the distribution of labels in the given data."""
    labels = [y for _, y in data]
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"\nLabel Distribution in {dataset_name}:")
    for label in range(10):
        count = distribution.get(label, 0)
        print(f"Label {label}: {count} samples")
    print()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Add text annotations.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


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
    plt.show()


def main():
    # Load data using scikit-learn with stratified splitting
    training_data, test_data = load_data_sklearn()
    print(f"Training data: {len(training_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Check label distribution in training and test data
    check_label_distribution(training_data, "Training Data")
    check_label_distribution(test_data, "Test Data")

    # Initialize the network
    net = Network([784, 30, 10])

    # Train the network using SGD
    epochs = 3  # Increased epochs for better learning
    mini_batch_size = 100  # Adjusted mini-batch size
    learning_rate = 0.1  # Adjusted learning rate
    print(f"Training the network for {epochs} epochs with mini-batch size {mini_batch_size} and learning rate {learning_rate}...")
    net.SGD(training_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=learning_rate, test_data=test_data)

    # Final Evaluation
    print("\nFinal Evaluation on Test Data:")
    final_accuracy = net.evaluate(test_data)
    print(f"Final Test Accuracy: {final_accuracy} / {len(test_data)} ({(final_accuracy / len(test_data)) * 100:.2f}%)")

    # Detailed Metrics
    print("\nCalculating Detailed Metrics...")
    predictions, true_labels = net.get_predictions(test_data)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(true_labels, predictions)
    print("\nClassification Report:")
    print(cr)

    # Plot Confusion Matrix
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)],
                          title='Confusion Matrix')

    # Visualize Some Predictions
    visualize_predictions(test_data, predictions, num_samples=10)


if __name__ == '__main__':
    main()
