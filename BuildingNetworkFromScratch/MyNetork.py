import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


"""
Helper Functions 

This section contains all helper functions used in the Neural Network class.
"""

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU activation function.
    """
    return np.maximum(0, x)

def reluDerivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU function.
    """
    return (x > 0).astype(float)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function in a numerically stable way.
    """
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    # Prevent overflow in exp
    x_shifted = np.clip(x_shifted, -500, 500)
    e_x = np.exp(x_shifted)
    return e_x / e_x.sum(axis=0, keepdims=True)

def MeanSquaredError(Actual: np.ndarray, Predicted: np.ndarray) -> float:
    """
    Compute the mean squared error.
    """
    return np.mean((Actual - Predicted) ** 2) / 2

def MseDerivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the mean squared error with respect to the output.
    """
    return x - y

def crossEntropyLoss(Actual: np.ndarray, Predicted: np.ndarray) -> float:
    """
    Compute the cross-entropy loss.
    """
    return -np.mean(np.sum(Actual * np.log(Predicted + 1e-15), axis=0))

def crossEntropyDerivative(Actual: np.ndarray, Predicted: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the cross-entropy loss with respect to the predicted output.
    """
    return Predicted - Actual

"""
Neural Network

This section contains the code for the Neural Network implementation.
"""

class NeuralNetwork:
    def __init__(self, sizes: List[int], activation: str = 'relu') -> None:
        """
        Initialize the neural network with the given architecture.
        """
        self.num_layers: int = len(sizes)
        self.sizes: List[int] = sizes
        self.activation = activation

        # Initialize weights with Xavier, He initialization, or random if different function are used
        self.weights: List[np.ndarray] = []
        for x, y in zip(sizes[:-1], sizes[1:]):
            if activation == 'relu':
                # He Initialization
                print("He Weight Intilaization For Relu")
                weight = np.random.randn(y, x) * np.sqrt(2. / x)
            elif activation == 'sigmoid':
                # Xavier Initialization
                print("Xavier Weight Intilaization For Sigmoid")
                weight = np.random.randn(y, x) * np.sqrt(1. / x)
            else:
                # Default to standard normal if unknown activation
                print("Random Weight Intilaization For Generic Activation Function")
                weight = np.random.randn(y, x)
            self.weights.append(weight)

        # Initialize biases with zeros for better symmetry breaking
        self.biases: List[np.ndarray] = [np.zeros((y, 1)) for y in sizes[1:]]

    def feed_forward(self, input_vector: np.ndarray, activationFunction) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform a feedforward pass through the network.
        """
        activation: np.ndarray = input_vector
        activations: List[np.ndarray] = [input_vector]  # Store all activations
        nonActivatedVectors: List[np.ndarray] = []       # Store pre-activation values

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases), start=1):
            nonActivatedVector: np.ndarray = np.matmul(weight, activation) + bias
            nonActivatedVectors.append(nonActivatedVector)
            #If not last later aplly activation function and 
            if index < self.num_layers - 1:
                activation = activationFunction(nonActivatedVector)  
            else:
                activation = nonActivatedVector  # No activation on output layer
            activations.append(activation)

        # Apply softmax to the output layer for probability distribution
        activations[-1] = softmax(activations[-1])
        return activations, nonActivatedVectors

    def backPropagation(self, x: np.ndarray, y: np.ndarray, activationFunction, activationFunctionDerivative, lossDerivative) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation to compute gradients for weights and biases.
        """
        activations, nonActivated = self.feed_forward(x, activationFunction)

        partialBiases: List[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        partialWeights: List[np.ndarray] = [np.zeros(w.shape) for w in self.weights]

        # Compute the initial error using the derivative of the loss function
        error: np.ndarray = lossDerivative(activations[-1], y)  # (output - actual)
        partialBiases[-1] = error
        partialWeights[-1] = np.matmul(error, activations[-2].T)

        # Backpropagate the error through the hidden layers
        for layer in range(2, self.num_layers):
            error = np.matmul(self.weights[-layer + 1].T, error) * activationFunctionDerivative(nonActivated[-layer])
            partialBiases[-layer] = error
            partialWeights[-layer] = np.matmul(error, activations[-layer - 1].T)
       
        return partialBiases, partialWeights

    def gradientDescent(
        self,
        trainingData: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        batchSize: int,
        learningRate: float,
        activationFunction=sigmoid,
        activationFunctionDerivative=sigmoidPrime,
        lossFunction=crossEntropyLoss,
        lossDerivative=crossEntropyDerivative
    ) -> None:
        """
        Perform stochastic gradient descent on the training data.
        """
        n: int = len(trainingData)
        for j in range(epochs):
            random.shuffle(trainingData)
            batches: List[List[Tuple[np.ndarray, np.ndarray]]] = [
                trainingData[k:k + batchSize] for k in range(0, n, batchSize)
            ]
            for miniBatch in batches:
                self.updateWeights(miniBatch, learningRate, activationFunction, activationFunctionDerivative, lossDerivative)
            print(f"Epoch {j + 1} complete", end="")
            
            # Evaluate after each epoch on a random subset of the training data
            correct_predictions = 0
            total_loss = 0.0
            for x, y in trainingData[:500]:
                output, _ = self.feed_forward(x, activationFunction)
                predicted_class = np.argmax(output[-1])
                actual_class = np.argmax(y)
                if predicted_class == actual_class:
                    correct_predictions += 1
                total_loss += lossFunction(y, output[-1])
            accuracy = correct_predictions / 500
            avg_loss = total_loss / 500
            print(f": Accuracy on evaluation subset: {accuracy * 100:.2f}%, Avg Loss: {avg_loss:.4f}")

    def updateWeights(self, batch: List[Tuple[np.ndarray, np.ndarray]], learningRate: float, activationFunction, activationFunctionDerivative, lossDerivative) -> None:
        accumulativeDW: List[np.ndarray] = [np.zeros(w.shape) for w in self.weights]
        accumulativeDB: List[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        
        for x, y in batch:
            db, dw = self.backPropagation(x, y, activationFunction, activationFunctionDerivative, lossDerivative)
            accumulativeDB = [nb + dnb for nb, dnb in zip(accumulativeDB, db)]
            accumulativeDW = [nw + dnw for nw, dnw in zip(accumulativeDW, dw)]
    
        # Gradient Clipping: Clip gradients to a maximum norm (e.g., 1.0)
        clippedDW = [np.clip(dw, -3.0, 3.0) for dw in accumulativeDW]
        clippedDB = [np.clip(db, -3.0, 3.0) for db in accumulativeDB]
    
        # Update weights and biases by subtracting the clipped gradient scaled by the learning rate
        self.weights = [
        w - (learningRate / len(batch)) * cw
            for w, cw in zip(self.weights, clippedDW)
        ]

        self.biases = [
            b - (learningRate / len(batch)) * cb
            for b, cb in zip(self.biases, clippedDB)
        ]
        
    def print_layers(self) -> None:
        """
        Print the shapes of weights and biases for each layer.
        Useful for verifying the network architecture.
        """
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases), start=1):
            print(f"Layer {i}:")
            print(f"  Weights shape: {weight.shape}")
            print(f"  Biases shape: {bias.shape}\n")

def displayingImagePrediction(network: NeuralNetwork, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Display a set of images along with their predicted and actual labels.

    Parameters:
    - network: The trained NeuralNetwork instance.
    - data: A list of tuples (input_vector, actual_vector).
    """
    save_path = "C:\\Users\\james\\Downloads\\MathematicsOfMachineLearningProject\\Graphs\\SelfMade\\PredictionVsActual.png"
    
    sampled_data = random.sample(data, 20)
    
    rows, cols = 4, 5
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, (x, y) in enumerate(data[:20], start=1):
        output, _ = network.feed_forward(x, relu)
        predicted_class = np.argmax(output[-1])
        actual_class = np.argmax(y)
        image = x.reshape(28, 28)
        
        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap='gray')
        plt.title(f"P: {predicted_class}\nA: {actual_class}")
        plt.axis('off')
    

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.close()

def confussionMatrix(network: NeuralNetwork, data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Display a Confussion Matrix for the given nueral network.

    Parameters:
    - network: The trained NeuralNetwork instance.
    - data: A list of tuples (input_vector, actual_vector).
    """
    save_path = "C:\\Users\\james\\Downloads\\MathematicsOfMachineLearningProject\\Graphs\\SelfMade\\ConfussionMatrix.png"
    
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]

    for i, (x, y) in enumerate(data, start=1):
        output, _ = network.feed_forward(x, relu)
        predicted_class = np.argmax(output[-1])
        actual_class = np.argmax(y)
        confusion_matrix[predicted_class][actual_class] += 1

    confusion_matrix = np.array(confusion_matrix)
    print("Confusion Matrix:")
    print(confusion_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)]
                )
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.title('Confusion Matrix')
    
    try:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.close()

def findAccuracyAndLoss(network: NeuralNetwork, activationFunction: function,lossFunction: function, trainingData: List[Tuple[np.ndarray, np.ndarray]]):
    """
    Finds the accuracy of the given model.

    Parameters:
    - network: The trained NeuralNetwork instance.
    - activationFunction: the given activation function used.
    - lossFunction: the given loss function used.
    - trainingData: the test data to find the accuracy.
    """
    correct_predictions = 0
    total_loss = 0.0
    for x, y in trainingData[:1000]:  # Evaluate on a subset for speed
        output, _ = network.feed_forward(x, activationFunction)  # Use relu as defined
        predicted_class = np.argmax(output[-1])
        actual_class = np.argmax(y)
        if predicted_class == actual_class:
            correct_predictions += 1
        total_loss += lossFunction(y, output[-1])

    #Calculates the Models accuracy
    accuracy = correct_predictions / len(trainingData[:1000])
    avg_loss = total_loss / len(trainingData[:1000])
    return accuracy, avg_loss

def main():
    print("Initializing Neural Network with MNIST Dataset...\n")

    # Load MNIST dataset
    print("Fetching MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)  # X shape: (70000, 784), y shape: (70000,)

    # One-Hot Encode the target labels
    print("One-Hot Encoding target labels...")
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # y_encoded shape: (70000, 10)

    # Split into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Normalize the input features
    print("Normalizing input features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on training data
    X_test = scaler.transform(X_test)        # Transform test data

    # Transpose for the network (features x samples)
    X_train = X_train.T  # Shape: (784, 56000)
    y_train = y_train.T  # Shape: (10, 56000)
    X_test = X_test.T    # Shape: (784, 14000)
    y_test = y_test.T    # Shape: (10, 14000)

    print(f"X_train shape: {X_train.shape}")  # Expected: (784, 56000)
    print(f"y_train shape: {y_train.shape}")  # Expected: (10, 56000)

    # Prepare training and test data as lists of tuples
    trainingData = [
        (X_train[:, i].reshape(-1, 1), y_train[:, i].reshape(-1, 1)) 
        for i in range(X_train.shape[1])
    ]
    testData = [
        (X_test[:, i].reshape(-1, 1), y_test[:, i].reshape(-1, 1)) 
        for i in range(X_test.shape[1])
    ]
    
    # Initialize and visualize the network structure
    network = NeuralNetwork([784, 128, 64, 10], activation='relu')
    network.print_layers()
    
    # Train the network
    epochs = 5
    batch_size = 128  
    learning_rate = .05 
    
    # Assign functions without trailing commas
    activationFunction = relu
    activationFunctionDerivative = reluDerivative
    lossFunction = MeanSquaredError
    lossDerivative = MseDerivative

    network.gradientDescent(
        trainingData,
        epochs,
        batch_size,
        learning_rate,
        activationFunction,
        activationFunctionDerivative,
        lossFunction,
        lossDerivative
    )
    
    # Display some image predictions
    print("\nDisplaying image predictions on the test set:")
    displayingImagePrediction(network, testData)

    #Displays Confussion Matrix
    print("Confussion Matrix")
    confussionMatrix(network, testData)

    #finds the accuracy and Loss
    print("Finding Accuracy")
    accuracy, avg_loss = findAccuracyAndLoss(network, activationFunction, lossFunction, testData)
    print(f"\nAccuracy on the training set subset: {accuracy * 100:.2f}%, Avg Loss: {avg_loss:.4f}")
if __name__ == '__main__':
    main()
