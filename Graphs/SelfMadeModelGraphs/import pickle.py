import pickle
import sys
import os

def load_model(filepath):
    """Load a serialized model from a pickle file."""
    if not os.path.isfile(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
    except pickle.UnpicklingError:
        print("Error: Failed to unpickle the file. The file may be corrupted or not a pickle file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """Main function to load and interact with the model."""
    if len(sys.argv) != 2:
        print("Usage: python load_model.py <path_to_pickle_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    model = load_model(filepath)
    
    # Example: Print model architecture
    try:
        print("\nModel Architecture:")
        print(f"Number of Layers: {model.num_layers}")
        print(f"Layer Sizes: {model.sizes}")
    except AttributeError:
        print("Warning: The loaded model does not have 'num_layers' or 'sizes' attributes.")
    
    # Further interactions can be added here
    # For example, evaluating the model on test data or making predictions

if __name__ == '__main__':
    main()
