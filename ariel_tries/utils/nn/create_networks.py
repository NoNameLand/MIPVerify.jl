import scipy.io as sio
import numpy as np

def save_fc_layers(input_size, layer_sizes, save_path):
    """
    Creates a model representation with input size and fully connected (FC) layers
    and saves it to a .mat file.

    Parameters:
        input_size (int): Size of the input layer.
        layer_sizes (list of int): List of sizes for each fully connected layer.
        save_path (str): Path to save the .mat file.
    """
    model_layers = {}
    
    # Store input size
    model_layers["input_size"] = input_size

    # Generate layers and weights for each FC layer
    previous_size = input_size
    for i, size in enumerate(layer_sizes):
        layer_name = f"layer_{i+1}"
        weights_name = f"{layer_name}/weight"
        biases_name = f"{layer_name}/bias"
        
        # Randomly initialize weights and biases
        weights = np.random.randn(previous_size, size)
        biases = np.random.randn(size)

        # Store weights and biases in the dictionary
        model_layers[weights_name] = weights
        model_layers[biases_name] = biases
        
        # Update previous size for the next layer
        previous_size = size

    # Save the model layers to a .mat file
    sio.savemat(save_path, model_layers)
    print(f"Model saved to {save_path}")

'''
# Example usage:
input_size = 128  # Example input size
layer_sizes = [64, 32, 16]  # Example layer sizes
save_path = "fc_layers_model.mat"  # Path to save the .mat file

save_fc_layers(input_size, layer_sizes, save_path)
'''
