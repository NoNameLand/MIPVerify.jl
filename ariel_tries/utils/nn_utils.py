import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import os
import re


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
    
    

def train_model(network_path, dataset_path, output_pth_path, output_mat_path, epochs=10, batch_size=32, learning_rate=0.001):
    print("Starting process...")

    # Step 1: Load the model from the .mat file
    print("Loading the model...")
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found at {network_path}")
    model_data = sio.loadmat(network_path)

    # Extract keys excluding MATLAB metadata keys
    keys = [key for key in model_data.keys() if not key.startswith('__')]

    # Build a dictionary of layer parameters
    layer_params = {}
    for key in keys:
        if 'weight' in key or 'bias' in key:
            match = re.match(r'layer_(\d+)/(weight|bias)', key)
            if match:
                layer_num = int(match.group(1))
                param_type = match.group(2)
                if layer_num not in layer_params:
                    layer_params[layer_num] = {}
                layer_params[layer_num][param_type] = model_data[key]
            else:
                print(f"Warning: Key '{key}' does not match expected pattern and will be ignored.")

    if not layer_params:
        raise ValueError("No valid layer parameters found in the network file.")

    # Sort layers by their layer number
    sorted_layers = sorted(layer_params.items())

    # Build the model using the extracted parameters
    layers = []
    num_layers = len(sorted_layers)
    for idx, (layer_num, params) in enumerate(sorted_layers):
        weight = params.get('weight')
        bias = params.get('bias')

        if weight is None or bias is None:
            raise ValueError(f"Layer {layer_num} is missing 'weight' or 'bias'.")

        # Create Linear layer
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        linear_layer = nn.Linear(in_features, out_features)
        linear_layer.weight.data = torch.from_numpy(weight).float()
        linear_layer.bias.data = torch.from_numpy(bias).float()
        layers.append(linear_layer)

        # Add ReLU activation after each layer except the last
        if idx < num_layers - 1:
            layers.append(nn.ReLU())

    # Create the sequential model
    model = nn.Sequential(*layers)
    print("Model loaded successfully.")

    # Step 2: Load the dataset from the .mat file
    print("Loading the dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    dataset = sio.loadmat(dataset_path)

    # Check for required keys in the dataset
    if "train_set" not in dataset or "train_labels" not in dataset:
        raise KeyError("The dataset file must contain 'train_set' and 'train_labels'.")

    train_set = torch.from_numpy(dataset["train_set"]).float()
    train_labels = torch.from_numpy(dataset["train_labels"]).float()
    print("Dataset loaded successfully.")

    # Step 3: Prepare DataLoader
    train_dataset = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Train the model
    print("Training the model...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    print("Training completed.")

    # Step 6: Save the trained model as a .pth file
    print("Saving the trained model as .pth file...")
    torch.save(model.state_dict(), output_pth_path)
    print(f"Model saved to {output_pth_path}")

    # Step 7: Save the trained model as a .mat file
    print("Saving the trained model as .mat file...")
    # Prepare the data in the same format as the input .mat file
    trained_params = {}
    layer_counter = 0  # Counter to track the linear layers
    for idx in range(len(model)):
        layer = model[idx]
        if isinstance(layer, nn.Linear):
            weight_key = f'layer{layer_counter + 1}_weight'
            bias_key = f'layer{layer_counter + 1}_bias'
            trained_params[weight_key] = layer.weight.data.numpy()
            trained_params[bias_key] = layer.bias.data.numpy()
            layer_counter += 1

    # Save the parameters to a .mat file
    sio.savemat(output_mat_path, trained_params)
    print(f"Model saved to {output_mat_path}")
    
    
    
def combine_mat_files(mat_file1, mat_file2, output_file):
    # Load the data from the two .mat files
    data1 = sio.loadmat(mat_file1)
    data2 = sio.loadmat(mat_file2)

    # Remove MATLAB metadata entries
    data1 = {key: value for key, value in data1.items() if not key.startswith('__')}
    data2 = {key: value for key, value in data2.items() if not key.startswith('__')}

    # Combine the dictionaries
    combined_data = {**data1, **data2}
    # Note: If there are duplicate keys, data from data2 will overwrite data1

    # Save the combined data into a new .mat file
    sio.savemat(output_file, combined_data)
    print(f"Combined .mat file saved to {output_file}")