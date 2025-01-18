import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import os
import re

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
            match = re.match(r'layer(\d+)_(weight|bias)', key)
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
    if "train_set" not in dataset or "train_labels" not in dataset or "test_set" not in dataset or "test_labels" not in dataset:
        raise KeyError("The dataset file must contain 'train_set', 'train_labels', 'test_set', and 'test_labels'.")

    train_set = torch.from_numpy(dataset["train_set"]).float()
    train_labels = torch.from_numpy(dataset["train_labels"]).float()
    test_set = torch.from_numpy(dataset["test_set"]).float()
    test_labels = torch.from_numpy(dataset["test_labels"]).float()
    print("Dataset loaded successfully.")

    # Step 3: Prepare DataLoader
    train_dataset = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_set, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Step 6: Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Step 7: Save the trained model as a .pth file
    print("Saving the trained model as .pth file...")
    torch.save(model.state_dict(), output_pth_path)
    print(f"Model saved to {output_pth_path}")

    # Step 8: Save the trained model as a .mat file
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
