import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import os
import re
from sklearn.model_selection import KFold
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt


def save_model_layers(layer_definitions, save_path):
    """
    Creates a model representation with specified layers and saves it to a .mat file.

    Parameters:
        layer_definitions (list of dict): List of layer definitions. Each layer definition is a dictionary with keys specifying layer type and parameters.
        save_path (str): Path to save the .mat file.
    """
    model_layers = {}

    for i, layer_def in enumerate(layer_definitions):
        layer_name = f"layer_{i+1}"
        layer_type = layer_def['type']
        model_layers[f"{layer_name}/type"] = layer_type  # Store layer type as string

        if layer_type == 'fc':
            # Fully Connected Layer
            weights_name = f"{layer_name}/weight"
            biases_name = f"{layer_name}/bias"

            in_features = layer_def['in_features']
            out_features = layer_def['out_features']

            # Randomly initialize weights and biases
            weights = np.random.randn(out_features, in_features)
            biases = np.random.randn(out_features)

            # Store weights and biases in the dictionary
            model_layers[weights_name] = weights
            model_layers[biases_name] = biases

        elif layer_type == 'conv':
            # Convolutional Layer
            weights_name = f"{layer_name}/weight"
            biases_name = f"{layer_name}/bias"
            stride_name = f"{layer_name}/stride"
            padding_name = f"{layer_name}/padding"

            in_channels = layer_def['in_channels']
            out_channels = layer_def['out_channels']
            kernel_size = layer_def['kernel_size']  # Should be an int or tuple
            stride = layer_def.get('stride', 1)
            padding = layer_def.get('padding', 0)

            # Ensure kernel_size is a tuple
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            # Randomly initialize weights and biases
            # Weights shape: (out_channels, in_channels, kernel_height, kernel_width)
            weights = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
            biases = np.random.randn(out_channels)

            # Store weights, biases, stride, and padding in the dictionary
            model_layers[weights_name] = weights
            model_layers[biases_name] = biases
            model_layers[f"{layer_name}/stride"] = np.array([stride])
            model_layers[f"{layer_name}/padding"] = np.array([padding])
            model_layers[f"{layer_name}/kernel_size"] = np.array(kernel_size)

        elif layer_type == 'pool':
            # Pooling Layer
            kernel_size_name = f"{layer_name}/kernel_size"
            stride_name = f"{layer_name}/stride"
            padding_name = f"{layer_name}/padding"
            pool_type_name = f"{layer_name}/pool_type"

            kernel_size = layer_def.get('kernel_size', 2)
            stride = layer_def.get('stride', kernel_size)
            padding = layer_def.get('padding', 0)
            pool_type = layer_def.get('pool_type', 'max')  # 'max' or 'avg'

            # Ensure kernel_size is a tuple
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)

            # Save the parameters
            model_layers[kernel_size_name] = np.array(kernel_size)
            model_layers[stride_name] = np.array(stride)
            model_layers[padding_name] = np.array([padding])
            model_layers[pool_type_name] = pool_type  # Store as string

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    # Save the model layers to a .mat file
    sio.savemat(save_path, model_layers)
    print(f"Model saved to {save_path}")


import os
import re
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import logging

def train_model(network_path, dataset_path, output_pth_path, output_mat_path, epochs=10, batch_size=32, learning_rate=0.001, num_folds=5, log_file_path='process_log.txt', 
                weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set up logging
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Starting process...")

    # Step 1: Load the model from the .mat file
    logging.info("Loading the model...")
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found at {network_path}")
    model_data = sio.loadmat(network_path)

    def build_model_from_mat(model_data):
        # Extract keys excluding MATLAB metadata keys
        keys = [key for key in model_data.keys() if not key.startswith('__')]

        # Build a dictionary of layer parameters
        layer_params = {}
        for key in keys:
            if any(param in key for param in ['weight', 'bias', 'stride', 'padding', 'type', 'pool_type', 'kernel_size']):
                match = re.match(r'layer_(\d+)/(weight|bias|stride|padding|type|pool_type|kernel_size)', key)
                if match:
                    layer_num = int(match.group(1))
                    param_type = match.group(2)
                    if layer_num not in layer_params:
                        layer_params[layer_num] = {}
                    layer_params[layer_num][param_type] = model_data[key]
                else:
                    logging.warning(f"Key '{key}' does not match expected pattern and will be ignored.")

        if not layer_params:
            raise ValueError("No valid layer parameters found in the network file.")

        # Sort layers by their layer number
        sorted_layers = sorted(layer_params.items())

        # Build the model using the extracted parameters
        layers = []
        num_layers = len(sorted_layers)
        prev_layer_type = None  # Keep track of the previous layer type
        for idx, (layer_num, params) in enumerate(sorted_layers):
            # Get layer type
            layer_type = params.get('type')
            if layer_type is None:
                raise ValueError(f"Layer {layer_num} is missing 'type'.")

            if isinstance(layer_type, np.ndarray):
                layer_type = layer_type.item()
            else:
                layer_type = str(layer_type)

            if layer_type == 'fc':
                weight = params.get('weight')
                bias = params.get('bias').squeeze()

                if weight is None or bias is None:
                    raise ValueError(f"Layer {layer_num} is missing 'weight' or 'bias'.")

                out_features, in_features = weight.shape
                linear_layer = nn.Linear(in_features, out_features)
                linear_layer.weight.data = torch.from_numpy(weight).float()
                linear_layer.bias.data = torch.from_numpy(bias).float()
                # Insert Flatten if previous layer was Conv2d or Pooling
                if prev_layer_type in ['Conv2d', 'MaxPool2d', 'AvgPool2d']:
                    layers.append(nn.Flatten())

                layers.append(linear_layer)
                prev_layer_type = 'Linear'

                # Add activation function after each layer except the last
                if idx < num_layers - 1:
                    layers.append(nn.ReLU())

            elif layer_type == 'conv':
                weight = params.get('weight')
                bias = params.get('bias').squeeze()
                stride = int(params.get('stride', 1))
                padding = int(params.get('padding', 0))

                if weight is None or bias is None:
                    raise ValueError(f"Layer {layer_num} is missing 'weight' or 'bias'.")

                out_channels, in_channels, kernel_height, kernel_width = weight.shape
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)
                conv_layer.weight.data = torch.from_numpy(weight).float()
                conv_layer.bias.data = torch.from_numpy(bias).float()
                layers.append(conv_layer)
                prev_layer_type = 'Conv2d'

                # Add activation function after each layer except the last
                if idx < num_layers - 1:
                    layers.append(nn.ReLU())

            elif layer_type == 'pool':
                pool_type = params.get('pool_type')
                kernel_size = params.get('kernel_size')
                stride = params.get('stride')
                padding = int(params.get('padding', 0))

                if pool_type is None or kernel_size is None:
                    raise ValueError(f"Layer {layer_num} is missing 'pool_type' or 'kernel_size'.")

                if isinstance(pool_type, np.ndarray):
                    pool_type = pool_type.item()
                else:
                    pool_type = str(pool_type)

                kernel_size = tuple(kernel_size.flatten())
                stride = tuple(stride.flatten()) if stride is not None else kernel_size

                if pool_type == 'max':
                    pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                elif pool_type == 'avg':
                    pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                else:
                    raise ValueError(f"Unsupported pool type: {pool_type}")
                layers.append(pool_layer)
                prev_layer_type = 'MaxPool2d' if pool_type == 'max' else 'AvgPool2d'

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Create the sequential model
        model = nn.Sequential(*layers)
        return model

    # Step 2: Load the dataset from the .mat file
    logging.info("Loading the dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    dataset = sio.loadmat(dataset_path)

    # Check for required keys in the dataset
    if "train_set" not in dataset or "train_labels" not in dataset:
        raise KeyError("The dataset file must contain 'train_set' and 'train_labels'.")

    train_set = dataset["train_set"]
    train_labels = dataset["train_labels"]

    # Build model
    model = build_model_from_mat(model_data)
    
    # Reshape and convert data based on the first layer
    if isinstance(model[0],nn.Conv2d):
        # Assuming images are stored as (num_samples, height, width) or (num_samples, channels, height, width)
        train_set = train_set.reshape(train_set.shape[0], 28, 28)  # Adjust based on actual image size
        if train_set.ndim == 3:
            # Add channel dimension
            train_set = np.expand_dims(train_set, axis=1)
        elif train_set.ndim == 4 and train_set.shape[1] > 1:
            # If data is already in (num_samples, channels, height, width), do nothing
            pass
        else:
            raise ValueError("Unsupported train_set shape for convolutional layers.")
    else:
        pass  # For fully connected layers, assume data is already flattened

    train_set = torch.from_numpy(train_set).float()

    # Process labels
    if train_labels.ndim > 1 and train_labels.shape[0] > 1:
        # One-hot encoded labels
        train_labels = np.argmax(train_labels, axis=0)
    elif train_labels.ndim > 1 and train_labels.shape[0] == 1:
        train_labels = train_labels.squeeze()
    train_labels = torch.from_numpy(train_labels).long()
    logging.info("Dataset loaded successfully.")

    # Step 3: Set up K-Fold Cross Validation
    logging.info(f"Performing {num_folds}-Fold Cross-Validation...")
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []
    best_accuracy = 0
    best_model = None
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        logging.info(f"\nFold {fold + 1}/{num_folds}")
        # Split data
        train_inputs, val_inputs = train_set[train_idx], train_set[val_idx]
        train_targets, val_targets = train_labels[train_idx], train_labels[val_idx]

        # Prepare DataLoader
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Step 4: Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Step 5: Train the model
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
            logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Step 6: Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        logging.info(f'Validation Accuracy for fold {fold + 1}: {val_accuracy:.2f}%')
        fold_results.append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model
            best_model = model
            logging.info(f"This fold ({fold + 1}) is the best yet, with accuracy of {val_accuracy:.2f}%")

    # Step 7: Report Cross-Validation Results
    avg_accuracy = sum(fold_results) / len(fold_results)
    logging.info(f'\nAverage Cross-Validation Accuracy: {avg_accuracy:.2f}%')

    # Step 11: Save the trained model as a .pth file
    logging.info("Saving the best trained model as .pth file...")
    torch.save(best_model, output_pth_path)
    logging.info(f"Model saved to {output_pth_path}")
    
    # Step 12: Save the trained model as a .mat file
    logging.info("Saving the trained model as .mat file...")
    # Prepare the data in the same format as the input .mat file
    trained_params = {}
    layer_counter = 0  # Counter to track the layers
    for idx in range(len(best_model)):
        layer = best_model[idx]
        if isinstance(layer, nn.Linear):
            weight_key = f'layer_{layer_counter + 1}/weight'
            bias_key = f'layer_{layer_counter + 1}/bias'
            trained_params[weight_key] = layer.weight.data.numpy()
            trained_params[bias_key] = layer.bias.data.numpy()
            layer_counter += 1
        elif isinstance(layer, nn.Conv2d):
            weight_key = f'layer_{layer_counter + 1}/weight'
            bias_key = f'layer_{layer_counter + 1}/bias'
            stride_key = f'layer_{layer_counter + 1}/stride'
            padding_key = f'layer_{layer_counter + 1}/padding'
            trained_params[weight_key] = layer.weight.data.numpy()
            trained_params[bias_key] = layer.bias.data.numpy()
            trained_params[stride_key] = np.array([layer.stride[0]])
            trained_params[padding_key] = np.array([layer.padding[0]])
            layer_counter += 1

    # Save the parameters to a .mat file
    sio.savemat(output_mat_path, trained_params)
    logging.info(f"Model saved to {output_mat_path}")
    
def adjust_model_weights(input_model_path, output_model_path):
    """
    Adjusts the weight matrices of a neural network model saved in a .mat file
    so that the weight matrices have the shape (in_features, out_features) for FC layers,
    which is required by MIPVerify.jl.

    Parameters:
        input_model_path (str): Path to the input .mat file containing the model.
        output_model_path (str): Path to save the adjusted model .mat file.
    """
    # Load the model from the .mat file
    model_data = sio.loadmat(input_model_path)

    # Extract keys excluding MATLAB metadata keys
    keys = [key for key in model_data.keys() if not key.startswith('__')]

    adjusted_model_data = {}

    for key in keys:
        # Check if the key corresponds to weight or bias or other parameters
        if 'weight' in key or 'bias' in key or 'stride' in key or 'padding' in key:
            # Match layer number and parameter type
            match = re.match(r'(layer_\d+)/(weight|bias|stride|padding)', key)
            if match:
                layer_name = match.group(1)
                param_type = match.group(2)
                param = model_data[key]

                # If it's a weight matrix for FC layer, transpose it
                """
                if param_type == 'weight':
                    if param.ndim == 2:
                        # Fully connected layer, transpose weight matrix
                        param = param.T  # Transpose the weight matrix
                    # Convolutional layers' weights do not need transposition
                """
                
                if param_type == 'bias':
                    if param.size[1] == 1:
                        param.reshape(param.size[1], param.size[0])
                

                # Save the adjusted parameter
                adjusted_model_data[f'{layer_name}/{param_type}'] = param
            else:
                print(f"Warning: Key '{key}' does not match expected pattern and will be ignored.")
        else:
            # Copy other data as is
            adjusted_model_data[key] = model_data[key]

    # Save the adjusted model to a .mat file
    sio.savemat(output_model_path, adjusted_model_data)
    print(f"Adjusted model saved to {output_model_path}")

# ...existing code...

def evaluate_network(model_path, test_data_path):
    """
    Evaluates the network's success rate on the test data.

    Parameters:
    model_path (str): Path to the .pth file containing the model.
    test_data_path (str): Path to the .mat file containing the test data.

    Returns:
    float: The success rate of the network on the test data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from the .pth file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = torch.load(model_path, weights_only=False) # weights_only is prone to attacks when False.
    model.to(device)
    model.eval()

    # Load the test data from the .mat file
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    test_data = sio.loadmat(test_data_path)

    # Check for required keys in the test data
    if "test_set" not in test_data or "test_labels" not in test_data:
        raise KeyError("The test data file must contain 'test_set' and 'test_labels'.")

    test_set = test_data["test_set"]
    test_labels = test_data["test_labels"]

    # Reshape and convert data based on the first layer
    if isinstance(model[0], nn.Conv2d):
        test_set = test_set.reshape(test_set.shape[0], 28, 28)  # Adjust based on actual image size
        if test_set.ndim == 3:
            test_set = np.expand_dims(test_set, axis=1)
        elif test_set.ndim == 4 and test_set.shape[1] > 1:
            pass
        else:
            raise ValueError("Unsupported test_set shape for convolutional layers.")
    else:
        pass  # For fully connected layers, assume data is already flattened

    test_set = torch.from_numpy(test_set).float().to(device)

    # Process labels
    if test_labels.ndim > 1 and test_labels.shape[0] > 1:
        test_labels = np.argmax(test_labels, axis=0)
    elif test_labels.ndim > 1 and test_labels.shape[0] == 1:
        test_labels = test_labels.squeeze()
    test_labels = torch.from_numpy(test_labels).long().to(device)

    # Evaluate the model
    test_dataset = TensorDataset(test_set, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    success_rate = correct / total
    return success_rate


def evaluate_network_and_save_activations(model_path, test_data_path, output_dir):
    """
    Evaluates the network's success rate on the test data and saves neuron activations for each layer.

    Parameters:
    model_path (str): Path to the .pth file containing the model.
    test_data_path (str): Path to the .mat file containing the test data.
    output_dir (str): Directory to save the neuron activations.

    Returns:
    float: The success rate of the network on the test data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from the .pth file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = torch.load(model_path, weights_only=False) # weights_only is prone to attacks when False.
    model.to(device)
    model.eval()

    # Load the test data from the .mat file
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    test_data = sio.loadmat(test_data_path)

    # Check for required keys in the test data
    if "test_set" not in test_data or "test_labels" not in test_data:
        raise KeyError("The test data file must contain 'test_set' and 'test_labels'.")

    test_set = test_data["test_set"]
    test_labels = test_data["test_labels"]

    # Reshape and convert data based on the first layer
    if isinstance(model[0], nn.Conv2d):
        test_set = test_set.reshape(test_set.shape[0], 28, 28)  # Adjust based on actual image size
        if test_set.ndim == 3:
            test_set = np.expand_dims(test_set, axis=1)
        elif test_set.ndim == 4 and test_set.shape[1] > 1:
            pass
        else:
            raise ValueError("Unsupported test_set shape for convolutional layers.")
    else:
        pass  # For fully connected layers, assume data is already flattened

    test_set = torch.from_numpy(test_set).float().to(device)

    # Process labels
    if test_labels.ndim > 1 and test_labels.shape[0] > 1:
        test_labels = np.argmax(test_labels, axis=0)
    elif test_labels.ndim > 1 and test_labels.shape[0] == 1:
        test_labels = test_labels.squeeze()
    test_labels = torch.from_numpy(test_labels).long().to(device)

    # Evaluate the model and save neuron activations
    test_dataset = TensorDataset(test_set, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    # Calculate the number of layers taht are linear of Conv2D
    num_layers = 0
    for layer in model:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            num_layers += 1
    neuron_activations = {f"layer_{i+1}": [] for i in range(num_layers)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = inputs
            real_idx = 0
            for i, layer in enumerate(model):
                outputs = layer(outputs)
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    neuron_activations[f"layer_{real_idx+1}"].append(outputs.cpu().numpy())
                    real_idx += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    success_rate = correct / total

    # Save neuron activations to .mat files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for layer, activations in neuron_activations.items():
        activations = np.concatenate(activations, axis=0)
        sio.savemat(os.path.join(output_dir, f"{layer}_activations.mat"), {layer: activations})

    return success_rate

def plot_neuron_statistics_per_layer(neuron_activations_dir, output_dir):
    """
    Plots neuron activation statistics for each layer and saves the plots.

    Parameters:
    neuron_activations_dir (str): Directory containing the neuron activations .mat files.
    output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(neuron_activations_dir):
        if file_name.endswith("_activations.mat"):
            layer_name = file_name.replace("_activations.mat", "")
            activations = sio.loadmat(os.path.join(neuron_activations_dir, file_name))[layer_name]

            means = np.mean(activations, axis=0)
            variances = np.var(activations, axis=0)
            max_val = np.max(np.abs(activations))

            plt.figure()
            for neuron in range(activations.shape[1]):
                x = np.linspace(-2 * max_val, 2 * max_val, max(int(np.ceil(10 * max_val)), 100))
                y = norm.pdf(x, means[neuron], np.sqrt(variances[neuron]))
                plt.plot(x, y, label=f"Neuron {neuron}")
                if layer_name == f"layer_{len(os.listdir(neuron_activations_dir))}":
                    peak_x = means[neuron]
                    peak_y = norm.pdf(peak_x, means[neuron], np.sqrt(variances[neuron]))
                    plt.annotate(f"{neuron}", (peak_x, peak_y), textcoords="offset points", xytext=(0,10), ha='center')

            plt.xlabel("Neuron Activation")
            plt.ylabel("Density")
            plt.title(f"Neuron Activation Statistics for {layer_name}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{layer_name}_statistics.png"))
            plt.close()