from nn_utils import train_model, save_model_layers, adjust_model_weights, evaluate_network
import json
import torch

nn_path = "ariel_tries/networks.mat"
dataset_path = "deps/datasets/mnist/mnist_data.mat" #TODO: Add path to mnist dataset

"""layer_definitions = [
    {
        'type': 'conv',
        'in_channels': 1,         # MNIST images have 1 channel (grayscale)
        'out_channels': 6,        # Small number of filters to keep the model small
        'kernel_size': 5,         # Kernel size of 5x5
        'stride': 1,
        'padding': 0              # No padding
    },
    {
        'type': 'fc',
        'in_features': 6 * 24 * 24,  # Output from conv layer flattened
        'out_features': 10           # 10 classes for MNIST digits
    }
] # Last layer
"""

"""layer_definitions = [
    # First Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 1,         # MNIST images have 1 channel (grayscale)
        'out_channels': 16,       # Increase the number of filters
        'kernel_size': 5,         # Kernel size of 5x5
        'stride': 1,
        'padding': 2              # Add padding to preserve spatial dimensions
    },
    # Second Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 16,
        'out_channels': 32,
        'kernel_size': 5,
        'stride': 1,
        'padding': 2
    },
    # Fully Connected Layer
    {
        'type': 'fc',
        'in_features': 32 * 28 * 28,  # Output from conv layers after 2x2 pooling
        'out_features': 128         # Dense layer with 128 units
    },
    # Final Classification Layer
    {
        'type': 'fc',
        'in_features': 128,
        'out_features': 10          # 10 classes for MNIST digits
    }
]"""
"""
layer_definitions = [
    # First Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 1,         # MNIST images have 1 channel (grayscale)
        'out_channels': 8,        # Reduced the number of filters
        'kernel_size': 3,         # Smaller kernel size of 3x3
        'stride': 1,
        'padding': 1              # Minimal padding to preserve spatial dimensions
    },
    # Second Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 8,
        'out_channels': 16,       # Fewer filters in the second layer
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    # Fully Connected Layer
    {
        'type': 'fc',
        'in_features': 16 * 28 * 28,  # Output from conv layers after 2x2 pooling
        'out_features': 64          # Smaller dense layer with 64 units
    },
    # Final Classification Layer
    {
        'type': 'fc',
        'in_features': 64,
        'out_features': 10          # 10 classes for MNIST digits
    }
]"""

"""
layer_definitions = [
    {
        'type': 'fc',
        'in_features': 784,
        'out_features': 512
    },
    {
        'type': 'fc',
        'in_features': 512,
        'out_features': 256
    },
    {
        'type': 'fc',
        'in_features': 256,
        'out_features': 128
    },
    {
        'type': 'fc',
        'in_features': 128,
        'out_features': 64
    },
    {
        'type': 'fc',
        'in_features': 64,
        'out_features': 32
    },
    {
        'type': 'fc',
        'in_features': 32,
        'out_features': 16
    },
    {
        'type': 'fc',
        'in_features': 16,
        'out_features': 10
    }
]
"""
"""
layer_definitions = [
    # First Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 1,         # MNIST images have 1 channel (grayscale)
        'out_channels': 16,       # Reduced number of filters
        'kernel_size': 3,         # Kernel size of 3x3
        'stride': 1,
        'padding': 1              # Padding to preserve spatial dimensions
    },
    # Second Convolutional Layer
    {
        'type': 'conv',
        'in_channels': 16,
        'out_channels': 32,       # Reduced number of filters
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    # Fully Connected Layer
    {
        'type': 'fc',
        'in_features': 32 * 28 * 28,  # Output from conv layers flattened
        'out_features': 128          # Dense layer with 128 units
    },
    # Final Classification Layer
    {
        'type': 'fc',
        'in_features': 128,
        'out_features': 10          # 10 classes for MNIST digits
    }
]
"""

layer_definitions = [
{
    'type': 'fc',
    'in_features': 784,  # MNIST images are 28x28 pixels, flattened to 784
    'out_features': 512  # First dense layer with 512 units
},
{
    'type': 'fc',
    'in_features': 512,
    'out_features': 256  # Second dense layer with 256 units
},
{
    'type': 'fc',
    'in_features': 256,
    'out_features': 128  # Third dense layer with 128 units
},
{
    'type': 'fc',
    'in_features': 128,
    'out_features': 64   # Fourth dense layer with 64 units
},
{
    'type': 'fc',
    'in_features': 64,
    'out_features': 10   # Final layer with 10 units for 10 classes
}

]

save_path = 'ariel_tries/networks/mnist_model.mat'

with open("ariel_tries/utils/params.json", "r") as f:
    params = json.load(f)

# Save the model to a .mat file
save_model_layers(layer_definitions, params["layers_def"])

# Train the model
"""
train_model(
    network_path=params["layers_def"],
    dataset_path=params["dataset_path"],  # Ensure this file contains your MNIST data
    output_pth_path=params["path_to_nn_pth"],
    output_mat_path=params["path_to_nn_mat"],
    log_file_path=params["log_file_path_train"],
    epochs=30,# Train for 10 epochs
    batch_size=128, # Batch size of 64, can be increased if you have more memory. 
    learning_rate=0.001, # Learning rate of 0.003
    weight_decay=1e-4, # Weight decay of 4e-3, to prevent overfitting. 
    num_folds=10 # Use 20 folds for cross-validation
)
"""
adjust_model_weights(save_path, params["path_to_nn_adjust"])


# Test the model
print(evaluate_network(params["path_to_nn_pth"], params["dataset_path"]))

# Adjust the model weights for MIPVerify.jl
# adjust_model_weights(save_path, params["path_to_nn_adjust"])
