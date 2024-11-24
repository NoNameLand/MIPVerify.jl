from nn_utils import train_model, save_model_layers, adjust_model_weights

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
]""" # Last layer

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
layer_definitions = [
    # First Convolutional Layer
    {
        type = :conv,
        in_channels = 1,         # MNIST images have 1 channel (grayscale)
        out_channels = 8,        # Reduced the number of filters
        kernel_size = 3,         # Smaller kernel size of 3x3
        stride = 1,
        padding = 1              # Minimal padding to preserve spatial dimensions
    },
    # Second Convolutional Layer
    {
        type = :conv,
        in_channels = 8,
        out_channels = 16,       # Fewer filters in the second layer
        kernel_size = 3,
        stride = 1,
        padding = 1
    },
    # Fully Connected Layer
    {
        type = :fc,
        in_features = 16 * 7 * 7,  # Output from conv layers after 2x2 pooling
        out_features = 64          # Smaller dense layer with 64 units
    },
    # Final Classification Layer
    {
        type = :fc,
        in_features = 64,
        out_features = 10          # 10 classes for MNIST digits
    }
]



# Save the model to a .mat file
save_model_layers(layer_definitions, 'ariel_tries/networks/small_mnist_model.mat')

# Train the model
train_model(
    network_path='ariel_tries/networks/small_mnist_model.mat',
    dataset_path='deps/datasets/mnist/mnist_data.mat',  # Ensure this file contains your MNIST data
    output_pth_path='ariel_tries/networks/trained_small_mnist_model.pth',
    output_mat_path='ariel_tries/networks/trained_small_mnist_model.mat',
    epochs=5,
    batch_size=64,
    learning_rate=0.001
)

# Adjust the model weights for MIPVerify.jl
adjust_model_weights('ariel_tries/networks/trained_small_mnist_model.mat', 'ariel_tries/networks/adjusted_small_mnist_model.mat')
