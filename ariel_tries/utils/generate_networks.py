from nn_utils import train_model, save_model_layers, adjust_model_weights

nn_path = "ariel_tries/networks/mnist2.mat"
dataset_path = "deps/datasets/mnist/mnist_data.mat" #TODO: Add path to mnist dataset

layer_definitions = [
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
adjust_model_weights('trained_small_mnist_model.mat', 'adjusted_small_mnist_model.mat')
