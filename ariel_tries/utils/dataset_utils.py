import scipy.io as sio
import numpy as np
from torchvision import datasets, transforms
import torch

def save_mnist_to_mat(mat_file_path):
    # Define transformations for the MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Get the full training data and labels
    train_data_iter = iter(train_loader)
    train_images, train_labels = next(train_data_iter)

    # Get the full test data and labels
    test_data_iter = iter(test_loader)
    test_images, test_labels = next(test_data_iter)

    # Convert images to numpy arrays and reshape them
    train_set = train_images.numpy().reshape(-1, 28*28)  # Flatten images
    train_labels = train_labels.numpy()

    test_set = test_images.numpy().reshape(-1, 28*28)
    test_labels = test_labels.numpy()

    # Create a dictionary to save to .mat file
    mnist_data = {
        'train_set': train_set,
        'train_labels': train_labels,
        'test_set': test_set,
        'test_labels': test_labels
    }

    # Save to .mat file
    sio.savemat(mat_file_path, mnist_data)
    print(f"MNIST data saved to {mat_file_path}")

# Example usage:
# save_mnist_to_mat('mnist_data.mat')

