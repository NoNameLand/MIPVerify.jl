import sys
import os

# Add the directory containing nn_utils.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/nn')))



import scipy.io
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nn_utils import evaluate_network
import json
import scipy.io as sio

# Set the JULIA_PROJECT environment variable to use the correct Julia environment
os.environ["JULIA_PROJECT"] = "../../"

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

def load_mat_file(file_path):
    try:
        # Try to load as HDF5 file
        with h5py.File(file_path, 'r') as f:
            return {key.replace('/', '_'): f[key][()] for key in f.keys() if not key.startswith('__')}
    except:
        # If it fails, try to load as a standard MATLAB file
        mat = scipy.io.loadmat(file_path)
        return {key.replace('/', '_'): value for key, value in mat.items() if not key.startswith('__')}

def compare_mat_files(mat_file_path1, mat_file_path2):
    mat1 = load_mat_file(mat_file_path1)
    mat2 = load_mat_file(mat_file_path2)
    
    keys1 = set(mat1.keys())
    keys2 = set(mat2.keys())
    
    if keys1 != keys2:
        return False, keys1.symmetric_difference(keys2)
    
    for key in keys1:
        if (np.abs(np.array(mat1[key]).T - np.array(mat2[key])) > 1e-10).any(): # Some tolerance
            return False, key
    
    return True, None

def compare_output_nn(nn_julia, nn_python, test_data):
    """
    nn_julia: A mat file of the neural network from Julia
    nn_python: A pth file of the neural network from Python
    """

    # Compare the output of the neural network

    # Loading and Evaluating the Neural Network Python
    # nn = torch.load(nn_python, weights_only=False)
    py_acc = evaluate_network(nn_python, test_data)

    # Loading and Evaluating the Neural Network Julia
    # Initialize Julia

    # Include the Julia script
    #jl.eval('using Pkg; Pkg.activate("../"); Pkg.instantiate()')
    #jl.eval('import Pkg; Pkg.add("Flux"); Pkg.add("ProgressMeter"); Pkg.add("MIPVerify")')
    # Main.eval('using Pkg; Pkg.activate("../"); Pkg.instantiate()')
    Main.eval('include("ariel_tries/tests/test_nn_jl.jl")') # mbe needed /home/ariel/CodeProjects/GlobalRobustnessProject/MIPVerify.jl/ariel_tries/tests/test_nn_jl.jl
    
    # Define a Python function to call the Julia function
    def test_model_accuracy(model_path, dataset_path):
        return Main.test_model_accuracy(model_path, dataset_path)
    
    acc_jl = test_model_accuracy(nn_julia, test_data)

    print(f"Python accuracy: {py_acc}")
    print(f"Julia accuracy: {acc_jl}")


import h5py

def load_mat_file(file_path):
    """
    Tries to load a .mat file using h5py. If it fails, falls back to scipy.io.loadmat.

    Parameters:
    file_path (str): Path to the .mat file.

    Returns:
    dict: Dictionary containing the data from the .mat file.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            return {key: np.array(f[key]) for key in f.keys()}
    except OSError:
        return sio.loadmat(file_path)

def compare_activations(single_mat_path, layers_dir):
    """
    Compares the activations stored in a single .mat file with the activations stored in multiple .mat files.

    Parameters:
    single_mat_path (str): Path to the .mat file containing activations of all layers.
    layers_dir (str): Path to the directory containing .mat files for each layer.

    Returns:
    bool: True if the activations are the same, False otherwise.
    """
    # Check if the single .mat file exists
    if not os.path.exists(single_mat_path):
        print(f"File not found: {single_mat_path}")
        return False

    # Load the single .mat file containing activations of all layers
    single_mat_data = load_mat_file(single_mat_path)

    # Iterate over the .mat files in the layers directory
    for layer_file in os.listdir(layers_dir):
        if layer_file.endswith(".mat"):
            layer_name = layer_file.replace("_activations.mat", "")
            layer_path = os.path.join(layers_dir, layer_file)

            # Check if the layer .mat file exists
            if not os.path.exists(layer_path):
                print(f"File not found: {layer_path}")
                return False

            print(f"Comparing activations for {layer_name}")

            # Load the layer .mat file
            layer_data = load_mat_file(layer_path)

            # Compare the activations
            if layer_name in single_mat_data and layer_name in layer_data:
                single_activations = single_mat_data[layer_name]
                layer_activations = layer_data[layer_name]

                if not np.array_equal(single_activations, layer_activations):
                    print(f"Mismatch found in {layer_name}")
                    return False
            else:
                print(f"Layer {layer_name} not found in one of the files")
                return False

    return True



# Loading params
with open("ariel_tries/utils/params.json", "r") as f:
    params = json.load(f)

path_julia = "ariel_tries/networks/julia_nn.mat"
path_py = "ariel_tries/networks/mnist_model_adjusted.mat"

result, key = compare_mat_files(path_julia, path_py)
if result:
    print("The files are the same")
else:
    print(f"The files differ at key {key}")

print("current working directory: ", os.getcwd())
result = compare_activations("/home/ariel/CodeProjects/GlobalRobustnessProject/results/neuron_statistics_jl/neuron_activations_jl.mat",
                              "../results/nn_statistics_test/")

#compare_output_nn(path_julia, path_py, params["dataset_path"])