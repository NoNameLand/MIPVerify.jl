import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
import sys
import json

# Add the directory containing nn_utils.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/nn')))
from nn_utils import evaluate_network_and_save_activations, plot_neuron_statistics_per_layer

# Import the params
with open('ariel_tries/utils/params.json') as json_file:
    params = json.load(json_file)

evaluate_network_and_save_activations(params["path_to_nn_pth"], params["dataset_path"],
                                      "../results/nn_statistics_test")
plot_neuron_statistics_per_layer("../results/nn_statistics_test", 
                                 "../results/nn_statistics_test/neuron_statistics_per_layer.png")