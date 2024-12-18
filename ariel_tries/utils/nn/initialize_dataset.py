from dataset_utils import save_mnist_to_mat
import json
import os

with open("ariel_tries/utils/params.json", 'r') as file:
    params = json.load(file)

save_mnist_to_mat(params["dataset_path"])