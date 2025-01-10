from dataset_utils import save_mnist_to_mat, save_mnist_to_mat_train, save_mnist_to_mat_test
import json
import os

with open("ariel_tries/utils/params.json", 'r') as file:
    params = json.load(file)

save_mnist_to_mat(params["dataset_path"])
save_mnist_to_mat_train(params["dataset_path_train"])
save_mnist_to_mat_test(params["dataset_path_test"])
