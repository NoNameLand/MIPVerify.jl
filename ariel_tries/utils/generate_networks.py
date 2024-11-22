from nn_utils import train_model, save_fc_layers, adjust_model_weights

nn_path = "ariel_tries/networks/mnist2.mat"
dataset_path = "deps/datasets/mnist/mnist_data.mat" #TODO: Add path to mnist dataset
save_fc_layers(784, [40, 10], nn_path)
train_model(nn_path, dataset_path, "ariel_tries/networks/mnist2.pth", nn_path)
adjust_model_weights(nn_path, nn_path) # Deal with mipverify notation
