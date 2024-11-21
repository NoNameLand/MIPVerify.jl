using MIPVerify
using Gurobi
using HiGHS
using Images
using MAT

# Including functions
include("utils/create_sequential_model.jl")

# Creating Model
println("The current dir is: ", pwd())
path_to_network = "ariel_tries/networks/mnist2.mat" # Path to network
model = create_sequential_model(path_to_network, "model.n1")


