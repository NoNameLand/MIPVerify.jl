using MIPVerify
using Gurobi
using HiGHS
using Images
using HDF5  # For saving data in HDF5 format
using JuMP
using Printf
using Dates
using MathOptInterface
using JSON


# loading params
params = JSON.parsefile("ariel_tries/utils/params.json")

# Utils Functions
function print_summary(d::Dict)
    # Helper function to print out output
    obj_val = JuMP.objective_value(d[:Model])
    solve_time = JuMP.solve_time(d[:Model])
    println("Objective Value: $(@sprintf("%.6f", obj_val)), Solve Time: $(@sprintf("%.2f", solve_time))")
end

function view_diff(diff::Array{<:Real, 2})
    n = 1001
    colormap("RdBu", n)[ceil.(Int, (diff .+ 1) ./ 2 .* n)]
end

# Including functions
include("utils/create_sequential_model.jl")

# Setting log outputs
MIPVerify.set_log_level!("debug")

# Loading MNIST dataset
mnist = MIPVerify.read_datasets("MNIST")

# Creating Model
println("The current dir is: ", pwd())
path_to_network = params["path_to_nn_adjust"]#"ariel_tries/networks/mnist_model.mat"  # Path to network
model = create_sequential_model(path_to_network, "model.n1")
println(model)

global image_num = 4 # The sample number
global classified_wrong = true
while classified_wrong
    # Choosing the input to find adversarial attack against
    global sample_image = MIPVerify.get_image(mnist.test.images, image_num)

    # Get class of network
    model_output = model(sample_image)
    global predicted_class = argmax(model_output)
    global real_class = MIPVerify.get_label(mnist.test.labels, image_num) + 1
    println("Predicted Class: ", predicted_class)
    println("The real label: ", real_class)

    if real_class != predicted_class
        println("The model classified the wanted sample wrong!")
    end

    if real_class != predicted_class
        global image_num += 1
    else 
        global classified_wrong = false
        println("The model classified the wanted sample correctly!")
    end
end

# Finding appropriate label
function exclude_number(n::Int)
    # Create an array of numbers from 1 to 10
    numbers = 1:10
    # Filter out the given number
    filtered_numbers = filter(x -> x != n, numbers)
    # Return the resulting array
    return collect(filtered_numbers)
end
# Define the perturbation family
pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.1)

# Compute neuron bounds
neuron_bounds = MIPVerify.compute_neuron_bounds(
    model,
    sample_image,
    pp;
    optimizer = Gurobi.Optimizer,
    optimizer_options = Dict("OutputFlag" => false),
    tightening_algorithm = :lp
)

# neuron_bounds now contains the upper and lower bounds for each neuron
# You can access them as follows:
for layer_idx in 1:length(neuron_bounds)
    layer_bounds = neuron_bounds[layer_idx]
    println("Layer $layer_idx:")
    println("Lower bounds: ", layer_bounds.lower)
    println("Upper bounds: ", layer_bounds.upper)
end
