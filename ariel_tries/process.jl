using MIPVerify
using Gurobi
using HiGHS
using Images
using MAT
using JuMP
using Printf

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
MIPVerify.set_log_level("debug")

# Loading mnist 
mnist = MIPVerify.read_datasets("MNIST")
mnist.train

# Creating Model
println("The current dir is: ", pwd())
path_to_network = "ariel_tries/networks/adjusted_small_mnist_model.mat" # Path to network
model = create_sequential_model(path_to_network, "model.n1")
println(model)

image_num = 4 # The sample number
# Choosing the input to find adverserial attack against
sample_image = MIPVerify.get_image(mnist.test.images, image_num)

# Get class of network
model_output = model(sample_image)
predicted_class = argmax(model_output)
println("Predicted Class: ", predicted_class)

if MIPVerify.get_label(mnist.test.labels, image_num) != predicted_class
    println("The model classified the wanted sample wrong!")
end

d = MIPVerify.find_adversarial_example(
    model,
    sample_image,
    8,
    Gurobi.Optimizer,
    Dict("output_flag" => false),
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.105),
    norm_order = Inf,
    tightening_algorithm = lp,
)
print_summary(d)

diff = value.(d[:Perturbation])
view_diff(diff[1, :, :, 1])

perturbed_input = JuMP.value.(d[:PerturbedInput])
colorview(Gray, perturbed_input[1, :, :, 1])
# colorview(Gray, sample_image[1, :, :, 1])