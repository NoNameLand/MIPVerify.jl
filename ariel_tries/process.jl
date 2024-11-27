using MIPVerify
using Gurobi
using HiGHS
using Images
using HDF5  # For saving data in HDF5 format
using JuMP
using Printf
using Dates

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
# MIPVerify.set_log_level("debug")

# Loading MNIST dataset
mnist = MIPVerify.read_datasets("MNIST")

# Creating Model
println("The current dir is: ", pwd())
path_to_network = "ariel_tries/networks/adjusted_small_mnist_model.mat"  # Path to network
model = create_sequential_model(path_to_network, "model.n1")
println(model)

image_num = 1  # The sample number
# Choosing the input to find adversarial attack against
sample_image = MIPVerify.get_image(mnist.test.images, image_num)

# Get class of network
model_output = model(sample_image)
predicted_class = argmax(model_output)
println("Predicted Class: ", predicted_class)

if MIPVerify.get_label(mnist.test.labels, image_num) != predicted_class
    println("The model classified the wanted sample wrong!")
end

# Finding the adversarial example
d = MIPVerify.find_adversarial_example(
    model,
    sample_image,
    8,
    Gurobi.Optimizer,
    Dict("output_flag" => false),
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.06),
    norm_order = Inf,
    tightening_algorithm = lp,
)
println("Solve Status is: ", d[:SolveStatus])
#print_summary(d)
if d[:SolveStatus] != "INFEASIBLE_OR_UNBOUNDED"
    # Get the perturbation and perturbed input
    diff = value.(d[:Perturbation])
    perturbed_input = value.(d[:PerturbedInput])

    # --- New Code Starts Here ---

    # Step 1: Create a folder named by the current date and time
    current_datetime = Dates.now()
    folder_name = Dates.format(current_datetime, "yyyy-mm-dd_HH-MM-SS")
    mkpath(folder_name)  # Creates the directory

    # Step 2: Save the results as an HDF5 file
    results_file = joinpath(folder_name, "results.h5")
    h5open(results_file, "w") do file
        # Save perturbation and perturbed input
        write(file, "diff", diff)  # Save the perturbation
        write(file, "perturbed_input", perturbed_input)  # Save the perturbed input

        # Save additional data like objective value and solve time
        write(file, "objective_value", JuMP.objective_value(d[:Model]))
        write(file, "solve_time", JuMP.solve_time(d[:Model]))

        # Save the path to the network for traceability
        write(file, "path_to_network", path_to_network)
    end

    # Save a summary of the `d` dictionary in a text file
    d_summary_file = joinpath(folder_name, "d_summary.txt")
    open(d_summary_file, "w") do file
        write(file, "Summary of `d` dictionary:\n$(d)\n")
    end

    # --- New Code Ends Here ---

    # Visualize the perturbation
    view_diff(diff[1, :, :, 1])

    # Visualize the perturbed input
    colorview(Gray, perturbed_input[1, :, :, 1])
else
    println("The NN is locally robust in that neighborhood")
end
