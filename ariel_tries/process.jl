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
using PrettyTables


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

# Finding the adversarial example
d = MIPVerify.find_adversarial_example(
    model,
    sample_image,
    exclude_number(predicted_class),
    Gurobi.Optimizer,
    Dict("output_flag" => false),
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.05),
    norm_order = Inf,
    tightening_algorithm = lp,
)
println("Solve Status is: ", d[:SolveStatus])
println("Num Variables: ", num_variables(d[:Model]))

# Getiing last Variables
last_var = all_variables(d[:Model])[end]
println("Last Variable: ", last_var)
println("is Int? ", is_integer(last_var))
println("Is Binary? ", is_binary(last_var))
#println("Lower Bound: ", lower_bound(last_var))
# println("Upper Bound: ", upper_bound(last_var))
println("Constraints in the model: ", num_constraints(d[:Model]; count_variable_in_set_constraints=true))
for constraint in all_constraints(d[:Model]; include_variable_in_set_constraints=true)
    #println(constraint)
end

## Trying to find last layer
last_layer = model.layers[end]
println("Last Layer Vars: ", size(last_layer.matrix))
println(size(d[:Output]))

function compute_bounds(aff_expr::AffExpr)
    # Initialize bounds with the constant term
    lb = aff_expr.constant
    ub = aff_expr.constant

    # Iterate over the terms
    for (var, coeff) in aff_expr.terms
        if coeff > 0
            lb += coeff * lower_bound(var)  # Positive coefficient → minimum contribution
            ub += coeff * upper_bound(var)  # Positive coefficient → maximum contribution
        else
            lb += coeff * upper_bound(var)  # Negative coefficient → minimum contribution
            ub += coeff * lower_bound(var)  # Negative coefficient → maximum contribution
        end
    end

    return [lb, ub]
end

# Compute bounds for the vector of affine expressions
bounds_matrix = [compute_bounds(expr) for expr in d[:Output]]
# Convert to a matrix with rows [lower_bound, upper_bound] for each expression
bounds_matrix = hcat(bounds_matrix)'
bounds_tuple = [(b[1], b[2]) for b in bounds_matrix']
println(size(bounds_matrix))
println("Bounds Out: ")
println(bounds_matrix)
# pretty_table(bounds_tuple, header=["Lower Bound", "Upper Bound"])

current_datetime = Dates.now()
global folder_name = joinpath(params["results_path"], Dates.format(current_datetime, "yyyy-mm-dd_HH-MM-SS"))
mkpath(folder_name)  # Creates the directory
if d[:SolveStatus] != MOI.INFEASIBLE_OR_UNBOUNDED
    println("Found an adversarial example")
    # Get the perturbation and perturbed input
    diff = value.(d[:Perturbation])
    perturbed_input = value.(d[:PerturbedInput])
    
    # Save the results as an HDF5 file
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

    # Visualize the perturbation
    view_diff(diff[1, :, :, 1])

    # Visualize the perturbed input
    colorview(Gray, perturbed_input[1, :, :, 1])
else
    println("The NN is locally robust in that neighborhood")
    results_file = joinpath(folder_name, "results.h5")
    h5open(results_file, "w") do file
        write(file, "time", d[:TotalTime])
        write(file, "path_to_network", path_to_network)
    end
end
