#=
quick_start:
- Julia version: 
- Author: root
- Date: 2024-11-17
=#
using MIPVerify
using Gurobi
using HiGHS
using Images
using MAT

mnist = MIPVerify.read_datasets("MNIST")
mnist.train

# Load network
println("The current dir is: ", pwd())
path_to_network = "ariel_tries/networks/mnist_model_weights.mat" # Path to network
data = matread(path_to_network) # read the .mat file
dict_data = Dict(data) # converting matdict to dict #TODO: understand what's the diffrence
for (key, value) in dict_data
    println("$key => $(typeof(value)), size: $(size(value))")
end
dense_1 = get_matrix_params(dict_data, "dense_1", (128, 64))
dense_2 = get_matrix_params(dict_data, "dense_2", (64, 10))
dense = get_matrix_params(dict_data, "dense", (784, 128))

mnist_nn = Sequential([
            Flatten(4),
            dense,
            ReLU(),
            dense_1,
            ReLU(),
            dense_2
            ], "MNIST.n1")

# Trying to find adverserial
using JuMP
using Printf

# mnist already ready
# mnist.n1 already ready as mnist_nn

sample_image = MIPVerify.get_image(mnist.test.images, 1)

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

d = MIPVerify.find_adversarial_example(
    mnist_nn,
    sample_image,
    9,
    HiGHS.Optimizer,
    Dict("output_flag" => false),
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.105),
    norm_order = Inf,
    tightening_algorithm = lp,
)
print_summary(d)

# Seeing The Difference 
perturbed_input = JuMP.value.(d[:PerturbedInput])
colorview(Gray, perturbed_input[1, :, :, 1])