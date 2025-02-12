using MIPVerify
using Gurobi
using JuMP
using MathOptInterface
using JSON
using Memento
using ProgressMeter
using Statistics
using StatsBase
using Distributions
using MAT

# --- Include Functions --- #
include("../src/logging.jl")
include("utils/cegis/spurious_functions.jl")
include("utils/create_sequential_model.jl")
include("utils/utils_functions.jl")
include("Partition.jl")
include("utils/partition_addons.jl")
include("utils/visual_functions.jl")

function frac_correct_mine(nn::NeuralNet, dataset::MIPVerify.LabelledDataset, num_samples::Integer, output_path::String)::Tuple{Real, Vector{Matrix{Float64}}}
    num_correct = 0.0
    num_samples = min(num_samples, MIPVerify.num_samples(dataset))
    p = Progress(num_samples, desc = "Computing fraction correct...", enabled = isinteractive())

    # Initialize a list of matrices to store neuron values for linear layers
    num_layers = length(nn.layers)
    neuron_values = [zeros(Float64, size(layer.bias, 1), num_samples) for layer in nn.layers if layer isa Linear]

    for sample_index in 1:num_samples
        input = MIPVerify.get_image(dataset.images, sample_index)
        actual_label = MIPVerify.get_label(dataset.labels, sample_index)
        
        # Forward pass through the network and store neuron values for linear layers
        x = input
        linear_layer_index = 1
        for layer in nn.layers
            x = layer(x)
            if layer isa Linear
                neuron_values[linear_layer_index][:, sample_index] = x
                linear_layer_index += 1
            end
        end

        predicted_label = (x |> MIPVerify.get_max_index) - 1 #TODO: - 1 Mbe Needed
        # println("Predicted Label is: ", predicted_label, " Actual Label is: ", actual_label)
        if actual_label == predicted_label
            num_correct += 1
        end
        next!(p)
    end

    accuracy = num_correct / num_samples

    # Save neuron activations to a .mat file
    neuron_activations = Dict{String, Any}()
    for (i, activations) in enumerate(neuron_values)
        neuron_activations["layer_$i"] = activations
    end
    matwrite(output_path, neuron_activations)

    return accuracy, neuron_values
end

function plot_neuron_statistics(neuron_values::Vector{Matrix{Float64}}, output_path::String)
    num_layers = length(neuron_values)
    layer_means = [mean(mean(neuron_values[layer], dims=2)) for layer in 1:num_layers]
    layer_variances = [mean(var(neuron_values[layer], dims=2)) for layer in 1:num_layers]
    max_vals = [maximum(abs.(neuron_values[layer])) for layer in 1:num_layers]

    plot()
    for layer in 1:num_layers
        max_val = max_vals[layer]
        x = range(-2 * max_val, 2 * max_val, length=100)
        y = pdf.(Normal(layer_means[layer], sqrt(layer_variances[layer])), x)
        println("Mean of layer $layer: ", layer_means[layer], "Std of layer $layer: ", sqrt(layer_variances[layer]))
        plot!(x, y, label="Layer $layer")
    end

    xlabel!("Neuron Activation")
    ylabel!("Density")
    title!("Neuron Activation Statistics")
    savefig(output_path)
end

function plot_neuron_statistics_per_layer(neuron_values::Vector{Matrix{Float64}}, output_dir::String)
    num_layers = length(neuron_values)
    for layer in 1:num_layers
        means = mean(neuron_values[layer], dims=2)
        variances = var(neuron_values[layer], dims=2)
        max_val = maximum(abs.(neuron_values[layer]))

        plot()
        for neuron in 1:size(neuron_values[layer], 1)
            x = range(-2 * max_val, 2 * max_val, length=max(Int(ceil(10*max_val)), 100))
            y = pdf.(Normal(means[neuron], sqrt(variances[neuron])), x)
            plot!(x, y, label="Neuron $neuron")
            if layer == num_layers
                peak_x = means[neuron]
                peak_y = pdf(Normal(means[neuron], sqrt(variances[neuron])), peak_x)
                annotate!(peak_x, peak_y, text("$neuron", :center, 8))
            end
        end

        xlabel!("Neuron Activation")
        ylabel!("Density")
        title!("Neuron Activation Statistics for Layer $layer")
        savefig(joinpath(output_dir, "neuron_statistics_layer_$layer.png"))
    end
end



function process_bounds()
    # --- Constans --- # 
    eps = 0.00
    norm_order = Inf
    tightening_algorithm = lp

    # --- Model Setup --- #
    # loading params
    params = JSON.parsefile("ariel_tries/utils/params.json")


    # Setting log outputs
    MIPVerify.set_log_level!("debug")

    # Loading MNIST dataset
    mnist = MIPVerify.read_datasets("MNIST") # MIPVerify.read_datasets("MNIST")

    # Creating Model
    println("The current dir is: ", pwd())
    path_to_network = params["path_to_nn_adjust"] #"ariel_tries/networks/mnist_model.mat"  # Path to network
    println("The path to the network is: ", path_to_network)
    model = create_sequential_model(path_to_network, "model2.n1")
    save_model_as_mat(model, "ariel_tries/networks/julia_nn.mat")
    println(model)

    # Testing Model Loading 
    img = MIPVerify.get_image(mnist.test.images, 1)
    println("Image Size:", size(img))
    layer = model.layers[1]
    flattend_img = layer(img)
    println("Flattened Image size: ", size(flattend_img))

    # Print Test Accuracy
    # labelled_data_set_test = 
    println("Size of input image: ", size(MIPVerify.get_image(mnist.test.images, 1)))
    # Reshape input image to (batch_size, height, width, channels)
    # mnist.test.images = reshape(mnist.test.images, (size(mnist.test.images, 1), 28, 28, 1))
    acc, neuron_vals = frac_correct_mine(model, mnist.test, length(mnist.test.labels), "../results/neuron_statistics_jl/neuron_activations_jl.mat")
    println("The test accuracy is: ", acc)
    # Plotting the neuron statistics
    plot_neuron_statistics(neuron_vals, "../results/neuron_statistics.png")
    plot_neuron_statistics_per_layer(neuron_vals, "../results/")

    """
    # --- Finding Image to Attack --- #
    # Finding an image to local verify around
    global image_num = 1 # The sample number
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
            notice(
                MIPVerify.LOGGER,
                "The model classified the wanted sample wrong!"
            )
        end

        if real_class != predicted_class
            global image_num += 1
        else 
            global classified_wrong = false
            println("The model classified the wanted sample correctly!")
            notice(
                MIPVerify.LOGGER,
                "The model classified the wanted sample correctly!"
            )
        end
    end


    #  --- Finding the adversarial example --- #
    # -- Without Partition -- #
    d_basic = MIPVerify.find_adversarial_example(
        model,
        sample_image,
        exclude_number(predicted_class),
        Gurobi.Optimizer,
        Dict("output_flag" => false),
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps),
        norm_order = norm_order,
        tightening_algorithm = tightening_algorithm,
    )
    println("Solve Status is: ", d_basic[:SolveStatus])
    notice(
        MIPVerify.LOGGER,
        "Solve Status is: $(d_basic[:SolveStatus])"
    )
    println("Time to solve is: ", d_basic[:TotalTime], " seconds")
    notice(
        MIPVerify.LOGGER,
        "Time to solve is: $(d_basic[:TotalTime]) seconds"
    )

    # -- With Partition -- #
    p = Partition(model) # The nn as a Sequential model
    index_partiotn_real = 3
    println("The index of the partition is: ", index_partiotn_real, " and the layer is: ", find_layer_index(p, index_partiotn_real), 
    " out of ", length(p.nns), " layers")
    PartitionByLayer(p, [find_layer_index(p, index_partiotn_real)]) # Splitting in half
    # Getting bounds for the first half
    d_1 = MIPVerify.find_adversarial_example(
        p.nns[1],
        sample_image,
        exclude_number(predicted_class),
        Gurobi.Optimizer,
        Dict("output_flag" => false),
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps),
        norm_order = norm_order,
        tightening_algorithm = tightening_algorithm,
    )
    println("Time it took for first half: ", d_1[:TotalTime], "s")    
    notice(
        MIPVerify.LOGGER,
        "Time it took for the first half: '$(d_1[:TotalTime])' s"
    )
    bounds_matrix = [compute_bounds(expr) for expr in d_1[:Output]]
    # bounds_matrix_og_nn = [compute_bounds(expr) for expr in d_basic[:Output]]
    push!(p.bounds, bounds_matrix)
    # println("The bound matrix after the first half is: " ,bounds_matrix)
    lbs = [bounds_matrix[i][1] for i in 1:size(bounds_matrix)[1]]
    ubs = [bounds_matrix[i][2] for i in 1:size(bounds_matrix)[1]]

    # Getting Bounds For the Second half
    d_2 = MIPVerify.find_adversarial_example(
        p.nns[2],
        lbs,
        exclude_number(predicted_class),
        Gurobi.Optimizer,
        Dict("output_flag" => false),
        pp = CostumeBoundedPerturbationFamily(lbs, ubs),
        norm_order = norm_order,
        tightening_algorithm = tightening_algorithm,
    )
    bounds_matrix = [compute_bounds(expr) for expr in d_2[:Output]]
    # Test if the bounds after over-approximation fully contain the bounds of the original model
    bounds_matrix_og_nn = [compute_bounds(expr) for expr in d_basic[:Output]]
    violation_bounds = false
    for i in 1:size(bounds_matrix)[1]
        println("Bounds of the approximated nn: ", bounds_matrix[i])
        println("Bounds of the original nn: ", bounds_matrix_og_nn[i])
        if bounds_matrix[i][1] > bounds_matrix_og_nn[i][1] || bounds_matrix[i][2] < bounds_matrix_og_nn[i][2]
            violation_bounds = true
            println("The bounds of the approximated nn do not fully contain the bounds of the original nn")
        end
    end
    if violation_bounds == false
        println("The bounds of the approximated nn fully contain the bounds of the original nn")
    end
    push!(p.bounds, bounds_matrix)

    # - Results - #
    println("Solve Status: ", d_2[:SolveStatus])
    notice(
        MIPVerify.LOGGER,
        "Solve Status Approx: '$(d_2[:SolveStatus])'"
    )
    println("Time it took for second half: ", d_2[:TotalTime], 's')
    notice(
        MIPVerify.LOGGER,
        "Time it took for the second half: '$(d_2[:TotalTime])'s"
    )
    
    notice(
        MIPVerify.LOGGER,
        "Time it took for full proof is: $(d_basic[:TotalTime])s"
    )
    notice(
        MIPVerify.LOGGER,
        "Time it took for approx proof is: $(d_1[:TotalTime] + d_2[:TotalTime])s"
    )
    notice(
        MIPVerify.LOGGER,
        "Did the approx proof work correctly? $(d_2[:SolveStatus] == d_basic[:SolveStatus])"
    )


    # --- Spurious Examples --- #
    # Testing spurious_functions
    # Trying to Convert
    # convert(Vector{<:Real}, value.(d_2[:PerturbedInput]))
    spurious_example = false # The spurious example
    if d_2[:SolveStatus] == MOI.OPTIMAL
        result =  verify_model2(
            model, 
            sample_image, 
            Gurobi.Optimizer,
            Dict("output_flag" => false, "MIPFocus" => 1), 
            value.(d_2[:PerturbedInput]),  
            MIPVerify.LInfNormBoundedPerturbationFamily(eps),
            tightening_algorithm, 
            MIPVerify.get_default_tightening_options(Gurobi.Optimizer)
            )

        # Step 7: Analyze the result
        println("Time to try and find spurious example was: $(result[:TotalTime])s")
        if result[:SolveStatus] == MOI.OPTIMAL
            println("Feasible solution found!")
            # println("Perturbed input: ", result[:PerturbedInput])
            println("The model calssified the pertubed input as: $(argmax(model(value.(result[:PerturbedInput]))))")
            println("The real classification is: $real_class")
        else
            println("No feasible solution found.")
        end
        spurious_example = value.(result[:PerturbedInput])
    end

    # --- Linear Constraints --- #
    bounds = p.bounds[1] # bounds_matrix_og_nn # p.bounds[1] # Bounds of the first layer
    num_vars = length(bounds)
    println("Size of bounds: ", length(bounds))
    println("Number of variables: ", num_vars)
    # Vector of vector to matrix
    bounds = hcat(bounds...)
    # FLoat64 ro Real
    bounds = convert(Matrix{Real}, bounds)

    linear_constraint_mat_bool = falses((num_vars, num_vars))
    for i in 1:num_vars
        for j in 1:num_vars
            # Create a linear constraint of coeffs [1*i, -1*j]* x <= 0 (n[i] <= n[j])
            coeffs = zeros(Real, (num_vars, 1))
            # Matrix to vector
            coeffs = vec(coeffs)
            # Float64 to Real
            coeffs = convert(Vector{<:Real}, coeffs)
            coeffs[i] = 1
            coeffs[j] = -1
            rhs = 0
            linear_constraint_mat_bool[i, j] = calc_linear_constraint(
                bounds,
                coeffs,
                rhs
            )
        end
    end
    plot_binary_matrix(linear_constraint_mat_bool)

    # --- Activation Pattern ---#
    # Getting the activation pattern
    # println("The spurious example is: ", spurious_example)
    if !(spurious_example isa Bool && spurious_example == false)
        activation_pattern = find_activation_pattern_spurious_example(model, spurious_example)
    end
    println("The activation pattern in the last layer is: ", activation_pattern[end])
    """
end

"""
    #  --- Linear Constraints --- #
    model_linear_constraints = p.nns[1]
    index1 = collect(1:length(model_linear_constraints.layers[end-1].bias))
    boolean_linear_constrains_mat = test_linear_constraint(
        model_linear_constraints, 
        sample_image, 
        Gurobi.Optimizer,
        Dict("output_flag" => false, "MIPFocus" => 1), 
        MIPVerify.LInfNormBoundedPerturbationFamily(eps),
        index1, # First index
        tightening_algorithm, 
        MIPVerify.get_default_tightening_options(Gurobi.Optimizer),
    )
    println("The matrix of the linear constraints is: ", boolean_linear_constrains_mat)
    plot_binary_matrix(boolean_linear_constrains_mat) # Plotting the matrix
"""