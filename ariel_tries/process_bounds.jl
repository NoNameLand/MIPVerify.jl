using MIPVerify
using Gurobi
using JuMP
using MathOptInterface
using JSON
using Memento

# --- Include Functions --- #
include("../src/logging.jl")
include("utils/cegis/spurious_functions.jl")
include("utils/create_sequential_model.jl")
include("utils/utils_functions.jl")
include("Partition.jl")
include("utils/partition_addons.jl")
include("utils/visual_functions.jl")

function process_bounds()
    # --- Constans --- # 
    eps = 0.008
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
    model = create_sequential_model(path_to_network, "model.n1")
    println(model)
    # Print Test Accuracy
    # labelled_data_set_test = 
    println("Size of input image: ", size(MIPVerify.get_image(mnist.test.images, 1)))
    # Reshape input image to (batch_size, height, width, channels)
    # mnist.test.images = reshape(mnist.test.images, (size(mnist.test.images, 1), 28, 28, 1))
    println("The test accuracy is: ", MIPVerify.frac_correct(model, mnist.test, length(mnist.test.labels)))
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
    println("Time it took for first half: ", d_1[:TotalTime])    
    notice(
        MIPVerify.LOGGER,
        "Time it took for the first half: '$(d_1[:TotalTime])'"
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
    println("Time it took for second half: ", d_2[:TotalTime])
    notice(
        MIPVerify.LOGGER,
        "Time it took for the second half: '$(d_2[:TotalTime])'"
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
        println("Time to try and find spurious example was: $(result[:TotalTime])")
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