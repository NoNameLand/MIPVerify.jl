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
using Memento

include("../src/logging.jl")
include("utils/cegis/spurious_functions.jl")

function process_bounds()
    # loading params
    params = JSON.parsefile("ariel_tries/utils/params.json")

    # Including functions
    include("utils/create_sequential_model.jl")
    include("utils/utils_functions.jl")
    include("Partition.jl")
    include("utils/partition_addons.jl")

    # Setting log outputs
    MIPVerify.set_log_level!("debug")

    # Loading MNIST dataset
    mnist = MIPVerify.read_datasets("MNIST")

    # Creating Model
    println("The current dir is: ", pwd())
    path_to_network = params["path_to_nn_adjust"]#"ariel_tries/networks/mnist_model.mat"  # Path to network
    model = create_sequential_model(path_to_network, "model.n1")
    println(model)

    # Finding an image to local verify around
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

    # Constants 
    eps = 0.05
    norm_order = Inf
    tightening_algorithm = lp

    # Finding the adversarial example
    # Without Partition
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

    # With Partition
    p = Partition(model) # The nn as a Sequential model
    EvenPartition(p, 2) # Splitting in half
    println(typeof(p.nns[1]))
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
    push!(p.bounds, bounds_matrix)
    lbs = [pair[1] for pair in p.bounds[1]]
    ubs = [pair[2] for pair in p.bounds[1]]

    println(typeof(CostumeBoundedPerturbationFamily(lbs, ubs)))
    println(typeof(MIPVerify.LInfNormBoundedPerturbationFamily(eps)))
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
    push!(p.bounds, bounds_matrix)

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

    # Testing spurious_functions
    println(vec(sample_image))
    println(size(vec(sample_image)))
    vec_sample_image = reshape(vec(sample_image), length(vec(sample_image)), 1)
    output_bounds = hcat(d_2[:PerturbedInput] .- 0.1, d_2[:PerturbedInput] .+ 0.1)
    input_bounds = hcat(vec_sample_image .- eps, vec_sample_image .+ eps)
    println(size(input_bounds))
    solve_status = verify_network(model, input_bounds, output_bounds, (1, 28, 28, 1))
    println("Spurious Solve Status is $solve_status")
end