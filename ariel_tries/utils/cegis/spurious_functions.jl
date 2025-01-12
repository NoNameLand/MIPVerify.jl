using MIPVerify
using JuMP
using Gurobi  # Replace with your solver if needed


function verify_network(model, input_bounds, output_constraints, input_shape)
    # Initialize the JuMP model with the specified solver
    m = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(m, "SolutionLimit", 1)  # Stop after 1 feasible solution
    set_optimizer_attribute(m, "OutputFlag", 0)       # Suppress output
    set_optimizer_attribute(m, "MIPFocus", 1) # Focus on feasbility
    # set_optimizer_attribute(m, "Time", 60) # time limit 60s
    set_time_limit_sec(m, 60)


    # Flatten input bounds (multi-dimensional input -> vectorized)
    flattened_bounds = vec(input_bounds)  # Convert to a 1D vector
    num_inputs = length(flattened_bounds)

    # Define variables for the flattened input
    # @variable(m, input_bounds[i, 1] <= x[i=1:size(input_bounds, 1)] <= input_bounds[i, 2])
    add_bounds!(model, input_bounds)
    println("Added variables")
    # Reshape variables back to the input's original shape for network evaluation
    reshaped_input = reshape(x, input_shape)  # Reshape to original dimensions

    # Propagate the reshaped input through the neural network
    output = model(reshaped_input)
    println("Calculated Output")
    # println(type(output))
    println(size(output))
    # Apply the output constraints
    for (i, (lower, upper)) in enumerate(output_constraints)
        # println("lower: $lower and upper $upper, i $i")
        @constraint(m, lower <= output[i] <= upper)
    end
    println("Added Constraints")
    
    # Solve the optimization problem
    optimize!(m)
    println("Optimized")

    # Check the result
    if termination_status(m) == MOI.OPTIMAL
        # Extract the input values that satisfy the constraints
        input_solution = value.(x)  # Flattened solution
        reshaped_solution = reshape(input_solution, input_shape)
        return true, reshaped_solution
    else
        return false, nothing
    end
end
    


function verify_model2(
        nn::NeuralNet,
        input::Array{<:Real, 4}, # Input to the neural network
        optimizer,
        main_solve_options::Dict, # Options for the main solve
        output_desired::Vector{<:Real}, # Desired output 
        pp::MIPVerify.PerturbationFamily, 
        tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
        tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),
    )::Dict
    
        time_verify = @elapsed begin
            d = Dict() # Empty dictionary to store results

            # Calculate predicted index
            predicted_output = input |> nn
            notice(
                MIPVerify.LOGGER,
                "Attempting to find an input that results in the output specified",
            )
            merge!(d, MIPVerify.get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm))
            m = d[:Model]
            
            # Defining output lower and upper constraints
            desired_output_lower = output_desired .- 1e-3 # Hardcoded tolerance
            desired_output_upper = output_desired .+ 1e-3 # Hardcoded tolerance
            
            # Add constraints for the desired output area
            @constraint(m, output_desired .<= desired_output_upper)
            @constraint(m, output_desired .>= desired_output_lower)

            # Change the objective to minimize the deviation from the desired output area
            @variable(m, deviation[1:length(output_desired)] >= 0)
            @constraint(m, deviation .>= output_desired .- desired_output_upper)
            @constraint(m, deviation .>= desired_output_lower .- output_desired)
            @objective(m, Min, sum(deviation))

            # Optimize the model
            set_optimizer(m, optimizer)
            set_optimizer_attributes(m, main_solve_options...)
            optimize!(m)
        

            d[:SolveStatus] = JuMP.termination_status(m)
            d[:SolveTime] = JuMP.solve_time(m)
        end
        
        d[:TotalTime] = time_verify
        return d
end


function get_constraints_index(
    nn::NeuralNet,
    layer_num::Int,
    neuron_num::Int,
)
    # Calculate the number of neurons in each layer, until the specified layer
    num_neurons = [size(layer.weights, 1) for layer in nn.layers if typeof(layer) == Linear]  # Assuming layer.W contains the weights
    # Calculate the index of the specified neuron in the specified layer
    println(num_neurons)
    index = sum(num_neurons) + neuron_num
    return index
end

function get_variable_from_index(
    model::Model,
    index::Int,
)
    return model[:x][index]
end

function test_linear_constraint(
    nn::NeuralNet,
    input::Array{<:Real, 4},
    optimizer,
    main_solve_options::Dict,
    pp::MIPVerify.PerturbationFamily, 
    index1::Int,
    index2::Int,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),

    )::Dict
    
    
    time_verify = @elapsed begin
        index1_full = index1# get_constraints_index(nn, length(nn.layers), index1)
        index2_full = index2 # get_constraints_index(nn, length(nn.layers), index2)
        d = Dict() # Empty dictionary to store results

        # Calculate predicted index
        predicted_output = input |> nn
        notice(
            MIPVerify.LOGGER,
            "Attempting to test if a linear constraint holds",
        )
        merge!(d, MIPVerify.get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm))
        m = d[:Model]        
        vars_model = all_variables(m)
        # Add the negation of the linear constraint index1 <= index2
        @constraint(m, vars_model[index1_full] >= vars_model[index2_full])

        # No need to define an objective function, just check for feasibility

        # Optimize the model
        set_optimizer(m, optimizer)
        set_optimizer_attributes(m, main_solve_options...)
        optimize!(m)
    

        d[:SolveStatus] = JuMP.termination_status(m)
        d[:SolveTime] = JuMP.solve_time(m)
    end
    d[:TotalTime] = time_verify
    return d
end