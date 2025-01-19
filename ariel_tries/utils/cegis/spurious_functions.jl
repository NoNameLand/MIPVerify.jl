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
            desired_output_lower = output_desired .- 1e-10 # Hardcoded tolerance
            desired_output_upper = output_desired .+ 1e-10 # Hardcoded tolerance
            
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


# --- Linear Constraint Testing --- #

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
        n_vars_actual = length(vars_model)
        println("Number of variables: $n_vars_actual")
        n_vars_calc = calc_num_expected_vars(nn, size(input))
        println("Number of variables calculated: $n_vars_calc")
        # Add the negation of the linear constraint index1 <= index2
        @constraint(m, d[:Output][index1_full] >= d[:Output][index2_full])

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

function test_linear_constraint(
    nn::NeuralNet,
    input::Array{<:Real, 4},
    optimizer,
    main_solve_options::Dict,
    pp::MIPVerify.PerturbationFamily, 
    index1::Array{Int},
    index2::Array{Int},
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),

    )::BitMatrix
    
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
        n_vars_actual = length(vars_model)
        println("Number of variables: $n_vars_actual")
        n_vars_calc = calc_num_expected_vars(nn, size(input))
        println("Number of variables calculated: $n_vars_calc")
        # Add the negation of the linear constraint index1 <= index2
        results_mat = falses(length(index1), length(index2))
        for (i, num1) in enumerate(index1)
            for (j, num2) in enumerate(index2)
                c = @constraint(m, d[:Output][num1] >= d[:Output][num2])
                # Optimize the model
                set_optimizer(m, optimizer)
                set_optimizer_attributes(m, main_solve_options...)
                optimize!(m)
                solve_status = JuMP.termination_status(m)
                result = solve_status == MOI.OPTIMAL
                # println("Result: $result")
                results_mat[i, j] = result
                delete(m, c)
            end
        end
        return results_mat
end

function test_linear_constraint(
    nn::NeuralNet,
    input::Array{<:Real, 4},
    optimizer,
    main_solve_options::Dict,
    pp::MIPVerify.PerturbationFamily, 
    index1::Array{Int},
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_options::Dict = MIPVerify.get_default_tightening_options(optimizer),

    )::BitMatrix
    
        index1_full = index1# get_constraints_index(nn, length(nn.layers), index1)
        index2_full = index1 # get_constraints_index(nn, length(nn.layers), index2)
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
        n_vars_actual = length(vars_model)
        println("Number of variables: $n_vars_actual")
        n_vars_calc = calc_num_expected_vars(nn, size(input))
        println("Number of variables calculated: $n_vars_calc")
        # Add the negation of the linear constraint index1 <= index2
        results_mat = falses(length(index1), length(index2))
        for (i, num1) in enumerate(index1)
            for (j, num2) in enumerate(index2)
                if num1 == num2
                    results_mat[i, j] = true
                    continue
                end
                if num1 > num2
                    continue
                end
                c = @constraint(m, d[:Output][num1] >= d[:Output][num2])
                # Optimize the model
                set_optimizer(m, optimizer)
                set_optimizer_attributes(m, main_solve_options...)
                optimize!(m)
                solve_status = JuMP.termination_status(m)
                result = solve_status == MOI.OPTIMAL
                # println("Result: $result")
                results_mat[i, j] = result
                delete(m, c)
            end
        end
        # Make it symmetric
        results_mat = results_mat .| transpose(results_mat)
        return results_mat
end


function calc_num_expected_vars(
    nn::NeuralNet,
    input_shape::Tuple,
)
    # Calculate the number of variables expected in the model
    num_vars = 0
    for (i, layer) in enumerate(nn.layers)
        if layer isa ReLU && nn.layers[i-1] isa Linear
            num_vars += size(nn.layers[i-1].bias)[1] # Add the number of bias variables, ReLU vars.
        end
    end
    num_vars += 2*(input_shape[1] * input_shape[2] * input_shape[3]) # Input Vars and Perturbation Vars
    return num_vars
end


"""
    calc_linear_constraint(bounds::Array{Real, 2}, coeffs::Array{Real, 1}, rhs::Real)
    A function to calculate if the linear constraint imposed by the coeffs is satisfied by the bounds.
    The linear constraint is of the form: coeffs * x <= rhs, where x is the input vector which is bounded by the bounds.
    The function returns a boolean value indicating if the constraint is satisfied or not.
"""
function calc_linear_constraint(
    bounds::Array{Real, 2},
    coeffs::Array{Real, 1}, 
    rhs::Real
)::Bool
    # Calculate the minimum and maximum values of the linear constraint
    # Calculating the minimum vqlue of the expression
    
    min_val = 0
    for i in 1:length(coeffs)
        min_val += if coeffs[i] > 0 minimum(bounds[:, i]) * coeffs[i] else maximum(bounds[:, i]) * coeffs[i] end
    end
    max_val = 0
    for i in 1:length(coeffs)
        max_val += if coeffs[i] > 0 maximum(bounds[:, i]) * coeffs[i] else minimum(bounds[:, i]) * coeffs[i] end
    end
    """
    for i in 1:length(coeffs)
        if coeffs[i] < 0
            bounds[:, i] = reverse(bounds[:, i])
        end
    end
    min_val = sum(minimum(bounds, dims=1) .* coeffs)
    max_val = sum(maximum(bounds, dims=1) .* coeffs)
    """
    # Check if the constraint is satisfied
    return max_val <= rhs 
end

# --- Finding Activation Pattern --- # 
"""
    find_activation_pattern_spurious_example
    A function to find the activation pattern of a spurious example.
    Arguments:
        - nn: The neural network model
        - input: The input to the neural network (the spurious example)
    Returns:
        - A dictionary containing the activation pattern of the spurious example
"""
function find_activation_pattern_spurious_example(
    nn::NeuralNet,
    input::Array{<:Real, 4},
)::Any
    # Initialize the dictionary to store the activation pattern
    activation_pattern = []
    # Loop through each layer of the neural network
    output_last_layer = input
    for (i, layer) in enumerate(nn.layers)
        # Check if the layer is a ReLU layer
        output_layer = layer(output_last_layer)
        if layer isa ReLU
            # Calculate the activation pattern of the ReLU layer
            push!(activation_pattern, output_layer .> 0)
        end
        output_last_layer = output_layer
    end
    return activation_pattern
end

""" Create a constraint for the activation pattern of the spurious example
    Arguments:
        - activation_pattern: The activation pattern of the spurious example
        - layer_num: The layer number to create the constraint for
        - neuron_num: The neuron number to create the constraint for
    Returns:
        - A constraint for the activation pattern of the spurious example
"""
function create_constraint_activation_pattern(
    activation_pattern::Array{Bool},
    model::JuMP.Model,
    output::Any, # Check the type 
)
    for i in 1:length(activation_pattern)
        @constraint(model, output[i] == 0)
    end
end
