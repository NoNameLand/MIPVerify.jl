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
        println("lower: $lower and upper $upper, i $i")
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
        input::Array{<:Real}, # Input to the neural network
        optimizer,
        main_solve_options::Dict, # Options for the main solve
        output_desired::Array{<:Real}; # Desired output 
        pp::MIPVerify.PerturbationFamily = UnrestrictedPerturbationFamily(),
        tightening_algorithm::TighteningAlgorithm = DEFAULT_TIGHTENING_ALGORITHM,
        tightening_options::Dict = get_default_tightening_options(optimizer),
    )::Dict
    
        total_time = @elapsed begin
        d = Dict() # Empty dictionary to store results

        # Calculate predicted index
        predicted_output = input |> nn
        notice(
            MIPVerify.LOGGER,
            "Attempting to find an input that results in the output specified",
        )
        merge!(d, get_model(nn, input, pp, optimizer, tightening_options, tightening_algorithm))
        m = d[:Model]
        
        # Defining output lower and upper constraints
        desired_output_lower = output_desired .- 1e-3 # Hardcoded tolerance
        desired_output_upper = output_desired .+ 1e-3 # Hardcoded tolerance
        
        # Add constraints for the desired output area
        @constraint(m, desired_output_lower .<= output .<= desired_output_upper)

        # Change the objective to minimize the deviation from the desired output area
        @variable(m, deviation[1:length(output)] >= 0)
        @constraint(m, deviation .>= output .- desired_output_upper)
        @constraint(m, deviation .>= desired_output_lower .- output)
        @objective(m, Min, sum(deviation))

        # Optimize the model
        set_optimizer(m, optimizer)
        set_optimizer_attributes(m, main_solve_options...)
        optimize!(m)

        d[:SolveStatus] = JuMP.termination_status(m)
        d[:SolveTime] = JuMP.solve_time(m)
        d[:TotalTime] = total_time
        return d
    end

    function get_model(
        nn::NeuralNet,
        input::Array{<:Real},
        pp::MIPVerify.PerturbationFamily,
        optimizer,
        tightening_options::Dict,
        tightening_algorithm::TighteningAlgorithm,
    )::Dict{Symbol,Any}
        notice(
            MIPVerify.LOGGER,
            "Determining upper and lower bounds for the input to each non-linear unit.",
        )
        m = Model(optimizer_with_attributes(optimizer, tightening_options...))
        m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)
    
        d_common = Dict(
            :Model => m,
            :PerturbationFamily => pp,
            :TighteningApproach => string(tightening_algorithm),
        )
    
        return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m))
    end
end