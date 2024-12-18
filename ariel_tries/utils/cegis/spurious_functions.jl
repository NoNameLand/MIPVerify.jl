using MIPVerify
using JuMP
using Gurobi  # Replace with your solver if needed

function verify_network(model, input_bounds, output_constraints, input_shape)
    # Initialize the JuMP model with the specified solver
    m = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(m, "SolutionLimit", 1)  # Stop after 1 feasible solution
    set_optimizer_attribute(m, "OutputFlag", 0)       # Suppress output

    # Flatten input bounds (multi-dimensional input -> vectorized)
    flattened_bounds = vec(input_bounds)  # Convert to a 1D vector
    num_inputs = length(flattened_bounds)

    # Define variables for the flattened input
    @variable(m, input_bounds[i, 1] <= x[i=1:size(input_bounds, 1)] <= input_bounds[i, 2])

    # Reshape variables back to the input's original shape for network evaluation
    reshaped_input = reshape(x, input_shape)  # Reshape to original dimensions

    # Propagate the reshaped input through the neural network
    output = model(reshaped_input)
    println(type(output))
    println(size(output))
    # Apply the output constraints
    for (i, (lower, upper)) in enumerate(output_constraints)
        println("lower: $lower and upper $upper, i $i")
        @constraint(m, lower <= output[i] <= upper)
    end
    
    # Solve the optimization problem
    optimize!(m)

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