using JuMP


## JuMP Utils

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

function exclude_number(n::Int)
    # Create an array of numbers from 1 to 10
    numbers = 1:10
    # Filter out the given number
    filtered_numbers = filter(x -> x != n, numbers)
    # Return the resulting array
    return collect(filtered_numbers)
end

