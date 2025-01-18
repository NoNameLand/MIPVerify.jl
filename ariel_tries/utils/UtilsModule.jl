"""
    use_local_MIPVerify()

- Julia version: 
- Author: root
- Date: 2024-11-19

# Arguments

# Examples

```jldoctest
julia>
```
"""

module UtilsModule
    export compute_bounds

    """
    Compute the lower and upper bounds of an affine expression.

    # Arguments
    - `aff_expr::AffExpr`: The affine expression for which to compute the bounds.

        terms_dict = Dict(aff_expr.terms)
        for (var, coeff) in terms_dict
    - A vector containing the lower bound and upper bound of the affine expression.

    # Examples
    ```julia
    aff_expr = AffExpr(constant=1.0, terms=[(x, 2.0), (y, -3.0)])
    bounds = compute_bounds(aff_expr)
    println(bounds)  # Output: [lower_bound, upper_bound]
    ```
    """
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

    

end
