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
