using MIPVerify
# include("../../src/models.jl")
# include("../../src/MIPVerify.jl")
struct CostumeBoundedPerturbationFamily <: MIPVerify.RestrictedPerturbationFamily
    lb::Vector{Real}
    ub::Vector{Real}
    

end
Base.show(io::IO, pp::CostumeBoundedPerturbationFamily) =

    print(io, "costume-norm-bounded-($(pp.lb),$(pp.ub)))")


function MIPVerify.get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::CostumeBoundedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        i -> @variable(m, lower_bound = 0, upper_bound = pp.ub[i]), # For numerical stability
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = pp.lb[i],
            upper_bound = pp.ub[i]
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end