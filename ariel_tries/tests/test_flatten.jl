using Test
using MIPVerify
# include("../../src/net_components/layers/flatten.jl")

function test_flatten()
    # Test 1: Check if the Flatten constructor works correctly
    perm = [1, 2, 3]
    flatten_layer = Flatten(perm)
    @test flatten_layer.n_dim == 3
    @test flatten_layer.perm == perm

    # Test 2: Check if the permutation is valid
    perm_invalid = [1, 3, 3]
    @test_throws DomainError Flatten(perm_invalid)

    # Test 3: Check if the permute_and_flatten function works correctly
    x = reshape(1:24, 2, 3, 4)
    perm = [3, 2, 1]
    flattened_x = permute_and_flatten(x, perm)
    expected_flattened_x = permutedims(x, perm)[:]
    @test flattened_x == expected_flattened_x

    # Test 4: Check if the Flatten layer works correctly with Real arrays
    x_real = reshape(1.0:24.0, 2, 3, 4)
    flatten_layer = Flatten([3, 2, 1])
    flattened_x_real = flatten_layer(x_real)
    expected_flattened_x_real = permutedims(x_real, [3, 2, 1])[:]
    @test flattened_x_real == expected_flattened_x_real

    # Test 5: Check if the Flatten layer works correctly with JuMPLinearType arrays
    using JuMP
    model = Model()
    x_jump = @variable(model, [1:2, 1:3, 1:4])
    flatten_layer = Flatten([3, 2, 1])
    flattened_x_jump = flatten_layer(x_jump)
    expected_flattened_x_jump = permutedims(x_jump, [3, 2, 1])[:]
    @test flattened_x_jump == expected_flattened_x_jump
end

# Run the tests
test_flatten()