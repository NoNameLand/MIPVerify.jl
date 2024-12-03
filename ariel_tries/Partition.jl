mutable struct Partition
    nn_tot::Sequential
    nns::Vector{Sequential}
    bounds::Vector{Vector{Vector{Float64}}}
end

function Partition(nn::Sequential)
    Partition(nn, Vector{Sequential}(), Vector{Vector{Vector{Float64}}}())
end

using UUIDs

function Sequential_Fake(layers::Array{Layer,1})
    Sequential(layers, string(uuid4()))
end

function EvenPartition(p::Partition, num_of_partitions::Int)
    num_layers = length(p.nn_tot.layers)
    size_each_partition = div(num_layers, num_of_partitions)
    nns = Sequential[]
    for i in 1:(num_of_partitions-1)
        start_idx = (i-1)*size_each_partition + 1
        end_idx = i*size_each_partition
        layers = p.nn_tot.layers[start_idx:end_idx]
        push!(nns, Sequential_Fake(layers))
    end
    # Handle the last partition
    start_idx = (num_of_partitions-1)*size_each_partition + 1
    end_idx = num_layers
    layers = p.nn_tot.layers[start_idx:end_idx]
    push!(nns, Sequential_Fake(layers))
    # Update the Partition object
    p.nns = nns
end


function AddBounds()

end
