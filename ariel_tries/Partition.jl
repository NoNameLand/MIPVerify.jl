using MIPVerify

mutable struct Partition
    nn_tot::MIPVerify.Sequential
    nns::Vector{MIPVerify.Sequential}
    bounds::Vector{Vector{Vector{Float64}}}
end

function Partition(nn::MIPVerify.Sequential)
    Partition(nn, Vector{MIPVerify.Sequential}(), Vector{Vector{Vector{Float64}}}())
end

using UUIDs

function Sequential_Fake(layers::Array{Layer,1})
    MIPVerify.Sequential(layers, string(uuid4()))
end

function EvenPartition(p::Partition, num_of_partitions::Int)
    num_layers = length(p.nn_tot.layers)
    size_each_partition = div(num_layers, num_of_partitions)
    nns = MIPVerify.Sequential[]
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

function EvenPartitionFixed(p::Partition, num_of_partitions::Int)
    # Extract layers and filter only Linear and Conv2d layers
    layers = p.nn_tot.layers
    relevant_layers = [layer for layer in layers if layer isa Linear || layer isa Conv2d]

    # Calculate the number of layers per partition
    num_relevant_layers = length(relevant_layers)
    size_each_partition = div(num_relevant_layers, num_of_partitions)
    remainder = mod(num_relevant_layers, num_of_partitions)

    # Initialize partitions
    partitions = MIPVerify.Sequential[]

    # Partition relevant layers
    start_idx = 1
    for i in 1:num_of_partitions
        end_idx = start_idx + size_each_partition - 1
        # Distribute any remainder layers
        if remainder > 0
            end_idx += 1
            remainder -= 1
        end
        push!(partitions, Sequential_Fake(relevant_layers[start_idx:end_idx]))
        start_idx = end_idx + 1
    end

    # Add non-relevant layers to respective partitions
    final_partitions = MIPVerify.Sequential[]
    current_idx = 1
    for partition in partitions
        partition_layers = []
        for layer in layers[current_idx:end]
            # Add layers to the partition until reaching the relevant layer boundary
            if layer in partition.layers
                push!(partition_layers, layer)
                current_idx += 1
            elseif !(layer isa Linear || layer isa Conv2d)
                push!(partition_layers, layer)
            end
        end
        push!(final_partitions, Sequential_Fake(partition_layers))
    end

    # Update the Partition object
    p.nns = final_partitions
end




function EvenPartitionByNeurons(p::Partition, num_of_partitions::Int)
    # Extract layers and neuron counts
    layers = p.nn_tot.layers
    println(layers[1])
    neuron_counts = [length(layer) for layer in layers]  # Assuming `num_neurons` exists for each layer
    
    total_neurons = sum(neuron_counts)
    target_neurons = div(total_neurons, num_of_partitions)  # Approximate neurons per partition

    # Initialize partitions
    partitions = []
    current_partition = []
    current_count = 0

    for (layer, neurons) in zip(layers, neuron_counts)
        if current_count + neurons <= target_neurons || length(partitions) == num_of_partitions - 1
            # Add layer to the current partition
            push!(current_partition, layer)
            current_count += neurons
        else
            # Finalize the current partition and start a new one
            push!(partitions, Sequential_Fake(current_partition))
            current_partition = [layer]
            current_count = neurons
        end
    end

    # Handle the last partition
    push!(partitions, Sequential_Fake(current_partition))

    # Update the Partition object
    p.nns = partitions
end


function AddBounds()

end
