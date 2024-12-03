mutable struct Partition
    nn_tot::Sequential
    nns::Vector{Sequential}
    bounds::Vector{Tuple{Float64, Float64}}
    
    function Partition(nn)
        new(nn, nothing, nothing)
    end

    function EvenPartition(p::Partition, num_of_partitions::Int)
        size_each_partition = floor(length(p.nn_tot)/num_of_partitions) # Number of layers in DNN
        nns = Sequential[] # Might Not Work
        for i in 1:(num_of_partitions-1)
            push!(nns, Sequential(p.nn_tot.layers([i:(i+size_each_partition-1)]))
        end
        if size_each_partition*(num_of_partitions) < length(p.nn_tot)
            push!(nns, Sequential(p.nn_tot.layers([length(p.nn_tot) - size_each_partition:length(p.nn_tot)])))
        else
            push!(nns, Sequential(p.nn_not.layers([length(p.nn_tot) - size_each_partition + 1:length(p.nn_tot)])))
    end
end
