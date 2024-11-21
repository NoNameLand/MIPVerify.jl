using MIPVerify
using Flux

function create_sequential_model(mat_file::String)
    # Read the neural network from the .mat file using MIPVerify
    net = read_network(mat_file)

    layers = []

    for l in 1:net.num_layers
        # Extract weights and biases for the current layer
        W = net.W[l]
        b = net.b[l]
        in_dim = size(W, 2)
        out_dim = size(W, 1)

        # Create a Dense layer with the extracted weights and biases
        fc_layer = Dense(in_dim, out_dim)
        fc_layer.W .= W
        fc_layer.b .= b

        push!(layers, fc_layer)

        # Add activation function if it's not the last layer
        if l < net.num_layers
            activation = net.activations[l]
            if activation == relu
                push!(layers, Flux.relu)
            else
                error("Unsupported activation function")
            end
        end
    end

    # Create a sequential model using Chain
    model = Chain(layers...)
    return model
end
