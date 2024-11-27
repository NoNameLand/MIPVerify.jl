using MIPVerify
using Flux
using MAT
using OrderedCollections

function extract_number(s::String)
    # Match the first sequence of digits in the string
    match_obj = match(r"\d+", s)
    
    # If a match is found, convert it to an integer
    if match_obj !== nothing
        return parse(Int, match_obj.match)  # Or Float64 if you need a floating-point number
    else
        error("No number found in the string")
    end
end

function create_sequential_model(mat_file::String, model_name::String)
    # Read the neural network from the .mat file
    data = matread(mat_file)
    dict_data = Dict(data)

    layers = []
    order_lst = []

    # Define the pattern to match layer weights
    pattern = r"^layer_\d+/weight$"

    for (key, value) in dict_data
        if occursin(pattern, string(key))
            println("Processing layer: ", string(key))
            name_val = split(string(key), "/")[1]  # Extract 'layer_X'
            size_val = size(value)
            layer_num = extract_number(string(name_val))

            # Determine layer type based on weight dimensions
            if length(size_val) == 2
                # Fully Connected Layer
                expected_size = size_val
                layer = get_matrix_params(
                    dict_data,
                    string(name_val),
                    expected_size
                )
            elseif length(size_val) == 4
                # Convolutional Layer
                # Adjust the weights to match MIPVerify's expected format
                weight_key = "$name_val/weight"
                weights = dict_data[weight_key]

                # Transpose weights from PyTorch format to MIPVerify format
                # PyTorch: (out_channels, in_channels, kernel_height, kernel_width)
                # MIPVerify: (kernel_height, kernel_width, in_channels, out_channels)
                weights = permutedims(weights, (3, 4, 2, 1))

                # Update the weights in dict_data
                dict_data[weight_key] = weights

                # Update expected_size after transposition
                expected_size = size(weights)

                # Retrieve stride and padding if available, else use defaults
                stride_key = "$name_val/stride"
                padding_key = "$name_val/padding"

                expected_stride = get(dict_data, stride_key, 1)
                #println("Expected Stride: ", expected_stride)
                padding = get(dict_data, padding_key, SamePadding())

                layer = get_conv_params(
                    dict_data,
                    string(name_val),
                    expected_size;
                    expected_stride = expected_stride,
                    padding = padding
                )
            else
                error("Unsupported layer type with weight dimensions: $(size_val)")
            end

            push!(layers, layer)
            push!(order_lst, layer_num)
        end
        # Optional: Print key information for debugging
        # println("$key => $(typeof(value)), size: $(size(value))")
    end

    # Order the layers based on their layer numbers
    index_ordered = sortperm(order_lst)
    layers = layers[index_ordered]
    println("Layers: ", layers)

    modified_layers = []

    # Determine if the first layer is convolutional
    if !isa(layers[1], Linear) #&& !isa(layers[2], Conv2d)
        # No need to flatten before convolutional layers
    else
        # Flatten the input if the first layer is fully connected
        push!(modified_layers, Flatten(4))
    end

    for i in 1:length(layers)
        push!(modified_layers, layers[i])

        # Add activation functions after each layer except the last
        if i < length(layers)
            push!(modified_layers, ReLU())
            if isa(layers[i+1], Linear) && !isa(layers[i], Linear)
                push!(modified_layers, Flatten(4)) #TOD: Test if 4 is always true
            end
        end
    end

    model = Sequential(modified_layers, model_name)

    return model
end