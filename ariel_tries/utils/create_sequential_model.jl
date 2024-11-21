using MIPVerify
using Flux
using MAT
using OrderedCollections

function create_sequential_model(mat_file::String, model_name::String)
    # Read the neural network from the .mat file using MIPVerify
    data = matread(path_to_network) # read the .mat file
    dict_data = Dict(data) # converting matdict to dict #TODO: understand what's the diffrence
    layers = []
    order_lst = []
    # Ordering the dictionary
    function sort_dict_by_key_number(dict::Dict)
        # Extract keys and sort them based on the number in the key
        sorted_keys = sort(collect(keys(dict)), by = key -> begin
            m = match(r"\d+", key)
            m !== nothing ? parse(Int, m.match) : typemax(Int)  # Place non-numeric keys at the end
        end)
        
        # Create an OrderedDict with sorted keys
        sorted_dict = OrderedDict(key => dict[key] for key in sorted_keys)
        
        return sorted_dict
    end

    function extract_number(s::String)
        # Match the first sequence of digits in the string
        match_obj = match(r"\d+", s)
        
        # If a match is found, convert it to an integer
        if match_obj !== nothing
            return parse(Int, match_obj.match)
        else
            return typemax(Int)  # Return a very large number if no number is found
        end
    end
    
    
    # dict_data = sort_dict_by_key_number(dict_data)

    # Defiing the regular expression
    pattern = r"^layer_\d+/weight$"

    for (key, value) in dict_data
        if occursin(pattern, string(key))
            println("Added the layer to the list: ", string(key))
            name_val = split(string(key), "/")[1]
            size_val = size(value)
            layer = get_matrix_params(dict_data, string(name_val), size_val)
            push!(layers, layer)
            push!(order_lst, extract_number(string(name_val)))

        end
        println("$key => $(typeof(value)), size: $(size(value))")
    end

    # Ordering the layers
    index_ordered = sortperm(order_lst)
    layers = layers[index_ordered]    

    modified_layers = []
    push!(modified_layers, Flatten(4))

    for i in 1:length(layers)
        push!(modified_layers, layers[i])
        if i < length(layers)
            push!(modified_layers, ReLU())
        end
    end

    model = Sequential(modified_layers, model_name)

    return model
end
