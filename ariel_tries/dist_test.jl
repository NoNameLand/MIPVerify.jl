using MIPVerify
using StatsBase # For entropy calculation
using KernelDensity # For kde 
using Plots
using JSON
using GraphPlot
using Colors

"""
    sample_image_random(image::AbstractArray{<:Real, 2}, bounds::Tuple{<:Real, <:Real}, num_samples::Int)
    
    Generates random samples around the original image within the specified bounds.
    
    # Arguments
    - `image::AbstractArray{<:Real, 2}`: The original image.
    - `bounds::Tuple{<:Real, <:Real}`: A tuple specifying the lower and upper bounds for the noise.
    - `num_samples::Int`: The number of samples to generate.
    
    # Returns
    - A 3D array of size (size(image, 1), size(image, 2), num_samples) containing the sampled images.
"""
function sample_image_random(
    image::AbstractArray{<:Real, 4},
    bounds::Tuple{<:Real, <:Real},
    num_samples::Int
)
    lower_bound, upper_bound = bounds
    image_size = size(image)
    sampled_images = zeros(image_size..., num_samples)
    for i in 1:num_samples
        noise = rand(image_size) .* (upper_bound - lower_bound) .+ lower_bound
        sampled_images[:, :, :, :, i] = image .+ noise
    end
    return sampled_images
end

"""
    feed_net(net::MIPVerify.Sequential, image::Array{<:Real, 2})
    
    Feeds an image through the neural network and returns the neuron values of each layer.
    
    # Arguments
    - `net::MIPVerify.Sequential`: The neural network.
    - `image::Array{<:Real, 2}`: The input image.
    
    # Returns
    - A vector of neuron values for each layer in the network.
"""
function feed_net(net::MIPVerify.Sequential, image::Array{<:Real, 4})
    #input_shape = net.layers[1].input_shape
    #x = reshape(image, input_shape)
    x = image
    xs = [x]
    neruons = []
    for layer in net.layers
        if typeof(layer) == MIPVerify.Flatten
            x = reshape(x, layer.output_shape)
        else
            x = layer(x)
        end
        output = x
        push!(neruons, output)
    end
    return neruons[2:end] # ignore flattened input
end

"""
    empirical_entropy(samples::Vector{<:Real})
    
    Calculates the empirical entropy of a vector of samples using kernel density estimation (KDE).
    
    # Arguments
    - `samples::Vector{<:Real}`: A vector of samples.
    
    # Returns
    - The empirical entropy of the samples.
"""
function empirical_entropy(samples::Vector{<:Real})
    kde = KernelDensity.kde(samples)
    # Filter out zero or negative density values
    valid_density = kde.density[kde.density .> 0]
    valid_x = kde.x[kde.density .> 0]
    
    # Normalize the density values to form a valid probability distribution
    valid_density /= sum(valid_density)
    
    # Calculate the entropy
    entropy = -sum(valid_density .* log.(valid_density)) * (valid_x[2] - valid_x[1])
    return entropy
end

"""
    plot_kde_density(kde)
    
    Plots the kernel density estimation (KDE) density.
    
    # Arguments
    - `kde`: The kernel density estimation object.
"""
function plot_kde_density(kde)
    plot(kde.density, kde.x, xlabel="Neuron Value", ylabel="Density", title="KDE Density")
    savefig("../results/dist_test/kde_density.png")
end

"""
    dist_test(net::MIPVerify.Sequential, image::Array{<:Real, 2}, bounds::Tuple{<:Real, <:Real}, num_samples::Int)
    
    Generates random samples around an image, feeds them through the network, and calculates the empirical entropy of the neuron values.
    
    # Arguments
    - `net::MIPVerify.Sequential`: The neural network.
    - `image::Array{<:Real, 2}`: The original image.
    - `bounds::Tuple{<:Real, <:Real}`: A tuple specifying the lower and upper bounds for the noise.
    - `num_samples::Int`: The number of samples to generate.
    
    # Returns
    - A vector of entropies for the neuron values of each sample.
"""
function dist_test(
    net::MIPVerify.Sequential,
    image::Array{<:Real, 4},
    bounds::Tuple{<:Real, <:Real},
    num_samples::Int
)
    sampled_images = sample_image_random(image, bounds, num_samples)
    neuron_values = [feed_net(net, sampled_images[:, :, :, :,  i]) for i in 1:num_samples]
    
    # Calculate the empirical entropy of the neuron values for each layer
    num_layers = length(net.layers) - 1  # Exclude the input layer
    entropies = []
    for layer_idx in 2:num_layers
        entropies_layer = []
        if typeof(net.layers[layer_idx]) == MIPVerify.Flatten || typeof(net.layers[layer_idx]) == MIPVerify.ReLU
            layer_idx += 1
        else
            for neruon_idx in 1:length(net.layers[layer_idx].bias)
                # println("Layer $layer_idx neuron $neruon_idx")
                layer_activations = [neuron_values[sample_idx][layer_idx][neruon_idx] for sample_idx in 1:num_samples]
                layer_activations_flat = vcat(layer_activations...)
                push!(entropies_layer, empirical_entropy(layer_activations_flat))
            end
            push!(entropies, entropies_layer)
        end
    end
    
    return entropies
end

"""
    plot_entropies(entropies::Vector{<:Any})
    
    Plots a histogram of the entropies.
    
    # Arguments
    - `entropies::Vector{<:Any}`: A vector of entropies.
"""
function plot_entropies(entropies::Vector{<:Any})
    entropies_flat = vcat(entropies...)
    histogram(entropies_flat, bins=100, xlabel="Entropy", ylabel="Frequency", title="Distribution of Neuron Entropies")
    savefig("../results/dist_test/neuron_entropies.png")
end

"""
    calculate_average_entropy_per_layer(entropies::Vector{<:Real})
    
    Calculates the average entropy per layer.
    
    # Arguments
    - `entropies::Vector{<:Real}`: A vector of entropies.
    
    # Returns
    - A vector of average entropies per layer.
"""
function calculate_average_entropy_per_layer(entropies::Vector{<:Any})
    # println("Number of layers: ", length(entropies))
    true_num_layers = length(entropies)
    avg_entropies = zeros(true_num_layers)
    for i in 1:true_num_layers
        # println(length(entropies[:][i]))
        entropies_current = entropies[:][i]
        avg_entropies[i] = mean(entropies_current)
    end
    return avg_entropies
end

"""
    plot_average_entropies(avg_entropies::Vector{<:Real})
    
    Plots the average entropies per layer.
    
    # Arguments
    - `avg_entropies::Vector{<:Real}`: A vector of average entropies per layer.
"""
function plot_average_entropies(avg_entropies::Vector{<:Real})
    plot(avg_entropies, xlabel="Layer", ylabel="Average Entropy", title="Average Entropy per Layer")
    savefig("../results/dist_test/average_entropy_per_layer.png")
end

"""
    plot_nn_entropy(net::MIPVerify.Sequential, entropies::Vector{<:Any})
    
    Plots a representation of the neural network as a graph where each node is a neuron and its color represents the entropy.
    
    # Arguments
    - `net::MIPVerify.Sequential`: The neural network.
    - `entropies::Vector{<:Any}`: A vector of entropies for each neuron in the network.
"""
function plot_nn_entropy(net::MIPVerify.Sequential, entropies::Vector{<:Any})
    num_layers = length(net.layers)
    g = SimpleGraph()
    node_labels = []
    node_colors = []

    # Add nodes and edges for each layer
    node_idx = 1
    for layer_idx in 1:num_layers
        if typeof(net.layers[layer_idx]) == MIPVerify.Flatten || typeof(net.layers[layer_idx]) == MIPVerify.ReLU
            continue
        else
            for neuron_idx in 1:length(net.layers[layer_idx].bias)
                add_vertex!(g)
                push!(node_labels, "L$layer_idx-N$neuron_idx")
                entropy = entropies[layer_idx][neuron_idx]
                color = get_color(entropy)
                push!(node_colors, color)
                node_idx += 1
            end
        end
    end

    # Add edges between layers
    for layer_idx in 1:(num_layers - 1)
        if typeof(net.layers[layer_idx]) == MIPVerify.Flatten || typeof(net.layers[layer_idx]) == MIPVerify.ReLU
            continue
        else
            for neuron_idx in 1:length(net.layers[layer_idx].bias)
                for next_neuron_idx in 1:length(net.layers[layer_idx + 1].bias)
                    add_edge!(g, neuron_idx, next_neuron_idx)
                end
            end
        end
    end

    # Plot the graph
    gplot(g, node_labels=node_labels, nodefillc=node_colors)
end

"""
    get_color(entropy::Real)
    
    Returns a color based on the entropy value.
    
    # Arguments
    - `entropy::Real`: The entropy value.
    
    # Returns
    - A color from green (low entropy) to red (high entropy).
"""
function get_color(entropy::Real)
    # Normalize entropy to [0, 1] range
    normalized_entropy = (entropy - minimum(entropy)) / (maximum(entropy) - minimum(entropy))
    return RGB(1 - normalized_entropy, normalized_entropy, 0)
end



"""
    main()
    
    Main function to load the neural network and image, perform the distribution test, and plot the results.
"""
function main()
    params = JSON.parsefile("ariel_tries/utils/params.json")

    # Loading MNIST dataset
    mnist = MIPVerify.read_datasets("MNIST") # MIPVerify.read_datasets("MNIST")

    # Creating Model
    println("The current dir is: ", pwd())
    path_to_network = params["path_to_nn_adjust"]#"ariel_tries/networks/mnist_model.mat"  # Path to network
    model = create_sequential_model(path_to_network, "model.n1")
    println(model)

    image_num = 1
    image =  MIPVerify.get_image(mnist.test.images, image_num)
    #image = reshape(image, 28, 28) # To work with the sample image
    bounds = (-0.1, 0.1)
    num_samples = 1000
    entropies = dist_test(model, image, bounds, num_samples)
    # println(entropies)
    plot_entropies(entropies)
    avg_entropies = calculate_average_entropy_per_layer(entropies)
    plot_average_entropies(avg_entropies)

    # Plot the neural network with entropy colors (should be cool)
    plot_nn_entropy(model, entropies)
end

main()