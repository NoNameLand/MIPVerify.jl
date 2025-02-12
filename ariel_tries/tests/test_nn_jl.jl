# using MIPVerify
using MAT
using Flux
using ProgressMeter

include("../utils/create_sequential_model.jl")

function load_dataset(mat_file::String)
    data = matread(mat_file)
    test_set = data["test_set"]
    test_labels = data["test_labels"]

    # Reshape and convert data based on the first layer
    if size(test_set, 4) == 1
        test_set = permutedims(test_set, (4, 3, 2, 1))
    end

    return test_set, test_labels
end

function test_model_accuracy(model_path::String, dataset_path::String)
    # Create the model from the .mat file
    model = create_sequential_model(model_path, "test_model")

    # Load the dataset
    test_set, test_labels = load_dataset(dataset_path)

    # Evaluate the model
    num_samples = size(test_set, 4)
    num_correct = 0.0
    p = Progress(num_samples, desc = "Computing accuracy...", enabled = isinteractive())
    for i in 1:num_samples
        input = test_set[:, :, :, i]
        actual_label = test_labels[i]
        predicted_label = (input |> model |> get_max_index)
        if actual_label == predicted_label
            num_correct += 1
        end
        next!(p)
    end

    accuracy = num_correct / num_samples
    println("Model accuracy: ", accuracy)
    return accuracy
end

