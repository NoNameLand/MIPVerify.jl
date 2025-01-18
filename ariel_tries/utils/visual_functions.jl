# --- Using Stuff --- #
using Plots
using JSON
# --- Include Functions --- #


# --- Constants and Params --- #
params = JSON.parsefile("ariel_tries/utils/params.json")

# --- Plotting Function Definitions --- #
function plot_binary_matrix(matrix::BitMatrix)
    heatmap(matrix, c=:greys, aspect_ratio=1, axis=false, color=:grays, legend=false, colorbar = true)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            annotate!(j, i, text("$(i),$(j)", :white, :center))
        end
    end
    savefig("../results/binary_matrix.png")
end

# --- Save the plot to a file --- #
function save_plot(plot, filename)
    savefig(plot, filename)
end


