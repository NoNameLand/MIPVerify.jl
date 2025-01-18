# --- Using Stuff --- #
using Plots
using JSON
# --- Include Functions --- #


# --- Constants and Params --- #
params = JSON.parsefile("params.json")

# --- Plotting Function Definitions --- #
function plot_binary_matrix(matrix::Array{Int, 2})
    heatmap(matrix, c=:greys, aspect_ratio=1, axis=false, color=:grays, legend=false)
    save_plot(plot, "../results/binary_matrix.png")
end

# --- Save the plot to a file --- #
function save_plot(plot, filename)
    savefig(plot, filename)
end


