# --- Using Stuff --- #
using Plots
using JSON
# --- Include Functions --- #


# --- Constants and Params --- #
params = JSON.parsefile("ariel_tries/utils/params.json")

# --- Plotting Function Definitions --- #
function plot_binary_matrix(matrix::BitMatrix)
    heatmap(matrix, c=[:red, :green], aspect_ratio=1, axis=:on, color=:grays, legend=false, colorbar=true)
    savefig("../results/binary_matrix.png")
end

# --- Save the plot to a file --- #
function save_plot(plot, filename)
    savefig(plot, filename)
end


