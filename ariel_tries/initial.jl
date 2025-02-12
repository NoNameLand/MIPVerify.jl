#=
initial.jl:
A file to run when activating the project before starting to edit and running some codes.
- Julia version: 
- Author: root
- Date: 2024-11-17
=#


#include("utils/UtilsModule.jl")
#using .UtilsModule
println("The current dir is: ", pwd())
using Pkg

Pkg.activate(".")
Pkg.add("MAT")
Pkg.add("Gurobi")
Pkg.add("Flux")
Pkg.add("Plots")
Pkg.add("MathOptInterface")
Pkg.add("JSON")
Pkg.add("UUIDs")
Pkg.add("ArgParse")
Pkg.add("StatsBase")
Pkg.add("Memento")
Pkg.add("KernelDensity")
Pkg.add("OrderedCollections")
Pkg.add("GraphPlot")
Pkg.add("Colors")