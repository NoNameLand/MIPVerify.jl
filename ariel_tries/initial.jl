#=
initial.jl:
A file to run when activating the project before starting to edit and running some codes.
- Julia version: 
- Author: root
- Date: 2024-11-17
=#


include("utils/UtilsModule.jl")
using .UtilsModule
println("The current dir is: ", pwd())
using Pkg

Pkg.activate(".")
Pkg.add("MAT")
Pkg.add("Gurobi")
Pkg.add("Images")