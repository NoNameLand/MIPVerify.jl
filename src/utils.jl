name = "utils" # Could be a serious error #TODO: Test this
for (_, _, files) in walkdir(joinpath(@__DIR__, name))
    for file in files
        endswith(file, ".jl") && include(joinpath(name, file))
    end
end
