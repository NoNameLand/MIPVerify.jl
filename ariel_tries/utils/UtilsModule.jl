"""
    use_local_MIPVerify()

- Julia version: 
- Author: root
- Date: 2024-11-19

# Arguments

# Examples

```jldoctest
julia>
```
"""

module UtilsModule
using Pkg
export use_local_MIPVerify

function use_local_MIPVerify(local_path::String)

    # Check if the path exists
    if !isdir(local_path)
        error("The specified path does not exist: $local_path")
    end

    # Use the local version of MIPVerify
    println("Switching to local version of MIPVerify at $local_path...")
    Pkg.develop(PackageSpec(path=local_path))
    println("Successfully switched to the local version.")
end

end
