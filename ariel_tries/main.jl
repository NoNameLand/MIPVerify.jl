using ArgParse

# Adding the function of the main
include("process_bounds.jl")


function main()
    # Define the argument parser
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--network_type"
            help = "The type of the network to test"
            arg_type = String
            default = ":)"
        "--output", "-o"
            help = "Path to the output file"
            arg_type = String
            default = "output.txt"
    end

    # Parse the arguments
    parsed_args = parse_args(ARGS, s)

    # Use the parsed arguments
    println("Input file: ", parsed_args["network_type"])
    println("Output file: ", parsed_args["output"])

    println("Processing inputs and writing to output file...")

    # Calling the function
    process_bounds()


end

# Run the script if executed directly
main()
