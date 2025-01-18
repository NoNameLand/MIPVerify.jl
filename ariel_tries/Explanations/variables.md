When we define a model the variables are set as follows:
Given an input of size [d1, d2, ... , dn], the model sets:
1. d1*d2...*dN variables as input variables with [l, u] = [0,1]
2. d1*d2...*dN variables as pertuabtrion varaibles with [l,u] = [-eps, eps]. Where eps is the l-inifinty pertubation strength.
3. For each ReLU layer that has a Linear layer before it (layers[i] is a ReLU and layer[i-1] is a Linear or CONV2D) we test what condition holds:
for all j in layer[i-1] neruons:
    a. layer[i-1][j] lower bound > 0 -> ReLU(x) = x -> NO NEW VARIABLE ADDED.
    b. layer[i-1][j] upper bounds < 0 -> ReLU(x) = 0 -> NO NEW VARIABLE ADDED.
    c. layer[i-1][j] lower bound < 0 and upper bound > 0 -> ReLU(x) = alpha * x, where alpha is a binary variable. NEW VARIABLE ADDED.

Overall, the total number of variables in the model is input_size * 2 (for input and pertubation) + #(unknown RELU's).
We can find an upper bound for that:
#vars_in_model <= 2*size_input + sum(layer.bias for layer is a linear in layers and layer next is a ReLU) # For FC Networks.

When we want to access a varialbe in the model we need to access them by order, the relation between the order to the neuron is defined by:
the input variable of location [i1,  ... , iN] is in variable of index CartesianIndices((i1, ... iN)). 
the pertubation variable of location [i1, ... , iN] is in variable of index size_input + CartesianIndices((i1, ... iN))

When we want to access the output of the network as a function of the variables:
d[:Output][i] = Output Neuron of index i. 
