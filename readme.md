# What is this
This is a package for writing differential programs (i.e. neural networks) in
rust.

# TODO Now
* GRU-RNN Example
    * Embedding layer
* Refactor Operations
    * Box slice inputs and losses to reflect static length
    ```
    struct ForwardArg {
        inputs: Box<[ArrayViewD<f32>]>,
        value: Box<[ArrayViewMutD<f32>]>,
    }
    struct BackwardArg {
        inputs: Box<[ArrayViewD<f32>]>,
        losses: Box<[ArrayMutD<f32>]>,
        value: ArrayViewD<f32>,
    }
    ```
* Parameters need `n_in` and `n_out` for Xavier initialization
* Refactor Optimizer trait / Adam Optimizer
    * Needs to keep a map of parameters and keep associated information
    * Register parameters
* Bias add
* Need to make CNN orders of magnitude more performant

# TODO Future
* Saving and Loading
* Ergonomics
    * Add groups of nodes at once e.g. (kernel, conv, bias, add, relu)
    * Graph builder that holds optional optimizers and Nodes
        * Parameter in graph builder is just a shape
        * Compilation initializes parameters with optimizer meta data
        * Optimizers can parameterize actual graph
    * Naming nodes, retrieval by name
* Optimize
    * Benchmark everything
    * Flame graphs for examples
* More Features
    * Batchnorm
    * dialated convolutions
    * LSTM
    * Limiting gradients (e.g for GANs)


# TODO Far Future
* GPU
* Inlining simple nodes to reduce allocations
    * If a node's outputs to only one node then it need not allocate loss
* backwards versions of ndarray operations where possible
