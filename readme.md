# What is this
This is a package for writing differential programs (i.e. neural networks) in
rust.


# Done
* Graph and Node architecture
* conv
* Relu, Sigmoid


# TODO Now
* Working Mnist example
    * GlobalAveragePool
    * Softmax-X-Entropy Loss
* Bias add
* Matrix multiply


# TODO Future
* GRU-RNN Example
* Saving and Loading
* Ergonomics
    * Add groups of nodes at once e.g. (conv, kernel, bias, add, relu)
    * Graph builder that holds optional optimizers and Nodes
        * Parameter in graph builder is just a shape
        * Compilation initializes parameters with optimizer meta data
        * Optimizers can parameterize actual graph
    * Naming nodes, retrieval by name
* Optimize
    * Benchmark everything
    * Flame graphs for examples
    * GPU?
* More Features
    * Batchnorm
    * strided / dialated convolutions
    * LSTM
    * Limiting gradients (e.g for GANs)



# TODO Far Future
* Inlining simple nodes to reduce allocations
    * If a node's outputs to only one node then it need not allocate loss
* backwards versions of ndarray operations where possible
