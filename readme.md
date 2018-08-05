# What is this
This is a package for writing differential programs (i.e. neural networks) in
rust. It is mostly inspired by PyTorch


### computation graph options

Idea                            Problem
----------------------------    ----------------------------------
Lifetimes hell                  Not easy to use, too many variables
Move when used                  Disallows skip connections
Rc<Refcell<Node>>               Kinda fucky
Enum wrapping computations      Not scalable
Heterogenous Arena              Type checker issues?

**ND array everywhere**         This might work

# TODO
* Figure out how to use nalgebra
* Xavier Initializer
* Computations
    * Convolution
    * Bias add
    * Sigmoid
    * Relu
    * Max Pooling
* Optimizers
    * SGD
    * SGD with Momentum
    * Adam
