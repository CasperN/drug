//! This module holds the nodes that build a computation graph.
//! * Node::Input(...) should be used to put boxed iterators into the graph
//! * The preferred way of creating operations is with `Node::operation(...)`.
//! This is easier than assembling the Node enum manually.
//! See the [impl Node](enum.Node.html) section.
//! * For parameters, use the graph's new_parameter method.

pub use self::activation::*;
pub use self::conv::Conv;
pub use self::conv::Padding;
pub use self::embedding::Embedding;
pub use self::global_pool::GlobalPool;
pub use self::matmul::MatMul;
use graph::Idx;
use ndarray::prelude::*;
use std::fmt::Debug;
mod activation;
mod conv;
mod embedding;
mod global_pool;
mod matmul;

/// Represents a differentiable function in a computation graph.
/// Operations hold their own hyperparameters but not their parameters, values or losses.
pub trait Operation: Debug {
    /// Mutates Outputs based on inputs.
    /// Future warning: TODO do this in place by passing references and slices`
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32>;

    /// Returns gradients of inputs wrt outputs.
    /// Note the inputs and output vectors should be the same length.
    /// Future warning: TODO do this in place by passing references and slices`
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>>;
}

#[derive(DebugStub)]
/// Nodes are the building blocks of the computation graph.
/// The variants of a node differ in how the value is produced and how loss is propagated back
pub enum Node {
    /// Produce Value from beyond the graph.
    /// * In a forward pass, its value is updates by the iterator.
    /// * In a backward pass, its losses are currently calculated but unused.
    Input(#[debug_stub = "Box<Iterator<Item=ArrayD<f32>>>"] Box<Iterator<Item = ArrayD<f32>>>),

    /// Parameter nodes only hold a shape. Its values are initialized when inserted into the graph
    /// using the graph's initializer.
    /// * In a foward pass, parameters are ignored.
    /// * In a backward pass, their losses are applied by the graph's optimizer.
    Parameter(Box<[usize]>),

    /// An Operation node holds an [Operation trait object](trait.Operation.html) and the indices
    /// referring to its input values.
    /// * In a forward pass, its value is updated by the `operation` and the values indexed by
    /// `inputs`.
    /// * In a backward pass, gradients are calculated and losses are propagated backwards and added
    /// to the losses indexed by `inputs`.
    Operation {
        inputs: Vec<Idx>,
        operation: Box<Operation>,
    },
}

/// The building blocks of a computation graph. Node methods are here to give a nicer interface for
/// instantiating operations in the computation graph as the number of required arguments of an
/// operation cannot be known at compile time.
impl Node {
    /// Builds an operation node directly from a struct that implements Operation.
    /// Prefer to use another method if implemented e.g. `Node::conv(kernel, img)`
    pub fn op(op: impl Operation + 'static, inputs: &[Idx]) -> Self {
        Node::Operation {
            operation: Box::new(op),
            inputs: inputs.to_vec(),
        }
    }
    /// Convolution operation that supports striding and padding.
    /// * Input and output arrays are `Batch * Height * Width * Channels`. Though the number of
    /// input and output channels may differ.
    /// * Kernel shape is `Kernel_height * Kernel_width * Channels_in * Channels_out`
    pub fn conv(kernel: Idx, img: Idx, padding: Padding, stride: usize) -> Self {
        Node::Operation {
            inputs: vec![kernel, img],
            operation: Box::new(Conv::new(padding, stride)),
        }
    }
    /// A pooling operation takes a `Batch * Height * Width * Channels` image and reduces it to
    /// a `Batch * Channels` vector.
    pub fn global_pool(input: Idx, pool: GlobalPool) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(pool),
        }
    }
    /// A Relu returns the elementwise maximum of the input array and 0.
    pub fn relu(x: Idx) -> Self {
        Node::Operation {
            inputs: vec![x],
            operation: Box::new(Relu(0.0)),
        }
    }
    /// Returns a new sigmoid activation operation, an
    /// elementwise application of $\frac{ 1 }{1 - e^{-x}}$.
    pub fn sigmoid(input: Idx) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(Sigmoid()),
        }
    }
    pub fn tanh(input: Idx) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(Tanh()),
        }
    }
    /// Returns a new matrix multiply operation.
    /// Assumes the value at `weights` is a N * M 2D array
    /// and the value at `input` is a Batch * N 2D array.
    /// Output value is Batch * M 2D array.
    pub fn matmul(weights: Idx, input: Idx) -> Self {
        Node::Operation {
            inputs: vec![weights, input],
            operation: Box::new(MatMul()),
        }
    }
}
