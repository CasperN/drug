//! This module holds the different types nodes that exist in a computation graph.
//! * See [Graph](../struct.Graph.html) for methods that create and register nodes.
//! * See [Node](enum.Node.html) for the types of node available.

pub use self::activation::*;
pub use self::arithmetic::{Add, Mult};
pub use self::conv::Conv;
pub use self::conv::Padding;
pub use self::embedding::Embedding;
pub use self::global_pool::GlobalPool;
pub use self::matmul::MatMul;

use erased_serde;
// use serde::Deserialize;
use graph::Idx;
use ndarray::prelude::*;
use std::fmt::Debug;
mod activation;
mod arithmetic;
mod conv;
mod embedding;
mod global_pool;
mod matmul;

/// Represents a differentiable function in a computation graph.
/// Operations hold their own hyperparameters but not their parameters, values or losses.
pub trait Operation: Debug + erased_serde::Serialize {
    /// Mutates Outputs based on inputs.
    /// Future warning: TODO do this in place by passing references and slices`
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32>;

    /// Returns gradients of inputs wrt outputs.
    /// Note the inputs and output vectors should be the same length.
    /// Future warning: TODO do this in place by passing references and slices`
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>>;
}
serialize_trait_object!(Operation);

#[derive(DebugStub, Serialize, Deserialize)]
/// Nodes are the building blocks of the [computation graph](../struct.Graph.html).
/// The variants of a node differ in how the value is produced and how loss is propagated back
pub enum Node {
    Conv {
        kernel: Idx,
        img: Idx,
        conv: Conv,
    },
    Add {
        xs: Vec<Idx>,
    },
    Mult {
        xs: Vec<Idx>,
    },
    MatMul {
        mat: Idx,
        v: Idx,
    },
    Activation {
        x: Idx,
        a: Activation,
    },
    Embedding {
        emb: Idx,
        code: Idx,
    },
    GlobalPool {
        pool: GlobalPool,
        x: Idx,
    },

    /// Produce Value from beyond the graph.
    /// * In a forward pass, its value is updates by the iterator.
    /// * In a backward pass, its losses are currently calculated but unused.

    #[serde(skip_serializing, skip_deserializing)]
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
        inputs: Box<[Idx]>,
        #[serde(skip_deserializing)]
        operation: Box<Operation>,
    },

    /// Ignored by the graph, you have to set the values yourself
    Constant,
}

impl Node {
    pub fn inputs(&self) -> Vec<Idx> {
        match self {
            Node::Conv { kernel, img, .. } => vec![*kernel, *img],
            Node::Add { xs } => xs.to_vec(),
            Node::Mult { xs } => xs.to_vec(),
            Node::MatMul { mat, v } => vec![*mat, *v],
            Node::Activation { x, .. } => vec![*x],
            Node::Embedding { emb, code } => vec![*emb, *code],
            Node::GlobalPool { x, .. } => vec![*x],
            Node::Operation { inputs, .. } => inputs.to_vec(),
            Node::Input(..) | Node::Parameter(..) | Node::Constant => vec![],
        }
    }
    pub fn forward(&mut self, inputs: Box<[ArrayViewD<f32>]>) -> Option<ArrayD<f32>> {
        match self {
            Node::Conv { conv, .. } => Some(conv.eval(inputs)),
            Node::Add { .. } => Some(Add().eval(inputs)),
            Node::Mult { .. } => Some(Mult().eval(inputs)),
            Node::MatMul { .. } => Some(MatMul().eval(inputs)),
            Node::Activation { a, .. } => Some(a.eval(inputs)),
            Node::Embedding { .. } => Some(Embedding().eval(inputs)),
            Node::GlobalPool { pool, .. } => Some(pool.eval(inputs)),
            Node::Operation { operation, .. } => Some(operation.eval(inputs)),
            Node::Input(ref mut it) => it.next(),
            Node::Parameter(..) | Node::Constant => None,
        }
    }
    pub fn backward(
        &self,
        inputs: Box<[ArrayViewD<f32>]>,
        loss: ArrayViewMutD<f32>,
    ) -> Vec<ArrayD<f32>> {
        match self {
            Node::Conv { conv, .. } => conv.grad(inputs, loss.view()),
            Node::Add { .. } => Add().grad(inputs, loss.view()),
            Node::Mult { .. } => Mult().grad(inputs, loss.view()),
            Node::MatMul { .. } => MatMul().grad(inputs, loss.view()),
            Node::Activation { a, .. } => a.grad(inputs, loss.view()),
            Node::Embedding { .. } => Embedding().grad(inputs, loss.view()),
            Node::GlobalPool { pool, .. } => pool.grad(inputs, loss.view()),
            Node::Operation { operation, .. } => operation.grad(inputs, loss.view()),
            Node::Input(..) | Node::Constant | Node::Parameter(..) => vec![],
        }
    }
}

// TODO figure out serialization and deserialization of operations
impl Default for Box<Operation> {
    fn default() -> Self {
        Box::new(arithmetic::Add())
    }
}
