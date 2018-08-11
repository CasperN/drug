use activation::{Relu, Sigmoid};
use conv::{Conv, Padding};
use global_pool::GlobalPool;
use graph::Idx;
use matmul::MatMul;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use std::fmt::Debug;

pub trait Operation: Debug {
    // Represents a differentiable function

    // Mutates Outputs in place based on Inputs
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32>;

    // Returns gradients of inputs wrt outputs
    // Note the inputs and output vectors should be the same length
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>>;
}

pub trait Optimizer: Debug {
    fn from_shape(&self, shape: &[usize]) -> Box<Optimizer>;
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>);
}

#[derive(DebugStub)]
pub enum Node {
    // These versions of node differ in how the value is produced and how loss is propagated back

    // Produce Value from beyond the graph, ignore loss

    Input(
        #[debug_stub = "Input"]
        Box<Iterator<Item = ArrayD<f32>>>),

    // Value is initialized at the start of the graph, loss is applied to value through optimizer
    Parameter(Box<[usize]>),

    // Value is determined by input values of other nodes and operation
    // Loss is propagated backwards and update the loss field of inputs
    Operation {
        inputs: Vec<Idx>,
        operation: Box<Operation>,
    },
}
impl Node {
    // Maybe just have only one field and call directly?
    pub fn relu(x: Idx) -> Self {
        Node::Operation {
            inputs: vec![x],
            operation: Box::new(Relu(0.0)),
        }
    }
    pub fn conv(kernel: Idx, img: Idx, padding: Padding, stride: usize) -> Self {
        Node::Operation {
            inputs: vec![kernel, img],
            operation: Box::new(Conv::new(padding, stride)),
        }
    }
    pub fn global_pool(input: Idx, pool: GlobalPool) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(pool),
        }
    }
    pub fn sigmoid(input: Idx) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(Sigmoid()),
        }
    }
    pub fn mat_mul(weights: Idx, input: Idx) -> Self {
        Node::Operation {
            inputs: vec![weights, input],
            operation: Box::new(MatMul()),
        }
    }
}
