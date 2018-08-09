use activation;
use conv::{Conv, Padding};
use global_pool::GlobalPool;
use graph::Idx;
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
    Input {
        #[debug_stub = "Box<dyn Iterator<Item = ArrayD<f32>>>"]
        dataset: Box<Iterator<Item = ArrayD<f32>>>,
    },

    // Value is initialized at the start of the graph, loss is applied to value through optimizer
    Parameter {
        optimizer: Box<Optimizer>,
    },

    // Value is determined by input values of other nodes and operation
    // Loss is propagated backwards and update the loss field of inputs
    Operation {
        inputs: Vec<Idx>,
        operation: Box<Operation>,
    },
}
impl Node {
    // Maybe just have only one field and call directly?
    pub fn input(dataset: Box<Iterator<Item = ArrayD<f32>>>) -> Self {
        Node::Input { dataset }
    }
    pub fn relu(x: Idx) -> Self {
        Node::Operation {
            inputs: vec![x],
            operation: Box::new(activation::Relu(0.0)),
        }
    }
    pub fn conv(kernel: Idx, img: Idx, padding: Padding) -> Self {
        Node::Operation {
            inputs: vec![kernel, img],
            operation: Box::new(Conv::new(padding)),
        }
    }
    pub fn global_average_pool(input: Idx) -> Self {
        Node::Operation {
            inputs: vec![input],
            operation: Box::new(GlobalPool::Average),
        }
    }
}
