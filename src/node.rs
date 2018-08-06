use graph::Idx;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

pub trait Operation {
    // Represents a differentiable function

    // Mutates Outputs in place based on Inputs
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32>;

    // Returns gradients of inputs wrt outputs
    // Note the inputs and output vectors should be the same length
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>>;
}

pub trait DataSet {
    fn next(&mut self) -> ArrayD<f32>;
}

pub trait Optimizer {
    fn from_shape(&self, shape: &[usize]) -> Box<Optimizer>;
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>);
}

pub enum Node {
    // These versions of node differ in how the value is produced and how loss is propagated back

    // Produce Value from beyond the graph, ignore loss
    Input {
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
