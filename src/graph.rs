// use std::cell::Ref;
use ndarray::{Array, ArrayD, ArrayViewD, ArrayViewMutD};
use optimizers::SGD;
use std::mem;
// use node::{NodeTy, Node};

type Idx = usize;

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
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>);
}

// TODO Come up with a better name
enum NodeTy {
    // These versions of node differ in how the value is produced and how loss is propagated back

    // Produce Value from beyond the graph, ignore loss
    Input { dataset: Box<DataSet> },

    // Value is initialized at the start of the graph, loss is applied to value through optimizer
    Parameter { optimizer: Box<Optimizer> },

    // Value is determined by input values of other nodes and operation
    // Loss is propagated backwards and update the loss field of inputs
    Operation { inp: Vec<Idx>, op: Box<Operation> },
}

pub struct Node {
    variant: NodeTy,
    value: ArrayD<f32>,
    loss: ArrayD<f32>, // should be same shape as value, probably ignored by Inputs
}

impl Node {
    fn new_tmp() -> Self {
        Node {
            variant: NodeTy::Parameter {
                optimizer: Box::new(SGD()),
            },
            value: Array::zeros([0; 4]).into_dyn(),
            loss: Array::zeros([0; 4]).into_dyn(),
        }
    }
}

pub struct Graph {
    pub nodes: Vec<Node>, // TODO
                          // initializer: Fn(&[usize]) -> ArrayD<f32>,
                          // optimizer: Box<OptimizerBuilder>
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }
    pub fn forward(&mut self) {
        for i in 0..self.nodes.len() {
            // Need to borrow self.nodes to get inputs, this is fine as no node has itself as input
            let mut tmp = Node::new_tmp();
            mem::swap(&mut tmp, &mut self.nodes[i]);

            // Reset losses
            tmp.loss.mapv_inplace(|_| 0.0);

            // Update Values
            match tmp {
                Node {
                    variant: NodeTy::Input { ref mut dataset },
                    ref mut value,
                    ..
                } => {
                    *value = dataset.next();
                }

                Node {
                    variant:
                        NodeTy::Operation {
                            ref mut inp,
                            ref mut op,
                        },
                    ref mut value,
                    ..
                } => {
                    let inputs = get_input_values(&inp, &self.nodes);
                    *value = op.eval(inputs);
                }

                Node {
                    variant: NodeTy::Parameter { .. },
                    ..
                } => {}
            }
            mem::swap(&mut tmp, &mut self.nodes[i]);
        }
    }
    pub fn backward(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            // Need to borrow self.nodes to get inputs, this is fine as no node has itself as input
            let mut tmp = Node::new_tmp();
            mem::swap(&mut tmp, &mut self.nodes[i]);

            match tmp {
                Node {
                    variant: NodeTy::Input { .. },
                    ..
                } => {}

                Node {
                    variant:
                        NodeTy::Operation {
                            ref inp,
                            ref mut op,
                        },
                    ref loss,
                    ..
                } => {
                    let gradients = op.grad(get_input_values(&inp, &self.nodes), loss.view());

                    for (grad, j) in gradients.iter().zip(inp.iter()) {
                        // TODO make this support broadcasting
                        self.nodes[*j].loss += grad;
                    }
                }

                Node {
                    variant: NodeTy::Parameter { ref mut optimizer },
                    ref loss,
                    ref mut value,
                } => {
                    optimizer.apply_gradient(loss.view(), value.view_mut());
                }
            }

            mem::swap(&mut tmp, &mut self.nodes[i]);
        }
    }
}

fn get_input_values<'a>(indices: &Vec<Idx>, nodes: &'a Vec<Node>) -> Vec<ArrayViewD<'a, f32>> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[*i].value.view());
    }
    vals
}
