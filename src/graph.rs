use std::convert::Into;
use std::mem;

use ndarray::{Array, ArrayD, ArrayViewD};

use node::Node;
use optimizers::SGD;

pub type Idx = usize;

pub struct Graph {
    pub nodes: Vec<RuntimeNode>,
    // TODO
    // initializer: Fn(&[usize]) -> ArrayD<f32>,
    // optimizer: Box<OptimizerBuilder>
}

pub struct RuntimeNode {
    pub variant: Node,
    pub value: ArrayD<f32>,
    pub loss: ArrayD<f32>, // should be same shape as value, probably ignored by Inputs
}
impl RuntimeNode {
    fn new_tmp() -> Self {
        RuntimeNode {
            variant: Node::Parameter {
                shape: Vec::new(),
                optimizer: Box::new(SGD()),
            },
            value: Array::zeros([0; 4]).into_dyn(),
            loss: Array::zeros([0; 4]).into_dyn(),
        }
    }
}

impl Into<RuntimeNode> for Node {
    fn into(self: Node) -> RuntimeNode {
        RuntimeNode {
            variant: self,
            value: Array::zeros([0; 4]).into_dyn(),
            loss: Array::zeros([0; 4]).into_dyn(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    pub fn register<T: Into<RuntimeNode>>(&mut self, node: T) -> Idx {
        self.nodes.push(node.into());
        self.nodes.len() - 1
    }

    pub fn forward(&mut self) {
        for i in 0..self.nodes.len() {
            // Need to borrow self.nodes to get inputs, this is fine as no node has itself as input
            let mut current_node = RuntimeNode::new_tmp();
            mem::swap(&mut current_node, &mut self.nodes[i]);

            // Reset losses
            current_node.loss.mapv_inplace(|_| 0.0);

            // Update Values
            match current_node {
                RuntimeNode {
                    variant: Node::Input { ref mut dataset },
                    ref mut value,
                    ..
                } => {
                    if let Some(v) = dataset.next() {
                        *value = v;
                    } else {
                        unimplemented!("TODO handle input exhaustion gracefully")
                    }
                }

                RuntimeNode {
                    variant:
                        Node::Operation {
                            ref mut inputs,
                            ref mut operation,
                        },
                    ref mut value,
                    ..
                } => {
                    let inputs = get_input_values(&inputs, &self.nodes);
                    *value = operation.eval(inputs);
                }

                RuntimeNode {
                    variant: Node::Parameter { .. },
                    ..
                } => {}
            }
            mem::swap(&mut current_node, &mut self.nodes[i]);
        }
    }
    pub fn backward(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            // Need to borrow self.nodes to get inputs, this is fine as no node has itself as input
            let mut current_node = RuntimeNode::new_tmp();
            mem::swap(&mut current_node, &mut self.nodes[i]);

            match current_node {
                RuntimeNode {
                    variant: Node::Input { .. },
                    ..
                } => {}

                RuntimeNode {
                    variant:
                        Node::Operation {
                            ref inputs,
                            ref mut operation,
                        },
                    ref loss,
                    ..
                } => {
                    let gradients =
                        operation.grad(get_input_values(&inputs, &self.nodes), loss.view());

                    for (grad, j) in gradients.iter().zip(inputs.iter()) {
                        // TODO make this support broadcasting
                        self.nodes[*j].loss += grad;
                    }
                }

                RuntimeNode {
                    variant:
                        Node::Parameter {
                            ref mut optimizer, ..
                        },
                    ref loss,
                    ref mut value,
                } => {
                    optimizer.apply_gradient(loss.view(), value.view_mut());
                }
            }

            mem::swap(&mut current_node, &mut self.nodes[i]);
        }
    }
}

fn get_input_values<'a>(
    indices: &Vec<Idx>,
    nodes: &'a Vec<RuntimeNode>,
) -> Vec<ArrayViewD<'a, f32>> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[*i].value.view());
    }
    vals
}
