use std::convert::Into;
use std::mem;

use ndarray::{Array, ArrayD, ArrayViewD};

use node::{Node, Optimizer};

use optimizers::SGD;
use xavier_initialize;

pub type Idx = usize;

pub struct Graph {
    pub nodes: Vec<RuntimeNode>,
    initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>,
    optimizer: Box<Optimizer>,
}

pub struct RuntimeNode {
    pub variant: Node,
    pub value: ArrayD<f32>,
    pub loss: ArrayD<f32>, // should be same shape as value, probably ignored by Inputs
}
// Shape information?

impl RuntimeNode {
    fn new_tmp() -> Self {
        RuntimeNode {
            variant: Node::Parameter {
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

impl Default for Graph {
    fn default() -> Self {
        Graph::new(Box::new(xavier_initialize), Box::new(SGD()))
    }
}

impl Graph {
    pub fn new(initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>, optimizer: Box<Optimizer>) -> Self {
        Graph {
            nodes: Vec::new(),
            initializer,
            optimizer,
        }
    }

    pub fn new_initialized_param(&mut self, param: ArrayD<f32>) -> Idx {
        let shape = param.shape().to_vec();
        let rtn = RuntimeNode {
            variant: Node::Parameter {
                optimizer: self.optimizer.from_shape(&shape[..]),
            },
            value: param,
            loss: Array::zeros(&shape[..]),
        };
        self.nodes.push(rtn);
        self.nodes.len() - 1
    }

    pub fn new_param(&mut self, shape: &[usize]) -> Idx {
        let rtn = RuntimeNode {
            variant: Node::Parameter {
                optimizer: self.optimizer.from_shape(shape),
            },
            value: (self.initializer)(shape),
            loss: Array::zeros(shape),
        };
        self.nodes.push(rtn);
        self.nodes.len() - 1
    }

    pub fn register<T: Into<RuntimeNode>>(&mut self, node: T) -> Idx {
        // TODO Check Idx makes sense for Operation inputs
        self.nodes.push(node.into());
        self.nodes.len() - 1
    }

    pub fn forward(&mut self) {
        for i in 0..self.nodes.len() {
            // Need to borrow self.nodes to get inputs, this is fine as no node has itself as input
            let mut current_node = RuntimeNode::new_tmp();
            mem::swap(&mut current_node, &mut self.nodes[i]);
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
            // reset losses
            current_node.loss = Array::zeros(current_node.value.shape());
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
