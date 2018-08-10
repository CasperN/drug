use std::convert::Into;
use std::{fmt, mem};

use ndarray::{Array, ArrayD, ArrayViewD};
use node::{Node, Optimizer};

use optimizers::SGD;
use xavier_initialize;

pub type Idx = usize;

#[derive(Debug)]
pub struct RuntimeNode {
    pub variant: Node,
    pub value: ArrayD<f32>,
    pub loss: ArrayD<f32>, // should be same shape as value, probably ignored by Inputs
}

impl fmt::Display for RuntimeNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        write!(
            f,
            "Node{{\n{:?} value shape: {:?}, loss shape: {:?}}}",
            self.variant,
            self.value.shape(),
            self.loss.shape()
        )
    }
}
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
    pub fn minimize(&mut self, learning_rate: f32) {
        // Adds value * learning_rate to loss, will be backpropagated upon graph backward pass
        self.loss
            .zip_mut_with(&self.value, |l, v| *l = *v * learning_rate);
    }
}

#[derive(DebugStub)]
pub struct Graph {
    pub nodes: Vec<RuntimeNode>,
    #[debug_stub = "Initializer function"]
    initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>,
    optimizer: Box<Optimizer>,
}
impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        writeln!(f, "Computation Graph with Optimizer: {:?}", self.optimizer)?;
        for (i, rtn) in self.nodes.iter().enumerate() {
            writeln!(
                f,
                "\n{}\t{:?}\n\tvalue shape: {:?}\tloss shape:  {:?}",
                i,
                rtn.variant,
                rtn.value.shape(),
                rtn.loss.shape()
            )?
        }
        Ok(())
    }
}

// Shape information?

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
            {
                let RuntimeNode {
                    ref mut variant,
                    ref mut value,
                    ..
                } = current_node;

                match variant {
                    Node::Input { ref mut dataset } => {
                        if let Some(v) = dataset.next() {
                            *value = v;
                        } else {
                            unimplemented!("TODO handle input exhaustion gracefully")
                        }
                    }

                    Node::Operation {
                        ref mut inputs,
                        ref mut operation,
                    } => {
                        let inputs = get_input_values(&inputs, &self.nodes);
                        *value = operation.eval(inputs);
                    }

                    Node::Parameter { .. } => {}
                }
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
            {
                let RuntimeNode {
                    ref mut variant,
                    ref mut value,
                    ref mut loss,
                } = current_node;

                match variant {
                    Node::Input { .. } => {}

                    Node::Operation {
                        ref inputs,
                        ref mut operation,
                    } => {
                        let gradients =
                            operation.grad(get_input_values(&inputs, &self.nodes), loss.view());

                        for (grad, j) in gradients.iter().zip(inputs.iter()) {
                            // TODO make this support broadcasting
                            // println!("loss {:?} grad {:?}", self.nodes[*j].loss, grad);
                            self.nodes[*j].loss += grad;
                        }
                    }

                    Node::Parameter {
                        ref mut optimizer, ..
                    } => {
                        optimizer.apply_gradient(loss.view(), value.view_mut());
                    }
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
