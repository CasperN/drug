use ndarray::{Array, ArrayD, ArrayViewD};
use nodes::Node;
use std::fmt;

use optimizers::{Optimizer, SGD};
use xavier_initialize;

pub type Idx = usize;

/// A differentiable computation graph. Use this struct to hold your differentiable program
/// composed of linked [`Nodes`](enum.Node.html).
/// The default graph comes with an xavier initializer and normal stochastic grapdient descent
/// initializer. The graph's `forward` and `backward` methods compute values and
/// backpropagates losses respectively. See [Node](enum.Node.html) for what can be put in a graph.
///
/// ## Planned Features i.e. expect breaking changes (TODO):
/// * Hide fields and allow indexing the graph itself directly so nodes, losses, and values are
/// created and destroyed together despite living in different vectors. his allows the user to
/// dynamically add and remove nodes while still being able to use the original indices. The
/// current hack is to first register immutable parts of the graph like parameters.
/// * Naming and indexing via string
/// * Saving / loading (need to distinguish parameters from other kinds of values)
/// * Freezing part of the graph for training (particularly for GANs)
/// * Building complexes of nodes such as (conv + bias + relu) or RNN cells while allowing for
/// parameter reuse
///
/// ## Wishlist
/// * Running in a GPU (once rust gets real GPGPU support)
/// * Automatic differentiation for things like wasserstien loss
///     * Automatically derive backwards versions of primatives
/// * A nice DSL overloading arithmetic operators and allowing you to implicitly build a graph
/// with Idxs
///     * Probably requires "Graph Cursor" structs that hold `Rc<Refcell<Graph>>` and a trait object
///     Giving it suitable methods
/// * Multithreaded / distributed graphs
///     * Asyncronous training with periodic merging of weights
/// * Graph analysis and inlining operations
#[derive(DebugStub)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub values: Vec<ArrayD<f32>>,
    pub losses: Vec<ArrayD<f32>>,

    #[debug_stub = "Initializer function"]
    initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>,
    optimizer: Box<Optimizer>,
}
impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        writeln!(f, "Computation Graph with Optimizer: {:?}", self.optimizer)?;
        for i in 0..self.nodes.len() {
            writeln!(
                f,
                "\n{}\t{:?}\n\tvalue shape: {:?}\tloss shape:  {:?}",
                i,
                self.nodes[i],
                self.values[i].shape(),
                self.losses[i].shape(),
            )?
        }
        Ok(())
    }
}

// Shape information?
impl Default for Graph {
    /// xavier initializer and normal gradient descent
    fn default() -> Self {
        Graph::new(Box::new(xavier_initialize), Box::new(SGD()))
    }
}

impl Graph {
    pub fn new(initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>, optimizer: Box<Optimizer>) -> Self {
        Graph {
            nodes: Vec::new(),
            values: Vec::new(),
            losses: Vec::new(),
            initializer,
            optimizer,
        }
    }
    /// Inserts a parameter of the given shape and initializes the value using the graph's
    /// initializer.
    pub fn new_param(&mut self, shape: &[usize]) -> Idx {
        self.nodes.push(Node::Parameter({
            let x: Vec<usize> = shape.iter().map(|x| *x).collect();
            x.into_boxed_slice()
        }));
        self.values.push((self.initializer)(shape));
        self.losses.push(Array::zeros(shape));
        self.nodes.len() - 1
    }
    /// Inserts the node into the graph and returns the index
    pub fn register(&mut self, node: Node) -> Idx {
        self.nodes.push(node);
        self.values.push(Array::zeros([0; 4]).into_dyn());
        self.losses.push(Array::zeros([0; 4]).into_dyn());
        self.nodes.len() - 1
    }
    pub fn forward(&mut self) {
        for i in 0..self.nodes.len() {
            match self.nodes[i] {
                Node::Input(ref mut dataset) => {
                    if let Some(v) = dataset.next() {
                        self.values[i] = v;
                    } else {
                        unimplemented!("TODO handle input exhaustion gracefully")
                    }
                }
                Node::Operation {
                    ref inputs,
                    ref mut operation,
                } => {
                    let v = operation.eval(view_at_idxs(&inputs, &self.values));
                    self.values[i] = v;
                }
                Node::Parameter(_) => {}
            }
            // reset losses
            self.losses[i] = Array::zeros(self.values[i].shape());
        }
    }
    pub fn backward(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            match self.nodes[i] {
                Node::Input(_) => {}
                Node::Parameter(_) => {
                    self.optimizer
                        .apply_gradient(self.losses[i].view(), self.values[i].view_mut());
                }
                Node::Operation {
                    ref inputs,
                    ref mut operation,
                } => {
                    let gradients =
                        operation.grad(view_at_idxs(&inputs, &self.values), self.losses[i].view());
                    for (grad, j) in gradients.iter().zip(inputs.iter()) {
                        self.losses[*j] += grad;
                    }
                }
            }
        }
    }
}

fn view_at_idxs<'a>(indices: &Vec<Idx>, nodes: &'a Vec<ArrayD<f32>>) -> Vec<ArrayViewD<'a, f32>> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[*i].view());
    }
    vals
}
