use ndarray::{Array, ArrayD, ArrayViewD};
use nodes::*;
use std::fmt;

use optimizers::{Optimizer, SGD};
use xavier_initialize;

#[derive(Debug, Clone)]
pub struct Idx{
    insert_num: usize,
}

/// A differentiable computation graph. Use this struct to hold your differentiable program
/// composed of linked [`Nodes`](enum.Node.html).
/// The default graph comes with an xavier initializer and normal stochastic grapdient descent
/// initializer. The graph's `forward` and `backward` methods compute values and
/// backpropagates losses respectively. This struct offers methods for easy construction
/// and insertion of common nodes.
///
/// ## Planned Features:
/// * **Breaking change:** Hide fields and allow indexing the graph itself directly so nodes,
/// losses, and values are created and destroyed together despite living in different vectors. This
/// should the user to dynamically add and remove nodes while still being able to use the original
/// indices. The current hack is to first register immutable parts of the graph like parameters.
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
    nodes: Vec<Node>,
    values: Vec<ArrayD<f32>>,
    losses: Vec<ArrayD<f32>>,
    num_inserted: usize,
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
    /// Consider using `Graph::default()` if you don't want to choose your own optimizer and
    /// initializer.
    pub fn new(initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>, optimizer: Box<Optimizer>) -> Self {
        Graph {
            nodes: Vec::new(),
            values: Vec::new(),
            losses: Vec::new(),
            num_inserted: 0,
            initializer,
            optimizer,
        }
    }
    /// Remove the node at `idx` as well as its associated value and loss.
    /// Tagged as unsafe because it invalidies previosly returned indices
    pub unsafe fn remove(&mut self, idx: Idx){
        unimplemented!()
    }
    pub fn set_value(&mut self, idx: Idx, val: ArrayD<f32>){
        self.values[idx.insert_num] = val;
    }
    pub fn get_value(&mut self, idx: Idx) -> ArrayViewD<f32> {
        self.values[idx.insert_num].view()
    }
    pub fn set_loss(&mut self, idx: Idx, loss: ArrayD<f32>) {
        self.losses[idx.insert_num] = loss;
    }
    pub fn replace_input_iterator(&mut self, idx: Idx, new: Box<Iterator<Item = ArrayD<f32>>>) -> Result<(), String> {
        if let Node::Input(ref mut old) = self.nodes[idx.insert_num] {
            *old = new;
            Ok(())
        } else {
            Err("Tried to replace iterator in a node that was not input".to_string())
        }
    }
    /// Inserts the node into the graph and returns the index
    pub fn register(&mut self, node: Node) -> Idx {
        self.nodes.push(node);
        self.values.push(Array::zeros([0; 4]).into_dyn());
        self.losses.push(Array::zeros([0; 4]).into_dyn());
        self.num_inserted += 1;
        Idx { insert_num: self.num_inserted - 1 }
    }
    /// Inserts a parameter of the given shape and initializes the value using the graph's
    /// initializer.
    pub fn param(&mut self, shape: &[usize]) -> Idx {
        self.nodes.push(Node::Parameter({
            let x: Vec<usize> = shape.iter().map(|x| *x).collect();
            x.into_boxed_slice()
        }));
        self.values.push((self.initializer)(shape));
        self.losses.push(Array::zeros(shape));
        self.num_inserted += 1;
        Idx { insert_num: self.num_inserted - 1 }
    }
    /// Registers an operation and its inputs
    pub fn op(&mut self, op: impl Operation + 'static, inputs: &[Idx]) -> Idx {
        // TODO Verify inputs
        let o = Node::Operation {
            operation: Box::new(op),
            inputs: inputs.to_vec(),
        };
        self.register(o)
    }
    /// Registers
    pub fn conv(&mut self, kernel: Idx, img: Idx, padding: Padding, stride: usize) -> Idx {
        self.op(Conv::new(padding, stride), &[kernel, img])
    }
    /// Registers a pooling operation takes a `Batch * Height * Width * Channels` image and reduces it to
    /// a `Batch * Channels` vector.
    pub fn global_pool(&mut self, input: Idx, pool: GlobalPool) -> Idx {
        self.op(pool, &[input])
    }
    /// Registers a Relu operation which takes the elementwise maximum of the input array and 0.
    pub fn relu(&mut self, x: Idx) -> Idx {
        self.op(Relu(0.0), &[x])
    }
    /// Registers a new sigmoid activation operation, an
    /// elementwise application of $\frac{ 1 }{1 - e^{-x}}$.
    pub fn sigmoid(&mut self, x: Idx) -> Idx {
        self.op(Sigmoid(), &[x])
    }
    /// Registers a Tanh operation.
    pub fn tanh(&mut self, x: Idx) -> Idx {
        self.op(Tanh(), &[x])
    }
    /// Registers a matrix multiplication of `input` by `weights`.
    pub fn matmul(&mut self, weights: Idx, input: Idx) -> Idx {
        self.op(MatMul(), &[weights, input])
    }
    /// Computes values for each node in insertion order.
    /// Parameters are unaffected.
    /// Inputs will set their value to the next output of their iterator,
    /// Operations will compute a new value based on the values of its inputs.
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
    /// Propagates gradients in reverse insertion order.
    /// Parameters will apply gradients with the graph's optimizer.
    /// Inputs are unaffected
    /// Operations will compute gradient given values from their inputs and gradients from its outputs
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
                        self.losses[j.insert_num] += grad;
                    }
                }
            }
        }
    }
}

fn view_at_idxs<'a>(indices: &Vec<Idx>, nodes: &'a Vec<ArrayD<f32>>) -> Vec<ArrayViewD<'a, f32>> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[i.insert_num].view());
    }
    vals
}
