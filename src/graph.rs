use ndarray::{Array, ArrayD, ArrayViewD};
use nodes::*;
use std::collections::BTreeMap;
use std::fmt;

use optimizers::{Optimizer, SGD};
use xavier_initialize;

/// A placeholder to help index into a graph. These should not be interchanged between graphs.
/// That may work but its undefined behaviour.
#[derive(Debug, Clone, Copy)]
pub struct Idx {
    idx: usize,
}

/// A differentiable computation graph. Use this struct to hold your differentiable program
/// composed of linked [`Nodes`](enum.Node.html).
/// The default graph comes with an xavier initializer and a vanilla stochastic gradient descent
/// optimizer. The graph's `forward` and `backward` methods compute values and
/// backpropagates losses respectively. This struct offers methods for easy construction
/// and insertion of common nodes.
///
/// ## Planned Features:
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
///     * Probably requires "Graph Cursor" trait that hold `Rc<Refcell<Graph>>` with various
///     implementations with methods that make sense to them.
/// * Multithreaded / distributed graphs
///     * Asyncronous training with periodic merging of weights
/// * Graph analysis and inlining operations
#[derive(DebugStub)]
pub struct Graph {
    nodes: BTreeMap<usize, Node>,
    values: BTreeMap<usize, ArrayD<f32>>,
    losses: BTreeMap<usize, ArrayD<f32>>,
    num_inserted: usize,
    #[debug_stub = "Initializer function"]
    initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>,
    optimizer: Box<Optimizer>,
}
impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        writeln!(f, "Computation Graph with Optimizer: {:?}", self.optimizer)?;
        for (i, node) in self.nodes.iter() {
            writeln!(
                f,
                "\n{}\t{:?}\n\tvalue shape: {:?}\tloss shape:  {:?}",
                i,
                node,
                self.values[&i].shape(),
                self.losses[&i].shape(),
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
            nodes: BTreeMap::new(),
            values: BTreeMap::new(),
            losses: BTreeMap::new(),
            num_inserted: 0,
            initializer,
            optimizer,
        }
    }
    pub fn clear_non_parameters(&mut self) {
        let mut keys = Vec::new();
        for (i, n) in self.nodes.iter() {
            if let Node::Parameter(_) = n {
                //pass
            } else {
                keys.push(*i);
            }
        }
        for k in keys.into_iter() {
            self.nodes.remove(&k);
            self.values.remove(&k);
            self.losses.remove(&k);
        }
    }
    /// Remove the node at `idx` as well as its associated value and loss.
    pub fn remove(&mut self, idx: Idx) {
        self.nodes.remove(&idx.idx);
        self.values.remove(&idx.idx);
        self.losses.remove(&idx.idx);
    }
    pub fn set_value(&mut self, idx: Idx, val: ArrayD<f32>) {
        if let None = self.values.insert(idx.idx, val) {
            panic!("Tried to set value at a removed index")
        }
    }
    pub fn get_value(&mut self, idx: Idx) -> ArrayViewD<f32> {
        self.values[&idx.idx].view()
    }
    pub fn set_loss(&mut self, idx: Idx, loss: ArrayD<f32>) {
        if let None = self.losses.insert(idx.idx, loss) {
            panic!("Tried to set loss at a removed index")
        }
    }
    pub fn replace_input_iterator(
        &mut self,
        idx: Idx,
        new: Box<Iterator<Item = ArrayD<f32>>>,
    ) -> Result<(), String> {
        if let Some(Node::Input(old)) = self.nodes.get_mut(&idx.idx) {
            *old = new;
            Ok(())
        } else {
            Err("Tried to replace iterator in a node that was not input".to_string())
        }
    }
    /// Inserts the node into the graph and returns the index
    pub fn register(&mut self, node: Node) -> Idx {
        let idx = self.num_inserted;
        self.nodes.insert(idx, node);
        self.values.insert(idx, Array::zeros(()).into_dyn());
        self.losses.insert(idx, Array::zeros(()).into_dyn());
        self.num_inserted += 1;
        Idx { idx }
    }
    /// Inserts a parameter of the given shape and initializes the value using the graph's
    /// initializer.
    pub fn param(&mut self, shape: &[usize]) -> Idx {
        let idx = self.num_inserted;
        self.nodes.insert(
            idx,
            Node::Parameter({
                let x: Vec<usize> = shape.iter().map(|x| *x).collect();
                x.into_boxed_slice()
            }),
        );
        self.values.insert(idx, (self.initializer)(shape));
        self.losses.insert(idx, Array::zeros(shape));
        self.num_inserted += 1;
        Idx { idx }
    }
    /// Registers an operation and its inputs
    pub fn op(&mut self, op: impl Operation + 'static, inputs: &[Idx]) -> Idx {
        // TODO Verify inputs
        let o = Node::Operation {
            operation: Box::new(op),
            inputs: inputs.to_vec().into_boxed_slice(),
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
    /// Registers an embedding later that converts A0 to vector representation
    pub fn embedding(&mut self, weights: Idx, code: Idx) -> Idx {
        self.op(Embedding(), &[weights, code])
    }

    /// Computes values for each node in insertion order.
    /// Parameters are unaffected.
    /// Inputs will set their value to the next output of their iterator,
    /// Operations will compute a new value based on the values of its inputs.
    pub fn forward(&mut self) {
        for (i, node) in self.nodes.iter_mut() {
            match node {
                Node::Constant => {}
                Node::Input(ref mut dataset) => {
                    if let Some(v) = dataset.next() {
                        self.values.insert(*i, v);
                    } else {
                        unimplemented!("TODO handle input exhaustion gracefully")
                    }
                }
                Node::Operation {
                    ref inputs,
                    ref operation,
                } => {
                    let v = operation.eval(view_at_idxs(&inputs, &self.values));
                    self.values.insert(*i, v);
                }
                Node::Parameter(_) => {}
            }
            // reset losses
            self.losses.insert(*i, Array::zeros(self.values[i].shape()));
        }
    }
    /// Propagates gradients in reverse insertion order.
    /// Parameters will apply gradients with the graph's optimizer.
    /// Inputs are unaffected
    /// Operations will compute gradient given values from their inputs and gradients from its outputs
    pub fn backward(&mut self) {
        for (i, node) in self.nodes.iter_mut().rev() {
            match node {
                Node::Constant => {}
                Node::Input(_) => {}
                Node::Parameter(_) => {
                    self.optimizer.apply_gradient(
                        self.losses[&i].view(),
                        self.values.get_mut(i).unwrap().view_mut(),
                    );
                }
                Node::Operation {
                    ref inputs,
                    ref operation,
                } => {
                    let gradients =
                        operation.grad(view_at_idxs(&inputs, &self.values), self.losses[&i].view());
                    for (grad, j) in gradients.iter().zip(inputs.iter()) {
                        self.losses.get_mut(&j.idx).map(|x| *x += grad);
                    }
                }
            }
        }
    }
}

fn view_at_idxs<'a>(
    indices: &Box<[Idx]>,
    nodes: &'a BTreeMap<usize, ArrayD<f32>>,
) -> Box<[ArrayViewD<'a, f32>]> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[&i.idx].view());
    }
    vals.into_boxed_slice()
}
