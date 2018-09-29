use ndarray::{Array, ArrayD, ArrayViewD};
use nodes::*;
use std::collections::BTreeMap;
use std::fmt;

use optimizers::Optimizer;
use xavier_initialize;

/// A placeholder to help index into a graph. These should not be interchanged between graphs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Idx {
    idx: usize,
}

/// A differentiable computation graph. Use this struct to hold your differentiable program
/// which is a directed acyclic graph of Nodes, their associated values
/// and losses (gradients). The graph computes values moving forward in insertion order (see
/// `forward` method) and propagates losses backwards in reverse insertion order (see `backward`
/// method). The default graph comes with an xavier initializer and a vanilla stochastic gradient
/// descent optimizer.
#[derive(DebugStub, Serialize, Deserialize)]
pub struct Graph {
    nodes: BTreeMap<usize, Node>,
    values: BTreeMap<usize, ArrayD<f32>>,
    losses: BTreeMap<usize, ArrayD<f32>>,
    num_inserted: usize,
    #[debug_stub = "Initializer function"]
    #[serde(skip)]
    initializer: Initializer,
    pub optimizer: Optimizer,
    pub named_idxs: BTreeMap<String, Idx>,
}

struct Initializer(Box<(Fn(&[usize]) -> ArrayD<f32>)>);

impl Default for Initializer {
    fn default() -> Self {
        Initializer(Box::new(xavier_initialize))
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Customize so only `x` and `y` are denoted.
        writeln!(f, "Computation Graph with Optimizer:\n\t{}", self.optimizer)?;
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
        Graph::new(Box::new(xavier_initialize), Optimizer::default())
    }
}

impl Graph {
    /// Consider using `Graph::default()` if you don't want to choose your own optimizer and
    /// initializer.
    pub fn new(initializer: Box<(Fn(&[usize]) -> ArrayD<f32>)>, optimizer: Optimizer) -> Self {
        Graph {
            nodes: BTreeMap::new(),
            values: BTreeMap::new(),
            losses: BTreeMap::new(),
            named_idxs: BTreeMap::new(),
            num_inserted: 0,
            initializer: Initializer(initializer),
            optimizer,
        }
    }
    /// Inserts the node into the graph and returns the index
    pub fn register(&mut self, node: Node) -> Idx {
        let idx = self.num_inserted;
        if let Node::Parameter(ref shape) = node {
            self.optimizer.register(Idx { idx }, shape)
        }
        self.nodes.insert(idx, node);
        self.values.insert(idx, Array::zeros(()).into_dyn());
        self.losses.insert(idx, Array::zeros(()).into_dyn());
        self.num_inserted += 1;
        Idx { idx }
    }
    /// Registers a parameter of the given shape and initializes the value using the graph's
    /// initializer.
    pub fn param(&mut self, shape: &[usize]) -> Idx {
        let idx = self.register(Node::Parameter({
            let x: Vec<usize> = shape.iter().map(|x| *x).collect(); // HACK
            x.into_boxed_slice()
        }));
        self.values.insert(idx.idx, (self.initializer.0)(shape));
        self.losses.insert(idx.idx, Array::zeros(shape));
        self.num_inserted += 1;
        idx
    }
    /// Registers an input node which advances the iterator `it` each forward pass.
    /// WARNING this prevents saving with serde.
    pub fn input(&mut self, it: Box<Iterator<Item = ArrayD<f32>>>) -> Idx {
        self.register(Node::Input(it))
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
    /// Registers a constant, sets its value to `c`, then returns the idx
    pub fn constant(&mut self, c: ArrayD<f32>) -> Idx {
        let idx = self.register(Node::Constant);
        self.set_value(idx, c);
        idx
    }
    fn _forward1(&mut self, i: &usize) {
        if let Some(n) = self.nodes.get_mut(&i) {
            let inps = n.inputs();
            if let Some(v) = n.forward(view_at_idxs(&inps, &self.values)) {
                self.values.insert(*i, v);
            }
        }
        // reset losses
        self.losses.insert(*i, Array::zeros(self.values[i].shape()));
    }
    fn _backward1(&mut self, i: &usize) {
        if let Some(n) = self.nodes.get_mut(&i) {
            if let Node::Parameter(..) = n {
                self.optimizer.apply_gradient(
                    &Idx { idx: *i },
                    self.values.get_mut(i).unwrap().view_mut(),
                    self.losses[&i].view(),
                );
            } else {
                let inps = n.inputs();
                let gradients = n.backward(
                    view_at_idxs(&inps, &self.values),
                    self.losses.get(&i).unwrap().view(),
                );
                for (grad, j) in gradients.iter().zip(inps.iter()) {
                    self.losses.get_mut(&j.idx).map(|x| *x += grad);
                }
            }
        }
    }
    /// Computes values for each node in insertion order.
    /// Parameters are unaffected.
    /// Inputs will set their value to the next output of their iterator,
    /// Operations will compute a new value based on the values of its inputs.
    pub fn forward(&mut self) {
        let keys: Vec<usize> = self.nodes.keys().map(|x| *x).collect();
        for i in keys.iter() {
            self._forward1(i);
        }
    }
    /// Propagates gradients in reverse insertion order.
    /// Parameters will apply gradients with the graph's optimizer.
    /// Inputs are unaffected
    /// Operations will compute gradient given values from their inputs and gradients from its outputs
    pub fn backward(&mut self) {
        let keys: Vec<usize> = self.nodes.keys().rev().map(|x| *x).collect();
        for i in keys.iter() {
            self._backward1(i);
        }
    }
    /// Updates value and resets losses for node with Idx `i`.
    pub fn forward1(&mut self, i: Idx) {
        self._forward1(&i.idx);
    }
    /// Back propagates losses for node with Idx `i`.
    pub fn backward1(&mut self, i: Idx) {
        self._backward1(&i.idx);
    }
    /// Remove the node at `idx` as well as its associated value and loss.
    pub fn remove(&mut self, idx: Idx) {
        self.nodes.remove(&idx.idx);
        self.values.remove(&idx.idx);
        self.losses.remove(&idx.idx);
    }
    /// This op removes every node from the graph that is not a parameter. This is useful for
    /// dynamic graphs and recurrent neural networks when you want to rebuild everything each
    /// forward and backward pass of the network.
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
    pub fn set_value(&mut self, idx: Idx, val: ArrayD<f32>) {
        if let None = self.values.insert(idx.idx, val) {
            panic!("Tried to set value at a removed index")
        }
    }
    pub fn get_value(&self, idx: Idx) -> ArrayViewD<f32> {
        self.values[&idx.idx].view()
    }
    pub fn set_loss(&mut self, idx: Idx, loss: ArrayD<f32>) {
        if let None = self.losses.insert(idx.idx, loss) {
            panic!("Tried to set loss at a removed index")
        }
    }
    pub fn get_loss(&self, idx: Idx) -> ArrayViewD<f32> {
        self.losses[&idx.idx].view()
    }
    /// Replace an Input node's iterator or converts Constant nodes into Input with this iterator.
    /// WARNING this prevents saving with serde (boxed iterator). Constants should be used instead.
    pub fn replace_input_iterator(
        &mut self,
        idx: Idx,
        new: Box<Iterator<Item = ArrayD<f32>>>,
    ) -> Result<(), String> {
        if let Some(n) = self.nodes.get_mut(&idx.idx) {
            match n {
                Node::Input(old) => *old = new,
                Node::Constant => *n = Node::Input(new),
                _ => {
                    return Err("Tried to replace input iter at non Input/Constant node.".to_string())
                }
            }
            Ok(())
        } else {
            Err("Tried to replace input iterator at invalid index.".to_string())
        }
    }
    pub fn add(&mut self, inputs: &[Idx]) -> Idx {
        self.register(Node::Add {
            xs: inputs.to_vec(),
        })
    }
    pub fn mult(&mut self, inputs: &[Idx]) -> Idx {
        self.register(Node::Mult {
            xs: inputs.to_vec(),
        })
    }
    /// Registers a convolution operation node and returns the index
    pub fn conv(&mut self, kernel: Idx, img: Idx, padding: Padding, stride: usize) -> Idx {
        self.register(Node::Conv {
            kernel,
            img,
            conv: Conv::new(padding, stride),
        })
    }
    /// Registers a pooling operation takes a `Batch * Height * Width * Channels` image and reduces
    /// it to a `Batch * Channels` vector.
    pub fn global_pool(&mut self, x: Idx, pool: GlobalPool) -> Idx {
        self.register(Node::GlobalPool { x, pool })
    }
    /// Registers a Relu operation which takes the elementwise maximum of the input array and 0.
    pub fn relu(&mut self, x: Idx) -> Idx {
        self.register(Node::Activation {
            x,
            a: Activation::Relu { leak: 0.0 },
        })
    }
    /// Registers a new sigmoid activation operation, an
    /// elementwise application of $\frac{ 1 }{1 - e^{-x}}$.
    pub fn sigmoid(&mut self, x: Idx) -> Idx {
        self.register(Node::Activation {
            x,
            a: Activation::Sigmoid,
        })
    }
    /// Registers a Tanh operation.
    pub fn tanh(&mut self, x: Idx) -> Idx {
        self.register(Node::Activation {
            x,
            a: Activation::Tanh,
        })
    }
    /// Registers a matrix multiplication of vectors `v` by matrix `mat`.
    pub fn matmul(&mut self, mat: Idx, v: Idx) -> Idx {
        self.register(Node::MatMul { mat, v })
    }
    /// Registers an embedding later that converts A0 to vector representation
    pub fn embedding(&mut self, emb: Idx, code: Idx) -> Idx {
        self.register(Node::Embedding { emb, code })
    }
}

fn view_at_idxs<'a>(
    indices: &Vec<Idx>,
    nodes: &'a BTreeMap<usize, ArrayD<f32>>,
) -> Box<[ArrayViewD<'a, f32>]> {
    let mut vals = Vec::new();
    for i in indices.iter() {
        vals.push(nodes[&i.idx].view());
    }
    vals.into_boxed_slice()
}
