use drug::*;
use ops::*;

/// All operations made by each instance of this trait share the same parameters.
/// This enables the same weights to be shared by a whole sequence.
pub trait RecurrentCell {
    fn new(g: &mut Graph, batch_size: usize, seq_in_dim: usize, hidden_dim: usize) -> Self;
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx;
    fn get_hidden0_idx(&self) -> Idx;
}

/// Holds stacked RecurrentCells in a graph
pub struct RecurrentLayers<T: RecurrentCell> {
    layers: Vec<T>,
}
impl<T: RecurrentCell> RecurrentLayers<T> {
    pub fn new(g: &mut Graph, batch_size: usize, dimensions: Vec<usize>) -> RecurrentLayers<T> {
        assert!(
            dimensions.len() > 1,
            "Need to specify at least 1 input and output layer"
        );
        let mut layers = vec![];
        for i in 0..dimensions.len() - 1 {
            layers.push(T::new(g, batch_size, dimensions[i], dimensions[i + 1]));
        }
        RecurrentLayers { layers }
    }
    pub fn get_hidden0_idxs(&self) -> Vec<Idx> {
        self.layers.iter().map(|l| l.get_hidden0_idx()).collect()
    }
    pub fn add_cells(&self, g: &mut Graph, hiddens: Vec<Idx>, seq_in: Idx) -> Vec<Idx> {
        assert_eq!(self.layers.len(), hiddens.len());
        let mut h = seq_in;
        let mut new_hiddens = vec![];
        for (l, hid) in self.layers.iter().zip(hiddens.iter()) {
            h = l.add_cell(g, *hid, h);
            new_hiddens.push(h)
        }
        new_hiddens
    }
}

pub struct RNNCell {
    hidden0: Idx,
    weights: Idx,
}

impl RecurrentCell for RNNCell {
    fn new(g: &mut Graph, batch_size: usize, seq_in_dim: usize, hidden_dim: usize) -> Self {
        RNNCell {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[batch_size, hidden_dim]),
            weights: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
        }
    }
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        let app = g.op(Append(), &[hidden_in, seq_in]);
        let update = g.matmul(self.weights, app);
        let hidden_out = g.tanh(update);
        hidden_out
    }
    fn get_hidden0_idx(&self) -> Idx {
        self.hidden0
    }
}

pub struct GatedRecurrentUnit {
    hidden0: Idx,
    forgets: Idx,
    resets: Idx,
    updates: Idx,
}
/// This is broken
impl RecurrentCell for GatedRecurrentUnit {
    /// Register the params for one gated recurrent unit
    fn new(g: &mut Graph, batch_size: usize, seq_in_dim: usize, hidden_dim: usize) -> Self {
        GatedRecurrentUnit {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[batch_size, hidden_dim]),
            forgets: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
            updates: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
            resets: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
        }
    }
    /// Add an instance of the gated recurrent unit
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        let app1 = g.op(Append(), &[hidden_in, seq_in]);

        // Forget Gate
        let f_matmul = g.matmul(self.forgets, app1);
        let forget = g.sigmoid(f_matmul);

        // Reset Gate
        let r_matmul = g.matmul(self.resets, app1);
        let reset = g.sigmoid(r_matmul);

        // New hidden
        let filtered = g.mult(&[forget, hidden_in]);
        let app2 = g.op(Append(), &[filtered, seq_in]);
        let u_matmul = g.matmul(self.updates, app2);
        let update = g.tanh(u_matmul);

        // Combine them and get predictions
        let hidden_out = g.op(ConvexCombine(), &[hidden_in, update, reset]);
        hidden_out
    }
    fn get_hidden0_idx(&self) -> Idx {
        self.hidden0
    }
}
