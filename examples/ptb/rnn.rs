use drug::*;
use ops::*;
use serde::Serialize;

pub trait RecurrentCell: Serialize {
    /// Constructor.
    fn new(g: &mut Graph, seq_in_dim: usize, hidden_dim: usize) -> Self;
    /// Adds an instance of itself, every instance shares the same parameters
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx;
    /// The index of hidden 0
    fn get_hidden0_idx(&self) -> Idx;
}

/// Holds stacked RecurrentCells in a graph
#[derive(Serialize, Deserialize)]
pub struct RecurrentLayers<T: RecurrentCell> {
    layers: Vec<T>,
}
impl<T: RecurrentCell> RecurrentLayers<T> {
    pub fn new(g: &mut Graph, dimensions: &[usize]) -> RecurrentLayers<T> {
        assert!(
            dimensions.len() > 1,
            "Need to specify at least 1 input and output layer"
        );
        let mut layers = vec![];
        for i in 0..dimensions.len() - 1 {
            layers.push(T::new(g, dimensions[i], dimensions[i + 1]));
        }
        RecurrentLayers { layers }
    }
    pub fn get_hidden0_idxs(&self) -> Vec<Idx> {
        self.layers.iter().map(|l| l.get_hidden0_idx()).collect()
    }
    pub fn add_cells(&self, g: &mut Graph, hiddens: &[Idx], seq_in: Idx) -> Vec<Idx> {
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

/// Basic vanilla RNN
#[derive(Serialize, Deserialize)]
pub struct RNNCell {
    hidden0: Idx,
    weights: Idx,
}

impl RecurrentCell for RNNCell {
    fn new(g: &mut Graph, seq_in_dim: usize, hidden_dim: usize) -> Self {
        RNNCell {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[1, hidden_dim]),
            weights: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
        }
    }
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        let app = g.op(Append(), &[hidden_in, seq_in]);
        let update = g.matmul(self.weights, app);
        g.tanh(update)
    }
    fn get_hidden0_idx(&self) -> Idx {
        self.hidden0
    }
}

/// Gated recurrent unit. Computes a feature vector and reset vector at each step given previous
/// state and input. The new state is a convex combination of the previous state and feature vector.
/// This is mediated by the reset vector.
#[derive(Serialize, Deserialize)]
pub struct GatedRecurrentUnit {
    hidden0: Idx,
    feature: Idx,
    resets: Idx,
}

impl RecurrentCell for GatedRecurrentUnit {
    /// Register the params for one gated recurrent unit
    fn new(g: &mut Graph, seq_in_dim: usize, hidden_dim: usize) -> Self {
        GatedRecurrentUnit {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[1, hidden_dim]),
            feature: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
            resets: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
        }
    }
    /// Add an instance of the gated recurrent unit
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        let app1 = g.op(Append(), &[hidden_in, seq_in]);

        // Extract features Gate
        let f_matmul = g.matmul(self.feature, app1);
        let feature = g.sigmoid(f_matmul);

        // Reset Gate
        let r_matmul = g.matmul(self.resets, app1);
        let reset = g.sigmoid(r_matmul);

        // Combine them and get predictions
        g.op(ConvexCombine(), &[hidden_in, feature, reset])
    }
    fn get_hidden0_idx(&self) -> Idx {
        self.hidden0
    }
}
