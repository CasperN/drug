#![feature(exact_chunks)]
#[macro_use]
extern crate ndarray;
extern crate drug;
extern crate rand;

use drug::*;
use ndarray::prelude::*;

mod ops;
mod text_dataset;
use ops::{Append, ConvexCombine};
use text_dataset::TextDataSet;

/// Single layer gated recurrent unit builder. Also includes embedding, prediction, and hidden0.
/// All Nodes built by this builder share the same weights.
struct GatedRecurrentUnit {
    embedding: Idx,
    hidden0: Idx,
    forgets: Idx,
    updates: Idx,
    prediction: Idx,
}
impl GatedRecurrentUnit {
    /// Register self.{weights} with the graph
    fn new(
        g: &mut Graph,
        batch_size: usize,
        embedding_len: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        pred_len: usize,
    ) -> Self {
        GatedRecurrentUnit {
            embedding: g.param(&[embedding_len, embedding_dim]),
            forgets: g.param(&[hidden_dim + embedding_dim, hidden_dim]),
            updates: g.param(&[hidden_dim + embedding_dim, hidden_dim]),
            prediction: g.param(&[hidden_dim, pred_len]),
            hidden0: g.param(&[batch_size, hidden_dim]),
            // TODO this is a bit of a hack as we're allowing the hidden state to vary within the
            // batch. This is because Append only appends rank 2 matrices. Ideally hidden0 weights
            // should be identical within accross the batch.
        }
    }
    /// Add a copy of the GRU into the graph.
    /// Given an input `hidden` state and next element of the sequence, `words`,
    /// output a `new_hidden` state and `predictions` of the following element of the sequence.
    fn add_cell(&self, g: &mut Graph, hidden: Idx, words: &ArrayD<f32>) -> (Idx, Idx) {
        // Use constant node as placeholder for current element of sequence
        let code = g.register(Node::Constant);
        g.set_value(code, words.to_owned());

        // Translate element into an embedding vector
        let emb = g.embedding(self.embedding, code);
        let appended = g.op(Append(), &[hidden, emb]);

        // Forget Gate
        let f_matmul = g.matmul(self.forgets, appended);
        let f_sig = g.sigmoid(f_matmul);

        // Update Gate
        let u_matmul = g.matmul(self.updates, appended);
        let u_tanh = g.tanh(u_matmul);

        // Combine them and get predictions
        let new_hidden = g.op(ConvexCombine(), &[u_tanh, hidden, f_sig]);
        let predictions = g.matmul(self.prediction, new_hidden);
        (new_hidden, predictions)
    }
}

#[allow(unused_variables)]
fn main() {
    let hidden_dim = 100;
    let embedding_dim = 6;
    let learning_rate = 0.001;
    let batch_size = 16;
    let max_train_steps = 100;

    let train = TextDataSet::new(batch_size);

    let mut g = Graph::default();
    let gru_cell = GatedRecurrentUnit::new(
        &mut g,
        batch_size,
        train.idx2char.len(), // embedding length -- number of possible input characters
        embedding_dim,
        hidden_dim,
        train.idx2char.len(), // prediction length -- number of possible output characters
    );

    for (step, line) in train.corpus[0..max_train_steps].iter().enumerate() {
        let mut hidden = gru_cell.hidden0;
        let mut output = vec![];

        // Build GRU-RNN sequence
        for code in line.iter() {
            let (hidden, pred) = gru_cell.add_cell(&mut g, hidden, code);
            output.push((pred, code));
        }
        g.forward();
        let mut total_loss = 0.0;
        for (pred, correct) in output.into_iter() {
            let correct: Vec<usize> = correct.iter().map(|x| *x as usize).collect();

            let (loss, grad) = softmax_cross_entropy_loss(g.get_value(pred), correct.as_slice());
            total_loss += loss;
            g.set_loss(pred, -grad * learning_rate)
        }
        total_loss /= line.len() as f32 * batch_size as f32;
        println!(
            "Step {:?} / {:?}\t Perplexity {:?}",
            step,
            train.corpus.len(),
            total_loss.exp2()
        );
        g.backward();
        g.clear_non_parameters();
    }
}
