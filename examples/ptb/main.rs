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

#[allow(unused_variables)]
fn main() {
    let hidden_dim = 100;
    let embedding_dim = 6;
    let learning_rate = 0.1;
    let batch_size = 4;

    let train = TextDataSet::new(batch_size);

    let mut g = Graph::default();
    // Declare weights
    let embedding_weights = g.param(&[train.idx2char.len(), embedding_dim]);
    let forget_weights = g.param(&[hidden_dim + embedding_dim, hidden_dim]);
    let update_weights = g.param(&[hidden_dim + embedding_dim, hidden_dim]);
    let pred_weights = g.param(&[hidden_dim, train.idx2char.len()]);
    let hidden0 = g.param(&[batch_size, hidden_dim]);
    // TODO this is a bit of a hack as we're allowing the hidden state to vary within the batch
    // This is because Append only appends rank 2 matrices

    // Convenience function to add a new GRU node.
    let add_gru_node = |graph: &mut Graph, hidden, words: &ArrayD<f32>| {
        // Append word embedding to previous hidden state
        let code = graph.register(Node::Constant);
        graph.set_value(code, words.to_owned());
        let emb = graph.embedding(embedding_weights, code);
        let appended = graph.op(Append(), &[hidden, emb]);

        // Forget Gate
        let f_matmul = graph.matmul(forget_weights, appended);
        let f_sig = graph.sigmoid(f_matmul);

        // Update Gate
        let u_matmul = graph.matmul(update_weights, appended);
        let u_tanh = graph.tanh(u_matmul);

        // Combine them and get predictions
        let new_hidden = graph.op(ConvexCombine(), &[u_tanh, hidden, f_sig]);
        let predictions = graph.matmul(pred_weights, new_hidden);
        (new_hidden, predictions)
    };

    for line in train.corpus.iter() {
        let mut hidden = hidden0;
        let mut output = vec![];

        // Build GRU-RNN sequence
        for code in line.iter() {
            let (hidden, pred) = add_gru_node(&mut g, hidden, code);
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
        g.backward();
        g.clear_non_parameters();
    }
}
