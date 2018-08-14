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

/// Adds batches of words to the graph by registering constants and passing the coded words through
/// an embedding vector. Despite being categorical, word_batch is a vector of positive integers
/// because the graph only holds ArrayD<f32>
struct Embedding(Idx);
impl Embedding {
    fn new(g: &mut Graph, embedding_len: usize, embedding_dim: usize) -> Self {
        Embedding(g.param(&[embedding_len, embedding_dim]))
    }
    // Add batch to graph and return Idx of its embedding
    fn add_word(&self, g: &mut Graph, word_batch: &ArrayD<f32>) -> Idx {
        let word = g.register(Node::Constant);
        g.set_value(word, word_batch.to_owned());
        g.embedding(self.0, word)
    }
}
struct Predict(Idx);
impl Predict {
    fn new(g: &mut Graph, hidden_dim: usize, pred_len: usize) -> Self {
        Predict(g.param(&[hidden_dim, pred_len]))
    }
    fn predict(&self, g: &mut Graph, hidden: Idx) -> Idx {
        g.matmul(self.0, hidden)
    }
}
/// Adds a Gated Recurrent Units into the graph. All operations made by each instance of the struct
/// share the same parameters.
struct GatedRecurrentUnit {
    hidden0: Idx,
    forgets: Idx,
    updates: Idx,
}
impl GatedRecurrentUnit {
    /// Register the params for one gated recurrent unit
    fn new(g: &mut Graph, batch_size: usize, seq_in_dim: usize, hidden_dim: usize) -> Self {
        GatedRecurrentUnit {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[batch_size, hidden_dim]),
            forgets: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
            updates: g.param(&[hidden_dim + seq_in_dim, hidden_dim]),
        }
    }
    /// Add an instance of the gated recurrent unit
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        let appended = g.op(Append(), &[hidden_in, seq_in]);

        // Forget Gate
        let f_matmul = g.matmul(self.forgets, appended);
        let f_sig = g.sigmoid(f_matmul);

        // Update Gate
        let u_matmul = g.matmul(self.updates, appended);
        let u_tanh = g.tanh(u_matmul);

        // Combine them and get predictions
        let hidden_out = g.op(ConvexCombine(), &[u_tanh, hidden_in, f_sig]);
        hidden_out
    }
}

#[allow(unused_variables)]
fn main() {
    let hidden_dim = 50;
    let embedding_dim = 40;
    let learning_rate = 0.001;
    let batch_size = 8;
    let max_train_steps = 500;
    let summary_every = 1;

    println!("Reading dataset...",);
    let train = TextDataSet::new(batch_size);
    let num_symbols = train.idx2char.len();
    println!("  Number of symbols:   {:?}", num_symbols);
    println!("  Number of sequences: {:?}\n", train.corpus.len());

    let mut g = Graph::default();
    // These structs hold Idx pointing to their parameters
    let embedding = Embedding::new(&mut g, num_symbols, embedding_dim);
    let predict = Predict::new(&mut g, hidden_dim, num_symbols);
    let gru = GatedRecurrentUnit::new(&mut g, batch_size, embedding_dim, hidden_dim);

    println!("Training...");
    for (step, sequence) in train.corpus.iter().enumerate() {
        let mut hidden = gru.hidden0;
        let mut output = vec![];
        let mut total_loss = 0f32;

        // Build GRU-RNN sequence dynamically based on the length of the sequence.
        for word_batch in sequence.iter() {
            let pred = predict.predict(&mut g, hidden);
            let emb = embedding.add_word(&mut g, word_batch);
            let hidden = gru.add_cell(&mut g, hidden, emb);

            output.push((pred, word_batch));
        }
        g.forward();
        for (pred, correct) in output.into_iter() {
            let correct: Vec<usize> = correct.iter().map(|x| *x as usize).collect();

            let (loss, grad) = softmax_cross_entropy_loss(g.get_value(pred), correct.as_slice());
            total_loss += loss;
            g.set_loss(pred, -grad * learning_rate)
        }
        g.backward();
        g.clear_non_parameters();

        if step % summary_every == 0 {
            total_loss /= sequence.len() as f32 * batch_size as f32;
            println!("Step: {:?}   \t Perplexity: {:2.2}", step, total_loss.exp());
        }
    }
}
