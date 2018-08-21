#![feature(exact_chunks)]
#[macro_use]
extern crate ndarray;
extern crate drug;
extern crate rand;

use drug::*;
use ndarray::prelude::*;
// use std::cmp::Ordering;
#[allow(dead_code, unused_variables, unused_imports)]
mod beam_search;
mod ops;
mod text_dataset;

use beam_search::BeamSearch;
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
    fn add_word(&self, g: &mut Graph, word_batch: ArrayViewD<f32>) -> Idx {
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
        let app1 = g.op(Append(), &[hidden_in, seq_in]);

        // Forget Gate
        let f_matmul = g.matmul(self.forgets, app1);
        let forget = g.sigmoid(f_matmul);

        // Update Gate
        let filtered = g.mult(&[forget, hidden_in]);
        let app2 = g.op(Append(), &[filtered, seq_in]);
        let u_matmul = g.matmul(self.updates, app2);
        let update = g.tanh(u_matmul);

        // Combine them and get predictions
        let hidden_out = g.op(ConvexCombine(), &[update, hidden_in, forget]);
        hidden_out
    }
}

/// Stacked `GatedRecurrentUnit`s
struct GruLayers {
    layers: Vec<GatedRecurrentUnit>,
}
impl GruLayers {
    /// dimensions is a vector at least length 2 specifying input dimension and all hidden dimensions
    fn new(g: &mut Graph, batch_size: usize, dimensions: Vec<usize>) -> Self {
        assert!(
            dimensions.len() > 1,
            "Need to specify at least 1 input and output layer"
        );
        let mut layers = vec![];
        for i in 0..dimensions.len() - 1 {
            layers.push(GatedRecurrentUnit::new(
                g,
                batch_size,
                dimensions[i],
                dimensions[i + 1],
            ));
        }
        GruLayers { layers }
    }
    fn get_hidden0_idxs(&self) -> Vec<Idx> {
        self.layers.iter().map(|l| l.hidden0).collect()
    }
    fn add_cells(&self, g: &mut Graph, hiddens: Vec<Idx>, seq_in: Idx) -> Vec<Idx> {
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

#[allow(unused_variables, unused_assignments)] // silly compiler
fn main() {
    // dimensions[0] is embedding dimension
    let dimensions = vec![50, 60, 60];
    let batch_size = 16;
    // Note the effective learning_rate is this * batch_size * sequence_len
    let learning_rate = 0.0001 as f32;
    let summary_every = 25;
    let num_epochs = 3;

    println!("Reading dataset...",);
    let train = TextDataSet::new(batch_size);
    let num_symbols = train.idx2char.len();
    println!("  Number of symbols:   {:?}", num_symbols);
    println!("  Number of sequences: {:?}\n", train.corpus.len());

    let mut g = Graph::default();
    // These structs hold Idx pointing to their parameters
    let embedding = Embedding::new(&mut g, num_symbols, dimensions[0]);
    let predict = Predict::new(&mut g, *dimensions.last().unwrap(), num_symbols);
    let gru = GruLayers::new(&mut g, batch_size, dimensions);

    println!("Training...");
    for epoch in 0..num_epochs {
        g.optimizer.set_learning_rate(learning_rate * (0.5f32).powi(epoch));

        for (step, sequence) in train.corpus.iter().enumerate() {
            let mut hiddens = gru.get_hidden0_idxs();
            let mut output = vec![];
            let mut total_loss = 0f32;

            // Build GRU-RNN sequence dynamically based on the length of the sequence.
            for word_batch in sequence.iter() {
                let pred = predict.predict(&mut g, *hiddens.last().unwrap());
                let emb = embedding.add_word(&mut g, word_batch.view());
                hiddens = gru.add_cells(&mut g, hiddens, emb);

                output.push((pred, word_batch));
            }
            g.forward();
            for (pred, correct) in output.into_iter() {
                let correct: Vec<usize> = correct.iter().map(|x| *x as usize).collect();

                let (loss, grad) =
                    softmax_cross_entropy_loss(g.get_value(pred), correct.as_slice());
                total_loss += loss;
                g.set_loss(pred, -grad)
            }
            g.backward();
            g.clear_non_parameters();

            if step % summary_every == 0 {
                total_loss /= sequence.len() as f32 * batch_size as f32;
                println!(
                    "Epoch: {:?} of {:?}\t Step: {:4} of {:?}\t Perplexity: {:2.2}",
                    epoch,
                    num_epochs,
                    step,
                    train.corpus.len(),
                    total_loss.exp()
                );
            }
        }
    }
    println!("Generating...");
    let beam_width = 25;
    let gen_len = 50;
    let mut beam_search = BeamSearch::new(beam_width);

    let mut hiddens = vec![];
    for h in gru.get_hidden0_idxs().iter() {
        let mean_h0 = g.get_value(*h).mean_axis(Axis(0));
        let h_dim = mean_h0.shape()[0];
        let hidden = Array::from_shape_fn([beam_width, h_dim], |(b, _h)| mean_h0[(b)]).into_dyn();
        hiddens.push(hidden);
    }

    for _ in 0..gen_len {
        // predict next characters based on hidden state
        let mut old_hidden_idxs = vec![];
        for h in hiddens.iter() {
            old_hidden_idxs.push(g.constant(h.to_owned()));
        }

        let pred_i = predict.predict(&mut g, *old_hidden_idxs.last().unwrap());
        g.forward1(pred_i);

        // Consider next hidden state and words based on probability of sequence
        let (mut hiddens, words) = beam_search.search(&hiddens, g.get_value(pred_i));
        let emb_i = embedding.add_word(&mut g, words.view());

        // Propagate hidden state
        let hidden_i = gru.add_cells(&mut g, old_hidden_idxs, emb_i);
        g.forward();

        for (i, idx) in gru.get_hidden0_idxs().iter().enumerate() {
            hiddens[i] = g.get_value(*idx).to_owned();
        }

        g.clear_non_parameters();
    }

    let res = beam_search.into_codes();
    for s in res.into_iter() {
        println!("{:?}", train.decode(s));
    }
}
