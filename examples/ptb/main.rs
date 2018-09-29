#![feature(exact_chunks)]
#[macro_use]
extern crate ndarray;
extern crate drug;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate ron;
extern crate serde;

use drug::*;
use ndarray::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
mod beam_search;
mod ops;
mod rnn;
mod text_dataset;

use beam_search::BeamSearch;
use rnn::*;
use text_dataset::TextDataSet;

static MODEL_DIR: &'static str = "/tmp/drug/ptb/";

/// Adds batches of words to the graph by registering constants and passing the coded words through
/// an embedding vector. Despite being categorical, word_batch is a vector of positive integers
/// because the graph only holds ArrayD<f32>
#[derive(Serialize, Deserialize)]
struct Embedding(Idx);
impl Embedding {
    fn new(g: &mut Graph, embedding_len: usize, embedding_dim: usize) -> Self {
        Embedding(g.param(&[embedding_len, embedding_dim]))
    }
    // Add batch to graph and return Idx of its embedding
    fn add_word(&self, g: &mut Graph, word_batch: ArrayViewD<f32>) -> Idx {
        let word = g.constant(word_batch.to_owned());
        g.embedding(self.0, word)
    }
}

#[derive(Serialize, Deserialize)]
struct Predict(Idx);
impl Predict {
    fn new(g: &mut Graph, hidden_dim: usize, pred_len: usize) -> Self {
        Predict(g.param(&[hidden_dim, pred_len]))
    }
    fn predict(&self, g: &mut Graph, hidden: Idx) -> Idx {
        g.matmul(self.0, hidden)
    }
}

fn save_model<T: RecurrentCell>(
    g: &Graph,
    r: &RecurrentLayers<T>,
) -> Result<(), Box<std::error::Error>> {
    create_dir_all(MODEL_DIR)?;
    let model_path = Path::new(MODEL_DIR).join("model.json");
    let mut f = File::create(&model_path)?;
    let gs = ron::ser::to_string(&g)?;
    f.write_all(gs.as_bytes())?;

    let rl_path = Path::new(MODEL_DIR).join("rl.json");
    let mut f = File::create(&rl_path)?;
    let rs = ron::ser::to_string(&r)?;
    f.write_all(rs.as_bytes())?;

    Ok(())
}

fn load_model<T: RecurrentCell>(
) -> Result<(Graph, Embedding, Predict, RecurrentLayers<T>), Box<std::error::Error>>
where
    T: serde::de::DeserializeOwned,
{
    let model_path = Path::new(MODEL_DIR).join("model.json");
    let f = File::open(&model_path)?;
    let g: Graph = ron::de::from_reader(f)?;

    let rl_path = Path::new(MODEL_DIR).join("rl.json");
    let f = File::open(&rl_path)?;
    let rl: RecurrentLayers<T> = ron::de::from_reader(&f)?;

    let emb_idx = *g
        .named_idxs
        .get("embedding")
        .expect("Expected named index `embedding`");
    let embedding = Embedding(emb_idx);

    let prd_idx = *g
        .named_idxs
        .get("predict")
        .expect("Expected named index `predict`");
    let predict = Predict(prd_idx);

    println!("Loaded saved model");
    Ok((g, embedding, predict, rl))
}

/// Architecture            Epoch 5 Train Perplexity
/// --------------------    ------------------------
/// GRU [30, 30, 30]        5.35
/// GRU [30, 30, 30, 30]    5.09
/// GRU [50, 50, 50]        4.69
/// GRU [50, 100, 100]      4.16
/// GRU [50, 250, 250]      3.86 - 3.74 (10 epochs)
fn main() {
    // dimensions[0] is embedding dimension, the rest are size of hidden dim in each layer
    let dimensions = vec![50, 50, 50];
    let batch_size = 16;
    let sequence_len = 50;
    // Note the effective learning_rate is this * batch_size * sequence_len
    let learning_rate = 0.01 as f32;
    let summary_every = 250;
    let num_epochs = 1;

    println!("Reading dataset...",);
    let train = TextDataSet::new(batch_size, sequence_len);
    let num_symbols = train.idx2char.len();
    println!("  Batch size {:?}", batch_size);
    println!("  Sequence len {:?}", sequence_len);
    println!("  Number of symbols:   {:?}", num_symbols);
    println!("  Number of sequences: {:?}\n", train.corpus.len());

    let (mut g, embedding, predict, rnn) = load_model().unwrap_or_else(|_| {
        println!("Defining new model");
        let mut g = Graph::default();
        g.optimizer.learning_rate = learning_rate;

        // These structs hold Idx pointing to their parameters and have methods adding operations to
        // the graph.
        let embedding = Embedding::new(&mut g, num_symbols, dimensions[0]);
        let predict = Predict::new(&mut g, *dimensions.last().unwrap(), num_symbols);
        let rnn = RecurrentLayers::<GatedRecurrentUnit>::new(&mut g, batch_size, dimensions);

        g.named_idxs.insert("embedding".to_string(), embedding.0);
        g.named_idxs.insert("predict".to_string(), predict.0);
        (g, embedding, predict, rnn)
    });

    println!("Training...");
    let mut total_loss = 0.0;
    let mut seen = 0;
    for epoch in 0..num_epochs {
        for (step, sequence) in train.corpus.iter().enumerate() {
            let mut hiddens = rnn.get_hidden0_idxs();
            let mut output = vec![];

            // Build RNN sequence dynamically based on the length of the sequence.
            for word_batch in sequence.iter() {
                let pred = predict.predict(&mut g, *hiddens.last().unwrap());
                let emb = embedding.add_word(&mut g, word_batch.view());
                hiddens = rnn.add_cells(&mut g, hiddens, emb);

                output.push((pred, word_batch));
            }
            g.forward();
            // Check 1 step predictions and compute loss
            for (pred, correct) in output.into_iter() {
                let correct: Vec<usize> = correct.iter().map(|x| *x as usize).collect();

                let (loss, grad) =
                    softmax_cross_entropy_loss(g.get_value(pred), correct.as_slice());
                total_loss += loss;
                g.set_loss(pred, -grad)
            }
            g.backward();
            g.clear_non_parameters();
            seen += sequence.len();

            if step % summary_every == 0 {
                total_loss /= seen as f32 * batch_size as f32;
                println!(
                    "Epoch: {:?} of {:?}\t Step: {:5} of {:?}\t Perplexity: {:2.2}",
                    epoch,
                    num_epochs,
                    step,
                    train.corpus.len(),
                    total_loss.exp()
                );
                total_loss = 0.0;
                seen = 0;
            }
        }
    }

    save_model(&g, &rnn).expect("Saving Error");

    // BUG forward pass will fail if beam width > num characters
    let beam_width = 30;
    let gen_len = 80;

    for temp in [1.0, 0.9, 0.8, 0.7].into_iter() {
        println!("\nGenerating with temp {:?}...", temp);

        let mut beam_search = BeamSearch::new(beam_width);
        let mut hiddens = vec![];

        for h in rnn.get_hidden0_idxs().iter() {
            let mean_h0 = g.get_value(*h).mean_axis(Axis(0));
            let h_dim = mean_h0.shape()[0];
            let hidden =
                Array::from_shape_fn([beam_width, h_dim], |(_b, h)| mean_h0[(h)]).into_dyn();
            hiddens.push(hidden);
        }

        for _ in 0..gen_len {
            // predict next characters based on hidden state
            let h = g.constant(hiddens.last().unwrap().to_owned());
            let pred_idx = predict.predict(&mut g, h);
            g.forward1(pred_idx);
            let next_word_logits = g.get_value(pred_idx).to_owned();

            // Consider next hidden state and words based on probability of sequence
            let (hiddens, words) = beam_search.search(&hiddens, next_word_logits.view(), *temp);
            let mut hidden_idxs = vec![];
            for h in hiddens.into_iter() {
                hidden_idxs.push(g.constant(h));
            }
            // Update hidden state
            let emb = embedding.add_word(&mut g, words.view());
            let hidden_idxs = rnn.add_cells(&mut g, hidden_idxs, emb);
            // g.forward();

            // Take it out of the graph
            let mut hiddens = vec![];
            for hi in hidden_idxs.into_iter() {
                hiddens.push(g.get_value(hi).to_owned());
            }
            g.clear_non_parameters();
        }

        let res = beam_search.into_codes();
        for s in res.iter() {
            println!("{:?}", train.decode(s));
        }
    }
}
