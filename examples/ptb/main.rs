use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

extern crate diff;
use diff::ndarray::prelude::*;
use diff::*;

static DATA_DIR: &'static str = "examples/data/";
static TRAIN: &'static str = "ptb.train.txt";
use std::io::Read;

#[derive(Default)]
struct TextDataSet {
    char2idx: HashMap<char, usize>,
    idx2char: Vec<char>,
    train_corpus: Vec<Vec<usize>>,
    // test_corpus: Vec<Vec<usize>,
}
impl TextDataSet {
    fn new() -> Self {
        let mut contents = String::new();
        let mut f = File::open(Path::new(DATA_DIR).join(TRAIN)).expect("train data not found");
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        let mut corpus = TextDataSet::default();

        for line in contents.lines() {
            for c in line.chars() {
                let mut line = Vec::new();
                if corpus.char2idx.contains_key(&c) {
                    let idx = corpus.char2idx.get(&c).unwrap();
                    line.push(*idx);
                } else {
                    let new_idx = corpus.idx2char.len();
                    corpus.char2idx.insert(c, new_idx);
                    corpus.idx2char.push(c);
                    line.push(new_idx);
                }
                corpus.train_corpus.push(line);
            }
        }
        corpus
    }
}
#[derive(Debug)]
struct ConvexCombine();
#[allow(unused_variables)]
impl nodes::Operation for ConvexCombine {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        unimplemented!()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        unimplemented!()
    }
}

#[derive(Debug)]
struct Stack();
#[allow(unused_variables)]
impl nodes::Operation for Stack {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        unimplemented!()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        unimplemented!()
    }
}

#[allow(unused_variables)]
fn main() {
    let hidden_dim = 100;
    let embedding_dim = 6;

    let t = TextDataSet::new();

    println!("{:?}", t.train_corpus.len());
    println!("{:?}", t.char2idx);
    println!("{:?}", t.idx2char);

    let mut g = Graph::default();
    // Declare weights
    let embedding_weights = g.param(&[t.idx2char.len(), embedding_dim]);
    let forget_weights = g.param(&[hidden_dim + embedding_dim, hidden_dim]);
    let update_weights = g.param(&[hidden_dim + embedding_dim, hidden_dim]);
    let pred_weights = g.param(&[hidden_dim, t.idx2char.len()]);
    let hidden0 = g.param(&[hidden_dim]);

    // Convenience function to add a new GRU node.
    let add_gru_node = |graph: &mut Graph, hidden, word| {
        let stacked = graph.op(Stack(), &[hidden, word]);

        let f_matmul = graph.matmul(forget_weights, stacked);
        let f_sig = graph.sigmoid(f_matmul);

        let u_matmul = graph.matmul(update_weights, stacked);
        let u_tanh = graph.tanh(u_matmul);

        let new_hidden = graph.op(ConvexCombine(), &[u_tanh, hidden, f_sig]);
        let predictions = graph.matmul(pred_weights, new_hidden);
        (new_hidden, predictions)
    };

    for line in t.train_corpus.iter() {
        for c in line.iter() {
            // Construct
            unimplemented!()
        }
        g.forward();
        for c in line.iter() {
            // Apply losses
            unimplemented!()
        }
        g.backward();
        // Remove everything but parameters
        unimplemented!()
    }
}
