use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

extern crate drug;
#[macro_use]
extern crate ndarray;
use drug::*;
use ndarray::prelude::*;

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
#[allow(unused_mut)] // silly compiler
impl nodes::Operation for ConvexCombine {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let mut x = inputs[0].to_owned().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let a = inputs[2].view().into_dimensionality::<Ix2>().unwrap();

        azip!(mut x, a, y in { *x = a * *x + (1.0 - a) * y});
        x.into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let mut x = inputs[0].to_owned().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let a = inputs[2].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();

        let batch_size = a.shape()[0];
        let num_channels = a.shape()[1];

        let mut a_grad = Array::zeros([batch_size, num_channels]);
        let mut x_grad = Array::zeros([batch_size, num_channels]);
        let mut y_grad = Array::zeros([batch_size, num_channels]);

        for b in 0..batch_size {
            for c in 0..num_channels {
                let ai = a[(b, c)];
                let xi = x[(b, c)];
                let yi = y[(b, c)];
                let li = loss[(b, c)];
                a_grad[(b, c)] += li * (xi - yi);
                x_grad[(b, c)] += ai * li;
                y_grad[(b, c)] += li * (1.0 - ai);
            }
        }
        vec![x_grad.into_dyn(), y_grad.into_dyn(), a_grad.into_dyn()]
    }
}

#[derive(Debug)]
struct Stack();
#[allow(unused_variables)]
impl nodes::Operation for Stack {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        let x = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let batch_size = x.shape()[0];
        assert_eq!(batch_size, y.shape()[0]);
        let x_c = x.shape()[1];
        let y_c = y.shape()[1];

        Array::from_shape_fn([batch_size, x_c + y_c], |(b, i)| {
            if i < x_c {
                x[(b, i)]
            } else {
                y[(b, i - x_c)]
            }
        }).into_dyn()
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
    // TODO matmul requires a batch dimension
    let add_gru_node = |graph: &mut Graph, hidden, word: usize| {
        // Stack word embedding on top of previous hidden state
        let code = graph.register(Node::Constant);
        graph.set_value(code, arr0(word as f32).into_dyn());
        let emb = graph.embedding(embedding_weights, code);
        let stacked = graph.op(Stack(), &[hidden, emb]);
        // Forget Gate
        let f_matmul = graph.matmul(forget_weights, stacked);
        let f_sig = graph.sigmoid(f_matmul);
        // Update Gate
        let u_matmul = graph.matmul(update_weights, stacked);
        let u_tanh = graph.tanh(u_matmul);
        // Combine them and get predictions
        let new_hidden = graph.op(ConvexCombine(), &[u_tanh, hidden, f_sig]);
        let predictions = graph.matmul(pred_weights, new_hidden);
        (new_hidden, predictions)
    };

    for line in t.train_corpus.iter() {
        let mut hidden = hidden0;
        let mut predictions = vec![];
        for code in line.iter() {
            let (hidden, pred) = add_gru_node(&mut g, hidden, *code);
            predictions.push(pred);
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
