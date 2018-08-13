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

#[allow(dead_code)]
struct TextDataSet {
    char2idx: HashMap<char, usize>,
    idx2char: Vec<char>,
    corpus: Vec<Vec<ArrayD<f32>>>,
}
impl TextDataSet {
    fn new(batch_size: usize) -> Self {
        let mut contents = String::new();
        let mut f = File::open(Path::new(DATA_DIR).join(TRAIN)).expect("train data not found");
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        let mut coded_lines = Vec::new();
        let mut char2idx = HashMap::new();
        let mut idx2char = Vec::new();

        for str_line in contents.lines() {
            let mut line = Vec::new();

            for c in str_line.chars() {
                if char2idx.contains_key(&c) {
                    let idx = char2idx.get(&c).unwrap();
                    line.push(*idx);
                } else {
                    let new_idx = idx2char.len();
                    char2idx.insert(c, new_idx);
                    idx2char.push(c);
                    line.push(new_idx);
                }
            }
            coded_lines.push(line);
        }
        coded_lines.sort_by(|a, b| a.len().cmp(&b.len()));

        let corpus = coded_lines
            .chunks(batch_size)
            .map(|chunk| {
                let batch_len = chunk.len();
                let sequence_len = chunk.last().unwrap().len();
                (0..batch_len)
                    .map(|b| {
                        Array::from_shape_fn([sequence_len], |s| {
                            if s < chunk[b].len() {
                                chunk[b][s] as f32
                            } else {
                                char2idx[&' '] as f32
                            }
                        }).into_dyn()
                    })
                    .collect()
            })
            .collect();

        TextDataSet {
            corpus,
            char2idx,
            idx2char,
        }
    }
}
#[derive(Debug)]
struct ConvexCombine();
#[allow(unused_mut)] // silly compiler
impl nodes::Operation for ConvexCombine {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let mut x = inputs[0]
            .to_owned()
            .into_dimensionality::<Ix2>()
            .expect("Append x dim");
        let y = inputs[1]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append y dim");
        let a = inputs[2]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append a dim");

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
// TODO call yasmeen 21:06

#[derive(Debug)]
struct Append();
#[allow(unused_variables)]
impl nodes::Operation for Append {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        // TODO this is failing because we are appending onto hidden0 which does not have a
        // batch dimension
        let x = inputs[0]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append x dim error");
        let y = inputs[1]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append y dim error");
        let batch_size = x.shape()[0];
        assert_eq!(batch_size, y.shape()[0]);
        let x_len = x.shape()[1];
        let y_len = y.shape()[1];

        Array::from_shape_fn([batch_size, x_len + y_len], |(b, i)| {
            if i < x_len {
                x[(b, i)]
            } else {
                y[(b, i - x_len)]
            }
        }).into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        let batch_size = x.shape()[0];
        assert_eq!(batch_size, y.shape()[0]);
        assert_eq!(batch_size, loss.shape()[0]);
        let (xlen, ylen) = (x.shape()[1], y.shape()[1]);
        let x_grad = Array::from_shape_fn([batch_size, xlen], |(b, xi)| loss[(b, xi)]);
        let y_grad = Array::from_shape_fn([batch_size, ylen], |(b, yi)| loss[(b, yi + xlen)]);
        vec![x_grad.into_dyn(), y_grad.into_dyn()]
    }
}

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
    let hidden0 = g.param(&[hidden_dim]);

    // Convenience function to add a new GRU node.
    // TODO sort by length of sequence, batching, padding if need be,
    // TODO matmul requires a batch dimension
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
