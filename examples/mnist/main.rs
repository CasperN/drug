use std::f32;

extern crate byteorder;
extern crate diff;
#[macro_use(s)]
extern crate ndarray;

use diff::*;
use ndarray::prelude::*;
use std::path::Path;
#[allow(dead_code)]
mod input;
use input::{images, labels};

static DATA: &'static str = "examples/data/";
static TR_IMG: &'static str = "train-images-idx3-ubyte";
static TR_LBL: &'static str = "train-labels-idx1-ubyte";
static TS_IMG: &'static str = "t10k-images-idx3-ubyte";
static TS_LBL: &'static str = "t10k-labels-idx1-ubyte";
static TR_LEN: u32 = 60_000;
static TS_LEN: u32 = 10_000;
static ROWS: usize = 28;
static COLS: usize = 28;

fn into_dataset(
    images: Vec<f32>,
    len: usize,
    batch_size: usize,
) -> Box<Iterator<Item = ArrayD<f32>>> {
    let images = Array::from_shape_vec([len, ROWS * COLS], images).unwrap();
    let data = (0..len / batch_size).map(move |i| {
        let idx = i * batch_size..(i + 1) * batch_size;
        images.slice(s!(idx, ..)).to_owned().into_dyn()
    });
    Box::new(data)
}

fn main() {
    let learning_rate = 0.25;
    let batch_size = 8;
    let train_steps = TR_LEN as usize / batch_size;

    println!("Reading data...",);
    let train_images = images(&Path::new(DATA).join(TR_IMG), TR_LEN);
    let train_labels = labels(&Path::new(DATA).join(TR_LBL), TR_LEN);
    let test_images = images(&Path::new(DATA).join(TS_IMG), TS_LEN);
    let test_labels = labels(&Path::new(DATA).join(TS_LBL), TS_LEN);

    // Convert Mnist data into ndarrays
    let train_images = into_dataset(train_images, TR_LEN as usize, batch_size);
    let test_images = into_dataset(test_images, TS_LEN as usize, batch_size);

    println!("Building graph...");
    let mut g = Graph::default();

    let imgs = g.register(Node::input(train_images));
    let weights_1 = g.new_param(&[784, 110]);
    let weights_2 = g.new_param(&[110, 10]);
    let mat_mul_1 = g.register(Node::mat_mul(weights_1, imgs));
    let sigmoid_1 = g.register(Node::sigmoid(mat_mul_1));
    let mat_mul_2 = g.register(Node::mat_mul(weights_2, sigmoid_1));
    let sigmoid_2 = g.register(Node::sigmoid(mat_mul_2));
    let out = sigmoid_2;

    println!("Training...");
    for step in 0..train_steps {
        g.forward();

        let labels = &train_labels[step * batch_size..(step + 1) * batch_size];
        let (loss, grad) = softmax_cross_entropy_loss(g.nodes[out].value.view(), labels);

        g.nodes[out].loss = -grad * learning_rate;
        g.backward();

        if step % 500 == 0 {
            println!("  Step: {:?}\t log loss: {:?}", step, loss);
        }
    }

    // old input node exhausted, refresh with test images
    g.nodes[imgs] = Node::input(test_images).into();
    let test_steps = TS_LEN as usize / batch_size;
    let mut num_correct = 0;
    println!("Testing...");
    for step in 0..test_steps {
        g.forward();
        let labels = &test_labels[step * batch_size..(step + 1) * batch_size];
        num_correct += count_correct(g.nodes[out].value.view(), labels);
    }
    println!(
        "Test accuracy: {:?}%",
        100.0 * num_correct as f32 / TS_LEN as f32
    );
}

fn count_correct(logits: ArrayViewD<f32>, labels: &[u8]) -> u32 {
    let logits = logits.to_owned().into_dimensionality::<Ix2>().unwrap();
    let batch_size = labels.len();
    let mut num_correct = 0;
    for b in 0..batch_size {
        let mut max = f32::MIN;
        let mut max_idx = 0;
        for i in 0..10 {
            if logits[(b, i)] > max {
                max = logits[(b, i)];
                max_idx = i;
            }
        }
        if max_idx == labels[b] as usize {
            num_correct += 1;
        }
    }
    num_correct
}
