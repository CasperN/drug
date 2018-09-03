//! Doc me por favor

use std::f32;
use std::fs::{create_dir_all, File};
use std::io;
use std::io::{Read, Write};
use std::path::Path;

extern crate byteorder;
extern crate drug;
#[macro_use(s)]
extern crate ndarray;
extern crate serde_json;

use drug::*;
use ndarray::prelude::*;
mod input;
use input::{images, labels};

static MODEL_DIR: &'static str = "/tmp/drug/mnist/";
static DATA: &'static str = "examples/data/";
static TR_IMG: &'static str = "train-images-idx3-ubyte";
static TR_LBL: &'static str = "train-labels-idx1-ubyte";
static TS_IMG: &'static str = "t10k-images-idx3-ubyte";
static TS_LBL: &'static str = "t10k-labels-idx1-ubyte";
static TR_LEN: u32 = 60_000;
static TS_LEN: u32 = 10_000;
static ROWS: usize = 28;
static COLS: usize = 28;

fn reshape_and_iter(
    data: Vec<f32>,    // The mnist data read from file
    batch_size: usize, // how many mnist examples to train with in one forward / backward pass
    as_vectors: bool,  // output vectors for a dense network instead of images for convolutional
) -> Box<Iterator<Item = ArrayD<f32>>> {
    let len = data.len() / ROWS / COLS;

    if as_vectors {
        // Iterate over dataset of batched vectors
        let flattened = Array::from_shape_vec([len, ROWS * COLS], data).unwrap();
        let vector_iterator = (0..len / batch_size).map(move |i| {
            let idx = i * batch_size..(i + 1) * batch_size;
            flattened.slice(s!(idx, ..)).to_owned().into_dyn()
        });

        Box::new(vector_iterator)
    } else {
        // Iterate over dataset of batched images
        let images = Array::from_shape_vec([len, ROWS, COLS, 1], data).unwrap();

        let image_iterator = (0..len / batch_size).map(move |i| {
            let idx = i * batch_size..(i + 1) * batch_size;
            images.slice(s!(idx, .., .., ..)).to_owned().into_dyn()
        });

        Box::new(image_iterator)
    }
}

/// Simple 3 layer neural network
fn dense_network(g: &mut Graph, imgs: Idx) -> Idx {
    let weights_1 = g.param(&[ROWS * COLS, 110]);
    let weights_2 = g.param(&[110, 10]);
    let mat_mul_1 = g.matmul(weights_1, imgs);
    let sigmoid_1 = g.sigmoid(mat_mul_1);
    let mat_mul_2 = g.matmul(weights_2, sigmoid_1);
    g.sigmoid(mat_mul_2)
}

/// 3 layer Convolutional neural network
fn conv_network(g: &mut Graph, imgs: Idx) -> Idx {
    let conv_block = |g: &mut Graph, in_idx, in_channels, out_channels| {
        // Repeating block of our cnn
        let kernel = g.param(&[3, 3, in_channels, out_channels]);
        let conv = g.conv(kernel, in_idx, Padding::Same, 1);
        let relu = g.relu(conv);
        relu
    };

    let b1 = conv_block(g, imgs, 1, 8);
    let b2 = conv_block(g, b1, 8, 16);
    let b3 = conv_block(g, b2, 16, 32);

    let kernel_1x1 = g.param(&[1, 1, 32, 10]);
    let conv_1x1 = g.conv(kernel_1x1, b3, Padding::Same, 1);

    g.global_pool(conv_1x1, GlobalPool::Average)
}

/// this is main
fn main() {
    let learning_rate = 0.25;
    let batch_size = 8;
    let train_steps = TR_LEN as usize / batch_size;
    let use_dense = true;
    let summary_every = 500;

    println!("Reading data...",);
    let train_images = images(&Path::new(DATA).join(TR_IMG), TR_LEN);
    let train_labels = labels(&Path::new(DATA).join(TR_LBL), TR_LEN);
    let test_images = images(&Path::new(DATA).join(TS_IMG), TS_LEN);
    let test_labels = labels(&Path::new(DATA).join(TS_LBL), TS_LEN);

    let mut train_images = reshape_and_iter(train_images, batch_size, use_dense);

    let (mut g, imgs, out) = load_model().unwrap_or_else(|_| {
        println!("Building graph...");
        let mut g = Graph::default();
        // FIXME Input Nodes prevent saving
        // let imgs = g.input(train_images);
        let imgs = g.constant(arr0(0.0).into_dyn());

        let out = if use_dense {
            dense_network(&mut g, imgs)
        } else {
            conv_network(&mut g, imgs)
        };

        g.named_idxs.insert("imgs".to_string(), imgs);
        g.named_idxs.insert("out".to_string(), out);
        (g, imgs, out)
    });

    g.optimizer.set_learning_rate(learning_rate);

    println!("{}", g);

    println!("Training...");
    for step in 0..train_steps {
        g.set_value(imgs, train_images.next().unwrap());
        g.forward();

        let labels = &train_labels[step * batch_size..(step + 1) * batch_size];
        let (loss, grad) = softmax_cross_entropy_loss(g.get_value(out), labels);

        g.set_loss(out, -grad);
        g.backward();

        if step % summary_every == 0 {
            println!("  Step: {:?}\t log loss: {:?}", step, loss);
        }
    }

    // old input node exhausted, refresh with test images
    let mut test_images = reshape_and_iter(test_images, batch_size, use_dense);
    // FIXME Input Nodes prevent saving
    // g.replace_input_iterator(imgs, test_images).unwrap();

    let test_steps = TS_LEN as usize / batch_size;
    let mut num_correct = 0;

    println!("Testing...");
    for step in 0..test_steps {
        g.set_value(imgs, test_images.next().unwrap());
        g.forward();
        let labels = &test_labels[step * batch_size..(step + 1) * batch_size];
        num_correct += count_correct(g.get_value(out), labels);
    }
    println!(
        "  Test accuracy: {:?}%",
        100.0 * num_correct as f32 / TS_LEN as f32
    );

    save_model(&g).expect("Error saving");
}

fn save_model(g: &Graph) -> Result<(), io::Error> {
    create_dir_all(MODEL_DIR)?;
    let model_path = Path::new(MODEL_DIR).join("model.json");
    let mut f = File::create(&model_path)?;
    let gs = serde_json::to_string(&g)?;
    f.write_all(gs.as_bytes())?;
    Ok(())
}

fn load_model() -> Result<(Graph, Idx, Idx), io::Error> {
    let model_path = Path::new(MODEL_DIR).join("model.json");
    let mut f = File::open(&model_path)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    let g: Graph = serde_json::from_str(&s).expect("Deserialize error");

    let imgs = *g.named_idxs.get("imgs").unwrap();
    let out = *g.named_idxs.get("out").unwrap();
    println!("Loaded saved model");
    Ok((g, imgs, out))
}

fn count_correct(logits: ArrayViewD<f32>, labels: &[usize]) -> u32 {
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
