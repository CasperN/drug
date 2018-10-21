//! Example dense net classifier with MNIST.
extern crate byteorder;
extern crate drug;
extern crate ndarray;
extern crate ron;
mod input;

use std::f32;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

use drug::*;
use input::{images, labels};
use ndarray::prelude::*;

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

// TODO: Replace with chunker, shuffle and batchify function for multiple epochs.
fn reshape_and_iter(
    data: &[f32],      // The mnist data read from file
    batch_size: usize, // how many mnist examples to train with in one forward / backward pass
    as_vectors: bool,  // output vectors for a dense network instead of images for convolutional
) -> Box<Iterator<Item = ArrayD<f32>>> {
    let shape = if as_vectors {
        vec![batch_size, ROWS * COLS]
    } else {
        vec![batch_size, ROWS, COLS]
    };
    let batched: Vec<ArrayD<f32>> = data
        .chunks_exact(batch_size * ROWS * COLS)
        .map(move |x| Array::from_shape_vec(shape.as_slice(), x.to_vec()).unwrap())
        .collect();

    Box::new(batched.into_iter())
}

/// Simple 3 layer neural network
fn dense_network(g: &mut Graph, imgs: Idx) -> Idx {
    let weights_1 = g.param(&[ROWS * COLS, 110]);
    let weights_2 = g.param(&[110, 10]);
    let mat_mul_1 = g.matmul(weights_1, imgs);
    let sigmoid = g.sigmoid(mat_mul_1);
    g.matmul(weights_2, sigmoid)
}

/// 3 layer Convolutional neural network
fn conv_network(g: &mut Graph, imgs: Idx) -> Idx {
    let conv_block = |g: &mut Graph, in_idx, in_channels, out_channels| {
        // Repeating block of our cnn
        let kernel = g.param(&[3, 3, in_channels, out_channels]);
        let conv = g.conv(kernel, in_idx, Padding::Same, 1);
        g.relu(conv)
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
    let learning_rate = 0.05;
    let batch_size = 8;
    let train_steps = TR_LEN as usize / batch_size;
    let use_dense = true;
    let summary_every = 500;

    println!("Reading data...",);
    let train_images = images(&Path::new(DATA).join(TR_IMG), TR_LEN);
    let train_labels = labels(&Path::new(DATA).join(TR_LBL), TR_LEN);
    let test_images = images(&Path::new(DATA).join(TS_IMG), TS_LEN);
    let test_labels = labels(&Path::new(DATA).join(TS_LBL), TS_LEN);

    let train_images = reshape_and_iter(&train_images, batch_size, use_dense);

    let (mut g, imgs, out) = load_model().unwrap_or_else(|e| {
        println!("Couldn't load graph because `{:?}`", e);
        println!("Building new graph...");

        let mut g = Graph::default();
        let imgs = g.input(None); // Set the iterator later

        let out = if use_dense {
            dense_network(&mut g, imgs)
        } else {
            conv_network(&mut g, imgs)
        };

        // Save input and output idxs for the model
        g.named_idxs.insert("imgs".to_string(), imgs);
        g.named_idxs.insert("out".to_string(), out);
        (g, imgs, out)
    });

    g.optimizer.learning_rate = learning_rate;

    println!("{}", g);

    g.replace_input_iterator(imgs, train_images).unwrap();
    println!("Training...");
    for step in 0..train_steps {
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
    let test_images = reshape_and_iter(&test_images, batch_size, use_dense);
    g.replace_input_iterator(imgs, test_images).unwrap();

    let test_steps = TS_LEN as usize / batch_size;
    let mut num_correct = 0;

    println!("Testing...");
    for step in 0..test_steps {
        g.forward();
        let labels = &test_labels[step * batch_size..(step + 1) * batch_size];
        num_correct += count_correct(&g.get_value(out), labels);
    }
    println!(
        "  Test accuracy: {:?}%",
        100.0 * num_correct as f32 / TS_LEN as f32
    );

    save_model(&g).expect("Error saving");
}

fn save_model(g: &Graph) -> Result<(), Box<std::error::Error>> {
    create_dir_all(MODEL_DIR)?;
    let model_path = Path::new(MODEL_DIR).join("model.bin");
    let mut f = File::create(&model_path)?;
    let gs = ron::ser::to_string(&g)?;
    f.write_all(gs.as_bytes())?;
    Ok(())
}

fn load_model() -> Result<(Graph, Idx, Idx), Box<std::error::Error>> {
    let model_path = Path::new(MODEL_DIR).join("model.bin");
    let f = File::open(&model_path)?;
    let g: Graph = ron::de::from_reader(&f)?;
    let imgs = *g
        .named_idxs
        .get("imgs")
        .expect("Expected named index `imgs`.");
    let out = *g
        .named_idxs
        .get("out")
        .expect("Expected named index `out`.");
    println!("Loaded saved model from {:?}", model_path);
    Ok((g, imgs, out))
}

fn count_correct(logits: &ArrayD<f32>, labels: &[usize]) -> u32 {
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
//
