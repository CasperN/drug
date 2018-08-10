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
// static TS_IMG: &'static str = "t10k-images-idx3-ubyte";
// static TS_LBL: &'static str = "t10k-labels-idx1-ubyte";
static TR_LEN: u32 = 60_000;
// static TS_LEN: u32 = 10_000;
// static ROWS: usize = 28;
// static COLS: usize = 28;

fn main() {
    println!("Reading data...",);
    let tr_img = images(&Path::new(DATA).join(TR_IMG), TR_LEN);
    let tr_lbl = labels(&Path::new(DATA).join(TR_LBL), TR_LEN);
    // let ts_img = images(&Path::new(DATA).join(TS_IMG), TS_LEN);
    // let _ts_lbl = labels(&Path::new(DATA).join(TS_LBL), TS_LEN);
    println!("Data read...");

    let batch_size = 32;

    // Convert Mnist data into ndarrays
    let tr_img = Array::from_shape_vec([TR_LEN as usize, 28, 28, 1], tr_img).unwrap();
    let data = (0..1000).map(move |i| {
        let idx = i * batch_size..(i + 1) * batch_size;
        tr_img.slice(s!(idx, .., .., ..)).to_owned().into_dyn()
    });

    // Build graph
    println!("Building graph...");
    let mut g = Graph::default();
    let imgs = g.register(Node::input(Box::new(data.into_iter())));

    let k1 = g.new_param(&[5, 5, 1, 8]);
    let conv1 = g.register(Node::conv(k1, imgs, Padding::Same, 2));
    let relu1 = g.register(Node::relu(conv1));

    let k2 = g.new_param(&[5, 5, 8, 16]);
    let conv2 = g.register(Node::conv(k2, relu1, Padding::Same, 2));
    let relu2 = g.register(Node::relu(conv2));

    // let k3 = g.new_param(&[3, 3, 16, 32]);
    // let conv3 = g.register(Node::conv(k3, relu2, Padding::Same, 2));
    // let relu3 = g.register(Node::relu(conv3));

    let k4 = g.new_param(&[1, 1, 16, 10]);
    let conv4 = g.register(Node::conv(k4, relu2, Padding::Same, 1));
    let avgp = g.register(Node::global_pool(conv4, GlobalPool::Average));

    let softmax = g.register(Node::softmax(avgp));

    print!("Showing gradients:\nStep");
    for i in 0..g.nodes.len() {
        print!("\tNode {:?}     ", i);
    }
    println!("\n");

    for step in 0..1000 {
        print!("{:?}", step);
        g.forward();

        let labels = &tr_lbl[step * batch_size..(step + 1) * batch_size];

        let losses = cross_entropy_loss(g.nodes[softmax].value.view(), labels);
        let _loss = losses.scalar_sum();

        g.nodes[softmax].loss = losses;
        g.backward();

        println!(
            "{}",
            g.nodes
                .iter()
                .map(|n| format!("\t{:.2e}   ", n.loss.mapv(|x| x * x).scalar_sum()))
                .collect::<String>()
        );
    }
}

fn cross_entropy_loss(softmax: ArrayViewD<f32>, lbls: &[u8]) -> ArrayD<f32> {
    let softmax = softmax.into_dimensionality::<Ix2>().unwrap();
    let mut grad = Array::zeros(softmax.shape());
    for (b, lbl) in lbls.iter().enumerate() {
        let correct = *lbl as usize;
        for i in 0..10 {
            let idx = Dim([b, i]);
            grad[idx] += if i == correct {
                1.0 / (1.0 - softmax[idx])
            } else {
                1.0 / softmax[idx]
            };
        }
    }
    grad.into_dyn()
}
