//! # ∂rug - Differentiable Rust Graph
//!
//! This crate is a collection of utilities to build build neural networks (differentiable
//! programs). See [examples source code](https://github.com/CasperN/drug/tree/master/examples)
//! for implementations of canonical neural networks. You may need to download those datasets
//! yourself to use them. Examples include:
//! * Mnist with dense networks
//! * Mnist with convolutional neural networks (though embarassingly slowly)
//! * Penn TreeBank character prediction with RNN and GRU
//!
//! ### Planned Changes
//! * Saving / loading
//!     * Naming and indexing via string
//! * Building complexes of nodes (conv + bias + relu) / RNN cells, with parameter reuse
//! * Subgraphs / updating subsets of graphs (e.g. for GAN)
//! * Parallel backprop multiple arguments of 1 node
//! * ndarray-parallel usage
//!
//! Reinforcement learning applications may also challenge the archiecture but I don't understand
//! the process well enough yet to consider adding it to the library.
//!
//! ### Wish list
//! * GPU integration (awaiting advancements in rust gp-gpu)
//! * Operator overloading API + Taking advantage of the type system and const generics
//!     * May require total overhaul.. or may be possible with a "Graph Cursor" trait and more
//!     sophisticaed handles beyond current Idxs
//! * Automatic differentiation of operations defined only from loops (proc macros?)
//! * Distributed training
//! * Other kinds of derivatives e.g. jacobian

#![feature(test)]
#[macro_use]
pub extern crate ndarray;
extern crate rand;
extern crate test;
#[macro_use]
extern crate debug_stub_derive;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate erased_serde;
extern crate serde;

#[cfg(test)]
#[macro_use(iproduct)]
extern crate itertools;

// pub use ndarray;
use ndarray::prelude::*;
use rand::distributions::{Distribution, Normal};
use rand::thread_rng;

mod graph;
pub mod nodes;
pub use nodes::{GlobalPool, Node, Padding};
pub mod optimizers;
pub use graph::*;

// TODO initializers file
/// The default (and only provided) initializer. Only works with convolution kernels and matrices.
pub fn xavier_initialize(shape: &[usize]) -> ArrayD<f32> {
    // let len: usize = shape.iter().product();
    let (n_in, n_out) = match shape.len() {
        4 => (shape[2], shape[3]), // Convolution kernel
        2 => (shape[0], shape[1]), // Matrix
        1 => (shape[0], shape[0]), // Vector
        x => unimplemented!("Initialize with {:?}", x),
    };
    let var = 2.0 / (n_in as f64 + n_out as f64);
    let normal = Normal::new(0.0, var.sqrt());
    let mut rng = thread_rng();
    ArrayD::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}

/// Take the softmax of an array of shape `batch_size * num_classes`
pub fn softmax(logits: ArrayViewD<f32>) -> Array2<f32> {
    let mut softmax = logits.to_owned().into_dimensionality::<Ix2>().unwrap();
    // Calculate softmax
    let max = softmax.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x = (*x - max[b]).exp();
    }
    let sum = softmax.sum_axis(Axis(1));
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x /= sum[b];
    }
    softmax
}

/// A loss function used for classification.
///
/// `logits` are a `batch_size * num_classes` array of values which will be compressed into the
/// `[0,1]` range by a softmax operation. Given the correct categories `labels`, this function will
/// calculate the negative log-probability of the logits and its gradient with respect to the logits.
pub fn softmax_cross_entropy_loss(logits: ArrayViewD<f32>, labels: &[usize]) -> (f32, ArrayD<f32>) {
    let mut softmax = softmax(logits);
    let mut log_loss = 0.0;
    // Turn softmax into gradient and add up log_loss
    for (b, lbl) in labels.iter().enumerate() {
        let correct = *lbl;
        log_loss -= softmax[(b, correct)].ln();
        softmax[(b, correct)] -= 1.0;
    }
    (log_loss, softmax.into_dyn())
}

#[cfg(test)]
mod libc {
    use super::*;
    use graph::Graph;
    use std::f32;
    #[test]
    fn param_initialize() {
        let mut g = Graph::default();
        let x = g.param(&[3, 3, 1, 8]);
        assert_eq!(g.get_value(x).shape(), [3, 3, 1, 8]);
    }
    #[test]
    fn softmax_vs_correct() {
        let logits = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let correct = arr2(&[
            [
                9.003057317038046e-2,
                0.24472847105479767,
                0.6652409557748219,
            ],
            [
                9.003057317038045e-2,
                0.24472847105479764,
                0.6652409557748219,
            ],
        ]);
        let softmax = softmax(logits.view().into_dyn());
        for i in 0..2 {
            for j in 0..3 {
                assert!((softmax[(i, j)] - correct[(i, j)]).abs() < f32::EPSILON);
            }
        }
    }
}
