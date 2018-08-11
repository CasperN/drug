#![feature(test)]
#[allow(unused_imports)]
#[macro_use(s)] // s! is used in tests
pub extern crate ndarray;
extern crate rand;
extern crate test;
#[macro_use]
extern crate debug_stub_derive;

#[macro_use(iproduct)]
extern crate itertools;

// pub use ndarray;
use ndarray::prelude::*;
use rand::distributions::{Distribution, Normal};
use rand::thread_rng;

pub mod activation;

mod conv;
mod global_pool;
mod graph;
mod matmul;
mod node;
mod optimizers;

pub use conv::{Conv, Padding};
pub use global_pool::GlobalPool;
pub use graph::*;
pub use node::{Node, Operation, Optimizer};

// TODO initializers file
pub fn xavier_initialize(shape: &[usize]) -> ArrayD<f32> {
    let len: usize = shape.iter().product();
    let normal = Normal::new(0.0, 1.0 / len as f64);
    let mut rng = thread_rng();
    ArrayD::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}

// TODO losses file
pub fn softmax_cross_entropy_loss(logits: ArrayViewD<f32>, labels: &[u8]) -> (f32, ArrayD<f32>) {
    let mut softmax = logits.to_owned().into_dimensionality::<Ix2>().unwrap();
    let mut log_loss = 0.0;
    // Calculate softmax
    let max = softmax.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x = (*x - max[b]).exp();
    }
    let sum = softmax.sum_axis(Axis(1));
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x /= sum[b];
    }
    // Turn softmax into gradient and add up log_loss
    for (b, lbl) in labels.iter().enumerate() {
        let correct = *lbl as usize;
        log_loss -= softmax[(b, correct)].ln();
        softmax[(b, correct)] -= 1.0;
    }
    (log_loss, softmax.into_dyn())
}

#[cfg(test)]
mod tests {
    use graph::Graph;
    #[test]
    fn test_param_initialize() {
        let mut g = Graph::default();
        assert_eq!(0, g.new_param(&[3, 3, 1, 8]));
        assert_eq!(g.values[0].shape(), [3, 3, 1, 8]);
    }
}
