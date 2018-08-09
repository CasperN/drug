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
use ndarray::ArrayD;
use rand::distributions::{Distribution, Normal};
use rand::thread_rng;

pub mod activation;
mod conv;
mod global_pool;
mod graph;
mod node;
mod optimizers;

pub use conv::{Conv, Padding};
pub use graph::{Graph, RuntimeNode};
pub use node::{Node, Operation, Optimizer};

// TODO initializers file
pub fn xavier_initialize(shape: &[usize]) -> ArrayD<f32> {
    let len: usize = shape.iter().product();
    let normal = Normal::new(0.0, 1.0 / len as f64);
    let mut rng = thread_rng();
    ArrayD::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}

#[cfg(test)]
mod tests {
    use graph::Graph;
    #[test]
    fn test_param_initialize() {
        let mut g = Graph::default();
        assert_eq!(0, g.new_param(&[3, 3, 1, 8]));
        assert_eq!(g.nodes[0].value.shape(), [3, 3, 1, 8]);
    }

}
