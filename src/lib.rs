extern crate ndarray;
extern crate rand;
#[macro_use]
extern crate itertools;

use ndarray::ArrayD;

use rand::distributions::{Distribution, Normal};
use rand::thread_rng;

mod conv;
pub mod graph;
pub mod node;
mod optimizers;

pub fn xavier_initialize(shape: &[usize]) -> ArrayD<f32> {
    let len: usize = shape.iter().product();
    let normal = Normal::new(0.0, 1.0 / len as f64);
    let mut rng = thread_rng();
    ArrayD::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::Graph;
    use optimizers::SGD;

    #[test]
    fn test_param_initialize() {
        let mut g = Graph::new(Box::new(xavier_initialize), Box::new(SGD()));
        assert_eq!(0, g.new_param(&[3, 3, 1, 8]));
        assert_eq!(g.nodes[0].value.shape(), [3, 3, 1, 8]);
    }

}
