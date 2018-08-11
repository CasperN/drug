use ndarray::{ArrayViewD, ArrayViewMutD};
use std::fmt::Debug;

pub trait Optimizer: Debug {
    fn from_shape(&self, shape: &[usize]) -> Box<Optimizer>;
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>);
}

#[derive(Debug)]
pub struct SGD();

impl Optimizer for SGD {
    fn from_shape(&self, _shape: &[usize]) -> Box<Optimizer> {
        Box::new(SGD())
    }
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, mut param: ArrayViewMutD<f32>) {
        param.zip_mut_with(&loss, |x, y| *x += *y);
    }
}
