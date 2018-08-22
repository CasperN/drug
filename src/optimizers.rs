use ndarray::{ArrayViewD, ArrayViewMutD};
use std::fmt::Debug;
use Idx;

pub trait Optimizer: Debug {
    fn register(&mut self, idx: Idx, shape: &[usize]);
    fn set_learning_rate(&mut self, learning_rate: f32);
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>);
    // TODO Optimizers should know index of where its applying the gradient so it can
    // keep the meta data
}

#[derive(Debug)]
pub struct SGD {
    learning_rate: f32,
}
impl SGD {
    pub fn new_boxed() -> Box<Self> {
        Box::new(SGD {
            learning_rate: 0.01,
        })
    }
}

impl Optimizer for SGD {
    fn register(&mut self, _idx: Idx, _shape: &[usize]) {}

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, mut param: ArrayViewMutD<f32>) {
        param.zip_mut_with(&loss, |x, y| *x += self.learning_rate * *y);
    }
}
