use ndarray::{ArrayViewD, ArrayViewMutD, Zip};
use node::Optimizer;

pub struct SGD();

impl Optimizer for SGD {
    fn apply_gradient(&mut self, loss: ArrayViewD<f32>, param: ArrayViewMutD<f32>) {
        Zip::from(param).and(loss).apply(|x, y| *x = *x + y);
    }
}
