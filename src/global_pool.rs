use ndarray::{ArrayD, ArrayViewD};
use node::Operation;

// TODO enum max pool, avg pool, sum pool, min pool
#[derive(Debug)]
pub enum GlobalPool {
    Average,
}

#[allow(unused_variables)]
impl Operation for GlobalPool {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        unimplemented!()
    }

    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        unimplemented!()
    }
}
