use ndarray::prelude::*;
use nodes::Operation;

#[derive(Debug)]
/// Elementwise addition operation
pub struct Add();

#[derive(Debug)]
/// Elementwise multiplication operation
pub struct Mult();


#[allow(unused_variables)]
impl Operation for Add {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32>{
        let mut res = inputs[0].to_owned();
        for i in 1..inputs.len() {
            res = res + inputs[i].view();
        }
        res
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        inputs.into_iter().map(|i| i.to_owned()).collect()
    }
}


#[allow(unused_variables)]
impl Operation for Mult {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32>{
        let mut res = inputs[0].to_owned();
        for i in 1..inputs.len() {
            res = res * inputs[i].view();
        }
        res
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let mut total_prod = inputs[0].to_owned();
        for i in 1..inputs.len() {
            total_prod = total_prod * inputs[i].view();
        }
        inputs
            .into_iter()
            .map(|v| {
                let mut v = v.to_owned();
                v.zip_mut_with(&total_prod, |v, tp| {
                    *v = tp / *v;
                });
                v
            })
            .collect()
    }
}
