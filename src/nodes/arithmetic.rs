use ndarray::prelude::*;
use nodes::Operation;

#[derive(Debug, Serialize, Deserialize)]
/// Elementwise addition operation
pub struct Add();

#[derive(Debug, Serialize, Deserialize)]
/// Elementwise multiplication operation
pub struct Mult();

impl Operation for Add {
    fn eval(&self, inputs: &[ArrayViewD<f32>]) -> ArrayD<f32> {
        let mut res = inputs[0].to_owned();
        for i in inputs {
            res += i;
        }
        res
    }
    fn grad(&self, inputs: &[ArrayViewD<f32>], _loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        inputs.into_iter().map(|i| i.to_owned()).collect()
    }
}

impl Operation for Mult {
    fn eval(&self, inputs: &[ArrayViewD<f32>]) -> ArrayD<f32> {
        let mut res = inputs[0].to_owned();
        for i in inputs {
            res *= i;
        }
        res
    }
    fn grad(&self, inputs: &[ArrayViewD<f32>], _loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 2);
        inputs.iter().rev().map(|v| v.to_owned()).collect()
    }
}
