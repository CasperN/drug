use ndarray::{ArrayD, ArrayViewD};
use nodes::Operation;

/// implements Relu [Operation](trait.Operation.html).
/// See [Graph](../struct.Graph.html) constructor for full description.
#[derive(Debug, Serialize, Deserialize)]
pub struct Relu(pub f32);

impl Operation for Relu {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        inputs[0].mapv(|x| if x > 0.0 { x } else { x * self.0 })
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        let mut res = loss.to_owned();
        res.zip_mut_with(&inputs[0], |l, i| {
            if *i < 0.0 {
                *l *= self.0
            }
        });
        vec![res]
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// implements Sigmoid [Operation](trait.Operation.html)
/// See [Graph](../struct.Graph.html) constructor for full description.
pub struct Sigmoid();

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Operation for Sigmoid {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Sigmoid accepts one input");
        inputs[0].mapv(|x| sig(x))
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        let mut res = loss.to_owned();
        res.zip_mut_with(&inputs[0], |l, i| {
            let s = sig(*i);
            *l *= s * (1.0 - s);
        });
        vec![res]
    }
}

#[derive(Debug, Serialize, Deserialize)]
/// implements  Tanh [Operation](trait.Operation.html)
/// See [Graph](../struct.Graph.html) constructor for full description.
pub struct Tanh();
impl Operation for Tanh {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Tanh accepts 1 input");
        inputs[0].mapv(|x| x.tanh())
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Tanh accepts one input");
        let mut res = loss.to_owned();
        res.zip_mut_with(&inputs[0], |l, i| {
            *l *= 1.0 - i.tanh().powi(2);
        });
        vec![res]
    }
}
