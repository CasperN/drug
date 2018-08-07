use ndarray::{ArrayD, ArrayViewD};
use node::Operation;

pub struct Relu {
    leak: f32,
}

impl Operation for Relu {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        inputs[0].mapv(|x| if x > 0.0 { x } else { x * self.leak })
    }
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        let mut res = loss.to_owned();
        res.zip_mut_with(&inputs[0], |l, i| {
            if *i < 0.0 {
                *l *= self.leak
            }
        });
        vec![res]
    }
}

pub struct Sigmoid();

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Operation for Sigmoid {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Sigmoid accepts one input");
        inputs[0].mapv(|x| sig(x))
    }
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Relu accepts one input");
        let mut res = loss.to_owned();
        res.zip_mut_with(&inputs[0], |l, i| {
            let s = sig(*i);
            *l *= s * (1.0 - s);
        });
        vec![res]
    }
}
