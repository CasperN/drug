use ndarray::{ArrayD, ArrayViewD};
use nodes::Operation;

/// Elementwise Relu [Operation](trait.Operation.html).
#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    Relu { leak: f32 },
    Sigmoid,
    Tanh,
}

impl Operation for Activation {
    fn eval(&self, inputs: &[ArrayViewD<f32>]) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Activation accepts one input");
        match self {
            Activation::Relu { leak } => inputs[0].mapv(|x| if x > 0.0 { x } else { x * leak }),
            Activation::Sigmoid => inputs[0].mapv(|x| sig(x)),
            Activation::Tanh => inputs[0].mapv(|x| x.tanh()),
        }
    }
    fn grad(&self, inputs: &[ArrayViewD<f32>], loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 1, "Activation accepts one input");

        let mut res = loss.to_owned();
        match self {
            Activation::Relu { leak } => {
                res.zip_mut_with(&inputs[0], |l, i| {
                    if *i < 0.0 {
                        *l *= leak
                    }
                });
            }
            Activation::Sigmoid => {
                res.zip_mut_with(&inputs[0], |l, i| {
                    let s = sig(*i);
                    *l *= s * (1.0 - s);
                });
            }
            Activation::Tanh => {
                res.zip_mut_with(&inputs[0], |l, i| {
                    *l *= 1.0 - i.tanh().powi(2);
                });
            }
        }
        vec![res]
    }
}
fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
