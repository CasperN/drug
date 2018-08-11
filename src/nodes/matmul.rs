use ndarray::prelude::*;
use nodes::Operation;

/// implements matrix multiply [Operation](trait.Operation.html).
/// See [Node](enum.Node.html) constructor for full description.
#[derive(Debug)]
pub struct MatMul();

impl Operation for MatMul {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 2);
        let weights = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let neurons = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let n_b = neurons.shape()[0];
        let n_i = weights.shape()[0];
        let n_j = weights.shape()[1];
        assert_eq!(n_i, neurons.shape()[1]);

        Array::from_shape_fn([n_b, n_j], |(b, j)| {
            let mut x = 0.0;
            for i in 0..n_i {
                x += weights[(i, j)] * neurons[(b, i)]
            }
            x
        }).into_dyn()
    }
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 2);
        let weights = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let neurons = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        let n_b = neurons.shape()[0];
        let n_i = weights.shape()[0];
        let n_j = weights.shape()[1];
        assert_eq!(n_i, neurons.shape()[1]);
        assert_eq!(n_b, loss.shape()[0]);
        assert_eq!(n_j, loss.shape()[1]);

        let grad_weights = Array::from_shape_fn([n_i, n_j], |(i, j)| {
            let mut x = 0.0;
            for b in 0..n_b {
                x += neurons[(b, i)] * loss[(b, j)];
            }
            x
        }).into_dyn();
        let grad_neurons = Array::from_shape_fn([n_b, n_i], |(b, i)| {
            let mut x = 0.0;
            for j in 0..n_j {
                x += weights[(i, j)] * loss[(b, j)];
            }
            x
        }).into_dyn();
        vec![grad_weights, grad_neurons]
    }
}
