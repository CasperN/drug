use ndarray::prelude::*;
use nodes::Operation;

/// Pulls out an embedding vector given the index and
#[derive(Debug)]
pub struct Embedding();

impl Operation for Embedding {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 2, "Embedding operation takes two inputs");
        let embedding = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let code = inputs[1].view().into_dimensionality::<Ix0>().unwrap();
        let code = code[()] as usize;

        embedding.slice(s!(code, ..)).to_owned().into_dyn()
    }

    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 2, "Embedding operation takes two inputs");
        let loss = loss.into_dimensionality::<Ix1>().unwrap();
        let embedding = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let code = inputs[1].view().into_dimensionality::<Ix0>().unwrap();
        let code = code[()] as usize;
        let num_codes = embedding.shape()[0];
        let code_len = embedding.shape()[1];

        let grad = Array::from_shape_fn([num_codes, code_len], |(c, i)| {
            if c == code {
                loss[i]
            } else {
                0.0
            }
        }).into_dyn();
        vec![Array::zeros([]).into_dyn(), grad]
    }
}
