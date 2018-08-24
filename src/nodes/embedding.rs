use ndarray::prelude::*;
use nodes::Operation;

/// Pulls out an embedding vector given the index and
#[derive(Debug, Serialize, Deserialize)]
pub struct Embedding();

impl Operation for Embedding {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 2, "Embedding operation takes two inputs");
        let embedding = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let code = inputs[1].view().into_dimensionality::<Ix1>().unwrap();
        let batch_size = code.shape()[0];
        let embedding_dim = embedding.shape()[1];

        Array::from_shape_fn([batch_size, embedding_dim], |(b, d)| {
            let x = code[(b)] as usize;
            embedding[(x, d)]
        }).into_dyn()
    }

    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 2, "Embedding operation takes two inputs");
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        let embedding = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let code = inputs[1].view().into_dimensionality::<Ix1>().unwrap();
        let batch_size = code.shape()[0];
        let num_embeddings = embedding.shape()[0];
        let embedding_dim = embedding.shape()[1];

        let mut grad = Array::zeros([num_embeddings, embedding_dim]);
        for b in 0..batch_size {
            let code = code[(b)] as usize;
            for d in 0..embedding_dim {
                grad[(code, d)] += loss[(b, d)]
            }
        }
        vec![grad.into_dyn(), Array::zeros([]).into_dyn()]
    }
}
