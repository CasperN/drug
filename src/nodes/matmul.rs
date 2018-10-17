use ndarray::prelude::*;
use nodes::Operation;

/// implements matrix multiply [Operation](trait.Operation.html).
/// See [Node](enum.Node.html) constructor for full description.
#[derive(Debug, Serialize, Deserialize)]
pub struct MatMul();

impl Operation for MatMul {
    fn eval(&self, inputs: &[ArrayViewD<f32>]) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 2);
        let weights = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let neurons = inputs[1].view().into_dimensionality::<Ix2>().unwrap();

        neurons.dot(&weights).into_dyn()
    }
    fn grad(&self, inputs: &[ArrayViewD<f32>], loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 2);
        let weights = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let neurons = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();

        let grad_weights = neurons.t().dot(&loss).into_dyn();
        let grad_neurons = loss.dot(&weights.t()).into_dyn();
        vec![grad_weights, grad_neurons]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use xavier_initialize;

    #[test]
    fn sample_eval() {
        let weights = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).into_dyn();
        let vecs = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        let m = MatMul();

        let o = m.eval(vec![weights.view(), vecs.view()].into_boxed_slice());
        assert_eq!(
            o,
            arr2(&[[11.0, 14.0, 17.0, 20.0], [23.0, 30.0, 37.0, 44.0]]).into_dyn()
        )
    }
    #[bench]
    fn bench_matmul_eval(b: &mut Bencher) {
        let weights = xavier_initialize(&[100, 150]);
        let vecs = xavier_initialize(&[8, 100]);
        let m = MatMul();
        b.iter(|| m.eval(vec![weights.view(), vecs.view()].into_boxed_slice()));
    }
    #[bench]
    fn bench_matmul_grad(b: &mut Bencher) {
        let weights = xavier_initialize(&[100, 150]);
        let vecs = xavier_initialize(&[8, 100]);
        let m = MatMul();
        let o = m.eval(vec![weights.view(), vecs.view()].into_boxed_slice());
        b.iter(|| {
            m.grad(
                vec![weights.view(), vecs.view()].into_boxed_slice(),
                o.view(),
            )
        });
    }
}
