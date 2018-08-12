use ndarray::prelude::*;
use nodes::Operation;

// TODO enum max pool, avg pool, sum pool, min pool
#[derive(Debug)]
/// Type of pooling operation (only supports average).
/// Implements [Operation](trait.Operation.html).
/// See [Node](enum.Node.html) constructor for full description.
pub enum GlobalPool {
    /// Reduces by taking the arithmetic mean
    Average,
}

#[allow(unused_variables)]
impl Operation for GlobalPool {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "GlobalPool takes one 4d-Array");
        let input = &inputs[0];

        match self {
            // Mean over axis 1 and 2 (but ndarray only supports mean over 1 axis at once)
            // In second mean_axis, axis 1 is original axis 2.
            GlobalPool::Average => input.mean_axis(Axis(1)).mean_axis(Axis(1)).into_dyn(),
        }
    }

    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        if let [n_b, n_i, n_j, n_c] = inputs[0].shape() {
            let res = match self {
                GlobalPool::Average => {
                    let scale = 1.0 / *n_i as f32 / *n_j as f32;
                    Array::from_shape_fn([*n_b, *n_i, *n_j, *n_c], |(b, _, _, c)| {
                        loss[(b, c)] * scale
                    }).into_dyn()
                }
            };
            vec![res]
        } else {
            unreachable!("Global pool grad should take in 2d-array or shape [batch, channels]")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_eval() {
        let x = Array::from_shape_vec([2, 3, 4, 5], (0..120).map(|x| x as f32).collect()).unwrap();
        let g = GlobalPool::Average;
        let avg = g
            .eval(vec![x.view().into_dyn()])
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(
            avg,
            aview2(&[
                [27.5, 28.5, 29.5, 30.5, 31.5],
                [87.5, 88.5, 89.5, 90.5, 91.5]
            ]),
            "\nFailed comparision with `np.array(range(120)).reshape([2,3,4,5]).mean(axis=(1,2))`"
        )
    }
    #[test]
    fn average_grad() {
        let inputs = Array::zeros([2, 3, 4, 5]).into_dyn();
        let losses = Array::ones([2, 5]).into_dyn();
        let g = GlobalPool::Average;
        let grad = g
            .grad(vec![inputs.view().into_dyn()], losses.view())
            .pop()
            .unwrap();
        // .into_dimensionality::<Ix4>()
        // .unwrap();
        assert_eq!(
            grad.into_dimensionality::<Ix4>().unwrap(),
            Array::ones([2, 3, 4, 5]) / 12.0,
            "\nFailed comparision with `np.array(range(120)).reshape([2,3,4,5]).mean(axis=(1,2))`"
        )
    }
}
