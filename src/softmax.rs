use ndarray::prelude::*;
use node::Operation;

#[derive(Debug)]
pub struct Softmax();

impl Operation for Softmax {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 1, "Softmax expects 1 argument");
        let mut res = inputs[0].to_owned().into_dimensionality::<Ix2>().unwrap();

        let max = res.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
        for ((b, _), x) in res.indexed_iter_mut() {
            *x = (*x - max[b]).exp();
        }

        let sum = res.sum_axis(Axis(1));
        for ((b, _), x) in res.indexed_iter_mut() {
            *x /= sum[b];
        }

        res.into_dyn()
    }
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        // OPTIMIZE, TODO shouldn't need to recompute value
        let val = self.eval(inputs).into_dimensionality::<Ix2>().unwrap();
        if let [n_b, n_c] = val.shape() {
            let mut softmax_grad: Array2<f32> = Array2::zeros([*n_b, *n_c]);

            // OPTIMIZE: vectorize?
            for b in 0..*n_b {
                for i in 0..*n_c {
                    for j in 0..*n_c {
                        if i == j {
                            softmax_grad[(b, i)] += val[(b, i)] * (1.0 - val[(b, i)]);
                        } else {
                            softmax_grad[(b, i)] -= val[(b, i)] * val[(b, j)];
                        }
                    }
                }
            }
            vec![(softmax_grad * loss).into_dyn()]
        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;

    #[test]
    fn softmax_eval() {
        let s = Softmax();
        let output = s
            .eval(vec![aview2(&[[1.0, 2.0, 3.0, 4.0, 5.0]]).into_dyn()])
            .into_dimensionality::<Ix2>()
            .unwrap();

        let correct = aview2(&[[
            1.1656230956039607e-2,
            3.168492079612427e-2,
            8.61285444362687e-2,
            0.23412165725273662,
            0.6364086465588308,
        ]]);

        assert!(
            (0..5)
                .into_iter()
                .all(|i| (output[(0, i)] - correct[(0, i)]).abs() < f32::EPSILON),
            "Output and precomputed output not close"
        );
    }
}
