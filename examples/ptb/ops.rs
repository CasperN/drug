use drug::*;
use ndarray::prelude::*;

#[derive(Debug)]
pub struct ConvexCombine();
#[allow(unused_mut)] // silly compiler
impl nodes::Operation for ConvexCombine {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let mut x = inputs[0]
            .to_owned()
            .into_dimensionality::<Ix2>()
            .expect("Append x dim");
        let y = inputs[1]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append y dim");
        let a = inputs[2]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append a dim");

        azip!(mut x, a, y in { *x = a * *x + (1.0 - a) * y});
        x.into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let mut x = inputs[0].to_owned().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let a = inputs[2].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();

        let batch_size = a.shape()[0];
        let num_channels = a.shape()[1];

        let mut a_grad = Array::zeros([batch_size, num_channels]);
        let mut x_grad = Array::zeros([batch_size, num_channels]);
        let mut y_grad = Array::zeros([batch_size, num_channels]);

        for b in 0..batch_size {
            for c in 0..num_channels {
                let ai = a[(b, c)];
                let xi = x[(b, c)];
                let yi = y[(b, c)];
                let li = loss[(b, c)];
                a_grad[(b, c)] += li * (xi - yi);
                x_grad[(b, c)] += ai * li;
                y_grad[(b, c)] += li * (1.0 - ai);
            }
        }
        vec![x_grad.into_dyn(), y_grad.into_dyn(), a_grad.into_dyn()]
    }
}

#[derive(Debug)]
pub struct Append();
#[allow(unused_variables)]
impl nodes::Operation for Append {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        // TODO this is failing because we are appending onto hidden0 which does not have a
        // batch dimension
        let x = inputs[0]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append x dim error");
        let y = inputs[1]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append y dim error");
        let batch_size = x.shape()[0];
        assert_eq!(
            batch_size,
            y.shape()[0],
            "Append: `x` and `y` batch sizes do not align."
        );

        let x_len = x.shape()[1];
        let y_len = y.shape()[1];

        Array::from_shape_fn([batch_size, x_len + y_len], |(b, i)| {
            if i < x_len {
                x[(b, i)]
            } else {
                y[(b, i - x_len)]
            }
        }).into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        let batch_size = x.shape()[0];
        assert_eq!(batch_size, y.shape()[0]);
        assert_eq!(batch_size, loss.shape()[0]);
        let (xlen, ylen) = (x.shape()[1], y.shape()[1]);
        let x_grad = Array::from_shape_fn([batch_size, xlen], |(b, xi)| loss[(b, xi)]);
        let y_grad = Array::from_shape_fn([batch_size, ylen], |(b, yi)| loss[(b, yi + xlen)]);
        vec![x_grad.into_dyn(), y_grad.into_dyn()]
    }
}
