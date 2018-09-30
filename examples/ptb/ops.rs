use drug::*;
use ndarray::prelude::*;

#[derive(Debug, Serialize, Deserialize)]
/// Operation that does [x, y, a] -> a * x + (1 - a) * y. This is used in gated recurrent units
/// forget gate.
pub struct ConvexCombine();

#[derive(Debug, Serialize, Deserialize)]
/// Operation that takes two batches of vectos xs, ys and appends ys below xs. Supports
/// broadcasting if the batch dimension of xs or ys is 1.
pub struct Append();

impl nodes::Operation for ConvexCombine {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");

        let y = inputs[1].to_owned();
        let a = inputs[2].to_owned();
        let mut x = inputs[0]
            .broadcast(y.shape())
            .expect("ConvexCombine: Broadcast Failed")
            .to_owned();

        azip!(mut x, a, y in { *x = a * *x + (1.0 - a) * y});
        x.into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert_eq!(inputs.len(), 3, "Convex combine takes 3 arguments x, y, a");
        let x = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let a = inputs[2].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();

        if x.shape() == y.shape() && x.shape() == a.shape() {}
        let x_bs = x.shape()[0];
        let y_bs = y.shape()[0];
        let a_bs = a.shape()[0];
        let num_channels = a.shape()[1];

        let mut a_grad = Array::zeros([a_bs, num_channels]);
        let mut x_grad = Array::zeros([x_bs, num_channels]);
        let mut y_grad = Array::zeros([y_bs, num_channels]);

        for b in 0..a_bs.max(x_bs).max(y_bs) {
            for c in 0..num_channels {
                // TODO make this prettier
                let ab = if a_bs == 1 { 0 } else { b };
                let xb = if x_bs == 1 { 0 } else { b };
                let yb = if y_bs == 1 { 0 } else { b };
                let ai = a[(ab, c)];
                let xi = x[(xb, c)];
                let yi = y[(yb, c)];
                let li = loss[(b, c)];
                a_grad[(b, c)] += li * (xi - yi);
                x_grad[(xb, c)] += li * ai;
                y_grad[(yb, c)] += li * (1.0 - ai);
            }
        }
        vec![x_grad.into_dyn(), y_grad.into_dyn(), a_grad.into_dyn()]
    }
}

impl nodes::Operation for Append {
    fn eval(&self, inputs: Box<[ArrayViewD<f32>]>) -> ArrayD<f32> {
        let x = inputs[0]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append x dim error");
        let y = inputs[1]
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Append y dim error");
        let x_bn = x.shape()[0];
        let y_bn = y.shape()[0];
        assert!(
            x_bn == y_bn || y_bn == 1 || x_bn == 1,
            "`Append::eval`: `x` and `y` batch sizes do not align and neither is 1."
        );

        let x_len = x.shape()[1];
        let y_len = y.shape()[1];

        Array::from_shape_fn([x_bn.max(y_bn), x_len + y_len], |(b, i)| {
            if i < x_len && x_bn == 1 {
                x[(0, i)]
            } else if i < x_len {
                x[(b, i)]
            } else if y_bn == 1 {
                y[(0, i - x_len)]
            } else {
                y[(b, i - x_len)]
            }
        }).into_dyn()
    }
    fn grad(&self, inputs: Box<[ArrayViewD<f32>]>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let x = inputs[0].view().into_dimensionality::<Ix2>().unwrap();
        let y = inputs[1].view().into_dimensionality::<Ix2>().unwrap();
        let loss = loss.into_dimensionality::<Ix2>().unwrap();
        let x_bn = x.shape()[0];
        let y_bn = y.shape()[0];
        assert!(
            x_bn == y_bn || y_bn == 1 || x_bn == 1,
            "`Append::grad`: `x` and `y` batch sizes do not align and neither is 1."
        );
        let (x_len, y_len) = (x.shape()[1], y.shape()[1]);

        let x_grad = if x_bn == 1 {
            loss.sum_axis(Axis(0))
                .slice_move(s![..x_len])
                .insert_axis(Axis(0))
        } else {
            Array::from_shape_fn([x_bn, x_len], |(b, xi)| loss[(b, xi)])
        };

        let y_grad = if y_bn == 1 {
            loss.sum_axis(Axis(0))
                .slice_move(s![x_len..])
                .insert_axis(Axis(0))
        } else {
            Array::from_shape_fn([y_bn, y_len], |(b, yi)| loss[(b, yi + x_len)])
        };

        vec![x_grad.into_dyn(), y_grad.into_dyn()]
    }
}
