use ndarray::{Array4, ArrayD, ArrayViewD, Ix4};
use nodes::Operation;
use std::cell::Cell;

/// implements convolution [Operation](trait.Operation.html).
/// See [Node](enum.Node.html) constructor for full description.
#[derive(Debug)]
pub struct Conv {
    _dialation: usize,
    stride: (usize, usize),
    padding: Padding,
    idx_ranges: Cell<[usize; 7]>,
}

/// Type of padding to use in a convolutional neural network. `No` padding means a non-strided
/// convolution will shrink by the dimensions of the kernel as pixels at the edge will not be the
/// center of a convolution. `Same` padding allows for convolution of edge pixels by assuming
/// the values beyond the images are equal to the edge. Other not implemented padding strategies
/// are "Zero" padding or "Reflection" padding.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum Padding {
    Same,
    No,
}

impl Default for Conv {
    fn default() -> Self {
        Conv {
            _dialation: 1,
            stride: (1, 1),
            padding: Padding::Same,
            idx_ranges: Cell::new([0; 7]),
        }
    }
}

impl Conv {
    #[allow(dead_code)]
    pub fn new(padding: Padding, stride: usize) -> Self {
        Conv {
            _dialation: 1,
            stride: (stride, stride),
            padding,
            idx_ranges: Cell::new([0; 7]),
        }
    }

    fn conv_point(&self, i: usize, j: usize, di: usize, dj: usize) -> Option<(usize, usize)> {
        // Returns the index of the point of the input image image multiplied by the Kernel
        // in the convolution.
        let [_, n_i, n_j, n_di, n_dj, _, _] = self.idx_ranges.get();
        let kernel_offset_i = n_di >> 1;
        let kernel_offset_j = n_dj >> 1;

        match self.padding {
            Padding::Same => {
                // subtract kernel size / 2 to center kernel
                let ci = (i * self.stride.0 + di)
                    .saturating_sub(kernel_offset_i)
                    .min(n_i - 1);
                let cj = (j * self.stride.1 + dj)
                    .saturating_sub(kernel_offset_j)
                    .min(n_j - 1);
                Some((ci, cj))
            }
            Padding::No => {
                // No padding so next image is di (dj) rows (cols) smaller
                if kernel_offset_i <= i && i + n_di < n_i {
                    let ci = i * self.stride.0 + di - kernel_offset_i;
                    if kernel_offset_j <= j && j + n_dj < n_j {
                        let cj = j * self.stride.1 + dj - kernel_offset_j;
                        return Some((ci, cj));
                    }
                }
                None
            }
        }
    }
}

impl Operation for Conv {
    #[allow(unused_mut)]
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert!(
            inputs.len() == 2,
            "Convolution operation takes two arguments"
        );
        let kernel = inputs[0].view().into_dimensionality::<Ix4>().unwrap();
        let image = inputs[1].view().into_dimensionality::<Ix4>().unwrap();

        // Attaching mut to n_i, n_j makes them variables that dont need to be mutable
        // but without it, they are references ==><==
        if let ([n_di, n_dj, n_c0, n_c1], [n_b, n_i, n_j, n_c0_]) = (kernel.shape(), image.shape())
        {
            assert_eq!(
                n_c0_, n_c0,
                "number of channels in image do not match kernel's"
            );
            // Striding shrinks image
            let (out_i, out_j) = match self.padding {
                Padding::Same => (n_i / self.stride.0, n_j / self.stride.1),
                Padding::No => ((n_i - n_di) / self.stride.0, (n_j - n_dj) / self.stride.1),
            };

            self.idx_ranges
                .set([*n_b, *n_i, *n_j, *n_di, *n_dj, *n_c0, *n_c1]);

            let mut output = Array4::zeros([*n_b, out_i, out_j, *n_c1]);

            for (b, i, j, di, dj) in iproduct!(0..*n_b, 0..out_i, 0..out_j, 0..*n_di, 0..*n_dj) {
                if let Some((ci, cj)) = self.conv_point(i, j, di, dj) {
                    for (c0, c1) in iproduct!(0..*n_c0, 0..*n_c1) {
                        output[(b, i, j, c1)] += kernel[(di, dj, c0, c1)] * image[(b, ci, cj, c0)];
                    }
                }
            }
            output.into_dyn()
        } else {
            unreachable!()
        }
    }
    fn grad(&self, inputs: Vec<ArrayViewD<f32>>, loss: ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        assert!(
            inputs.len() == 2,
            "Convolution operation takes two arguments"
        );
        let kernel = inputs[0].view().into_dimensionality::<Ix4>().unwrap();
        let image = inputs[1].view().into_dimensionality::<Ix4>().unwrap();
        let loss = loss.into_dimensionality::<Ix4>().unwrap();

        let [n_b, n_i, n_j, n_di, n_dj, n_c0, n_c1] = self.idx_ranges.get();
        let out_i = n_i / self.stride.0;
        let out_j = n_j / self.stride.1;

        let mut grad_kernel = Array4::zeros([n_di, n_dj, n_c0, n_c1]);
        let mut grad_image = Array4::zeros([n_b, n_i, n_j, n_c0]);

        for (b, i, j, di, dj) in iproduct!(0..n_b, 0..out_i, 0..out_j, 0..n_di, 0..n_dj) {
            if let Some((ci, cj)) = self.conv_point(i, j, di, dj) {
                for (c0, c1) in iproduct!(0..n_c0, 0..n_c1) {
                    grad_kernel[(di, dj, c0, c1)] += loss[(b, i, j, c1)] * image[(b, ci, cj, c0)];
                    grad_image[(b, ci, cj, c0)] += loss[(b, i, j, c1)] * kernel[(di, dj, c0, c1)];
                }
            }
        }

        vec![grad_kernel.into_dyn(), grad_image.into_dyn()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;
    use std::f32;
    use test::Bencher;
    use xavier_initialize;

    #[test]
    fn conv_point_same_padding() {
        let ker = Array4::zeros([3, 3, 1, 1]).into_dyn();
        let img = Array4::zeros([4, 4, 4, 1]).into_dyn();
        let c = Conv::new(Padding::Same, 1);
        c.eval(vec![ker.view(), img.view()]);
        assert_eq!(c.idx_ranges.get(), [4, 4, 4, 3, 3, 1, 1]);
        assert_eq!(
            c.conv_point(0, 0, 0, 0),
            Some((0, 0)),
            "Top left going up and left"
        );
        assert_eq!(
            c.conv_point(0, 3, 2, 2),
            Some((1, 3)),
            "Top right going down and right"
        );
        assert_eq!(
            c.conv_point(2, 2, 1, 1),
            Some((2, 2)),
            "Center going center"
        );
        assert_eq!(
            c.conv_point(3, 3, 0, 0),
            Some((2, 2)),
            "Bottom right going up and left"
        );
        assert_eq!(
            c.conv_point(3, 3, 0, 2),
            Some((2, 3)),
            "Bottom right going down and left"
        );
    }

    // #[test] TODO
    // fn conv_point_no_padding() {
    //     unimplemented!()
    // }

    fn stripe_detector_kernel(horizontal: bool) -> ArrayD<f32> {
        Array4::from_shape_fn([3, 3, 1, 1], move |(row, col, _, _)| {
            if (horizontal && row == 1) || (!horizontal && col == 1) {
                1.0 / 3.0
            } else {
                -1.0 / 6.0
            }
        }).into_dyn()
    }

    fn stripes(horizontal: bool) -> ArrayD<f32> {
        Array4::from_shape_fn(
            [1, 10, 10, 1],
            move |(_, row, col, _)| if horizontal { row % 2 } else { col % 2 } as f32,
        ).into_dyn()
    }

    #[test]
    fn stripe_detectors() {
        for (padding, det, st) in iproduct!(
            [Padding::Same, Padding::No].into_iter(),
            [true, false].into_iter(),
            [true, false].into_iter()
        ) {
            println!("{:?}", (*padding, *det, *st));
            let kernel = stripe_detector_kernel(*det);
            let stripes = stripes(*st);
            let conv = Conv::new(*padding, 1);
            let detections = conv.eval(vec![kernel.view(), stripes.view()]);
            let detections = detections.slice(s!(0, .., .., 0));
            if *det != *st {
                assert!(
                    detections.iter().all(|x| x.abs() < f32::EPSILON),
                    "padding: {:?}; h_detector: {:?}; h_stripes: {:?}; detected orthogonal lines\n{:?}",
                    padding,
                    *det,
                    *st,
                    detections
                );
            } else {
                assert!(
                    detections.iter().any(|x| x.abs() != 0.0),
                    "padding: {:?}; h_detector: {:?}; h_stripes: {:?}; detected nothing\n{:?}",
                    padding,
                    *det,
                    *st,
                    detections
                );
            }
        }
    }

    #[test]
    fn identity_kernel_eval() {
        let identity_kernel = Array4::from_shape_fn([3, 3, 1, 1], |(di, dj, c0, c1)| {
            if di == 1 && dj == 1 && c0 == c1 {
                1.0
            } else {
                0.0
            }
        }).into_dyn();

        let img = stripes(true);
        let conv = Conv::new(Padding::Same, 1);
        let res = conv.eval(vec![identity_kernel.view(), img.view()]);
        let conv = res.slice(s!(0, .., .., 0));
        let orig = img.slice(s!(0, .., .., 0));

        assert_eq!(orig, conv, "Identity Kernel failed\n");
    }

    #[test]
    fn identity_kernel_grad() {
        let identity_kernel = Array4::from_shape_fn([3, 3, 1, 1], |(di, dj, c0, c1)| {
            if di == 1 && dj == 1 && c0 == c1 {
                1.0
            } else {
                0.0
            }
        }).into_dyn();

        let orig = stripes(true);
        let conv = Conv::new(Padding::Same, 1);
        let eval = conv.eval(vec![identity_kernel.view(), orig.view()]);
        let grad = conv.grad(vec![identity_kernel.view(), orig.view()], eval.view());
        assert_eq!(grad.len(), 2);
        let g_img = grad[1].view();
        assert_eq!(g_img, orig.view(), "backwards identity");
    }

    #[test]
    fn minimize_from_positive_image() {
        let mut rng = thread_rng();
        let unif = Uniform::new(1.0, 2.0);
        let conv = Conv::new(Padding::Same, 1);
        let mut kernel = xavier_initialize(&[3, 3, 2, 2]);

        for _ in 0..5 {
            for _ in 0..3 {
                let img = Array4::from_shape_fn([4, 5, 5, 2], |_| unif.sample(&mut rng)).into_dyn();
                conv.eval(vec![kernel.view(), img.view()]);
                let grad = conv.grad(vec![kernel.view(), img.view()], img.view());
                let g_ker = grad[0].view();
                kernel = kernel - g_ker
            }
            assert!(
                kernel.iter().all(|x| *x < 0.0),
                "Kernel failed to learn to be all negative\n{:?}",
                kernel.view()
            )
        }
    }

    #[bench]
    fn eval_3x3x8_kernel_64x64x3_img(b: &mut Bencher) {
        let kernel = xavier_initialize(&[3, 3, 3, 8]);
        let conv = Conv::new(Padding::Same, 1);
        let img = xavier_initialize(&[1, 64, 64, 3]);

        b.iter(|| conv.eval(vec![kernel.view(), img.view()]));
    }
    #[bench]
    fn grad_3x3x8_kernel_64x64x3_img(b: &mut Bencher) {
        let kernel = xavier_initialize(&[3, 3, 3, 8]);
        let conv = Conv::new(Padding::Same, 1);
        let img = xavier_initialize(&[1, 64, 64, 3]);
        let out = conv.eval(vec![kernel.view(), img.view()]);

        b.iter(|| conv.grad(vec![kernel.view(), img.view()], out.view()));
    }

}