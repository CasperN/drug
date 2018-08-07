use ndarray::{Array4, ArrayD, ArrayViewD, Ix4};
use node::Operation;
use std::cell::Cell;

pub struct Conv {
    _dialation: usize,
    _stride: usize,
    padding: Padding,
    idx_ranges: Cell<[usize; 7]>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum Padding {
    Same,
    No,
}

impl Conv {
    #[allow(dead_code)]
    pub fn new(padding: Padding) -> Self {
        Conv {
            _dialation: 1,
            _stride: 1,
            padding,
            idx_ranges: Cell::new([0; 7]),
        }
    }

    fn iter_idxs(&self) -> impl Iterator<Item = [usize; 7]> {
        // Returns a boxed iterator to iterate over the indices b/c impl Iter is not good enough
        // makes for nicer code but consider profiling -- may need to OPTIMIZE
        // TODO refactor this into the others

        let [n_b, n_i, n_j, n_di, n_dj, n_c0, n_c1] = self.idx_ranges.get();

        iproduct!(0..n_b, 0..n_i, 0..n_j, 0..n_di, 0..n_dj, 0..n_c0, 0..n_c1)
            .map(|(b, i, j, di, dj, c0, c1)| [b, i, j, di, dj, c0, c1])
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
                let ci = (i + di)
                    .checked_sub(kernel_offset_i)
                    .unwrap_or(0)
                    .min(n_i - 1);
                let cj = (j + dj)
                    .checked_sub(kernel_offset_j)
                    .unwrap_or(0)
                    .min(n_j - 1);
                Some((ci, cj))
            }
            Padding::No => {
                // No padding so next image is di (dj) rows (cols) smaller
                if kernel_offset_i <= i && i + n_di < n_i {
                    let ci = i + di - kernel_offset_i;
                    if kernel_offset_j <= j && j + n_dj < n_j {
                        let cj = j + dj - kernel_offset_j;
                        return Some((ci, cj));
                    }
                }
                None
            }
        }
    }

    fn zeroed_output(&self) -> Array4<f32> {
        let [n_b, n_i, n_j, n_di, n_dj, _, n_c1] = self.idx_ranges.get();
        match self.padding {
            Padding::Same => Array4::zeros([n_b, n_i, n_j, n_c1]),
            Padding::No => Array4::zeros([n_b, n_i - n_di, n_j - n_dj, n_c1]),
        }
    }
}

impl Operation for Conv {
    fn eval(&self, inputs: Vec<ArrayViewD<f32>>) -> ArrayD<f32> {
        assert!(
            inputs.len() == 2,
            "Convolution operation takes two arguments"
        );
        let kernel = inputs[0].view().into_dimensionality::<Ix4>().unwrap();
        let image = inputs[1].view().into_dimensionality::<Ix4>().unwrap();

        if let ([n_di, n_dj, n_c0, n_c1], [n_b, n_i, n_j, n_c0_]) = (kernel.shape(), image.shape())
        {
            assert_eq!(
                n_c0_, n_c0,
                "number of channels in image do not match kernel's"
            );

            self.idx_ranges
                .set([*n_b, *n_i, *n_j, *n_di, *n_dj, *n_c0, *n_c1]);

            let mut output = self.zeroed_output();

            for [b, i, j, di, dj, c0, c1] in self.iter_idxs() {
                if let Some((ci, cj)) = self.conv_point(i, j, di, dj) {
                    println!("conv point{:?}", (i, j, di, dj, ci, cj));
                    output[(b, i, j, c1)] += kernel[(di, dj, c0, c1)] * image[(b, ci, cj, c0)];
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

        let mut grad_kernel = Array4::zeros([n_di, n_dj, n_c0, n_c1]);
        let mut grad_image = Array4::zeros([n_b, n_i, n_j, n_c0]);

        for [b, i, j, di, dj, c0, c1] in self.iter_idxs() {
            if let Some((ci, cj)) = self.conv_point(i, j, di, dj) {
                grad_kernel[(di, dj, c0, c1)] += loss[(b, i, j, c1)] * image[(b, ci, cj, c0)];
                grad_image[(b, ci, cj, c0)] += loss[(b, i, j, c1)] * kernel[(di, dj, c0, c1)];
            }
        }
        vec![grad_kernel.into_dyn(), grad_image.into_dyn()]
    }
}

#[cfg(test)]
mod tests {
    use conv::{Conv, Padding};
    use graph::Graph;
    use itertools::repeat_call;
    use ndarray::{Array, Array4, ArrayD};
    use node::{Node, Operation};
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;
    use std::f32;

    // TODO more tests for
    // panic if image channels do not match kernel
    // gradient kernel
    // gradient image
    // Padding::No (all 3 functions)

    #[test]
    fn conv_point_same_padding() {
        let ker = Array4::zeros([3, 3, 1, 1]).into_dyn();
        let img = Array4::zeros([4, 4, 4, 1]).into_dyn();
        let c = Conv::new(Padding::Same);
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
            let conv = Conv::new(*padding);
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
    fn identity_kernel() {
        let identity_kernel = Array::from_shape_fn([3, 3, 1, 1], |(di, dj, c0, c1)| {
            if di == 1 && dj == 1 && c0 == c1 {
                1.0
            } else {
                0.0
            }
        }).into_dyn();

        let img = stripes(true);
        let conv = Conv::new(Padding::Same);
        let res = conv.eval(vec![identity_kernel.view(), img.view()]);
        let conv = res.slice(s!(0, .., .., 0));
        let orig = img.slice(s!(0, .., .., 0));

        assert_eq!(orig, conv, "Identity Kernel failed\n");
    }

    // #[test]
    #[allow(dead_code, unused_variables)]
    fn _minimize_from_positive_image() {
        // Generate Arrays that sample unif[1, 2]
        let gen = repeat_call(move || {
            Array::from_shape_fn([4, 5, 5, 1], |_| {
                let mut rng = thread_rng();
                let unif = Uniform::new(1.0, 2.0);
                unif.sample(&mut rng)
            }).into_dyn()
        });
        // Construct graph
        let mut g = Graph::default();
        let input = g.register(Node::Input {
            dataset: Box::new(gen),
        });
        let kernel = g.new_param(&[3, 3, 1, 1]);
        let original_kernel = g.nodes[kernel].value.to_owned();
        let conv = g.register(Node::Operation {
            inputs: vec![kernel, input],
            operation: Box::new(Conv::new(Padding::Same)),
        });
        // Run graph
        for _ in 0..5 {
            g.forward();
            assert!(
                g.nodes[input].value.iter().all(|x| *x > 0.0),
                "Image should be positive"
            );
            g.nodes[conv].loss = 0.1 * g.nodes[conv].value.to_owned();
            g.backward();
        }

        println!("Final Kernel\n{:?}\n", g.nodes[kernel].value);
        // println!("Final Image\n{:?}\n", g.nodes[input].value);
        // println!("Final Conv\n{:?}\n", g.nodes[conv].value);

        assert!(
            g.nodes[kernel].value.iter().all(|x| *x < 0.0) & false,
            "Conv failed to be all negative"
        )
    }
}
