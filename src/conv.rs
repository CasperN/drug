use std::cell::Cell;

use ndarray::{Array4, ArrayD, ArrayViewD, Ix4};
use node::Operation;

pub struct Conv {
    _dialation: usize,
    _stride: usize,
    padding: Padding,
    idx_ranges: Cell<[usize; 7]>,
}

#[allow(dead_code)]
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
        match self.padding {
            Padding::Same => Some(((i + di).min(n_i - 1), (j + dj).min(n_j - 1))),
            Padding::No => {
                // No padding so next image is di (dj) rows (cols) smaller
                if i < n_i - n_di && j < n_j - n_dj {
                    Some((i + di, j + dj))
                } else {
                    None
                }
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
                n_c0, n_c0_,
                "number of channels in image do not match kernel's"
            );

            self.idx_ranges
                .set([*n_b, *n_i, *n_j, *n_di, *n_dj, *n_c0, *n_c1]);

            let mut output = self.zeroed_output();

            for [b, i, j, di, dj, c0, c1] in self.iter_idxs() {
                if let Some((ci, cj)) = self.conv_point(i, j, di, dj) {
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
mod forward {

    use super::*;
    use graph::Graph;
    use node::Node;
    use std::f32;

    // TODO more tests for
    // grad_param
    // grad_input
    // Padding::No (all 3 functions)

    fn stripes(horizontal: bool) -> Node {
        Node::Input {
            dataset: Box::new(
                vec![
                    Array4::from_shape_fn(
                        [1, 10, 10, 1],
                        move |(_, row, col, _)| if horizontal { row % 2 } else { col % 2 } as f32,
                    ).into_dyn(),
                ].into_iter(),
            ),
        }
    }

    fn stripe_detector_kernel(horizontal: bool) -> ArrayD<f32> {
        Array4::from_shape_fn([3, 3, 1, 1], move |(row, col, _, _)| {
            if (horizontal && row == 1) || (!horizontal && col == 1) {
                1.0 / 3.0
            } else {
                -1.0 / 6.0
            }
        }).into_dyn()
    }

    #[test]
    fn h_stripe_v_detector() {
        let mut g = Graph::default();
        let input = g.register(stripes(true));
        let kernel = g.new_initialized_param(stripe_detector_kernel(false));
        let conv = g.register(Node::Operation {
            inputs: vec![kernel, input],
            operation: Box::new(Conv::new(Padding::Same)),
        });
        g.forward();
        assert!(
            g.nodes[conv].value.iter().all(|x| x.abs() < f32::EPSILON),
            "{:?}",
            g.nodes[conv].value
        )
    }
    #[test]
    fn v_stripe_h_detector() {
        let mut g = Graph::default();
        let input = g.register(stripes(false));
        let kernel = g.new_initialized_param(stripe_detector_kernel(true));
        let conv = g.register(Node::Operation {
            inputs: vec![kernel, input],
            operation: Box::new(Conv::new(Padding::Same)),
        });
        g.forward();
        assert!(
            g.nodes[conv].value.iter().all(|x| x.abs() < f32::EPSILON),
            "{:?}",
            g.nodes[conv].value
        )
    }
    #[test]
    fn v_stripe_v_detector() {
        let mut g = Graph::default();
        let input = g.register(stripes(false));
        let kernel = g.new_initialized_param(stripe_detector_kernel(false));
        let conv = g.register(Node::Operation {
            inputs: vec![kernel, input],
            operation: Box::new(Conv::new(Padding::Same)),
        });
        g.forward();
        assert!(
            g.nodes[conv].value.iter().any(|x| x.abs() != 0.0),
            "{:?}",
            g.nodes[conv].value
        )
    }
    #[test]
    fn h_stripe_h_detector() {
        let mut g = Graph::default();
        let input = g.register(stripes(false));
        let kernel = g.new_initialized_param(stripe_detector_kernel(false));
        let conv = g.register(Node::Operation {
            inputs: vec![kernel, input],
            operation: Box::new(Conv::new(Padding::Same)),
        });
        g.forward();
        assert!(
            g.nodes[conv].value.iter().any(|x| x.abs() != 0.0),
            "{:?}",
            g.nodes[conv].value
        )
    }
}
