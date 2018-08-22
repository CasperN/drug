use drug::softmax;
use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::HashSet;

#[derive(Debug)]
struct Beam {
    sequence: Vec<usize>,
    log_prob: f32,
}
#[derive(Debug)]
pub struct BeamSearch {
    beams: Vec<Beam>,
    width: usize,
}
// TODO support N layers of GRU
impl BeamSearch {
    pub fn new(width: usize) -> Self {
        let mut beams = vec![];
        for _ in 0..width {
            beams.push(Beam {
                sequence: vec![],
                log_prob: 0.0,
            })
        }
        BeamSearch { beams, width }
    }
    pub fn into_codes(self) -> Vec<Vec<usize>> {
        self.beams.into_iter().map(|b| b.sequence).collect()
    }
    /// Return next hidden state and next words of the RNN sequence
    pub fn search(
        &mut self,
        hidden: &Vec<ArrayD<f32>>,
        logits: ArrayViewD<f32>,
        temperature: f32,
    ) -> (Vec<ArrayD<f32>>, ArrayD<f32>) {
        // Find likelihood of all "next elements" of every sequence
        let mut top = vec![];
        let probs = softmax(logits.into_dyn().mapv(|x| x / temperature).view());
        for b in 0..self.width {
            let codes = weighted_sample(probs.slice(s!(b, ..)), self.width);

            for code in codes.iter() {
                let lp = probs[(b, *code)].ln();
                let new_log_prob = self.beams[b].log_prob + lp;
                top.push((b, *code, new_log_prob));
            }
        }
        // Sort descending
        top.sort_by(|a, b| {
            if let Some(Ordering::Less) = (a.2).partial_cmp(&b.2) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        // Choose the top `self.width` (different) most likely sequences
        // Update beams, next hidden state and next words
        // let hidden_dim = hidden.shape()[1];
        let mut seen = HashSet::new();
        let mut new_beams = vec![];
        let mut new_words = vec![];
        let mut hidden: Vec<(ArrayView2<f32>, Array2<f32>)> = hidden
            .into_iter()
            .map(|h| {
                let h = h.view().into_dimensionality::<Ix2>().unwrap();
                (h, h.to_owned())
            })
            .collect();
        let mut i = 0;
        for (b, code, log_prob) in top.into_iter() {
            let mut sequence = self.beams[i].sequence.to_vec();
            sequence.push(code);

            if !seen.contains(&sequence) {
                for (orig, new) in hidden.iter_mut() {
                    let dim = orig.shape()[1];
                    for d in 0..dim {
                        new[(i, d)] = orig[(b, d)];
                    }
                }
                new_words.push(code as f32);
                seen.insert(sequence.to_vec());
                new_beams.push(Beam { sequence, log_prob });
                i += 1;
                if i == self.width {
                    break;
                }
            }
        }
        let new_hidden = hidden
            .into_iter()
            .map(|(orig, new)| new.into_dyn())
            .collect();
        self.beams = new_beams;
        let new_words = Array::from_shape_vec([self.width], new_words)
            .expect("Oh No")
            .into_dyn();

        (new_hidden, new_words)
    }
}

/// Returns width samples from each column from weights.
fn weighted_sample(weights: ArrayView1<f32>, width: usize) -> Vec<usize> {
    let len = weights.shape()[0];
    let unif = Uniform::new(0.0, 1.0);
    let mut rng = thread_rng();
    let mut res = HashSet::new();

    while res.len() < width {
        let mut x = unif.sample(&mut rng);
        let mut code = 0;
        for i in 0..len {
            x -= weights[i];
            if x > 0.0 {
                code += 1;
            } else {
                break;
            }
        }
        res.insert(code);
    }
    res.into_iter().collect()
}
