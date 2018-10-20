use drug::softmax;
use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

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
    /// Find likelihood of all "next elements" of every sequence.
    /// Returns next hidden state and next words of the RNN sequence
    pub fn search(
        &mut self,
        hidden: &[ArrayD<f32>],
        logits: ArrayViewD<f32>,
        temperature: f32,
    ) -> (Vec<ArrayD<f32>>, ArrayD<f32>) {
        let mut b = 0;
        let mut top = HashMap::new();
        let probs = softmax(logits.into_dyn().mapv(|x| x / temperature).view());

        while top.len() < self.width {
            let codes = weighted_sample(probs.slice(s!(b, ..)), 1);
            for code in codes.iter() {
                let new_log_prob = self.beams[b].log_prob + probs[(b, *code)].ln();
                let mut new_seq = self.beams[b].sequence.to_vec();
                new_seq.push(*code);
                top.insert(new_seq, (new_log_prob, b));
            }
            b += 1;
            b %= self.width;
        }

        let mut top: Vec<(Beam, usize)> = top
            .into_iter()
            .map(|(sequence, (log_prob, b))| (Beam { sequence, log_prob }, b))
            .collect();

        top.sort_by(|a, b| {
            if let Some(Ordering::Less) = (a.0.log_prob).partial_cmp(&b.0.log_prob) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        top.truncate(self.width);

        // Next set of words, hidden states and update beams
        let new_words = Array::from_iter(
            top.iter()
                .map(|(beam, _)| *beam.sequence.last().expect("Empty beam?") as f32),
        ).into_dyn();

        let mut new_hidden = vec![];
        for hid in hidden.iter() {
            let hdim = hid.shape()[1];
            let new_hid = Array::from_shape_fn([top.len(), hdim], |(b, d)| {
                let orig = top[b].1;
                hid[Dim([orig, d])]
            }).into_dyn();

            new_hidden.push(new_hid);
        }
        self.beams = top.into_iter().map(|(beam, _b)| beam).collect();

        (new_hidden, new_words)
    }
}

/// Returns width samples from each column from weights.
fn weighted_sample(weights: ArrayView1<f32>, width: usize) -> Vec<usize> {
    let len = weights.shape()[0];
    let unif = Uniform::new(0.0, 1.0);
    let mut rng = thread_rng();
    let mut res = HashSet::new();

    while res.len() < width.min(weights.len()) {
        let mut x = unif.sample(&mut rng);
        let mut code = 0;
        for w in weights.iter().take(len) {
            x -= w;
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
