use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

static DATA_DIR: &'static str = "examples/data/";
static TRAIN: &'static str = "ptb.train.txt";

#[allow(dead_code)]
pub struct TextDataSet {
    pub char2idx: HashMap<char, usize>,
    pub idx2char: Vec<char>,
    pub corpus: Vec<Vec<ArrayD<f32>>>,
}
impl TextDataSet {
    pub fn decode(&self, codes: Vec<usize>) -> String {
        codes.into_iter().map(|c| self.idx2char[c]).collect()
    }

    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        let mut contents = String::new();
        let mut f = File::open(Path::new(DATA_DIR).join(TRAIN)).expect("train data not found");
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        let mut coded_lines = Vec::new();
        let mut char2idx = HashMap::new();
        let mut idx2char = Vec::new();

        // Tokenize characters
        for str_line in contents.lines() {
            let mut line = Vec::new();

            for c in str_line.chars() {
                if char2idx.contains_key(&c) {
                    let idx = char2idx.get(&c).unwrap();
                    line.push(*idx);
                } else {
                    let new_idx = idx2char.len();
                    char2idx.insert(c, new_idx);
                    idx2char.push(c);
                    line.push(new_idx);
                }
            }
            coded_lines.push(line);
        }
        // Sort lines by length so sequences in the same batch are the same length

        let mut truncated: Vec<Vec<usize>> = coded_lines
            .into_iter()
            .flat_map(|l| {
                let v: Vec<Vec<usize>> = l.exact_chunks(seq_len).map(|x| x.to_vec()).collect();
                v.into_iter()
            })
            .collect();

        thread_rng().shuffle(truncated.as_mut_slice());

        let corpus: Vec<Vec<ArrayD<f32>>> = truncated
            .exact_chunks(batch_size)
            .map(|chunk| {
                let mut batch = vec![];
                for s in 0..seq_len {
                    let x = Array::from_shape_fn([batch_size], |b| chunk[b][s] as f32).into_dyn();
                    batch.push(x);
                }
                batch
            })
            .collect();

        TextDataSet {
            corpus,
            char2idx,
            idx2char,
        }
    }
}
