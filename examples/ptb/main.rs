use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

extern crate diff;
use diff::*;

static DATA_DIR: &'static str = "examples/data/";
static TRAIN: &'static str = "ptb.train.txt";
use std::io::Read;

#[derive(Default)]
struct TextDataSet {
    char2idx: HashMap<char, usize>,
    idx2char: Vec<char>,
    train_corpus: Vec<Vec<usize>>,
    // test_corpus: Vec<Vec<usize>,
}
impl TextDataSet {
    fn new() -> Self {
        let mut contents = String::new();
        let mut f = File::open(Path::new(DATA_DIR).join(TRAIN)).expect("train data not found");
        f.read_to_string(&mut contents).expect("something went wrong reading the file");

        let mut corpus = TextDataSet::default();

        for line in contents.lines() {
            for c in line.chars() {
                let mut line = Vec::new();
                if corpus.char2idx.contains_key(&c) {
                    let idx = corpus.char2idx.get(&c).unwrap();
                    line.push(*idx);
                } else {
                    let new_idx = corpus.idx2char.len();
                    corpus.char2idx.insert(c, new_idx);
                    corpus.idx2char.push(c);
                    line.push(new_idx);
                }
                corpus.train_corpus.push(line);
            }
        }
        corpus
    }
}

fn main() {
    let t = TextDataSet::new();

    println!("{:?}", t.train_corpus.len());
    println!("{:?}", t.char2idx);
    println!("{:?}", t.idx2char);

    let g = Graph::new();
    let


}
