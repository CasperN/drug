extern crate drug;
extern crate erased_serde;
extern crate ndarray;
extern crate ron;

use ndarray::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn save() {
    let mut g = drug::Graph::default();
    let a = g.constant(arr1(&[1.0, 2.0]).into_dyn());
    let b = g.constant(arr1(&[3.0, 4.0]).into_dyn());
    let c = g.mult(&[a, b]);
    g.forward();
    assert_eq!(*g.get_value(c), arr1(&[3.0, 8.0]).into_dyn());
    g.named_idxs.insert("c".to_string(), c);

    let path = Path::new("/tmp/drug.json");
    let mut file = File::create(&path).expect("File creation error");

    println!("Writing graph:\n{}", g);
    let g_str = ron::ser::to_string(&g).expect("Error serealizing graph");
    file.write_all(g_str.as_bytes()).expect("Could not write");
}

fn load() {
    let path = Path::new("/tmp/drug.json");
    let f = File::open(&path).expect("File open error");

    let g: drug::Graph = ron::de::from_reader(&f).unwrap();
    let c = &g.named_idxs["c"];

    assert_eq!(*g.get_value(*c), arr1(&[3.0, 8.0]).into_dyn());
    println!("Read graph:\n{}", g);
}

fn main() {
    save();
    load();
}
