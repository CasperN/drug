extern crate drug;
extern crate erased_serde;
extern crate ndarray;
extern crate serde_json;

use ndarray::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn save() {
    let mut g = drug::Graph::default();
    let a = g.constant(arr1(&[1.0, 2.0]).into_dyn());
    let b = g.constant(arr1(&[3.0, 4.0]).into_dyn());
    let c = g.mult(&[a, b]);
    g.forward();
    assert_eq!(g.get_value(c), arr1(&[3.0, 8.0]).into_dyn());
    let s = serde_json::to_string(&g).expect("Error serealizing graph");
    let path = Path::new("/tmp/drug.json");
    let mut f = File::create(&path).expect("File creation error");

    println!("Writing graph:\n{}", g);
    f.write_all(s.as_bytes()).expect("Could not write");
}

fn load() {
    let path = Path::new("/tmp/drug.json");
    let mut f = File::open(&path).expect("File open error");
    let mut s = String::new();
    f.read_to_string(&mut s).expect("Could not read to string");

    let g: drug::Graph = serde_json::from_str(&s).unwrap();

    // assert_eq!(g.get_value(c), arr1(&[4.0, 6.0]).into_dyn());

    println!("Read graph:\n{}", g);
}

/// Saving is not fully implemented because I haven't figued out how to make serde work with boxed
/// traits, like node with operations in them. Also the ony way to access nodes in the graph is
/// through Idx structs which are only created when something is added to the graph. Those have to
/// be saved seperately but I'm planning to be able to get Idxs from the graph using strings or
/// name-spaces somehow.
fn main() {
    save();
    load();
}
