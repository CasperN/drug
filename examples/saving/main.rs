extern crate drug;
extern crate ndarray;
use ndarray::prelude::*;


fn main() {
    let mut g = drug::Graph::default();

    let a = g.constant(arr1(&[1.0, 2.0]).into_dyn());
    let b = g.constant(arr1(&[3.0, 4.0]).into_dyn());
    let c = g.add(&[a, b]);
    g.forward();
    assert_eq!(g.get_value(c), arr1(&[4.0, 6.0]).into_dyn());

}
