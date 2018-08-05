extern crate ndarray;
// extern crate rand;

pub mod graph;
pub mod node;
mod optimizers;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
