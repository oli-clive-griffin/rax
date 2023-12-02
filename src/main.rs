mod ops;
mod backward;
mod node;

use std::iter::zip;

use crate::ops::{add, mul, sq};
use crate::node::Node;
use crate::backward::{backward, accum_grads};

fn model(params: &Vec<f64>) -> Vec<f64> {
    if params.len() != 3 { panic!() }
    let a = Node::param(params[0], "a");
    let b = Node::param(params[1], "b");
    let c = Node::param(params[2], "c");
    let expr = mul(add(mul(a.clone(), b.clone()), sq(c.clone())), c.clone());
    let backward_graph = backward(expr, 1.);
    let grads_map = accum_grads(backward_graph);
    return grads_map.values().map(|x| x.clone()).collect();
}
fn update(params: &Vec<f64>, grads: &Vec<f64>) -> Vec<f64> {
    zip(params, grads).map(|(p, g)| p - 0.001 * g).collect()
}

fn main() {
    let mut params = vec![2., 3., 4.];
    loop {
        let grads = model(&params);
        params = update(&params, &grads);
        println!("grads: {:#?}", grads);
    }
}

