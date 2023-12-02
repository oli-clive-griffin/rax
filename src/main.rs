mod ops;
mod backward;
mod node;

use crate::ops::{add, mul, sq};
use crate::node::Node;
use crate::backward::{backward, accum_grads};

fn main() {
    let a = Node::param(2.0, "a");
    let b = Node::param(3.0, "b");
    let c = Node::param(5.0, "c");
    let expr = mul(add(mul(a.clone(), b.clone()), sq(c.clone())), c.clone());

    let backward_graph = backward(expr, 1.);
    let grads_map = accum_grads(backward_graph);
    let grads = grads_map.values();
    println!("{:#?}", grads);
}

