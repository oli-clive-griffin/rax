mod ops;
mod backward;
mod node;

use crate::ops::{add, mul, sq};
use crate::node::Node;
use crate::backward::back_trace;

fn main() {
    let a = Node::param(1.0);
    let b = Node::param(2.0);
    let c = Node::param(3.0);
    let res2 =
        mul(
            add(
                mul(
                    a.clone(),
                    b.clone(),
                ),
                sq(
                    c.clone()
                )
            ),
            c.clone(),
        );

    let trace = back_trace(res2, 1.);
    println!("{:#?}", trace);
}

