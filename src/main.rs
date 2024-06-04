mod backward;
mod node;
mod ops;
mod optimizer;
mod tensor;

use std::collections::HashMap;
use std::rc::Rc;

use backward::GradMap;
use node::Param;

use crate::backward::accum_grads;
use crate::node::Node;
use crate::ops::{add, mul, sq};
use crate::optimizer::{Optimizer, SGD};

type ParamMap = HashMap<String, Param>;

fn demo_model(params: &ParamMap) -> (f64, GradMap) {
    let a = Rc::new(Node::Param(params.get("a").unwrap().clone()));
    let b = Rc::new(Node::Param(params.get("b").unwrap().clone()));
    let c = Rc::new(Node::Param(params.get("c").unwrap().clone()));

    let expr = sq(mul(add(a, b), c));
    let backward_graph = expr.backwards();
    let grads_map = accum_grads(backward_graph);

    (expr.val(), grads_map)
}

fn train() {
    let mut params = ParamMap::new();
    params.insert("a".to_string(), Param::new(1.0, "a"));
    params.insert("b".to_string(), Param::new(2.0, "b"));
    params.insert("c".to_string(), Param::new(3.0, "c"));

    let optim = SGD { lr: 0.01 };

    loop {
        let (val, grads_map) = demo_model(&params);
        params = optim.update(params, grads_map);
        if val < 1e-6 {
            break;
        }
    }
    println!("{:#?}", params);
}

fn main() {
    train();
}
