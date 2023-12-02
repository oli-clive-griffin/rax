use std::rc::Rc;

use crate::node::{UnaryOpResult, Node, BinaryOpResult};

#[derive(Debug)]
struct AddRes {
    val: f64,
    result_of: (Rc<Node>, Rc<Node>),
}

#[derive(Debug)]
struct MulRes {
    val: f64,
    result_of: (Rc<Node>, Rc<Node>),
}

#[derive(Debug)]
struct SqRes {
    val: f64,
    result_of: Rc<Node>,
}

impl BinaryOpResult for AddRes {
    fn get_args(&self) -> (Rc<Node>, Rc<Node>) {
        self.result_of.clone()
    }
    fn get_grads(&self) -> (f64, f64) {
        (1., 1.,)
    }
    fn op_name(&self) -> &'static str {
        "Add"
    }
    fn val(&self) -> f64 {
        self.val
    }
}

impl BinaryOpResult for MulRes {
    fn get_args(&self) -> (Rc<Node>, Rc<Node>) {
        self.result_of.clone()
    }
    fn get_grads(&self) -> (f64, f64) {
        (self.result_of.1.val(), self.result_of.0.val())
    }
    fn op_name(&self) -> &'static str {
        "Mul"
    }
    fn val(&self) -> f64 {
        self.val
    }
}


impl UnaryOpResult for SqRes {
    fn get_arg(&self) -> Rc<Node> { self.result_of.clone()
    }
    fn get_grad(&self) -> f64 {
        2. * self.result_of.val()
    }
    fn op_name(&self) -> &'static str {
        "Square"
    }
    fn val(&self) -> f64 {
        self.val
    }
}


pub fn add(a: Rc<Node>, b: Rc<Node>) -> Rc<Node> {
    Node::bin_res(AddRes {
        val: a.val() + b.val(),
        result_of: (a, b),
    })
}

pub fn mul(a: Rc<Node>, b: Rc<Node>) -> Rc<Node> {
    Node::bin_res(MulRes {
        val: a.val() * b.val(),
        result_of: (a, b),
    })
}

pub fn sq(x: Rc<Node>) -> Rc<Node> {
    Node::unr_res(SqRes {
        val: x.val() * x.val(),
        result_of: x,
    })
}
