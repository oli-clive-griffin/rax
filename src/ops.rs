use std::rc::Rc;

use crate::node::{Node, BinaryOp, UnaryOp};

#[derive(Debug)]
struct AddOp;
impl BinaryOp for AddOp {
    fn get_grads(&self, _args: (f64, f64)) -> (f64, f64) {
        (1., 1.)
    }
    fn op_name(&self) -> &'static str {
        "Add"
    }
}

#[derive(Debug)]
struct MulOp;
impl BinaryOp for MulOp {
    fn get_grads(&self, args: (f64, f64)) -> (f64, f64) {
        (args.1, args.0)
    }
    fn op_name(&self) -> &'static str {
        "Mul"
    }
}

#[derive(Debug)]
struct SqrOp;
impl UnaryOp for SqrOp {
    fn get_grads(&self, arg: f64) -> f64 {
        2. * arg
    }
    fn op_name(&self) -> &'static str {
        "Sqr"
    }
}

pub fn add(a: Rc<Node>, b: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        AddOp,
        (a.clone(), b.clone()),
        a.val() + b.val(),
    )
}

pub fn mul(a: Rc<Node>, b: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        MulOp,
        (a.clone(), b.clone()),
        a.val() * b.val(),
    )
}

pub fn sq(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(
        SqrOp,
        x.clone(),
        x.val() * x.val(),
    )
}
