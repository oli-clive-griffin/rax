use std::rc::Rc;

use crate::node::{Node, BinaryOpResult, UnaryOpResult};

#[derive(Debug)]
pub enum DTrace {
    BinOp{ _op_name: &'static str, _original_val: f64, _arg1: Box<DTrace>, _arg2: Box<DTrace> },
    UnaryOp{ _op_name: &'static str, _original_val: f64, _arg: Box<DTrace> },
    DParamDX{ _original_val: f64, d_val: f64 },
}

pub fn back_trace(node: Rc<Node>, upstream: f64) -> DTrace {
    match &*node {
        Node::BinaryOpResult(op_result) =>
            binary_op_res_back_trace(op_result, upstream),
        Node::UnaryOpResult(op_result) =>
            unary_op_trace(op_result, upstream),
        Node::Param(_original_val) =>
            DTrace::DParamDX { _original_val: *_original_val, d_val: upstream }
    }
}

fn unary_op_trace(
    op_result: &Box<dyn UnaryOpResult>,
    upstream: f64,
) -> DTrace {
    let arg = op_result.get_arg();
    let d = op_result.get_grad();
    let trace = back_trace(arg, d * upstream);
    DTrace::UnaryOp {
        _op_name: op_result.op_name(),
        _original_val: op_result.val(),
        _arg: Box::new(trace),
    }
}

fn binary_op_res_back_trace(
    op_result: &Box<dyn BinaryOpResult>,
    upstream: f64,
) -> DTrace {
    let (arg1, arg2) = op_result.get_args();
    let (d_arg1, d_arg2) = op_result.get_grads();
    let a_trace = back_trace(arg1, d_arg1 * upstream);
    let b_trace = back_trace(arg2, d_arg2 * upstream);

    DTrace::BinOp {
        _op_name: op_result.op_name(),
        _original_val: op_result.val(),
        _arg1: Box::new(a_trace),
        _arg2: Box::new(b_trace),
    }
}
