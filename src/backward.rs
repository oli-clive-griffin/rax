use std::collections::HashMap;
use std::rc::Rc;
use crate::node::{Node, BinaryOpResult, UnaryOpResult};

#[derive(Debug)]
pub struct BinOp {
    arg1: Box<DTrace>,
    arg2: Box<DTrace>,
    _op_name: &'static str,
    _original_val: f64,
}

#[derive(Debug)]
pub struct UnaryOp {
    arg: Box<DTrace>,
    _op_name: &'static str,
    _original_val: f64,
}

#[derive(Debug)]
pub struct DParamDX {
    d_val: f64,
    original_ptr: *const Node,
    _original_val: f64,
    _var_name: &'static str,
}

#[derive(Debug)]
pub enum DTrace {
    BinOp(BinOp),
    UnaryOp(UnaryOp),
    DParamDX(DParamDX),
}

pub fn backward(node: Rc<Node>, upstream: f64) -> DTrace {
    match &*node {
        Node::Param(original_val, name) => param_trace(upstream, Rc::as_ptr(&node), *original_val, name),
        Node::UnaryOpResult(op_result) => unary_op_trace(upstream, op_result),
        Node::BinaryOpResult(op_result) => binary_op_res_back_trace(upstream, op_result),
    }
}

fn param_trace(upstream: f64, original_ptr: *const Node, original_val: f64, name: &'static str) -> DTrace {
    DTrace::DParamDX(DParamDX {
        d_val: upstream,
        original_ptr, 
        _var_name: name, 
        _original_val: original_val,
    })
}

fn unary_op_trace(
    upstream: f64,
    op_result: &Box<dyn UnaryOpResult>,
) -> DTrace {
    let arg = op_result.get_arg();
    let grad = op_result.get_grad();
    let trace = backward(arg, grad * upstream);
    DTrace::UnaryOp(UnaryOp {
        _op_name: op_result.op_name(),
        _original_val: op_result.val(),
        arg: Box::new(trace),
    })
}

fn binary_op_res_back_trace(
    upstream: f64,
    op_result: &Box<dyn BinaryOpResult>,
) -> DTrace {
    let (arg1, arg2) = op_result.get_args();
    let (d_arg1, d_arg2) = op_result.get_grads();
    let trace1 = backward(arg1, d_arg1 * upstream);
    let trace2 = backward(arg2, d_arg2 * upstream);

    DTrace::BinOp(BinOp {
        _op_name: op_result.op_name(),
        _original_val: op_result.val(),
        arg1: Box::new(trace1),
        arg2: Box::new(trace2),
    })
}

type GradMap = HashMap<*const Node, f64>;

pub fn accum_grads(node: DTrace) -> GradMap {
    let mut map = GradMap::new();
    _accum_grads(&node, &mut map);
    map
}

fn _accum_grads(node: &DTrace, map: &mut GradMap) {
    match node {
        DTrace::BinOp(op) => {
            _accum_grads(&op.arg1, map);
            _accum_grads(&op.arg2, map);
        }
        DTrace::UnaryOp(op) => {
            _accum_grads(&op.arg, map)
        }
        DTrace::DParamDX(param) => {
            let current_value = map.entry(param.original_ptr).or_insert(0.0);
            *current_value += param.d_val;
        }
    }
}
