use std::{collections::HashMap, ptr};
use crate::node::{Node, BinaryOpResult, UnaryOpResult};

#[derive(Debug)]
pub struct BinOpTrace {
    arg1: Box<DTrace>,
    arg2: Box<DTrace>,
    _op_name: &'static str,
    _original_val: f64,
}

#[derive(Debug)]
pub struct UnaryOpTrace {
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
    BinOp(BinOpTrace),
    UnaryOp(UnaryOpTrace),
    DParamDX(DParamDX),
}

impl Node {
    pub fn back(&self, upstream: f64) -> DTrace {
        match self {
            Node::BinaryOpResult(res) => res.back(upstream),
            Node::UnaryOpResult(res) => res.back(upstream),
            Node::Param(val, var_name) =>
                DTrace::DParamDX(DParamDX {
                    d_val: upstream,
                    original_ptr: ptr::addr_of!(*self),
                    _original_val: *val,
                    _var_name: var_name,
                }),
        }
    }
}

impl BinaryOpResult {
    fn back(&self, upstream: f64) -> DTrace {
        let (g1, g2) = self.op.get_grads((self.args.0.val(), self.args.1.val()));
        return DTrace::BinOp(BinOpTrace {
            arg1: Box::new(self.args.0.back(g1 * upstream)),
            arg2: Box::new(self.args.1.back(g2 * upstream)),
            _op_name: self.op.op_name(),
            _original_val: self.value,
        });
    }
}

impl UnaryOpResult {
    fn back(&self, upstream: f64) -> DTrace {
        let g = self.op.get_grads(self.arg.val());
        return DTrace::UnaryOp(UnaryOpTrace {
            arg: Box::new(self.arg.back(g * upstream)),
            _op_name: self.op.op_name(),
            _original_val: self.value,
        });
    }
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
