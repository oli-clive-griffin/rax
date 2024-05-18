use crate::node::{BinaryOpResult, Node, Param, UnaryOpResult};
use std::{collections::HashMap, ptr};

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
    // original_ptr: *const Node,
    param_name: &'static str,
    _param_val: f64,
    _var_name: &'static str,
}

#[derive(Debug)]
pub enum DTrace {
    BinOp(BinOpTrace),
    UnaryOp(UnaryOpTrace),
    DParamDX(DParamDX),
}

impl Node {
    pub fn backwards(&self) -> DTrace {
        self.back_impl(1.)
    }

    pub fn back_impl(&self, upstream: f64) -> DTrace {
        match self {
            Node::BinaryOpResult(res) => res.back(upstream),
            Node::UnaryOpResult(res) => res.back(upstream),
            Node::Param(Param { val, name }) => DTrace::DParamDX(DParamDX {
                d_val: upstream,
                param_name: name,
                // original_ptr:
                _param_val: *val,
                _var_name: name,
            }),
        }
    }
}

impl BinaryOpResult {
    fn back(&self, upstream: f64) -> DTrace {
        let (g1, g2) = self.op.get_grads((self.args.0.val(), self.args.1.val()));
        return DTrace::BinOp(BinOpTrace {
            arg1: Box::new(self.args.0.back_impl(g1 * upstream)),
            arg2: Box::new(self.args.1.back_impl(g2 * upstream)),
            _op_name: self.op.op_name(),
            _original_val: self.value,
        });
    }
}

impl UnaryOpResult {
    fn back(&self, upstream: f64) -> DTrace {
        let g = self.op.get_grads(self.arg.val());
        return DTrace::UnaryOp(UnaryOpTrace {
            arg: Box::new(self.arg.back_impl(g * upstream)),
            _op_name: self.op.op_name(),
            _original_val: self.value,
        });
    }
}

pub type GradMap = HashMap<String, f64>;

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
        DTrace::UnaryOp(op) => _accum_grads(&op.arg, map),
        DTrace::DParamDX(param) => {
            let current_value = map.entry(param.param_name.to_string()).or_insert(0.0);
            *current_value += param.d_val;
        }
    }
}
