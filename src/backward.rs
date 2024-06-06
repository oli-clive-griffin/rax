use crate::{node::{BinaryOpResult, Node, ReduceOpResult, UnaryOpResult}, tensor::{self, Tensor}};
use std::collections::HashMap;

#[derive(Debug)]
pub struct BinOpTrace {
    arg1: Box<DTrace>,
    arg2: Box<DTrace>,
    op_name: &'static str,
}

#[derive(Debug)]
pub struct UnaryOpTrace {
    arg: Box<DTrace>,
    op_name: &'static str,
}

#[derive(Debug)]
pub struct DParamDX {
    d_val: Tensor,
    param_name: &'static str,
}

#[derive(Debug)]
pub enum DTrace {
    BinOp(BinOpTrace),
    UnaryOp(UnaryOpTrace),
    DParamDX(DParamDX),
}

impl Node {
    pub fn backwards(&self) -> DTrace {
        self.back_impl(&Tensor::from(1.))
    }

    pub fn back_impl(&self, upstream: &Tensor) -> DTrace {
        match self {
            Node::BinaryOp(res) => res.back(upstream),
            Node::UnaryOp(res) => res.back(upstream),
            Node::ReduceOp(res) => res.back(upstream),
            Node::TensorParam(_t, name) => {
                DTrace::DParamDX(DParamDX {
                    d_val: upstream.clone(),
                    param_name: name
                })
            }
        }
    }
}

impl BinaryOpResult {
    fn back(&self, upstream: &Tensor) -> DTrace {
        let (g1, g2) = self.op.get_grads(upstream, (&self.args.0.val(), &self.args.1.val()));
        DTrace::BinOp(BinOpTrace {
            arg1: Box::new(self.args.0.back_impl(&g1)),
            arg2: Box::new(self.args.1.back_impl(&g2)),
            op_name: self.op.op_name(),
        })
    }
}

impl UnaryOpResult {
    fn back(&self, upstream: &Tensor) -> DTrace {
        let g = self.op.get_grads(upstream, &self.arg.val());
        DTrace::UnaryOp(UnaryOpTrace {
            arg: Box::new(self.arg.back_impl(&tensor::mul(&g, upstream).unwrap())),
            op_name: self.op.op_name(),
        })
    }
}

impl ReduceOpResult {
    fn back(&self, _upstream: &Tensor) -> DTrace {
        todo!()
    }
}

pub type GradMap = HashMap<String, Tensor>;

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
            // I'm realllly not sure this is right at all
            let current_value = map.entry(param.param_name.to_string()).or_insert(Tensor::from(0.));
            let res = tensor::add(current_value, &param.d_val).unwrap_or_else(|_| {
                panic!("could not add tensors with shapes {:?} and {:?}", current_value.size(), param.d_val.size())
            });
            *current_value = res
        }
    }
}
