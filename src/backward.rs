use std::{collections::HashMap, rc::Rc};

use crate::node::{BinaryOpResult, Node, ReduceOpResult, UnaryOpResult};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct BinOpTrace {
    arg1: Box<DTrace>,
    arg2: Box<DTrace>,
    // name: &'static str,
}

#[derive(Debug)]
pub struct UnaryOpTrace {
    arg: Box<DTrace>,
    // name: &'static str,
}

#[derive(Debug)]
pub struct DParamDX {
    d_val: Rc<Tensor>,
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
        self.back_impl(Rc::new(Tensor::from(1.)))
    }

    pub fn back_impl(&self, upstream: Rc<Tensor>) -> DTrace {
        match self {
            Node::BinaryOp(res) => res.back(upstream),
            Node::UnaryOp(res) => res.back(upstream),
            Node::ReduceOp(res) => res.back(upstream),
            Node::TensorParam(_t, name) => DTrace::DParamDX(DParamDX {
                d_val: upstream.clone(),
                param_name: name,
            }),
        }
    }
}

impl BinaryOpResult {
    fn back(&self, upstream: Rc<Tensor>) -> DTrace {
        let (g1, g2) = self.op.get_grads(
            upstream,
            (Rc::new(self.args.0.val()), Rc::new(self.args.1.val())),
        );
        DTrace::BinOp(BinOpTrace {
            arg1: Box::new(self.args.0.back_impl(g1)),
            arg2: Box::new(self.args.1.back_impl(g2)),
            // name: self.op.name(),
        })
    }
}

impl UnaryOpResult {
    fn back(&self, upstream: Rc<Tensor>) -> DTrace {
        let g = self.op.get_grads(upstream.clone(), Rc::new(self.arg.val()));
        DTrace::UnaryOp(UnaryOpTrace {
            arg: Box::new(
                self.arg
                    .back_impl(Rc::new(Tensor::mul(&g, &upstream).unwrap())),
            ),
            // name: self.op.name(),
        })
    }
}

impl ReduceOpResult {
    fn back(&self, _upstream: Rc<Tensor>) -> DTrace {
        todo!()
    }
}

pub type GradMap = HashMap<String, Tensor>;

pub fn accum_grads(node: DTrace) -> GradMap {
    let mut map = GradMap::new();
    _accum_grads(&node, &mut map);
    map
}

/// Traverse the trace tree, accumulating gradients and summing gradients
/// for the same parameter (deduping by name)
fn _accum_grads(node: &DTrace, map: &mut GradMap) {
    match node {
        DTrace::BinOp(op) => {
            _accum_grads(&op.arg1, map);
            _accum_grads(&op.arg2, map);
        }
        DTrace::UnaryOp(op) => _accum_grads(&op.arg, map),
        DTrace::DParamDX(param) => {
            let name = param.param_name.to_string();

            if let Some(current_value) = map.get_mut(&name) {
                *current_value = Tensor::add(current_value, &param.d_val).unwrap_or_else(|_| {
                    panic!(
                        "could not add tensors with shapes {:?} and {:?}",
                        current_value.size(),
                        param.d_val.size()
                    )
                });
                return;
            };

            map.insert(name, Tensor::from(0.));
        }
    }
}
