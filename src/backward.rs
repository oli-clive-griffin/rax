use std::{collections::HashMap, rc::Rc};

use crate::node::{BinaryOpResult, Node, ReduceOpResult, UnaryOpResult};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct BinOpTrace {
    arg1: Box<DTrace>,
    arg2: Box<DTrace>,
}

#[derive(Debug)]
pub struct UnaryOpTrace {
    arg: Box<DTrace>,
}

#[derive(Debug)]
pub struct ReduceOpTrace {
    arg: Box<DTrace>,
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
    ReduceOp(ReduceOpTrace),
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
            arg: Box::new(self.arg.back_impl(g))
        })
    }
}

impl ReduceOpResult {
    fn back(&self, upstream: Rc<Tensor>) -> DTrace {
        let g = self.op.get_grads(upstream, Rc::new(self.arg.val()));
        DTrace::ReduceOp(ReduceOpTrace {
            arg: Box::new(self.arg.back_impl(g)),
        })
    }
}

pub type GradMap = HashMap<String, Tensor>;

#[macro_export]
macro_rules! grad {
    ($forward_fn:expr, $($arg:expr),*) => {
        {
            let output = $forward_fn($($arg),*);
            let dtrace = output.backwards();
            (output.val(), accum_grads(dtrace))
        }
    };
}

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
        DTrace::ReduceOp(op) => _accum_grads(&op.arg, map),
        DTrace::DParamDX(param) => {
            let name = param.param_name.to_string();

            let current_value = map.entry(name).or_insert(Tensor::from(0.));

            *current_value = Tensor::add(current_value, &param.d_val).unwrap_or_else(|_| {
                panic!(
                    "could not add tensors with shapes {:?} and {:?}",
                    current_value.size(),
                    param.d_val.size()
                )
            });

        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{add, mean, mul};

    use super::*;

    #[test]
    fn test_grads_1() {
        fn forward(x: Tensor) -> Rc<Node> {
            let x = Rc::new(Node::TensorParam(x, "x"));
            mean(x)
        }

        let x = Tensor::from(&[1., 2., 3., 4.] as &[f64]);

        let (val, grads_map) = grad!(forward, x);
    }

    #[test]
    fn test_grads_2() {
        fn forward(x: Tensor, y: Tensor) -> Rc<Node> {
            let x = Rc::new(Node::TensorParam(x, "x"));
            let y = Rc::new(Node::TensorParam(y, "y"));
            mean(add(x, y))
        }

        let x = Tensor::from(&[1., 2., 3., 4.] as &[f64]);
        let y = Tensor::from(&[1., 2., 3., 4.] as &[f64]);

        let (val, grads_map) = grad!(forward, x, y);
    }


    #[test]
    fn test_grads_3() {
        fn forward(x: Tensor, y: Tensor) -> Rc<Node> {
            let x = Rc::new(Node::TensorParam(x, "x"));
            let y = Rc::new(Node::TensorParam(y, "y"));
            mul(mean(x), y)
        }

        let x = Tensor::from(&[1., 2., 3., 4.] as &[f64]);
        let y = Tensor::from(&[8.] as &[f64]);

        let (_val, grads_map) = grad!(forward, x, y);

        assert_eq!(grads_map.len(), 2);

        // gradient of x:
        // explanation: x.n_elem = 4, y = 8, 1/4 * 8 = 2
        // each element of x should be 2
        assert_eq!(grads_map.get("x").unwrap().data, &[2., 2., 2., 2.]);

        // gradient of y
        // explanation: mean(x) = 2.5
        assert_eq!(grads_map.get("y").unwrap().data, &[2.5]);
    }
}
