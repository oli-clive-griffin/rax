use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct ParamsMap(pub HashMap<String, Tensor>);
impl ParamsMap {
    pub fn new() -> Self {
        ParamsMap(HashMap::new())
    }
}

#[derive(Debug)]
pub enum Node {
    BinaryOp(BinaryOpResult),
    UnaryOp(UnaryOpResult),
    ReduceOp(ReduceOpResult),
    TensorParam(Tensor, &'static str), // (Tensor, name) // this is gross but works for now
}

#[derive(Debug)]
pub struct BinaryOpResult {
    pub op: Box<dyn BinaryOp>,
    pub args: (Rc<Node>, Rc<Node>),
    pub value: Tensor,
}

#[derive(Debug)]
pub struct UnaryOpResult {
    pub op: Box<dyn UnaryOp>,
    pub arg: Rc<Node>,
    pub value: Tensor,
}

#[derive(Debug)]
pub struct ReduceOpResult {
    pub op: Box<dyn ReduceOp>,
    pub arg: Rc<Node>,
    pub value: Tensor,
}

pub trait BinaryOp: Debug {
    fn get_grads(&self, upstream: &Tensor, args: (&Tensor, &Tensor)) -> (Tensor, Tensor);
    fn op_name(&self) -> &'static str;
}

pub trait UnaryOp: Debug {
    fn get_grads(&self, upstream: &Tensor, arg: &Tensor) -> Tensor;
    fn op_name(&self) -> &'static str;
}

pub trait ReduceOp: Debug {
    fn get_grads(&self, arg: &Tensor) -> Tensor;
    fn op_name(&self) -> &'static str;
}

impl Node {
    pub fn new_unr_res(op: impl UnaryOp + 'static, arg: Rc<Node>, value: Tensor) -> Rc<Node> {
        Rc::new(Node::UnaryOp(UnaryOpResult {
            op: Box::new(op),
            arg,
            value,
        }))
    }

    pub fn new_bin_res(
        op: impl BinaryOp + 'static,
        args: (Rc<Node>, Rc<Node>),
        value: Tensor,
    ) -> Rc<Node> {
        Rc::new(Node::BinaryOp(BinaryOpResult {
            op: Box::new(op),
            args,
            value,
        }))
    }

    pub fn new_red_res(op: impl ReduceOp + 'static, arg: Rc<Node>, value: Tensor) -> Rc<Node> {
        Rc::new(Node::ReduceOp(ReduceOpResult {
            op: Box::new(op),
            arg,
            value,
        }))
    }

    pub fn val(&self) -> Tensor {
        match self {
            Node::TensorParam(tensor, _) => tensor.clone(),
            Node::BinaryOp(res) => res.value.clone(),
            Node::UnaryOp(res) => res.value.clone(),
            Node::ReduceOp(res) => res.value.clone(),
        }
    }
}
