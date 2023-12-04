use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
pub enum Node {
    BinaryOpResult(BinaryOpResult),
    UnaryOpResult(UnaryOpResult),
    Param(f64, &'static str),
}

#[derive(Debug)]
pub struct BinaryOpResult {
    pub op: Box<dyn BinaryOp>,
    pub args: (Rc<Node>, Rc<Node>),
    pub value: f64,
}

#[derive(Debug)]
pub struct UnaryOpResult {
    pub op: Box<dyn UnaryOp>,
    pub arg: Rc<Node>,
    pub value: f64,
}

pub trait BinaryOp: Debug {
    fn get_grads(&self, args: (f64, f64)) -> (f64, f64);
    fn op_name(&self) -> &'static str;
}

pub trait UnaryOp: Debug {
    fn get_grads(&self, arg: f64) -> f64;
    fn op_name(&self) -> &'static str;
}

impl Node {
    pub fn new_unr_res(op: impl UnaryOp + 'static, arg: Rc<Node>, value: f64) -> Rc<Node> {
        Rc::new(Node::UnaryOpResult(UnaryOpResult { op: Box::new(op), arg, value }))
    }

    pub fn new_bin_res(op: impl BinaryOp + 'static, args: (Rc<Node>, Rc<Node>), value: f64) -> Rc<Node> {
        Rc::new(Node::BinaryOpResult(BinaryOpResult { op: Box::new(op), args, value }))
    }

    pub fn param(val: f64, name: &'static str) -> Rc<Node> {
        Rc::new(Node::Param(val, name))
    }

    pub fn val(&self) -> f64 {
        match self {
            Node::BinaryOpResult(res) => res.value,
            Node::UnaryOpResult(res) => res.value,
            Node::Param(val, _name) => *val,
        }
    }
}
