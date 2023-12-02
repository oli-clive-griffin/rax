use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
pub enum Node {
    BinaryOpResult(Box<dyn BinaryOpResult>),
    UnaryOpResult(Box<dyn UnaryOpResult>),
    Param(f64),
}

pub trait BinaryOpResult: Debug {
    fn val(&self) -> f64;
    fn get_grads(&self) -> (f64, f64);
    fn get_args(&self) -> (Rc<Node>, Rc<Node>);
    fn op_name(&self) -> &'static str;
}

pub trait UnaryOpResult: Debug {
    fn val(&self) -> f64;
    fn get_grad(&self) -> f64;
    fn get_arg(&self) -> Rc<Node>;
    fn op_name(&self) -> &'static str;
}

impl Node {
    pub fn unr_res(res: impl UnaryOpResult + 'static) -> Rc<Node> {
        Rc::new(Node::UnaryOpResult(Box::new(res)))
    }

    pub fn bin_res(res: impl BinaryOpResult + 'static) -> Rc<Node> {
        Rc::new(Node::BinaryOpResult(Box::new(res)))
    }

    pub fn param(val: f64) -> Rc<Node> {
        Rc::new(Node::Param(val))
    }

    pub fn val(&self) -> f64 {
        match self {
            Node::BinaryOpResult(res) => res.val(),
            Node::UnaryOpResult(res) => res.val(),
            Node::Param(val) => *val,
        }
    }
}
