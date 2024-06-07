use std::rc::Rc;

use crate::node::{BinaryOp, Node, ReduceOp, UnaryOp};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct MMulOp;
impl BinaryOp for MMulOp {
    fn get_grads(&self, upstream: Rc<Tensor>, (l, r): (Rc<Tensor>, Rc<Tensor>)) -> (Rc<Tensor>, Rc<Tensor>) {
        let l_grad = Tensor::mmul(&upstream, &r.transpose(0, 1));
        let r_grad = Tensor::mmul(&l.transpose(0, 1), &upstream);
        (Rc::new(l_grad), Rc::new(r_grad))
    }

    fn name(&self) -> &'static str {
        "MatMul"
    }
    fn forward(&self, left: Rc<Tensor>, right: Rc<Tensor>) -> Tensor {
        Tensor::mmul(&left, &right)
    }
}

#[derive(Debug)]
pub struct AddOp;
impl BinaryOp for AddOp {
    fn get_grads(&self, upstream: Rc<Tensor>, _args: (Rc<Tensor>, Rc<Tensor>)) -> (Rc<Tensor>, Rc<Tensor>) {
        (upstream.clone(), upstream.clone())
    }
    fn name(&self) -> &'static str {
        "Add"
    }
    fn forward(&self, left: Rc<Tensor>, right: Rc<Tensor>) -> Tensor {
        Tensor::add(&left, &right).unwrap()
    }
}

#[derive(Debug)]
pub struct SubOp;
impl BinaryOp for SubOp {
    fn get_grads(&self, upstream: Rc<Tensor>, _args: (Rc<Tensor>, Rc<Tensor>)) -> (Rc<Tensor>, Rc<Tensor>) {
        (
            upstream.clone(),
            Rc::new(Tensor::mul(&upstream.clone(), &Tensor::from(-1.)).unwrap()),
        )
    }
    fn name(&self) -> &'static str {
        "Sub"
    }
    fn forward(&self, left: Rc<Tensor>, right: Rc<Tensor>) -> Tensor {
        Tensor::sub(&left, &right).unwrap()
    }
}

#[derive(Debug)]

pub struct MulOp;
impl BinaryOp for MulOp {
    fn get_grads(&self, upstream: Rc<Tensor>, (l, r): (Rc<Tensor>, Rc<Tensor>)) -> (Rc<Tensor>, Rc<Tensor>) {
        let asdf = (
            Rc::new(Tensor::mul(&r.clone(), &upstream.clone()).unwrap()),
            Rc::new(Tensor::mul(&l.clone(), &upstream.clone()).unwrap()),
        );
        asdf
    }
    fn name(&self) -> &'static str {
        "Mul"
    }
    fn forward(&self, left: Rc<Tensor>, right: Rc<Tensor>) -> Tensor {
        Tensor::mul(&left, &right).unwrap()
    }
}

#[derive(Debug)]
pub struct SqrOp;
impl UnaryOp for SqrOp {
    fn get_grads(&self, upstream: Rc<Tensor>, _arg: Rc<Tensor>) -> Rc<Tensor> {
        Rc::new(Tensor::mul(&Tensor::from(2.), &upstream).unwrap())
    }

    fn name(&self) -> &'static str {
        "Sqr"
    }
}

#[derive(Debug)]
pub struct MeanOp {
    input_n_elements: usize,
}
impl ReduceOp for MeanOp {
    fn name(&self) -> &'static str {
        "Mean"
    }
    /// gradient of mean is 1/n
    /// where n is the number of elements in the input tensor
    fn get_grads(&self, upstream: Rc<Tensor>, arg: Rc<Tensor>) -> Rc<Tensor> {
        let upstream2  = Tensor::div(&upstream, &Tensor::from(self.input_n_elements as f64)).unwrap();
        let out = Tensor::mul(&upstream2, &Tensor::ones(arg.size())).unwrap();
        Rc::new(out)
    }
}

#[derive(Debug)]
pub struct ReluOp {
    // which elements in the input were greater than 0.
    // used to zero out the upstream gradient.
    input_gt_zero_mask: Tensor,
}
impl UnaryOp for ReluOp {
    fn get_grads(&self, upstream: Rc<Tensor>, _arg: Rc<Tensor>) -> Rc<Tensor> {
        Rc::new(Tensor::mul(&self.input_gt_zero_mask.clone(), &upstream.clone()).unwrap())
    }

    fn name(&self) -> &'static str {
        "Relu"
    }
}

macro_rules! create_binary_op {
    ($name:ident, $op:ident) => {
        pub fn $name(l: Rc<Node>, r: Rc<Node>) -> Rc<Node> {
            Node::new_bin_res($op, l, r)
        }
    };
}

create_binary_op!(add, AddOp);
create_binary_op!(mul, MulOp);
create_binary_op!(mmul, MMulOp);
create_binary_op!(sub, SubOp);

// UNARY
//

pub fn sqr(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(SqrOp, x.clone(), Tensor::mul(&x.val(), &x.val()).unwrap())
}

pub fn neg(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(
        SqrOp,
        x.clone(),
        Tensor::mul(&x.val(), &Tensor::from(-1.)).unwrap(),
    )
}

pub fn relu(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(
        ReluOp {
            input_gt_zero_mask: Tensor::gt(&x.val(), 0.),
        },
        x.clone(),
        Tensor::relu(&x.val()),
    )
}

// REDUCE
// 

pub fn mean(x: Rc<Node>) -> Rc<Node> {
    Node::new_red_res(
        MeanOp {
            input_n_elements: x.val().n_elements(),
        },
        x.clone(),
        Tensor::mean(&x.val()),
    )
}