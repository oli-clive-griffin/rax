use std::rc::Rc;

use crate::{
    node::{BinaryOp, Node, ReduceOp, UnaryOp},
    tensor::{self, Tensor},
};

#[derive(Debug)]
pub struct MMulOp;
impl BinaryOp for MMulOp {
    fn get_grads(&self, upstream: &Tensor, (l, r): (&Tensor, &Tensor)) -> (Tensor, Tensor) {
        let l_grad = tensor::mmul(upstream, &r.transpose(0, 1));
        let r_grad = tensor::mmul(&l.transpose(0, 1), upstream);
        (l_grad, r_grad)
    }

    fn op_name(&self) -> &'static str {
        "MatMul"
    }
}

#[derive(Debug)]
pub struct AddOp;
impl BinaryOp for AddOp {
    fn get_grads(&self, upstream: &Tensor, _args: (&Tensor, &Tensor)) -> (Tensor, Tensor) {
        (upstream.clone(), upstream.clone())
    }
    fn op_name(&self) -> &'static str {
        "Add"
    }
}

#[derive(Debug)]
pub struct SubOp;
impl BinaryOp for SubOp {
    fn get_grads(&self, upstream: &Tensor, _args: (&Tensor, &Tensor)) -> (Tensor, Tensor) {
        (
            upstream.clone(),
            tensor::mul(&upstream.clone(), &Tensor::from(-1.)).unwrap(),
        )
    }
    fn op_name(&self) -> &'static str {
        "Sub"
    }
}

#[derive(Debug)]

pub struct MulOp;
impl BinaryOp for MulOp {
    fn get_grads(&self, upstream: &Tensor, (l, r): (&Tensor, &Tensor)) -> (Tensor, Tensor) {
        (
            tensor::mul(&l.clone(), &upstream.clone()).unwrap(),
            tensor::mul(&r.clone(), &upstream.clone()).unwrap(),
        )
    }
    fn op_name(&self) -> &'static str {
        "Mul"
    }
}

#[derive(Debug)]
pub struct SqrOp;
impl UnaryOp for SqrOp {
    fn get_grads(&self, upstream: &Tensor, _arg: &Tensor) -> Tensor {
        tensor::mul(&Tensor::from(2.), upstream).unwrap()
    }

    fn op_name(&self) -> &'static str {
        "Sqr"
    }
}

#[derive(Debug)]
pub struct MeanOp {
    input_n_elements: usize,
}
impl ReduceOp for MeanOp {
    fn op_name(&self) -> &'static str {
        "Mean"
    }
    /// gradient of mean is 1/n
    /// where n is the number of elements in the input tensor
    fn get_grads(&self, upstream: &Tensor) -> Tensor {
        return tensor::div(upstream, &Tensor::from(self.input_n_elements as f64)).unwrap();
    }
}

#[derive(Debug)]
pub struct ReluOp {
    // which elements in the input were greater than 0.
    // used to zero out the upstream gradient.
    input_gt_zero_mask: Tensor,
}
impl UnaryOp for ReluOp {
    fn get_grads(&self, upstream: &Tensor, arg: &Tensor) -> Tensor {
        tensor::mul(&self.input_gt_zero_mask.clone(), &upstream.clone()).unwrap()
    }

    fn op_name(&self) -> &'static str {
        "Relu"
    }
}

pub fn add(l: Rc<Node>, r: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        AddOp,
        (l.clone(), r.clone()),
        tensor::add(&l.val(), &r.val()).unwrap(),
    )
}

pub fn mul(l: Rc<Node>, r: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        MulOp,
        (l.clone(), r.clone()),
        tensor::mul(&l.val(), &r.val()).unwrap(),
    )
}

pub fn sqr(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(SqrOp, x.clone(), tensor::mul(&x.val(), &x.val()).unwrap())
}

pub fn mmul(l: Rc<Node>, r: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        MMulOp,
        (l.clone(), r.clone()),
        tensor::mmul(&l.val(), &r.val()),
    )
}

pub fn sub(l: Rc<Node>, r: Rc<Node>) -> Rc<Node> {
    Node::new_bin_res(
        SubOp,
        (l.clone(), neg(r.clone())),
        tensor::sub(&l.val(), &r.val()).unwrap(),
    )
}

pub fn neg(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(
        SqrOp,
        x.clone(),
        tensor::mul(&x.val(), &Tensor::from(-1.)).unwrap(),
    )
}

pub fn mean(x: Rc<Node>) -> Rc<Node> {
    Node::new_red_res(
        MeanOp {
            input_n_elements: x.val().n_elements(),
        },
        x.clone(),
        tensor::mean(&x.val()),
    )
}

pub fn relu(x: Rc<Node>) -> Rc<Node> {
    Node::new_unr_res(
        ReluOp {
            input_gt_zero_mask: tensor::gt(&x.val(), 0.),
        },
        x.clone(),
        tensor::relu(&x.val()),
    )
}
