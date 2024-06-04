use crate::{backward::GradMap, node::{Param, Params}};

pub trait Optimizer {
    fn update(&self, params: Params, grads: GradMap) -> Params;
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn update(&self, mut params: Params, grads: GradMap) -> Params {
        for Param { val, name } in params.values_mut() {
            let grad = grads.get(*name).unwrap();
            *val -= self.lr * grad;
        }
        params
    }
}
