use crate::{backward::GradMap, node::Param, ParamMap};

pub trait Optimizer {
    fn update(&self, params: ParamMap, grads: GradMap) -> ParamMap;
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn update(&self, mut params: ParamMap, grads: GradMap) -> ParamMap {
        for Param { val, name } in params.values_mut() {
            let grad = grads.get(*name).unwrap();
            *val -= self.lr * grad;
        }
        return params;
    }
}
