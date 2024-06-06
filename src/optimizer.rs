use std::collections::HashMap;

use crate::backward::GradMap;
use crate::tensor::Tensor;

pub trait Optimizer {
    fn update(&self, params: ParamsMap, grads: GradMap) -> ParamsMap;
}

#[derive(Debug)]
pub struct ParamsMap(pub HashMap<String, Tensor>);
impl ParamsMap {
    pub fn new() -> Self {
        ParamsMap(HashMap::new())
    }
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn update(&self, mut params: ParamsMap, grads: GradMap) -> ParamsMap {
        for (name, param) in params.0.iter_mut() {
            let grad = grads.get(name).unwrap();
            let update = Tensor::mul(grad, &Tensor::from(self.lr)).unwrap();
            let new = Tensor::sub(&param, &update).unwrap();
            *param = new;
        }
        params
    }
}

const DEFAULT_LR: f64 = 1e-3;
impl Default for SGD {
    fn default() -> Self {
        SGD { lr: DEFAULT_LR }
    }
}