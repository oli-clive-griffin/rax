use crate::{backward::GradMap, node::ParamsMap, tensor::{self, Tensor}};

pub trait Optimizer {
    fn update(&self, params: ParamsMap, grads: GradMap) -> ParamsMap;
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn update(&self, mut params: ParamsMap, grads: GradMap) -> ParamsMap {
        // println!("params: {:?}", params.0);
        // println!("gradients: {:?}", grads);
        for (name, param) in params.0.iter_mut() {
            let grad = grads.get(name).unwrap();
            *param = tensor::sub(&param, &tensor::mul(grad, &Tensor::from(self.lr)).unwrap()).unwrap();
        }
        params
    }
}

const DEFAULT_LR: f64 = 1e-6;
impl Default for SGD {
    fn default() -> Self {
        SGD { lr: DEFAULT_LR }
    }
}