use std::iter::zip;
use rand::Rng;

#[derive(Debug, Default, Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    shape_postfix_product: Vec<usize>,
}


impl Tensor {

    fn get_postfix_prod(shape: &Vec<usize>) -> Vec<usize> {
        let l = shape.len();
        let mut out: Vec<usize> = vec![0;l];
        let mut n = 1;
        for i in (0..l).rev() {
            out[i] = n;
            n *= shape[i];
        }
        out
    }
    pub fn zeroes(shape: &Vec<usize>) -> Tensor {
        let default = Default::default();
        let capacity = shape.iter().product();
        Tensor {
            shape: shape.clone(),
            data: vec![default; capacity],
            shape_postfix_product: Tensor::get_postfix_prod(shape),
        }
    }
    pub fn rand(shape: &Vec<usize>) -> Tensor {
        let mut rng = rand::thread_rng();
        let mut base = Tensor::zeroes(shape);
        for n in base.data.iter_mut() {
            *n = rng.gen();
        }
        base
    }
    fn flat_idx(&self, indices: &Vec<usize>) -> usize {
        zip(indices, &self.shape_postfix_product)
            .map(|(idx, prod)| idx * prod)
            .sum()
    }
    pub fn at(&self, indices: Vec<usize>) -> f64 {
        let idx = self.flat_idx(&indices);
        self.data[idx]
    }
    pub fn at_mut(&mut self, indices: Vec<usize>) -> &mut f64 {
        let idx = self.flat_idx(&indices);
        self.data.get_mut(idx).unwrap()
    }
    pub fn matmul(self, rhs: Self) -> Tensor {
        assert!(self.shape.len() == 2 && rhs.shape.len() == 2);
        assert!(self.shape[1] == rhs.shape[0]);
        let h = self.shape[0];
        let w = rhs.shape[1];
        let inner_dim = self.shape[1];
        let mut out = Tensor::zeroes(&vec![h, w]);
        for i in 0..h {
            for j in 0..w {
                let mut sum: f64 = 0.;
                for k in 0..inner_dim {
                    let l = self.at(vec![i, k]);
                    let r = rhs.at(vec![k, j]);
                    sum += l + r;
                }
                *out.at_mut(vec![i, j]) = sum;
            }
        }
        out
    }
}

pub fn mmul(l: Tensor, r: Tensor) -> Tensor {
    l.matmul(r)
}
