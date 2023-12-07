use std::{iter::zip, ops::Add, vec};
use rand::Rng;

#[derive(Debug, Default, Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    stride: Vec<usize>,
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
            stride: Tensor::get_postfix_prod(shape),
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
        zip(indices, &self.stride)
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

    pub fn transpose(&self, dim_idx_1: usize, dim_idx_2: usize) -> Tensor {
        let mut new_t = self.clone();

        let dim_1 = new_t.shape[dim_idx_1];
        new_t.shape[dim_idx_1] = new_t.shape[dim_idx_2];
        new_t.shape[dim_idx_2] = dim_1;

        let stride_1 = new_t.stride[dim_idx_1];
        new_t.stride[dim_idx_1] = new_t.stride[dim_idx_2];
        new_t.stride[dim_idx_2] = stride_1;

        new_t
    }
}

#[derive(PartialEq, Eq, Debug)]
enum BroadcastDir { UnNeeded, Right, Left }
fn broadcastable(shape_l: Vec<usize>, shape_r: Vec<usize>) -> Option<Vec<BroadcastDir>> {
    let l = if shape_l.len() > shape_r.len() { shape_l.len() } else { shape_r.len() };
    let shape_l_len = shape_l.len();
    let shape_r_len = shape_r.len();

    let mut out: Vec<BroadcastDir> = vec![];

    for i in 1..=l {
        let dim_l = if i <= shape_l_len { shape_l.get(shape_l_len - i) } else { None };
        let dim_r = if i <= shape_r_len { shape_r.get(shape_r_len - i) } else { None };

        let res = match (dim_l, dim_r) {
            (Some(l), Some(r)) => match (l == &1, r == &1) {
                (true, true) => Some(BroadcastDir::UnNeeded),
                (true, false) => Some(BroadcastDir::Right),
                (false, true) => Some(BroadcastDir::Left),
                (false, false) => if l == r { Some(BroadcastDir::UnNeeded) } else { None }, // Dims are different and non-zero
            }
            (None, Some(_)) => Some(BroadcastDir::Right),
            (Some(_), None) => Some(BroadcastDir::Left),
            (None, None) => panic!("this should not happen"),
        };

        if let Some(dim) = res {
            out.push(dim);
        } else {
            return None
        }
    }
    out.reverse();
    Some(out)
}

pub fn mmul(l: Tensor, r: Tensor) -> Tensor {
    l.matmul(r)
}

#[cfg(test)]
mod tests {
    use std::thread::panicking;

    use super::*;

    #[test]
    fn test_t() {
        let a = Tensor::rand(&vec![4, 4]);
        let a_t = a.transpose(0, 1);
        assert!(a.at(vec![2, 3]) == a_t.at(vec![3, 2]));
    }

    #[test]
    fn test_broadcasting_1() {
        let shape_l: Vec<usize> = vec![1, 2, 3];
        let shape_r: Vec<usize> = vec![1, 2, 3];
        let expected = vec![
            BroadcastDir::UnNeeded,
            BroadcastDir::UnNeeded,
            BroadcastDir::UnNeeded,
        ];
        let out = broadcastable(shape_l, shape_r);
        assert_eq!(out.unwrap(), expected);
    }

    #[test]
    fn test_broadcasting_2() {
        let shape_l: Vec<usize> = vec![1, 2, 3];
        let shape_r: Vec<usize> = vec![2, 3];
        assert!(broadcastable(shape_l, shape_r).is_some())
    }

    #[test]
    fn test_broadcasting_3() {
        let shape_l: Vec<usize> = vec![1, 2, 3];
        let shape_r: Vec<usize> = vec![2, 4];
        assert!(broadcastable(shape_l, shape_r).is_none())
    }

    #[test]
    fn test_broadcasting_4() {
        let shape_l: Vec<usize> = vec![1, 2, 3];
        let shape_r: Vec<usize> = vec![1, 2, 3];
        assert!(broadcastable(shape_l, shape_r).is_some())
    }

    #[test]
    fn test_broadcasting_5() {
        let shape_l: Vec<usize> = vec![1, 2, 1];
        let shape_r: Vec<usize> = vec![1, 1, 3];
        let expected = vec![
            BroadcastDir::UnNeeded,
            BroadcastDir::Left, // right broadcasts to left
            BroadcastDir::Right, // left broadcasts to right
        ];

        let result = broadcastable(shape_l, shape_r);
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_broadcasting_6() {
        let shape_l: Vec<usize> = vec![2, 1, 2, 1];
        let shape_r: Vec<usize> = vec![   1, 1, 3];
        let expected = vec![
            BroadcastDir::Left,
            BroadcastDir::UnNeeded,
            BroadcastDir::Left, // right broadcasts to left
            BroadcastDir::Right, // left broadcasts to right
        ];

        let result = broadcastable(shape_l, shape_r);
        assert_eq!(result.unwrap(), expected)
    }
}



