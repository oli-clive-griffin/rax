use rand::Rng;
use std::{
    fmt::{Debug, Write},
    iter::zip,
    rc::Rc,
    vec,
};

#[derive(Clone)]
pub struct Tensor {
    data: Rc<Vec<f64>>,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Clone)]
pub struct VecTensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl VecTensor {
    fn to_tensor(self) -> Tensor {
        Tensor {
            data: Rc::new(self.data),
            shape: self.shape,
            stride: self.stride,
        }
    }

    pub fn zeroes(shape: &Vec<usize>) -> VecTensor {
        let capacity = shape.iter().product();
        VecTensor {
            data: vec![0.; capacity],
            shape: shape.clone(),
            stride: Tensor::get_postfix_prod(shape),
        }
    }

    pub fn at_mut(&mut self, indices: &Vec<usize>) -> &mut f64 {
        let idx = self.flat_idx(indices);
        self.data.get_mut(idx).unwrap()
    }
}

impl Tensor {
    fn unsqueeze(&self, dim_index: usize) -> Tensor {
        let mut shape = self.shape.clone();
        let mut stride = self.stride.clone();

        shape.insert(dim_index, 0);
        stride.insert(dim_index, 0);

        Tensor {
            data: self.data.clone(),
            stride,
            shape,
        }
    }

    fn unsqueeze_(&mut self, dim_index: usize) {
        self.shape.insert(dim_index, 1);
        self.stride.insert(dim_index, 1);
    }
}

// TODO clean this up
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn nested_loop(
            t: &Tensor,
            idx_stack: &mut Vec<usize>,
            f: &mut std::fmt::Formatter,
            indent: u8,
        ) {
            let depth = idx_stack.len();
            let full_depth = t.shape.len();
            let is_leaf = depth == full_depth;

            if is_leaf {
                f.write_str(&format!("{}", &t.at(idx_stack).to_string()));

                let isnt_last_val =
                    *idx_stack.last().unwrap_or(&0) != t.shape.last().unwrap_or(&0) - 1;

                if isnt_last_val {
                    f.write_str(", ");
                }

                return;
            } else {
                f.write_str("\n");
            }

            for i in 0..indent {
                f.write_char(' ');
            }

            f.write_str("[");
            for dim_idx in 0..t.shape[depth] {
                idx_stack.push(dim_idx);
                nested_loop(t, idx_stack, f, indent + 2);
                idx_stack.pop();
            }

            if depth != full_depth - 1 {
                f.write_str("\n");
                for i in 0..indent {
                    f.write_char(' ');
                }
            }

            f.write_str("],");
        }

        let mut idx_stack: Vec<usize> = vec![];
        nested_loop(self, &mut idx_stack, f, 0);
        Ok(())
    }
}

trait Stridable {
    fn get_stride(&self) -> &Vec<usize>;
}

trait Indexable: Stridable {
    fn flat_idx(&self, indices: &Vec<usize>) -> usize {
        zip(indices, self.get_stride())
            .map(|(idx, prod)| idx * prod)
            .sum()
    }
}

impl Stridable for Tensor {
    fn get_stride(&self) -> &Vec<usize> {
        return &self.stride;
    }
}

impl Stridable for VecTensor {
    fn get_stride(&self) -> &Vec<usize> {
        return &self.stride;
    }
}

impl Indexable for Tensor {}
impl Indexable for VecTensor {}

impl Tensor {
    fn get_postfix_prod(shape: &Vec<usize>) -> Vec<usize> {
        let l = shape.len();
        let mut out: Vec<usize> = vec![0; l];
        let mut n = 1;
        for i in (0..l).rev() {
            out[i] = n;
            n *= shape[i];
        }
        out
    }

    pub fn rand(shape: &Vec<usize>) -> Tensor {
        let mut rng = rand::thread_rng();
        let mut base = VecTensor::zeroes(shape);
        for n in base.data.iter_mut() {
            *n = rng.gen();
        }
        base.to_tensor()
    }

    pub fn at(&self, indices: &Vec<usize>) -> f64 {
        assert!(indices.len() == self.shape.len());
        let idx = self.flat_idx(indices);
        self.data[idx]
    }

    // todo extend to arbitrary dimensions
    pub fn matmul(&self, rhs: &Self) -> Tensor {
        assert!(self.shape.len() == 2 && rhs.shape.len() == 2);
        assert!(self.shape[1] == rhs.shape[0]);
        let h = self.shape[0];
        let w = rhs.shape[1];
        let inner_dim = self.shape[1];
        let mut out = VecTensor::zeroes(&vec![h, w]);
        for i in 0..h {
            for j in 0..w {
                let mut sum: f64 = 0.;
                for k in 0..inner_dim {
                    let l = self.at(&vec![i, k]);
                    let r = rhs.at(&vec![k, j]);
                    sum += l + r;
                }
                *out.at_mut(&vec![i, j]) = sum;
            }
        }
        out.to_tensor()
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
enum BroadcastDir {
    UnNeeded,
    LTR, // left broadcasts to right,
    RTL, // right broadcasts to left,
}
fn broadcastable(shape_l: Vec<usize>, shape_r: Vec<usize>) -> Option<Vec<BroadcastDir>> {
    let l = if shape_l.len() > shape_r.len() {
        shape_l.len()
    } else {
        shape_r.len()
    };
    let shape_l_len = shape_l.len();
    let shape_r_len = shape_r.len();

    let mut out: Vec<BroadcastDir> = vec![];

    for i in 1..=l {
        let dim_l = if i <= shape_l_len {
            shape_l[shape_l_len - i]
        } else {
            1
        };
        let dim_r = if i <= shape_r_len {
            shape_r[shape_r_len - i]
        } else {
            1
        };

        let broadcast_dir = match (dim_l, dim_r) {
            (1, 1) => BroadcastDir::UnNeeded,
            (1, _) => BroadcastDir::LTR,
            (_, 1) => BroadcastDir::RTL,
            (_, _) => {
                if dim_l == dim_r {
                    BroadcastDir::UnNeeded
                } else {
                    return None;
                }
            }
        };

        out.push(broadcast_dir);
    }
    out.reverse(); // TODO just iterate forwards by prealocating max length vec
    Some(out)
}

pub fn mmul(l: &Tensor, r: &Tensor) -> Tensor {
    l.matmul(r)
}

#[derive(Debug)]
pub struct ShapeError;

fn elementwise_broadcasted_map(
    r: &Tensor,
    l: &Tensor,
    func: &impl Fn(f64, f64) -> f64,
) -> Result<Tensor, ShapeError> {
    let broadcast_dirs = broadcastable(l.shape.clone(), r.shape.clone()).ok_or(ShapeError)?;

    let mut r = r.clone();
    let mut l = l.clone();

    while r.shape.len() < broadcast_dirs.len() {
        r.unsqueeze_(0);
    }
    while l.shape.len() < broadcast_dirs.len() {
        l.unsqueeze_(0);
    }

    let mut idx_stack: Vec<usize> = vec![];
    let mut r_stack: Vec<usize> = vec![];
    let mut l_stack: Vec<usize> = vec![];

    let elemwise_max = elemwise_max(&l, &r);
    let mut out = VecTensor::zeroes(&elemwise_max);

    fn inner(
        r: &Tensor,
        r_stack: &mut Vec<usize>,
        l: &Tensor,
        l_stack: &mut Vec<usize>,
        out: &mut VecTensor,
        out_stack: &mut Vec<usize>,
        broadcast_dirs: &Vec<BroadcastDir>,
        func: &impl Fn(f64, f64) -> f64,
    ) {
        let full_depth = broadcast_dirs.len();
        let depth = out_stack.len();

        if depth == full_depth {
            *out.at_mut(out_stack) = func(r.at(r_stack), l.at(l_stack));
            return;
        }

        let broadcast_direction_for_depth = &broadcast_dirs[depth];
        let out_dim_size_for_depth = out.shape[depth];

        for dim_idx in 0..out_dim_size_for_depth {
            out_stack.push(dim_idx);

            // if theres a broadcast to be done for this
            // dimension, push `0` onto the appropriate
            // stack so that the pointer stays pointing
            // at index 0 for that dimension
            r_stack.push(if broadcast_direction_for_depth == &BroadcastDir::RTL {
                0
            } else {
                dim_idx
            });
            l_stack.push(if broadcast_direction_for_depth == &BroadcastDir::LTR {
                0
            } else {
                dim_idx
            });

            inner(r, r_stack, l, l_stack, out, out_stack, broadcast_dirs, func);

            out_stack.pop();
            r_stack.pop();
            l_stack.pop();
        }
    }

    inner(
        &r,
        &mut r_stack,
        &l,
        &mut l_stack,
        &mut out,
        &mut idx_stack,
        &broadcast_dirs,
        func,
    );

    Ok(out.to_tensor())
}

pub fn add(l: &Tensor, r: &Tensor) -> Result<Tensor, ShapeError> {
    elementwise_broadcasted_map(&r, &l, &|a, b| a + b)
}

pub fn mul(l: &Tensor, r: &Tensor) -> Result<Tensor, ShapeError> {
    elementwise_broadcasted_map(&r, &l, &|a, b| a * b)
}

pub fn sub(l: &Tensor, r: &Tensor) -> Result<Tensor, ShapeError> {
    elementwise_broadcasted_map(&r, &l, &|a, b| a - b)
}

pub fn div(l: &Tensor, r: &Tensor) -> Result<Tensor, ShapeError> {
    elementwise_broadcasted_map(&r, &l, &|a, b| a / b)
}

fn elemwise_max(l: &Tensor, r: &Tensor) -> Vec<usize> {
    zip(&l.shape, &r.shape).map(|(a, b)| max(*a, *b)).collect()
}

fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let a = Tensor::rand(&vec![4, 4]);
        let a_t = a.transpose(0, 1);
        assert!(a.at(&vec![2, 3]) == a_t.at(&vec![3, 2]));
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
            BroadcastDir::RTL, // right broadcasts to left
            BroadcastDir::LTR, // left broadcasts to right
        ];

        let result = broadcastable(shape_l, shape_r);
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_broadcasting_6() {
        let shape_l: Vec<usize> = vec![2, 1, 2, 1];
        let shape_r: Vec<usize> = vec![1, 1, 3];
        let expected = vec![
            BroadcastDir::RTL,
            BroadcastDir::UnNeeded,
            BroadcastDir::RTL,
            BroadcastDir::LTR,
        ];

        let result = broadcastable(shape_l, shape_r);
        assert_eq!(result.unwrap(), expected)
    }

    #[test]
    fn test_add() {
        let l = Tensor {
            data: Rc::new(vec![1., 2., 3.]),
            shape: vec![1, 3],
            stride: vec![3, 1],
        };

        let r = Tensor {
            data: Rc::new(vec![7., 2.]),
            shape: vec![1, 2, 1],
            stride: Tensor::get_postfix_prod(&vec![1, 2, 1]),
        };

        let res = add(&l, &r).unwrap();
        println!("{:?}", res);
        // todo implement actual test
    }

    #[test]
    fn test_display() {
        let md = Tensor {
            data: Rc::new((0..24).map(|i| i as f64).collect::<Vec<f64>>()),
            shape: vec![4, 2, 3],
            stride: Tensor::get_postfix_prod(&vec![4, 2, 3]),
        };
        println!("{:?}", md);
    }
}
