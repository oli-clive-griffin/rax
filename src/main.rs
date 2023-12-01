use std::fmt::Debug;
use std::ops::{Add, Mul};

type DVal = f64;

#[derive(Debug)]
enum Num {
    Res(Box<dyn Res>),
    Param(f64),
}
impl Num {
    fn val(&self) -> f64 {
        match self {
            Num::Res(res) => res.val(),
            Num::Param(val) => *val,
        }
    }
}

trait Res: Debug {
    fn back(&self, upstream: f64) -> Vec<DVal>;
    fn val(&self) -> f64;
}

impl Res for Num {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        match self {
            Num::Res(res) => res.back(upstream),
            Num::Param(_) => vec![upstream],
        }
    }
    fn val(&self) -> f64 {
        match self {
            Num::Res(res) => res.val(),
            Num::Param(val) => *val,
        }
    }
}

#[derive(Debug)]
struct AddRes {
    val: f64,
    of: Vec<Num>,
}

#[derive(Debug)]
struct MulRes {
    val: f64,
    of: (Num, Num),
}

impl Res for AddRes {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        return self.of.iter().flat_map(|x| x.back(upstream)).collect();
    }
    fn val(&self) -> f64 { self.val }
}

impl Res for MulRes {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        let (a, b) = &self.of;
        let a_d = a.back(upstream * b.val());
        let b_d = b.back(upstream * a.val());
        a_d.into_iter().chain(b_d.into_iter()).collect()
    }
    fn val(&self) -> f64 { self.val }
}


impl Add for Num {
    type Output = Num;

    fn add(self, other: Num) -> Num {
        Num::Res(Box::new(AddRes {
            val: self.val() + other.val(),
            of: vec![self, other],
        }))
    }
}

impl Mul for Num {
    type Output = Num;

    fn mul(self, other: Num) -> Num {
        Num::Res(Box::new(MulRes {
            val: self.val() * other.val(),
            of: (self, other),
        }))
    }
}


fn main() {
    println!("Hello, world!");
    let a = Num::Param(1.0);
    let b = Num::Param(2.0);
    let c = Num::Param(3.0);

    let res = (a * b) + c;

    let d_params = res.back(1.);
    println!("{:?}", d_params);
}
