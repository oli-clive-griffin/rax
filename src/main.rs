use std::fmt::Debug;
use std::rc::Rc;

type DVal = f64;

#[derive(Debug)]
enum Num {
    Res(Box<dyn Backable>),
    Param(f64),
}

type DNum = Num;

impl Num {
    fn rc_res(res: impl Backable + 'static) -> Rc<Num> {
        Rc::new(Num::Res(Box::new(res)))
    }

    fn param(val: f64) -> Rc<Num> {
        Rc::new(Num::Param(val))
    }
}

impl Num {
    fn val(&self) -> f64 {
        match self {
            Num::Res(res) => res.val(),
            Num::Param(val) => *val,
        }
    }
}

trait Backable: Debug {
    fn val(&self) -> f64;
    fn back_params(&self, upstream: f64) -> Vec<DVal>;
    fn back_graph(&self, upstream: f64) -> DNum;
}

impl Backable for Num {
    fn back_params(&self, upstream: f64) -> Vec<DVal> {
        match self {
            Num::Res(res) => res.back_params(upstream),
            Num::Param(_) => vec![upstream],
        }
    }
    fn back_graph(&self, upstream: f64) -> DNum {
        match self {
            Num::Res(res) => res.back_graph(upstream),
            Num::Param(_) => DNum::Param(upstream),
        }
    }
    fn val(&self) -> f64 {
        match self {
            Num::Res(res) => res.val(),
            Num::Param(val) => *val,
        }
    }
}

// NOTE:
// usage of non-boxed `Num`s here means we can't
// construct graphs with splits (there's gotta be a proper name for that in graph theory)
// i.e. a Num can't be used in 2 ops.

#[derive(Debug)]
struct AddRes {
    val: f64,
    result_of: Vec<Rc<Num>>,
}

#[derive(Debug)]
struct MulRes {
    val: f64,
    result_of: (Rc<Num>, Rc<Num>),
}

#[derive(Debug)]
struct SqRes {
    val: f64,
    result_of: Rc<Num>,
}

impl Backable for AddRes {
    fn back_params(&self, upstream: f64) -> Vec<DVal> {
        self.result_of
            .iter()
            .flat_map(|arg| arg.back_params(upstream))
            .collect()
    }
    fn back_graph(&self, upstream: f64) -> DNum {
        DNum::Res(Box::new(AddRes {
            val: upstream,
            result_of: self.result_of
                .iter()
                .map(|arg| {
                    Rc::new(arg.back_graph(upstream)) // is this a sign that back_graph should just return `Rc`s? probably
                })
                .collect(),
        }))
    }
    fn val(&self) -> f64 {
        self.val
    }
}

impl Backable for MulRes {
    fn back_params(&self, upstream: f64) -> Vec<DVal> {
        let a = &self.result_of.0;
        let b = &self.result_of.1;
        let a_d = a.back_params(upstream * b.val());
        let b_d = b.back_params(upstream * a.val());
        a_d.into_iter().chain(b_d).collect()
    }
    fn back_graph(&self, upstream: f64) -> DNum {
        let a = &self.result_of.0;
        let b = &self.result_of.1;

        DNum::Res(Box::new(MulRes {
            val: upstream,
            result_of: (
                Rc::new(a.back_graph(upstream * b.val())), // note reversal
                Rc::new(b.back_graph(upstream * a.val())),
            )
        }))
    }
    fn val(&self) -> f64 {
        self.val
    }
}


impl  SqRes {
    fn d(&self, upstream: f64) -> f64 {
        2. * self.result_of.val() * upstream
    }
}

impl Backable for SqRes {
    fn back_params(&self, upstream: f64) -> Vec<DVal> {
        self.result_of.back_params(self.d(upstream))
    }
    fn back_graph(&self, upstream: f64) -> DNum {
        DNum::Res(Box::new(SqRes {
            val: upstream,
            result_of: Rc::new(self.result_of.back_graph(2. * self.result_of.val() * upstream)),
        }))
    }
    fn val(&self) -> f64 {
        self.val
    }
}

fn add(a: Rc<Num>, b: Rc<Num>) -> Rc<Num> {
    Num::rc_res(AddRes {
        val: a.val() + b.val(),
        result_of: vec![a, b],
    })
}

fn mul(a: Rc<Num>, b: Rc<Num>) -> Rc<Num> {
    Num::rc_res(MulRes {
        val: a.val() * b.val(),
        result_of: (a, b),
    })
}

fn sq(x: Rc<Num>) -> Rc<Num> {
    Num::rc_res(SqRes {
        val: x.val() * x.val(),
        result_of: x,
    })
}

fn main() {
    let a = Num::param(1.0);
    let b = Num::param(2.0);
    let c = Num::param(3.0);

    let res1 = add(mul(a.clone(), b.clone()), sq(c.clone()));

    let res2 = add(res1.clone(), c.clone());

    let d_params = res2.back_params(1.);
    println!("{:?}", d_params);

    let d_graph = res2.back_graph(1.);
    println!("{:?}", d_graph);
}

// could probably create a BinaryOp Trait which is
// differentiable and has an abstract fn d(self) -> (f64, f64)
