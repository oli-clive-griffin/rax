use std::fmt::Debug;
use std::rc::Rc;

type DVal = f64;

#[derive(Debug)]
enum Num {
    Res(Box<dyn Res>),
    Param(f64),
}

impl Num {
    fn rc_res(res: impl Res + 'static) -> Rc<Num> {
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

impl Res for AddRes {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        self.result_of
            .iter()
            .flat_map(|arg| arg.back(upstream))
            .collect()
    }
    fn val(&self) -> f64 {
        self.val
    }
}

impl Res for MulRes {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        let a = &self.result_of.0;
        let b = &self.result_of.1;
        let a_d = a.back(upstream * b.val());
        let b_d = b.back(upstream * a.val());
        a_d.into_iter().chain(b_d).collect()
    }
    fn val(&self) -> f64 {
        self.val
    }
}

impl Res for SqRes {
    fn back(&self, upstream: f64) -> Vec<DVal> {
        self.result_of.back(2. * self.val * upstream)
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

    let d_params = res2.back(1.);
    println!("{:?}", d_params);
}
