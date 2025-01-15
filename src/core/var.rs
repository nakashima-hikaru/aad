use crate::core::operations::Operation;
use crate::core::operations::{
    Arithmetic, BinaryFn, BinaryFnPayload, Derive, UnaryFn, UnaryFnPayload,
};
use crate::core::tape::Tape;
use std::ops::{Add, Mul};

#[derive(Clone, Copy)]
pub struct Var<'a> {
    pub(crate) idx: usize,
    pub(crate) tape: &'a Tape,
}

impl<'a> Var<'a> {
    pub fn value(&self) -> f64 {
        unsafe { *(*self.tape.values.get()).get_unchecked(self.idx) }
    }

    pub fn grad(&self) -> f64 {
        unsafe { *(*self.tape.grads.get()).get_unchecked(self.idx) }
    }

    pub fn backward(&self) {
        unsafe {
            let grads = &mut (*self.tape.grads.get());
            grads.resize((*self.tape.values.get()).len(), 0.0);
            *grads.get_unchecked_mut(self.idx) += 1.0;
        }
        self.tape.replay();
    }

    pub fn sin(&self) -> Var<'a> {
        self.define_unary_fn(f64::sin, f64::cos)
    }

    pub fn ln(&self) -> Var<'a> {
        self.define_unary_fn(f64::ln, f64::recip)
    }

    #[inline(always)]
    fn define_unary_fn(&self, f: UnaryFn<f64>, df: UnaryFn<f64>) -> Var<'a> {
        unsafe {
            let idx = (*self.tape.values.get()).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let v = &mut (*self.tape.values.get());
            let res = f(*v.get_unchecked(self.idx));
            v.push(res);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: Derive::Fn(df),
            };

            self.tape.record(Operation::UnaryFn(payload));
            result
        }
    }

    #[inline(always)]
    fn define_binary_fn(
        &self,
        other: &Var<'a>,
        f: BinaryFn<f64>,
        dfdx: BinaryFn<f64>,
        dfdy: BinaryFn<f64>,
    ) -> Var<'a> {
        unsafe {
            let idx = (*self.tape.values.get()).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let v = &mut (*self.tape.values.get());
            let res = f(*v.get_unchecked(self.idx), *v.get_unchecked(other.idx));
            v.push(res);

            let payload = BinaryFnPayload {
                x: self.idx,
                y: other.idx,
                z: idx,
                dfdx,
                dfdy,
            };

            self.tape.record(Operation::BinaryFn(payload));
            result
        }
    }
}

impl<'a> Add<Var<'a>> for Var<'a> {
    type Output = Self;

    fn add(self, other: Var<'a>) -> Self::Output {
        self.define_binary_fn(&other, |x, y| x + y, |_, _| 1.0, |_, _| 1.0)
    }
}

impl<'a> Mul<Var<'a>> for Var<'a> {
    type Output = Self;

    fn mul(self, other: Var<'a>) -> Self::Output {
        self.define_binary_fn(&other, |x, y| x * y, |_, y| y, |x, _| x)
    }
}

impl<'a> Add<f64> for Var<'a> {
    type Output = Var<'a>;

    fn add(self, scalar: f64) -> Var<'a> {
        unsafe {
            let idx = (*self.tape.values.get()).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let v = &mut (*self.tape.values.get());
            let res = *v.get_unchecked(self.idx) + scalar;
            v.push(res);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: Derive::Scalar(1.0, Arithmetic::Add),
            };

            self.tape.record(Operation::UnaryFn(payload));
            result
        }
    }
}

impl<'a> Add<Var<'a>> for f64 {
    type Output = Var<'a>;

    fn add(self, var: Var<'a>) -> Var<'a> {
        var + self
    }
}

impl<'a> Mul<f64> for Var<'a> {
    type Output = Var<'a>;

    fn mul(self, scalar: f64) -> Var<'a> {
        unsafe {
            let idx = (*self.tape.values.get()).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let v = &mut (*self.tape.values.get());
            let res = *v.get_unchecked(self.idx) * scalar;
            v.push(res);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: Derive::Scalar(scalar, Arithmetic::Mul),
            };

            self.tape.record(Operation::UnaryFn(payload));
            result
        }
    }
}

impl<'a> Mul<Var<'a>> for f64 {
    type Output = Var<'a>;

    fn mul(self, var: Var<'a>) -> Var<'a> {
        var * self
    }
}
