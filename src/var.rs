use crate::grads::Grads;
use crate::operations::Operation;
use crate::operations::{BinaryFnPayload, UnaryFnPayload};
use crate::tape::Tape;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy)]
pub struct Var<'a> {
    pub(crate) idx: usize,
    pub(crate) tape: &'a Tape,
}

unsafe impl Send for Var<'_> {}
unsafe impl Sync for Var<'_> {}

type BinaryFn<T> = fn(T, T) -> T;
type UnaryFn<T> = fn(T) -> T;

impl<'a> Var<'a> {
    pub fn value(&self) -> f64 {
        unsafe { *(*self.tape.values.get()).get_unchecked(self.idx) }
    }

    pub fn backward(&self) -> Grads {
        unsafe {
            let mut grads = vec![0.0; (*self.tape.values.get()).len()];
            *grads.get_unchecked_mut(self.idx) = 1.0;
            self.tape.replay(&mut grads);
            Grads(grads)
        }
    }

    pub fn sin(&self) -> Var<'a> {
        self.custom_unary_fn(f64::sin, f64::cos)
    }

    pub fn ln(&self) -> Var<'a> {
        self.custom_unary_fn(f64::ln, f64::recip)
    }

    pub fn powf(&self, power: f64) -> Var<'a> {
        self.custom_scalar_fn(
            |s, x| f64::powf(x, s),
            |power, x| power * x.powf(power - 1.0),
            power,
        )
    }

    pub fn exp(&self) -> Var<'a> {
        self.custom_unary_fn(f64::exp, f64::exp)
    }

    pub fn sqrt(&self) -> Var<'a> {
        self.custom_unary_fn(f64::sqrt, |x| 0.5 * x.sqrt().recip())
    }

    pub fn cos(&self) -> Var<'a> {
        self.custom_unary_fn(f64::cos, |x| -f64::sin(x))
    }

    pub fn tan(&self) -> Var<'a> {
        self.custom_unary_fn(f64::tan, |x| f64::cos(x).powi(2).recip())
    }

    pub fn sinh(&self) -> Var<'a> {
        self.custom_unary_fn(f64::sinh, f64::cosh)
    }

    pub fn cosh(&self) -> Var<'a> {
        self.custom_unary_fn(f64::cosh, f64::sinh)
    }

    pub fn tanh(&self) -> Var<'a> {
        self.custom_unary_fn(f64::tanh, |x| f64::cosh(x).powi(2).recip())
    }

    pub fn recip(&self) -> Var<'a> {
        self.custom_unary_fn(f64::recip, |x| -(x * x).recip())
    }

    #[inline(always)]
    pub fn custom_unary_fn(&self, f: UnaryFn<f64>, df: UnaryFn<f64>) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(f(x));

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: df(x),
            };

            self.tape.record(Operation::Unary(payload));
            result
        }
    }

    #[inline(always)]
    pub fn custom_scalar_fn(&self, f: BinaryFn<f64>, df: BinaryFn<f64>, scalar: f64) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(f(scalar, x));

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: df(scalar, x),
            };

            self.tape.record(Operation::Unary(payload));
            result
        }
    }

    #[inline(always)]
    pub fn custom_binary_fn(
        &self,
        other: &Var<'a>,
        f: BinaryFn<f64>,
        dfdx: BinaryFn<f64>,
        dfdy: BinaryFn<f64>,
    ) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            let y = *(*values).get_unchecked(other.idx);
            (*values).push(f(x, y));

            let payload = BinaryFnPayload {
                x: self.idx,
                y: other.idx,
                z: idx,
                dfdx: dfdx(x, y),
                dfdy: dfdy(x, y),
            };

            self.tape.record(Operation::Binary(payload));
            result
        }
    }
}

impl Neg for Var<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(-x);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: -1.0,
            };

            self.tape.record(Operation::Unary(payload));
            result
        }
    }
}

impl<'a> Add<Var<'a>> for Var<'a> {
    type Output = Self;

    fn add(self, other: Var<'a>) -> Self::Output {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            let y = *(*values).get_unchecked(other.idx);
            (*values).push(x + y);

            let payload = BinaryFnPayload {
                x: self.idx,
                y: other.idx,
                z: idx,
                dfdx: 1.0,
                dfdy: 1.0,
            };

            self.tape.record(Operation::Binary(payload));
            result
        }
    }
}

impl<'a> Sub<Var<'a>> for Var<'a> {
    type Output = Self;

    fn sub(self, other: Var<'a>) -> Self::Output {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            let y = *(*values).get_unchecked(other.idx);
            (*values).push(x - y);

            let payload = BinaryFnPayload {
                x: self.idx,
                y: other.idx,
                z: idx,
                dfdx: 1.0,
                dfdy: -1.0,
            };

            self.tape.record(Operation::Binary(payload));
            result
        }
    }
}

impl<'a> Mul<Var<'a>> for Var<'a> {
    type Output = Self;

    fn mul(self, other: Var<'a>) -> Self::Output {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            let y = *(*values).get_unchecked(other.idx);
            (*values).push(x * y);

            let payload = BinaryFnPayload {
                x: self.idx,
                y: other.idx,
                z: idx,
                dfdx: y,
                dfdy: x,
            };

            self.tape.record(Operation::Binary(payload));
            result
        }
    }
}

impl<'a> Div<Var<'a>> for Var<'a> {
    type Output = Self;

    fn div(self, other: Var<'a>) -> Self::Output {
        self.custom_binary_fn(&other, |x, y| x / y, |_, y| y.recip(), |x, y| -x / (y * y))
    }
}

impl<'a> Add<f64> for Var<'a> {
    type Output = Var<'a>;

    fn add(self, scalar: f64) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(scalar + x);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: 1.0,
            };

            self.tape.record(Operation::Unary(payload));
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

impl<'a> Sub<f64> for Var<'a> {
    type Output = Var<'a>;

    fn sub(self, scalar: f64) -> Self::Output {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(x - scalar);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: 1.0,
            };

            self.tape.record(Operation::Unary(payload));
            result
        }
    }
}

impl<'a> Sub<Var<'a>> for f64 {
    type Output = Var<'a>;

    fn sub(self, var: Var<'a>) -> Var<'a> {
        -var + self
    }
}

impl<'a> Mul<f64> for Var<'a> {
    type Output = Var<'a>;

    fn mul(self, scalar: f64) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(scalar * x);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: scalar,
            };

            self.tape.record(Operation::Unary(payload));
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

impl<'a> Div<f64> for Var<'a> {
    type Output = Var<'a>;

    fn div(self, scalar: f64) -> Var<'a> {
        unsafe {
            let values = self.tape.values.get();
            let idx = (*values).len();
            let result = Var {
                idx,
                tape: self.tape,
            };

            let x = *(*values).get_unchecked(self.idx);
            (*values).push(x / scalar);

            let payload = UnaryFnPayload {
                x: self.idx,
                y: idx,
                dfdx: scalar.recip(),
            };

            self.tape.record(Operation::Unary(payload));
            result
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<Var<'a>> for f64 {
    type Output = Var<'a>;

    fn div(self, var: Var<'a>) -> Var<'a> {
        var.recip() * self
    }
}

impl AddAssign<f64> for Var<'_> {
    fn add_assign(&mut self, scalar: f64) {
        *self = *self + scalar
    }
}

impl<'a> AddAssign<Var<'a>> for Var<'a> {
    fn add_assign(&mut self, other: Var<'a>) {
        *self = *self + other
    }
}

impl SubAssign<f64> for Var<'_> {
    fn sub_assign(&mut self, scalar: f64) {
        *self = *self - scalar
    }
}

impl<'a> SubAssign<Var<'a>> for Var<'a> {
    fn sub_assign(&mut self, other: Var<'a>) {
        *self = *self - other
    }
}

impl MulAssign<f64> for Var<'_> {
    fn mul_assign(&mut self, scalar: f64) {
        *self = *self * scalar
    }
}

impl<'a> MulAssign<Var<'a>> for Var<'a> {
    fn mul_assign(&mut self, other: Var<'a>) {
        *self = *self * other
    }
}

impl DivAssign<f64> for Var<'_> {
    fn div_assign(&mut self, scalar: f64) {
        *self = *self / scalar
    }
}

impl<'a> DivAssign<Var<'a>> for Var<'a> {
    fn div_assign(&mut self, other: Var<'a>) {
        *self = *self / other
    }
}
