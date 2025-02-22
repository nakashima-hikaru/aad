use crate::{FloatLike, variable::Variable};
use std::cmp::Ordering;

impl From<Variable<'_, f64>> for f64 {
    fn from(value: Variable<'_, f64>) -> Self {
        value.value
    }
}

impl From<Variable<'_, Variable<'_, f64>>> for f64 {
    fn from(value: Variable<'_, Variable<'_, f64>>) -> Self {
        value.value.value
    }
}

impl FloatLike<f64> for Variable<'_, f64> {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }

    #[inline]
    fn sinh(self) -> Self {
        self.sinh()
    }

    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }

    #[inline]
    fn tanh(self) -> Self {
        self.tanh()
    }

    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn log(self, base: f64) -> Self {
        self.log(base)
    }

    #[inline]
    fn log2(self) -> Self {
        self.log2()
    }

    #[inline]
    fn log10(self) -> Self {
        self.log10()
    }

    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }

    #[inline]
    fn exp2(self) -> Self {
        self.exp2()
    }

    #[inline]
    fn powf(self, exponent: f64) -> Self {
        self.powf(exponent)
    }

    #[inline]
    fn powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn cbrt(self) -> Self {
        self.cbrt()
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn asin(self) -> Self {
        self.asin()
    }

    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }

    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }

    #[inline]
    fn asinh(self) -> Self {
        self.asinh()
    }

    #[inline]
    fn acosh(self) -> Self {
        self.acosh()
    }

    #[inline]
    fn atanh(self) -> Self {
        self.atanh()
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        self.hypot(other)
    }
}

impl<F: FloatLike<f64>> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        self.apply_unary_function(F::sin, F::cos)
    }

    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        self.apply_unary_function(F::cos, |x| -x.sin())
    }

    #[inline]
    #[must_use]
    pub fn tan(self) -> Self {
        self.apply_unary_function(F::tan, |x| x.cos().powi(2).recip())
    }

    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        self.apply_unary_function(F::ln, F::recip)
    }

    #[inline]
    #[must_use]
    pub fn log(self, base: f64) -> Self {
        self.apply_scalar_function(F::log, |x, b| x.recip().mul(b.ln().recip()), base)
    }

    #[inline]
    #[must_use]
    pub fn powf(self, power: f64) -> Self {
        self.apply_scalar_function(F::powf, |x, p| x.powf(p - 1.0).mul(p), power)
    }

    #[inline]
    #[must_use]
    pub fn powi(self, power: i32) -> Self {
        self.apply_scalar_function(F::powi, |x, p| F::from(f64::from(p)) * x.powi(p - 1), power)
    }

    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        self.apply_unary_function(F::exp, F::exp)
    }

    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.apply_unary_function(F::sqrt, |x| x.sqrt().recip().div(F::one() + F::one()))
    }

    #[inline]
    #[must_use]
    pub fn cbrt(self) -> Self {
        self.apply_unary_function(F::cbrt, |x| {
            x.powf(-(f64::from(2) / f64::from(3))) / f64::from(3)
        })
    }

    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        self.apply_unary_function(F::recip, |x| x.powi(2).recip().neg())
    }

    #[inline]
    #[must_use]
    pub fn exp2(self) -> Self {
        self.apply_unary_function(F::exp2, |x| F::ln(F::one() + F::one()) * F::exp2(x))
    }

    #[inline]
    #[must_use]
    pub fn log2(self) -> Self {
        self.apply_unary_function(F::log2, |x| x.recip() * F::ln(F::one() + F::one()).recip())
    }

    #[inline]
    #[must_use]
    pub fn log10(self) -> Self {
        self.apply_unary_function(F::log10, |x| x.recip() * 10.0_f64.ln().recip())
    }

    #[inline]
    #[must_use]
    pub fn hypot(self, other: Self) -> Self {
        self.apply_binary_function(&other, F::hypot, |x, y| {
            let denom = x.hypot(y);
            (x / denom, y / denom)
        })
    }

    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        todo!()
        // self.apply_unary_function(Variable::abs, Variable::signum)
    }

    #[inline]
    #[must_use]
    pub fn sinh(self) -> Self {
        self.apply_unary_function(F::sinh, F::cosh)
    }

    #[inline]
    #[must_use]
    pub fn cosh(self) -> Self {
        self.apply_unary_function(F::cosh, F::sinh)
    }

    #[inline]
    #[must_use]
    pub fn tanh(self) -> Self {
        self.apply_unary_function(F::tanh, |x| x.cosh().powi(2).recip())
    }

    #[inline]
    #[must_use]
    pub fn asin(self) -> Self {
        self.apply_unary_function(F::asin, |x| (F::one() - x * x).sqrt().recip())
    }

    #[inline]
    #[must_use]
    pub fn acos(self) -> Self {
        self.apply_unary_function(F::acos, |x| -(F::one() - x * x).sqrt().recip())
    }

    #[inline]
    #[must_use]
    pub fn atan(self) -> Self {
        self.apply_unary_function(F::atan, |x| (F::one() + x * x).recip())
    }

    #[inline]
    #[must_use]
    pub fn asinh(self) -> Self {
        self.apply_unary_function(F::asinh, |x| (x * x + F::one()).sqrt().recip())
    }

    #[inline]
    #[must_use]
    pub fn acosh(self) -> Self {
        self.apply_unary_function(F::acosh, |x| (x * x - F::one()).sqrt().recip())
    }

    #[inline]
    #[must_use]
    pub fn atanh(self) -> Self {
        self.apply_unary_function(F::atanh, |x| (F::one() - x * x).recip())
    }
}

impl FloatLike<f64> for Variable<'_, Variable<'_, f64>> {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }

    #[inline]
    fn sinh(self) -> Self {
        self.sinh()
    }

    #[inline]
    fn cosh(self) -> Self {
        self.cosh()
    }

    #[inline]
    fn tanh(self) -> Self {
        self.tanh()
    }

    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn log(self, base: f64) -> Self {
        self.log(base)
    }

    #[inline]
    fn log2(self) -> Self {
        self.log2()
    }

    #[inline]
    fn log10(self) -> Self {
        self.log10()
    }

    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }

    #[inline]
    fn exp2(self) -> Self {
        self.exp2()
    }

    #[inline]
    fn powf(self, exponent: f64) -> Self {
        self.powf(exponent)
    }

    #[inline]
    fn powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn cbrt(self) -> Self {
        self.cbrt()
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn asin(self) -> Self {
        self.asin()
    }

    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }

    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }

    #[inline]
    fn asinh(self) -> Self {
        self.asinh()
    }

    #[inline]
    fn acosh(self) -> Self {
        self.acosh()
    }

    #[inline]
    fn atanh(self) -> Self {
        self.atanh()
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        self.hypot(other)
    }
}

impl<T, F: PartialOrd<T>> PartialOrd<T> for Variable<'_, F>
where
    Self: PartialEq<T>,
{
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.value.partial_cmp(other)
    }
}

impl<T, F: PartialEq<T>> PartialEq<T> for Variable<'_, F> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.value == *other
    }
}
