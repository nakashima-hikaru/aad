use crate::variable::Variable;
use num_traits::{Float, Inv, Zero};
use std::ops::{Mul, Neg};

impl<F: Copy + Inv<Output = F> + Zero + Mul<F, Output = F> + Neg<Output = F>> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub fn inv(self) -> Self {
        self.apply_unary_function(F::inv, |x| x.mul(x).inv().neg())
    }
}

impl<F: Float> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub fn nan() -> Self {
        Self::constant(F::nan())
    }

    #[inline]
    #[must_use]
    pub fn infinity() -> Self {
        Self::constant(F::infinity())
    }

    #[inline]
    #[must_use]
    pub fn neg_infinity() -> Self {
        Self::constant(F::neg_infinity())
    }

    #[inline]
    #[must_use]
    pub fn neg_zero() -> Self {
        Self::constant(F::neg_zero())
    }

    #[inline]
    #[must_use]
    pub fn min_value() -> Self {
        Self::constant(F::min_value())
    }

    #[inline]
    #[must_use]
    pub fn max_value() -> Self {
        Self::constant(F::max_value())
    }

    #[inline]
    #[must_use]
    pub fn min_positive_value() -> Self {
        Self::constant(F::min_positive_value())
    }

    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    #[inline]
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }

    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    #[inline]
    #[must_use]
    pub fn is_normal(self) -> bool {
        self.value.is_normal()
    }

    #[inline]
    #[must_use]
    pub fn is_subnormal(self) -> bool {
        self.value.is_subnormal()
    }

    #[inline]
    #[must_use]
    pub fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }

    #[inline]
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }

    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        self.apply_unary_function(F::ln, F::recip)
    }

    #[inline]
    #[must_use]
    pub fn log(self, base: F) -> Self {
        self.apply_scalar_function(F::log, |x, b| x.recip().mul(b.ln().recip()), base)
    }
    #[inline]
    #[must_use]
    pub fn powf(self, power: F) -> Self {
        self.apply_scalar_function(F::powf, |x, p| p.mul(x.powf(p.sub(F::one()))), power)
    }

    #[inline]
    #[must_use]
    pub fn powi(self, power: i32) -> Self {
        self.apply_scalar_function(F::powi, |x, p| F::from(p).unwrap() * x.powi(p - 1), power)
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
            x.powf(-(F::from(2).unwrap() / F::from(3).unwrap()))
                / F::from(3).unwrap()
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
        self.apply_unary_function(F::log10, |x| {
            x.recip()
                * F::ln(F::from(10).unwrap())
                    .recip()
        })
    }

    #[inline]
    #[must_use]
    pub fn hypot(self, other: Self) -> Self {
        self.apply_binary_function(other, F::hypot, |x, y| {
            let denom = x.hypot(y);
            (x / denom, y / denom)
        })
    }

    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        self.apply_unary_function(F::abs, F::signum)
    }

    #[inline]
    #[must_use]
    pub fn mul_add(self, a: F, b: F) -> Self {
        self.apply_scalar_function(|x, (a, b)| F::mul_add(x, a, b), |_, (a, _)| a, (a, b))
    }

    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        self.apply_unary_function(F::sin, F::cos)
    }

    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        self.apply_unary_function(F::cos, |x| -F::sin(x))
    }

    #[inline]
    #[must_use]
    pub fn tan(self) -> Self {
        self.apply_unary_function(F::tan, |x| F::cos(x).powi(2).recip())
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
        self.apply_unary_function(F::tanh, |x| F::cosh(x).powi(2).recip())
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

    #[inline]
    #[must_use]
    pub fn exp_m1(self) -> Self {
        self.apply_unary_function(F::exp_m1, |x| F::exp(x).sub(F::one()))
    }

    #[inline]
    #[must_use]
    pub fn ln_1p(self) -> Self {
        self.apply_unary_function(F::ln_1p, |x| F::ln(F::one() + x))
    }
}
