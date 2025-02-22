use num_traits::{One, Zero};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait FloatLike<Scalar>:
    Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + Add<Scalar, Output = Self>
    + Sub<Scalar, Output = Self>
    + Mul<Scalar, Output = Self>
    + Div<Scalar, Output = Self>
    + for<'a> Add<&'a Scalar, Output = Self>
    + for<'a> Sub<&'a Scalar, Output = Self>
    + for<'a> Mul<&'a Scalar, Output = Self>
    + for<'a> Div<&'a Scalar, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + AddAssign<Scalar>
    + SubAssign<Scalar>
    + MulAssign<Scalar>
    + DivAssign<Scalar>
    + for<'a> AddAssign<&'a Scalar>
    + for<'a> SubAssign<&'a Scalar>
    + for<'a> MulAssign<&'a Scalar>
    + for<'a> DivAssign<&'a Scalar>
    + Sum<Self>
    + for<'a> Sum<&'a Self>
    + Sized
    + Clone
    + Copy
    + Zero
    + One
    + PartialOrd
    + PartialOrd<Scalar>
    + PartialEq
    + PartialEq<Scalar>
    + std::fmt::Debug
    + From<Scalar>
    + Into<Scalar>
{
    #[must_use]
    fn sin(self) -> Self;
    #[must_use]
    fn cos(self) -> Self;
    #[must_use]
    fn tan(self) -> Self;
    #[must_use]
    fn sinh(self) -> Self;
    #[must_use]
    fn cosh(self) -> Self;
    #[must_use]
    fn tanh(self) -> Self;

    #[must_use]
    fn ln(self) -> Self;
    #[must_use]
    fn log(self, base: Scalar) -> Self;
    #[must_use]
    fn log2(self) -> Self;
    #[must_use]
    fn log10(self) -> Self;
    #[must_use]
    fn exp(self) -> Self;
    #[must_use]
    fn exp2(self) -> Self;
    #[must_use]
    fn powf(self, exponent: Scalar) -> Self;
    #[must_use]
    fn powi(self, exponent: i32) -> Self;

    #[must_use]
    fn sqrt(self) -> Self;
    #[must_use]
    fn cbrt(self) -> Self;
    #[must_use]
    fn recip(self) -> Self;
    #[must_use]
    fn abs(self) -> Self;

    #[must_use]
    fn asin(self) -> Self;
    #[must_use]
    fn acos(self) -> Self;
    #[must_use]
    fn atan(self) -> Self;
    #[must_use]
    fn asinh(self) -> Self;
    #[must_use]
    fn acosh(self) -> Self;
    #[must_use]
    fn atanh(self) -> Self;

    #[must_use]
    fn hypot(self, other: Self) -> Self;
}
