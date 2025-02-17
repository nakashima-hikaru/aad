use crate::operation_record::OperationRecord;
use crate::Variable;
use num_traits::{Inv, One, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

macro_rules! impl_scalar_add {
    ($scalar:ty) => {
        impl<'a> Add<$scalar> for &Variable<'a, $scalar>
        where
            for<'b> &'b $scalar: Add<$scalar, Output = $scalar>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn add(self, rhs: $scalar) -> Self::Output {
                let value = &self.value + rhs;
                match &self.index {
                    Some((i, tape)) => Variable {
                        index: {
                            let mut operations = tape.operations.borrow_mut();
                            let count = operations.len();
                            operations.push(OperationRecord([
                                (*i, <$scalar>::one()),
                                (usize::MAX, <$scalar>::zero()),
                            ]));
                            Some((count, tape))
                        },
                        value,
                    },
                    None => Variable { index: None, value },
                }
            }
        }

        impl<'a> Add<$scalar> for Variable<'a, $scalar>
        where
            for<'b> &'b Variable<'a, $scalar>: Add<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn add(self, rhs: $scalar) -> Self::Output {
                &self + rhs
            }
        }

        impl<'a> Add<Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>: Add<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn add(self, rhs: Self::Output) -> Self::Output {
                &rhs + self
            }
        }

        impl<'a, 'b> Add<&'b Variable<'a, $scalar>> for $scalar
        where
            for<'c> &'c Variable<'a, $scalar>: Add<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn add(self, rhs: &'b Variable<'a, $scalar>) -> Self::Output {
                rhs + self
            }
        }

        impl AddAssign<$scalar> for Variable<'_, $scalar>
        where
            for<'a, 'b> &'a Variable<'b, $scalar>: Add<$scalar, Output = Variable<'b, $scalar>>,
        {
            #[inline]
            fn add_assign(&mut self, rhs: $scalar) {
                *self = &*self + rhs;
            }
        }
    };
}

macro_rules! impl_scalar_sub {
    ($scalar:ty) => {
        impl<'a> Sub<$scalar> for &Variable<'a, $scalar>
        where
            for<'b> &'b $scalar: Sub<$scalar, Output = $scalar>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn sub(self, rhs: $scalar) -> Self::Output {
                let value = &self.value - rhs;
                match &self.index {
                    Some((i, tape)) => Variable {
                        index: {
                            let mut operations = tape.operations.borrow_mut();
                            let count = operations.len();
                            operations.push(OperationRecord([
                                (*i, <$scalar>::one()),
                                (usize::MAX, <$scalar>::zero()),
                            ]));
                            Some((count, tape))
                        },
                        value,
                    },
                    None => Variable { index: None, value },
                }
            }
        }

        impl<'a> Sub<$scalar> for Variable<'a, $scalar>
        where
            for<'b> &'b Variable<'a, $scalar>: Sub<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn sub(self, rhs: $scalar) -> Self::Output {
                &self - rhs
            }
        }

        impl<'a> Sub<Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>:
                Sub<$scalar, Output = Variable<'a, $scalar>> + Neg<Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn sub(self, rhs: Self::Output) -> Self::Output {
                -(&rhs - self)
            }
        }

        impl<'a, 'b> Sub<&'b Variable<'a, $scalar>> for $scalar
        where
            for<'c> &'c Variable<'a, $scalar>:
                Sub<$scalar, Output = Variable<'a, $scalar>> + Neg<Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn sub(self, rhs: &'b Variable<'a, $scalar>) -> Self::Output {
                rhs - self
            }
        }

        impl SubAssign<$scalar> for Variable<'_, $scalar>
        where
            for<'a, 'b> &'a Variable<'b, $scalar>: Sub<$scalar, Output = Variable<'b, $scalar>>,
        {
            #[inline]
            fn sub_assign(&mut self, rhs: $scalar) {
                *self = &*self - rhs;
            }
        }
    };
}

macro_rules! impl_scalar_mul {
    ($scalar:ty) => {
        impl<'a> Mul<$scalar> for &Variable<'a, $scalar>
        where
            for<'b> &'b $scalar: Mul<$scalar, Output = $scalar>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                let value = &self.value * rhs;
                match &self.index {
                    Some((i, tape)) => Variable {
                        index: {
                            let mut operations = tape.operations.borrow_mut();
                            let count = operations.len();
                            operations.push(OperationRecord([
                                (*i, rhs),
                                (usize::MAX, <$scalar>::zero()),
                            ]));
                            Some((count, tape))
                        },
                        value,
                    },
                    None => Variable { index: None, value },
                }
            }
        }

        impl<'a> Mul<$scalar> for Variable<'a, $scalar>
        where
            for<'b> &'b Variable<'a, $scalar>: Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                &self * rhs
            }
        }

        impl<'a> Mul<Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>: Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn mul(self, rhs: Self::Output) -> Self::Output {
                &rhs * self
            }
        }

        impl<'a, 'b> Mul<&'b Variable<'a, $scalar>> for $scalar
        where
            for<'c> &'c Variable<'a, $scalar>: Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn mul(self, rhs: &'b Variable<'a, $scalar>) -> Self::Output {
                rhs * self
            }
        }

        impl MulAssign<$scalar> for Variable<'_, $scalar>
        where
            for<'a, 'b> &'a Variable<'b, $scalar>: Mul<$scalar, Output = Variable<'b, $scalar>>,
        {
            #[inline]
            fn mul_assign(&mut self, rhs: $scalar) {
                *self = &*self * rhs;
            }
        }
    };
}

macro_rules! impl_scalar_div {
    ($scalar:ty) => {
        impl<'a> Div<$scalar> for &Variable<'a, $scalar>
        where
            for<'b> &'b $scalar: Div<$scalar, Output = $scalar>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn div(self, rhs: $scalar) -> Self::Output {
                let value = &self.value / rhs;
                match &self.index {
                    Some((i, tape)) => Variable {
                        index: {
                            let operations = &mut tape.operations.borrow_mut();
                            let count = operations.len();
                            operations.push(OperationRecord([
                                (*i, rhs.recip()),
                                (usize::MAX, <$scalar>::zero()),
                            ]));
                            Some((count, tape))
                        },
                        value,
                    },
                    None => Variable { index: None, value },
                }
            }
        }

        impl<'a> Div<$scalar> for Variable<'a, $scalar>
        where
            for<'b> &'b Variable<'a, $scalar>: Div<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn div(self, rhs: $scalar) -> Self::Output {
                &self / rhs
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a> Div<Variable<'a, $scalar>> for $scalar
        where
            Variable<'a, $scalar>:
                Inv<Output = Variable<'a, $scalar>> + Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn div(self, rhs: Self::Output) -> Self::Output {
                rhs.inv() * self
            }
        }

        impl<'a, 'b> Div<&'b Variable<'a, $scalar>> for $scalar
        where
            for<'c> &'c Variable<'a, $scalar>: Div<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn div(self, rhs: &'b Variable<'a, $scalar>) -> Self::Output {
                rhs / self
            }
        }

        impl DivAssign<$scalar> for Variable<'_, $scalar>
        where
            for<'a, 'b> &'a Variable<'b, $scalar>: Div<$scalar, Output = Variable<'b, $scalar>>,
        {
            #[inline]
            fn div_assign(&mut self, rhs: $scalar) {
                *self = &*self / rhs;
            }
        }
    };
}

impl_scalar_add!(f32);
impl_scalar_sub!(f32);
impl_scalar_mul!(f32);
impl_scalar_div!(f32);

impl_scalar_add!(f64);
impl_scalar_sub!(f64);
impl_scalar_mul!(f64);
impl_scalar_div!(f64);

impl_scalar_add!(i8);
impl_scalar_sub!(i8);
impl_scalar_mul!(i8);

impl_scalar_add!(i16);
impl_scalar_sub!(i16);
impl_scalar_mul!(i16);

impl_scalar_add!(i32);
impl_scalar_sub!(i32);
impl_scalar_mul!(i32);

impl_scalar_add!(i64);
impl_scalar_sub!(i64);
impl_scalar_mul!(i64);

impl_scalar_add!(i128);
impl_scalar_sub!(i128);
impl_scalar_mul!(i128);

impl_scalar_add!(isize);
impl_scalar_sub!(isize);
impl_scalar_mul!(isize);

impl_scalar_add!(u8);
impl_scalar_sub!(u8);
impl_scalar_mul!(u8);

impl_scalar_add!(u16);
impl_scalar_sub!(u16);
impl_scalar_mul!(u16);

impl_scalar_add!(u32);
impl_scalar_sub!(u32);
impl_scalar_mul!(u32);

impl_scalar_add!(u64);
impl_scalar_sub!(u64);
impl_scalar_mul!(u64);

impl_scalar_add!(u128);
impl_scalar_sub!(u128);
impl_scalar_mul!(u128);

impl_scalar_add!(usize);
impl_scalar_sub!(usize);
impl_scalar_mul!(usize);
