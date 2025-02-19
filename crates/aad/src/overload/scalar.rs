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

        impl<'a, 'b> Add<$scalar> for &Variable<'a, Variable<'b, $scalar>>
        where
            for<'c> &'c $scalar: Add<$scalar, Output = $scalar>,
        {
            type Output = Variable<'a, Variable<'b, $scalar>>;

            #[inline]
            fn add(self, rhs: $scalar) -> Self::Output {
                let value = &self.value + rhs;
                match &self.index {
                    Some((i, tape)) => Variable {
                        index: {
                            let mut operations = tape.operations.borrow_mut();
                            let count = operations.len();
                            operations.push(OperationRecord([
                                (*i, Variable::one()),
                                (usize::MAX, Variable::zero()),
                            ]));
                            Some((count, tape))
                        },
                        value,
                    },
                    None => Variable { index: None, value },
                }
            }
        }

        impl<'a> Add<&Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>: Add<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn add(self, rhs: &Variable<'a, $scalar>) -> Self::Output {
                rhs + self
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

        impl<'a> Sub<&Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b $scalar: Sub<$scalar, Output = $scalar>,
            for<'b> &'b Variable<'a, $scalar>: Neg<Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn sub(self, rhs: &Variable<'a, $scalar>) -> Self::Output {
                -(rhs - self)
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

        impl<'a> Mul<&Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>: Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn mul(self, rhs: &Variable<'a, $scalar>) -> Self::Output {
                rhs * self
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

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a> Div<&Variable<'a, $scalar>> for $scalar
        where
            Variable<'a, $scalar>:
                Inv<Output = Variable<'a, $scalar>> + Mul<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn div(self, rhs: &Variable<'a, $scalar>) -> Self::Output {
                rhs.inv() * self
            }
        }
    };
}

macro_rules! impl_scalar_ops_float {
    ($($scalar:ty),*) => {
        $(
            impl_scalar_add!($scalar);
            impl_scalar_sub!($scalar);
            impl_scalar_mul!($scalar);
            impl_scalar_div!($scalar);
        )*
    };
}

macro_rules! impl_scalar_ops_int {
    ($($scalar:ty),*) => {
        $(
            impl_scalar_add!($scalar);
            impl_scalar_sub!($scalar);
            impl_scalar_mul!($scalar);
        )*
    };
}

impl_scalar_ops_float!(f32, f64);
impl_scalar_ops_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! impl_scalar_op {
    ($scalar:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<'a> $trait<$scalar> for Variable<'a, $scalar>
        where
            for<'b> &'b Variable<'a, $scalar>: $trait<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn $method(self, rhs: $scalar) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl<'a> $trait<Variable<'a, $scalar>> for $scalar
        where
            for<'b> &'b Variable<'a, $scalar>: $trait<$scalar, Output = Variable<'a, $scalar>>,
        {
            type Output = Variable<'a, $scalar>;

            #[inline]
            fn $method(self, rhs: Variable<'a, $scalar>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl $assign_trait<$scalar> for Variable<'_, $scalar>
        where
            for<'a, 'b> &'a Variable<'b, $scalar>: $trait<$scalar, Output = Variable<'b, $scalar>>,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: $scalar) {
                *self = self.$method(rhs);
            }
        }
    };
}

macro_rules! impl_scalar_ops_add_sub_mul {
    ($scalar:ty) => {
        impl_scalar_add_mul!($scalar);
        impl_scalar_op!($scalar, Sub, sub, SubAssign, sub_assign);
    };
}

macro_rules! impl_scalar_add_mul {
    ($scalar:ty) => {
        impl_scalar_op!($scalar, Add, add, AddAssign, add_assign);
        impl_scalar_op!($scalar, Mul, mul, MulAssign, mul_assign);
    };
}

macro_rules! impl_scalar_ops_float {
    ($scalar:ty) => {
        impl_scalar_ops_add_sub_mul!($scalar);
        impl_scalar_op!($scalar, Div, div, DivAssign, div_assign);
    };
}

impl_scalar_ops_float!(f32);
impl_scalar_ops_float!(f64);
impl_scalar_ops_add_sub_mul!(i8);
impl_scalar_ops_add_sub_mul!(i16);
impl_scalar_ops_add_sub_mul!(i32);
impl_scalar_ops_add_sub_mul!(i64);
impl_scalar_ops_add_sub_mul!(i128);
impl_scalar_ops_add_sub_mul!(isize);
impl_scalar_add_mul!(u8);
impl_scalar_add_mul!(u16);
impl_scalar_add_mul!(u32);
impl_scalar_add_mul!(u64);
impl_scalar_add_mul!(u128);
impl_scalar_add_mul!(usize);
