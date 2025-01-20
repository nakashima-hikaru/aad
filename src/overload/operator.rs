use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::operation_record::OperationRecord;
use crate::variable::Variable;

impl Neg for Variable<'_> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Variable {
            index: {
                let mut operations = self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, -1.0), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: -self.value,
        }
    }
}

impl<'a> Add<Self> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, 1.0), (other.index, 1.0)]));
                count
            },
            tape: self.tape,
            value: self.value + other.value,
        }
    }
}

impl<'a> Sub<Self> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Variable<'a>) -> Self::Output {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, 1.0), (other.index, -1.0)]));
                count
            },
            tape: self.tape,
            value: self.value - other.value,
        }
    }
}

impl<'a> Mul<Self> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Variable<'a>) -> Self::Output {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([
                    (self.index, other.value),
                    (other.index, self.value),
                ]));
                count
            },
            tape: self.tape,
            value: self.value * other.value,
        }
    }
}

impl<'a> Div<Self> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn div(self, other: Variable<'a>) -> Self::Output {
        self.apply_binary_function(&other, |x, y| x / y, |_, y| y.recip(), |x, y| -x / (y * y))
    }
}

impl<'a> Add<f64> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn add(self, scalar: f64) -> Self::Output {
        Variable {
            index: {
                let mut operations = self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, 1.0), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: scalar + self.value,
        }
    }
}

impl<'a> Add<Variable<'a>> for f64 {
    type Output = Variable<'a>;

    #[inline]
    fn add(self, var: Self::Output) -> Self::Output {
        var + self
    }
}

impl<'a> Sub<f64> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn sub(self, scalar: f64) -> Self::Output {
        Variable {
            index: {
                let mut operations = self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, 1.0), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: self.value - scalar,
        }
    }
}

impl<'a> Sub<Variable<'a>> for f64 {
    type Output = Variable<'a>;

    #[inline]
    fn sub(self, var: Self::Output) -> Self::Output {
        -var + self
    }
}

impl<'a> Mul<f64> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        Variable {
            index: {
                let mut operations = self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, scalar), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: scalar * self.value,
        }
    }
}

impl<'a> Mul<Variable<'a>> for f64 {
    type Output = Variable<'a>;

    #[inline]
    fn mul(self, var: Self::Output) -> Self::Output {
        var * self
    }
}

impl<'a> Div<f64> for Variable<'a> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: f64) -> Self::Output {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, scalar.recip()), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: self.value / scalar,
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<Variable<'a>> for f64 {
    type Output = Variable<'a>;

    #[inline]
    fn div(self, var: Self::Output) -> Self::Output {
        var.recip() * self
    }
}

impl AddAssign<f64> for Variable<'_> {
    #[inline]
    fn add_assign(&mut self, scalar: f64) {
        *self = *self + scalar
    }
}

impl AddAssign<Self> for Variable<'_> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}

impl SubAssign<f64> for Variable<'_> {
    #[inline]
    fn sub_assign(&mut self, scalar: f64) {
        *self = *self - scalar
    }
}

impl SubAssign<Self> for Variable<'_> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl MulAssign<f64> for Variable<'_> {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        *self = *self * scalar
    }
}

impl MulAssign<Self> for Variable<'_> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other
    }
}

impl DivAssign<f64> for Variable<'_> {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        *self = *self / scalar
    }
}

impl DivAssign<Self> for Variable<'_> {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other
    }
}