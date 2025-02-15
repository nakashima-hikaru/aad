use crate::operation_record::OperationRecord;
use crate::variable::Variable;
use crate::Tape;
use num_traits::{Float, Inv, One, Zero};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<F: Neg<Output = F> + One + Zero> Neg for Variable<'_, F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let value = self.value.neg();
        match self.index {
            Some((i, tape)) => Variable {
                index: {
                    let operations = &mut tape.operations.borrow_mut();
                    let count = (*operations).len();
                    (*operations).push(OperationRecord([
                        (i, F::one().neg()),
                        (usize::MAX, F::zero()),
                    ]));
                    Some((count, tape))
                },
                value,
            },
            None => Variable { index: None, value },
        }
    }
}

impl<'tape, F: Neg<Output = F> + One + Zero + Copy> Neg for &Variable<'tape, F> {
    type Output = Variable<'tape, F>;
    #[inline]
    fn neg(self) -> Self::Output {
        (*self).neg()
    }
}

impl<'tape, F: Neg<Output = F> + One + Zero + Copy> Neg for &mut Variable<'tape, F> {
    type Output = Variable<'tape, F>;
    #[inline]
    fn neg(self) -> Self::Output {
        (*self).neg()
    }
}

impl<F: Add<F, Output = F> + One> Add<Self> for Variable<'_, F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[inline]
        fn create_index<F: Add<F, Output = F> + One>(
            idx: [usize; 2],
            tape: &Tape<F>,
        ) -> (usize, &Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            (*operations).push(OperationRecord([(idx[0], F::one()), (idx[1], F::one())]));
            (count, tape)
        }

        let value = self.value + rhs.value;

        match (self.index, rhs.index) {
            (Some((i, tape)), Some((j, _))) => Variable {
                index: Some(create_index([i, j], tape)),
                value,
            },
            (None, None) => Variable { index: None, value },
            (None, Some((j, tape))) => Variable {
                index: Some(create_index([usize::MAX, j], tape)),
                value,
            },
            (Some((i, tape)), None) => Variable {
                index: Some(create_index([i, usize::MAX], tape)),
                value,
            },
        }
    }
}

impl<F: Sub<F, Output = F> + One + Neg<Output = F>> Sub<Self> for Variable<'_, F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[inline]
        fn create_index<F: Sub<F, Output = F> + One + Neg<Output = F>>(
            i: usize,
            j: usize,
            tape: &Tape<F>,
        ) -> (usize, &Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            (*operations).push(OperationRecord([(i, F::one()), (j, -F::one())]));
            (count, tape)
        }

        let value = self.value - rhs.value;

        match (self.index, rhs.index) {
            (Some((i, tape)), Some((j, _))) => Variable {
                index: Some(create_index(i, j, tape)),
                value,
            },
            (None, None) => Variable { index: None, value },
            (None, Some((j, tape))) => Variable {
                index: Some(create_index(usize::MAX, j, tape)),
                value,
            },
            (Some((i, tape)), None) => Variable {
                index: Some(create_index(i, usize::MAX, tape)),
                value,
            },
        }
    }
}

impl<'tape, F: Sub<F, Output = F> + One + Neg<Output = F> + Copy> Sub<Self>
    for &Variable<'tape, F>
{
    type Output = Variable<'tape, F>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (*self).sub(*rhs)
    }
}

impl<'tape, F: Sub<F, Output = F> + One + Neg<Output = F> + Copy> Sub<Variable<'tape, F>>
    for &Variable<'tape, F>
{
    type Output = Variable<'tape, F>;
    #[inline]
    fn sub(self, rhs: Variable<'tape, F>) -> Self::Output {
        (*self).sub(rhs)
    }
}

impl<F: Mul<F, Output = F> + Copy> Mul<Self> for Variable<'_, F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[inline]
        fn create_index<F: Mul<F, Output = F> + Copy>(
            value: F,
            rhs: F,
            i: usize,
            j: usize,
            tape: &Tape<F>,
        ) -> (usize, &Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            (*operations).push(OperationRecord([(i, rhs), (j, value)]));
            (count, tape)
        }

        let value = self.value * rhs.value;

        match (self.index, rhs.index) {
            (Some((i, tape)), Some((j, _))) => Variable {
                index: Some(create_index(self.value, rhs.value, i, j, tape)),
                value,
            },
            (None, None) => Variable { index: None, value },
            (None, Some((j, tape))) => Variable {
                index: Some(create_index(self.value, rhs.value, usize::MAX, j, tape)),
                value,
            },
            (Some((i, tape)), None) => Variable {
                index: Some(create_index(self.value, rhs.value, i, usize::MAX, tape)),
                value,
            },
        }
    }
}

impl<'tape, F: Mul<F, Output = F> + Copy> Mul<Self> for &Variable<'tape, F> {
    type Output = Variable<'tape, F>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        (*self).mul(*rhs)
    }
}

impl<'tape, F: Mul<F, Output = F> + Copy> Mul<Variable<'tape, F>> for &Variable<'tape, F> {
    type Output = Variable<'tape, F>;
    #[inline]
    fn mul(self, rhs: Variable<'tape, F>) -> Self::Output {
        (*self).mul(rhs)
    }
}

impl<F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>> Div<Self>
    for Variable<'_, F>
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.apply_binary_function(rhs, |x, y| x / y, |x, y| (y.inv(), -x / (y * y)))
    }
}

impl<'tape, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<Self> for &Variable<'tape, F>
{
    type Output = Variable<'tape, F>;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        (*self).div(*rhs)
    }
}

impl<'tape, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<Variable<'tape, F>> for &Variable<'tape, F>
{
    type Output = Variable<'tape, F>;
    #[inline]
    fn div(self, rhs: Variable<'tape, F>) -> Self::Output {
        (*self).div(rhs)
    }
}

impl<F: Copy + One + Zero> AddAssign<Self> for Variable<'_, F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<'tape, F: Copy + One + Zero> AddAssign<&Variable<'tape, F>> for Variable<'tape, F> {
    #[inline]
    fn add_assign(&mut self, rhs: &Variable<'tape, F>) {
        *self = *self + *rhs;
    }
}

impl<F: Copy + One + Neg<Output = F> + Sub<F, Output = F>> SubAssign<Self> for Variable<'_, F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<'tape, F: Copy + One + Neg<Output = F> + Sub<F, Output = F>> SubAssign<&Variable<'tape, F>>
    for Variable<'tape, F>
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Variable<'tape, F>) {
        *self = *self - *rhs;
    }
}
impl<F: Copy + Mul<F, Output = F>> MulAssign<Self> for Variable<'_, F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'tape, F: Copy + Mul<F, Output = F>> MulAssign<&Variable<'tape, F>> for Variable<'tape, F> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Variable<'tape, F>) {
        *self = *self * *rhs;
    }
}

impl<F: Copy + Div<Self, Output = Self> + Float + Inv<Output = F>> DivAssign<Self>
    for Variable<'_, F>
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<'tape, F: Copy + Div<Self, Output = Self> + Float + Inv<Output = F>>
    DivAssign<&Variable<'tape, F>> for Variable<'tape, F>
{
    #[inline]
    fn div_assign(&mut self, rhs: &Variable<'tape, F>) {
        *self = *self / *rhs;
    }
}

impl<F: Copy + Zero + Add<F, Output = F> + One> Sum for Variable<'_, F> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Variable::zero(), |acc, x| acc + x)
    }
}

impl<'tape, 'a, F: Copy + Zero + Add<F, Output = F> + One> Sum<&'a Variable<'tape, F>>
    for Variable<'tape, F>
{
    #[inline]
    fn sum<I: Iterator<Item = &'a Variable<'tape, F>>>(iter: I) -> Self {
        iter.fold(Variable::zero(), |acc, x| acc + *x)
    }
}

macro_rules! impl_add_ref {
    (for $lhs:ty, rhs: & $rhs_inner:ty, deref_rhs) => {
        impl<'tape, F: Add<F, Output = F> + One + Copy> Add<&$rhs_inner> for $lhs {
            type Output = Variable<'tape, F>;
            #[inline]
            fn add(self, rhs: &$rhs_inner) -> Self::Output {
                (*self).add(*rhs)
            }
        }
    };

    (for $lhs:ty, rhs: &mut $rhs_inner:ty, deref_rhs) => {
        impl<'tape, F: Add<F, Output = F> + One + Copy> Add<&mut $rhs_inner> for $lhs {
            type Output = Variable<'tape, F>;
            #[inline]
            fn add(self, rhs: &mut $rhs_inner) -> Self::Output {
                (*self).add(*rhs)
            }
        }
    };

    (for $lhs:ty, rhs: $rhs:ty, no_deref_rhs) => {
        impl<'tape, F: Add<F, Output = F> + One + Copy> Add<$rhs> for $lhs {
            type Output = Variable<'tape, F>;
            #[inline]
            fn add(self, rhs: $rhs) -> Self::Output {
                (*self).add(rhs)
            }
        }
    };
}

impl_add_ref!(for &Variable<'tape, F>, rhs: &Variable<'tape, F>, deref_rhs);
impl_add_ref!(for &Variable<'tape, F>, rhs: Variable<'tape, F>, no_deref_rhs);
impl_add_ref!(for &Variable<'tape, F>, rhs: &mut Variable<'tape, F>, deref_rhs);

impl_add_ref!(for &mut Variable<'tape, F>, rhs: Variable<'tape, F>, no_deref_rhs);
impl_add_ref!(for &mut Variable<'tape, F>, rhs: &Variable<'tape, F>, deref_rhs);
impl_add_ref!(for &mut Variable<'tape, F>, rhs: &mut Variable<'tape, F>, deref_rhs);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    #[test]
    fn test_sum() {
        let tape = Tape::new();
        let values = [1.0, 2.0, 3.0];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.into_iter().sum();

        assert_eq!(sum.value, 6.0);
        assert!(std::ptr::eq(sum.index.unwrap().1, &tape));
    }

    #[test]
    fn test_sum_empty() {
        let tape = Tape::new();
        let values = [];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.into_iter().sum();

        assert_eq!(sum.value, 0.0);
        assert!(sum.index.is_none());
    }

    #[test]
    fn test_sum_empty2() {
        let tape = Tape::new();
        let values = [];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.into_iter().sum();

        let x = tape.create_variable(5.0);

        let y = sum + x;

        assert_eq!(y.value, 5.0);
        assert!(std::ptr::eq(y.index.unwrap().1, &tape));
    }
}
