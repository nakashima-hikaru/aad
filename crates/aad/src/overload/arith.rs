use crate::operation_record::OperationRecord;
use crate::variable::Variable;
use crate::Tape;
use num_traits::{Inv, One, Zero};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<'a, F: Neg<Output = F> + One + Zero + Copy> Neg for &Variable<'a, F> {
    type Output = Variable<'a, F>;
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

impl<'a, F> Neg for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Neg<Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;
    #[inline]
    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}

impl<'a, F: Add<F, Output = F> + One + Copy> Add<Self> for &Variable<'a, F> {
    type Output = Variable<'a, F>;

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

impl<'a, F> Add<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Add<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<'a, F> Add<Variable<'a, F>> for &Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Add<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;

    #[inline]
    fn add(self, rhs: Variable<'a, F>) -> Self::Output {
        self.add(&rhs)
    }
}

impl<'a, F> Add<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Add<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;

    #[inline]
    fn add(self, rhs: &Variable<'a, F>) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<'a, F: Sub<F, Output = F> + One + Neg<Output = F> + Copy> Sub<Self> for &Variable<'a, F> {
    type Output = Variable<'a, F>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[inline]
        fn create_index<F: Sub<F, Output = F> + One + Neg<Output = F>>(
            idx: [usize; 2],
            tape: &Tape<F>,
        ) -> (usize, &Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            (*operations).push(OperationRecord([
                (idx[0], F::one()),
                (idx[1], F::one().neg()),
            ]));
            (count, tape)
        }

        let value = self.value - rhs.value;

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

impl<'a, F> Sub<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Sub<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl<'a, F> Sub<Variable<'a, F>> for &Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Sub<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;

    #[inline]
    fn sub(self, rhs: Variable<'a, F>) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<'a, F> Sub<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Sub<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;

    #[inline]
    fn sub(self, rhs: &Variable<'a, F>) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<'a, F: Mul<F, Output = F> + Copy> Mul<Self> for &Variable<'a, F> {
    type Output = Variable<'a, F>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[inline]
        fn create_index<F: Mul<F, Output = F> + Copy>(
            value: F,
            rhs: F,
            idx: [usize; 2],
            tape: &Tape<F>,
        ) -> (usize, &Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            (*operations).push(OperationRecord([(idx[0], rhs), (idx[1], value)]));
            (count, tape)
        }

        let value = self.value * rhs.value;

        match (self.index, rhs.index) {
            (Some((i, tape)), Some((j, _))) => Variable {
                index: Some(create_index(self.value, rhs.value, [i, j], tape)),
                value,
            },
            (None, None) => Variable { index: None, value },
            (None, Some((j, tape))) => Variable {
                index: Some(create_index(self.value, rhs.value, [usize::MAX, j], tape)),
                value,
            },
            (Some((i, tape)), None) => Variable {
                index: Some(create_index(self.value, rhs.value, [i, usize::MAX], tape)),
                value,
            },
        }
    }
}

impl<'a, F> Mul<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Mul<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<'a, F> Mul<Variable<'a, F>> for &Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Mul<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;
    #[inline]
    fn mul(self, rhs: Variable<'a, F>) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<'a, F> Mul<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Mul<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    type Output = Variable<'a, F>;

    #[inline]
    fn mul(self, rhs: &Variable<'a, F>) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<'a, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<Self> for &Variable<'a, F>
{
    type Output = Variable<'a, F>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.apply_binary_function(rhs, |x, y| x / y, |x, y| (y.inv(), -x / (y * y)))
    }
}

impl<'a, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<Self> for Variable<'a, F>
{
    type Output = Variable<'a, F>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl<'a, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<&Variable<'a, F>> for Variable<'a, F>
{
    type Output = Variable<'a, F>;

    #[inline]
    fn div(self, rhs: &Variable<'a, F>) -> Self::Output {
        (&self).div(rhs)
    }
}

impl<'a, F: Copy + Div<F, Output = F> + Inv<Output = F> + Neg<Output = F> + Mul<Output = F>>
    Div<Variable<'a, F>> for &Variable<'a, F>
{
    type Output = Variable<'a, F>;

    #[inline]
    fn div(self, rhs: Variable<'a, F>) -> Self::Output {
        self.div(&rhs)
    }
}

impl<'a, F> AddAssign<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Add<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl<'a, F> AddAssign<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Add<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Variable<'a, F>) {
        *self = &*self + rhs;
    }
}

impl<'a, F> SubAssign<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Sub<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<'a, F> SubAssign<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Sub<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Variable<'a, F>) {
        *self = &*self - rhs;
    }
}

impl<'a, F> MulAssign<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Mul<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}

impl<'a, F> MulAssign<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Mul<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &Variable<'a, F>) {
        *self = &*self * rhs;
    }
}

impl<'a, F> DivAssign<Self> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Div<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = &*self / &rhs;
    }
}

impl<'a, F> DivAssign<&Variable<'a, F>> for Variable<'a, F>
where
    for<'b> &'b Variable<'a, F>: Div<&'b Variable<'a, F>, Output = Variable<'a, F>>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &Variable<'a, F>) {
        *self = &*self / rhs;
    }
}

impl<'a, 'b, F> Sum<&'a Variable<'b, F>> for Variable<'b, F>
where
    for<'c> &'c Variable<'b, F>: Add<&'c Variable<'b, F>, Output = Variable<'b, F>>,
    Variable<'b, F>: Zero,
{
    #[inline]
    fn sum<I: Iterator<Item = &'a Variable<'b, F>>>(iter: I) -> Self {
        iter.fold(Variable::zero(), |acc, x| &acc + x)
    }
}

impl<'a, F> Sum<Variable<'a, F>> for Variable<'a, F>
where
    Variable<'a, F>: Zero,
{
    #[inline]
    fn sum<I: Iterator<Item = Variable<'a, F>>>(iter: I) -> Self {
        iter.fold(Variable::zero(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    #[test]
    fn test_sum() {
        let tape = Tape::new();
        let values = [1.0, 2.0, 3.0];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.iter().sum();

        assert_eq!(sum.value, 6.0);
        assert!(std::ptr::eq(sum.index.unwrap().1, &tape));
    }

    #[test]
    fn test_sum_empty() {
        let tape = Tape::new();
        let values = [];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.iter().sum();

        assert_eq!(sum.value, 0.0);
        assert!(sum.index.is_none());
    }

    #[test]
    fn test_sum_empty2() {
        let tape = Tape::new();
        let values = [];
        let variables = tape.create_variables(&values);

        let sum: Variable<f64> = variables.iter().sum();

        let x = tape.create_variable(5.0);

        let y = sum + x;

        assert_eq!(y.value, 5.0);
        assert!(std::ptr::eq(y.index.unwrap().1, &tape));
    }
}
