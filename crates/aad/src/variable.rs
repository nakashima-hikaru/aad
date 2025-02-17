use std::{
    cmp::Ordering,
    ops::{Add, Mul},
};

use crate::gradients::Gradients;
use crate::operation_record::OperationRecord;
use crate::tape::Tape;
use num_traits::{One, Zero};

#[derive(Clone, Copy, Debug)]
/// A variable type that tracks operations for automatic differentiation.
///
/// This struct represents a variable in the computation graph, storing its value
/// and maintaining references to the tape that records operations performed on it.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the reference to the tape
/// * `F` - The underlying numeric type (typically `f32` or `f64`)
///
/// # Fields
///
/// * `index` - The unique index of this variable in the computation tape
/// * `tape` - Reference to the tape that records operations on this variable
/// * `value` - The current value of the variable
pub struct Variable<'a, F> {
    pub(crate) index: Option<(usize, &'a Tape<F>)>,
    pub(crate) value: F,
}

type BinaryFn<T, S = T> = fn(T, S) -> T;
type UnaryFn<T> = fn(T) -> T;
type BinaryPairFn<T> = fn(T, T) -> (T, T);

impl<F: Copy> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub const fn value(&self) -> F {
        self.value
    }
    #[inline]
    #[must_use]
    pub fn apply_binary_function(&self, rhs: &Self, f: BinaryFn<F>, dfdx: BinaryPairFn<F>) -> Self {
        #[inline]
        fn create_index<'a, F>(
            value: F,
            rhs: Variable<'a, F>,
            dfdx: fn(F, F) -> (F, F),
            idx: [usize; 2],
            tape: &'a Tape<F>,
        ) -> (usize, &'a Tape<F>) {
            let operations = &mut tape.operations.borrow_mut();
            let count = (*operations).len();
            let df = dfdx(value, rhs.value);
            (*operations).push(OperationRecord([(idx[0], df.0), (idx[1], df.1)]));
            (count, tape)
        }
        let value = f(self.value, rhs.value);
        match (self.index, rhs.index) {
            (Some((i, tape)), Some((j, _))) => Variable {
                index: Some(create_index(self.value, *rhs, dfdx, [i, j], tape)),
                value,
            },
            (None, None) => Variable { index: None, value },
            (None, Some((j, tape))) => Variable {
                index: Some(create_index(self.value, *rhs, dfdx, [usize::MAX, j], tape)),
                value,
            },
            (Some((i, tape)), None) => Variable {
                index: Some(create_index(self.value, *rhs, dfdx, [i, usize::MAX], tape)),
                value,
            },
        }
    }
}

impl<F: Copy + Zero> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub fn apply_unary_function(&self, f: UnaryFn<F>, df: UnaryFn<F>) -> Self {
        let value = f(self.value);
        match self.index {
            Some((i, tape)) => Variable {
                index: {
                    let operations = &mut tape.operations.borrow_mut();
                    let count = (*operations).len();
                    (*operations).push(OperationRecord([
                        (i, df(self.value)),
                        (usize::MAX, F::zero()),
                    ]));
                    Some((count, tape))
                },
                value,
            },
            None => Variable { index: None, value },
        }
    }

    #[inline]
    #[must_use]
    pub fn apply_scalar_function<T: Copy>(
        &self,
        f: BinaryFn<F, T>,
        df: BinaryFn<F, T>,
        scalar: T,
    ) -> Self {
        let value = f(self.value, scalar);
        match self.index {
            Some((i, tape)) => Variable {
                index: {
                    let operations = &mut tape.operations.borrow_mut();
                    let count = (*operations).len();
                    (*operations).push(OperationRecord([
                        (i, df(self.value, scalar)),
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

impl<F: Copy + One + Zero> Variable<'_, F> {
    #[inline]
    #[must_use]
    pub fn compute_gradients(&self) -> Gradients<F> {
        let operations = &mut self.index.unwrap().1.operations.borrow_mut();
        let mut grads = vec![F::zero(); (*operations).len()];
        grads[self.index.unwrap().0] = F::one();

        for (i, operation) in (*operations).iter().enumerate().rev() {
            let grad = grads[i];
            if grad.is_zero() {
                continue;
            }
            for j in 0..2 {
                let (idx, val) = operation.0[j];
                if idx == usize::MAX {
                    continue;
                }
                grads[idx] = grads[idx] + val * grad;
            }
        }

        Gradients(grads)
    }
}

impl<F: PartialOrd> PartialOrd for Variable<'_, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F: PartialOrd<F>> PartialOrd<F> for Variable<'_, F> {
    #[inline]
    fn partial_cmp(&self, other: &F) -> Option<Ordering> {
        self.value.partial_cmp(other)
    }
}

impl<F: PartialOrd> PartialEq for Variable<'_, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: PartialEq<F>> PartialEq<F> for Variable<'_, F> {
    #[inline]
    fn eq(&self, other: &F) -> bool {
        self.value == *other
    }
}

impl<F: Zero> Zero for Variable<'_, F>
where
    Self: Add<Self, Output = Self>,
{
    #[inline]
    #[must_use]
    fn zero() -> Self {
        Self::constant(F::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    #[inline]
    fn set_zero(&mut self) {
        *self = Self::zero();
    }
}

impl<F: One> One for Variable<'_, F>
where
    Self: Mul<Self, Output = Self>,
{
    #[inline]
    #[must_use]
    fn one() -> Self {
        Self::constant(F::one())
    }

    #[inline]
    fn set_one(&mut self) {
        *self = Self::one();
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == Self::one()
    }
}

impl<'a, F> Variable<'a, F> {
    #[inline]
    #[must_use]
    pub fn constant(value: F) -> Variable<'a, F> {
        Variable { index: None, value }
    }
}

impl<F> From<F> for Variable<'_, F> {
    #[inline]
    fn from(value: F) -> Self {
        Self::constant(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_second_gradients() {
        let tape = Tape::new();
        let tape2 = Tape::new();
        let [x, y] = tape.create_variables(&[1.0, 2.0]);
        let [x, y] = tape2.create_variables(&[x, y]);
        let z = x * x + y;
        let grads = z.compute_gradients();
        let grad = grads.get_gradient(&x);
        let z = grad.compute_gradients();
        let grad2 = z.get_gradient(&x.value);
        assert_eq!(grad2, 2.0);
    }
}
