use crate::gradients::Gradients;
use crate::operation_record::OperationRecord;
use crate::tape::Tape;

#[derive(Clone, Copy, Debug)]
pub struct Variable<'a> {
    pub(crate) index: usize,
    pub(crate) tape: &'a Tape,
    pub(crate) value: f64,
}

type BinaryFn<T, S = T> = fn(T, S) -> T;
type UnaryFn<T> = fn(T) -> T;

impl Variable<'_> {
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    #[inline]
    pub fn compute_gradients(&self) -> Gradients {
        let operations = &mut self.tape.operations.borrow_mut();
        let count = (*operations).len();
        let mut grads = vec![0.0; count];
        grads[self.index] = 1.0;

        for (i, operation) in (*operations).iter().enumerate().rev() {
            let grad = grads[i];
            if grad == 0.0 {
                continue;
            }
            for j in 0..2 {
                grads[operation.0[j].0] += operation.0[j].1 * grad;
            }
        }

        Gradients(grads)
    }

    #[inline(always)]
    pub fn apply_unary_function(&self, f: UnaryFn<f64>, df: UnaryFn<f64>) -> Self {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([(self.index, df(self.value)), (0, 0.0)]));
                count
            },
            tape: self.tape,
            value: f(self.value),
        }
    }

    #[inline(always)]
    pub fn apply_scalar_function<T: Copy>(
        &self,
        f: BinaryFn<f64, T>,
        df: BinaryFn<f64, T>,
        scalar: T,
    ) -> Self {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([
                    (self.index, df(self.value, scalar)),
                    (0, 0.0),
                ]));
                count
            },
            tape: self.tape,
            value: f(self.value, scalar),
        }
    }

    #[inline(always)]
    pub fn apply_binary_function(
        &self,
        other: &Self,
        f: BinaryFn<f64>,
        dfdx: BinaryFn<f64>,
        dfdy: BinaryFn<f64>,
    ) -> Self {
        Variable {
            index: {
                let operations = &mut self.tape.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(OperationRecord([
                    (self.index, dfdx(self.value, other.value)),
                    (other.index, dfdy(self.value, other.value)),
                ]));
                count
            },
            tape: self.tape,
            value: f(self.value, other.value),
        }
    }
}
