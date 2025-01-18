use crate::var::Var;
use std::cell::UnsafeCell;
use crate::operations::Operation;

#[derive(Default)]
pub struct Tape {
    operations: UnsafeCell<Vec<Operation>>,
    pub(crate) count: UnsafeCell<usize>,
}

impl Tape {
    pub fn var(&self, value: f64) -> Var {
        unsafe {
            let count = self.count.get();
            let ret = Var {
                idx: *count,
                tape: self,
                value,
            };
            *count += 1;
            ret
        }
    }

    #[inline(always)]
    pub(crate) fn record(&self, operation: Operation) {
        unsafe {
            (*self.operations.get()).push(operation);
        }
    }

    #[inline]
    pub(crate) fn replay(&self, grads: &mut [f64], count: usize) {
        unsafe {
            for (i, operation) in (*self.operations.get()).iter().rev().enumerate() {
                let grad = *grads.get_unchecked(count - i - 1);
                // if grad == 0.0 {
                //     continue;
                // }
                operation.backward(grads, grad);
            }
        }
    }
}
