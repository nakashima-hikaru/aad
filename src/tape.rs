use crate::operations::Operation;
use crate::var::Var;
use std::cell::UnsafeCell;

#[derive(Default)]
pub struct Tape {
    pub(crate) operations: UnsafeCell<Vec<Operation>>,
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

    #[inline]
    pub(crate) fn record(&self, operation: Operation) {
        unsafe {
            (*self.operations.get()).push(operation);
        }
    }
}
