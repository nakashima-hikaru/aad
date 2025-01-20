use crate::operations::Operation;
use crate::var::Var;
use std::cell::RefCell;

#[derive(Debug, Default)]
pub struct Tape {
    pub(crate) operations: RefCell<Vec<Operation>>,
}

impl Tape {
    #[inline]
    pub fn var(&self, value: f64) -> Var {
        Var {
            idx: {
                let mut operations = self.operations.borrow_mut();
                let count = (*operations).len();
                (*operations).push(Operation([(0, 0.0), (0, 0.0)]));
                count
            },
            tape: self,
            value,
        }
    }
}
