use std::ops::AddAssign;


pub(crate) struct Operation {
    pub x: [usize; 2],
    pub dfdx: [f64; 2],
}

impl Operation {
    #[inline]
    pub(crate) fn backward(&self, grads: &mut [f64], grad: f64) {
        unsafe {
            for (x, dfdx) in self.x.iter().zip(self.dfdx) {
                grads
                    .get_unchecked_mut(*x)
                    .add_assign(dfdx * grad);
            }
        }
    }
}

