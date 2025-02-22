use crate::variable::Variable;
use num_traits::{Inv, Zero};
use std::ops::{Mul, Neg};

impl<'a, F: Copy + Zero + Inv<Output = F> + Mul<Output = F> + Neg<Output = F>> Inv
    for Variable<'a, F>
{
    type Output = Variable<'a, F>;

    #[inline]
    fn inv(self) -> Self::Output {
        self.apply_unary_function(F::inv, |x| x.mul(x).inv().neg())
    }
}
