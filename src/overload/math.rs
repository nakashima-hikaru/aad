use crate::variable::Variable;

impl<'a> Variable<'a> {
    #[inline]
    pub fn sin(self) -> Self {
        self.apply_unary_function(f64::sin, f64::cos)
    }

    #[inline]
    pub fn ln(self) -> Self {
        self.apply_unary_function(f64::ln, f64::recip)
    }

    #[inline]
    pub fn powf(self, power: f64) -> Self {
        self.apply_scalar_function(f64::powf, |x, power| power * x.powf(power - 1.0), power)
    }

    #[inline]
    pub fn powi(self, power: i32) -> Self {
        self.apply_scalar_function::<i32>(
            f64::powi,
            |x, power| power as f64 * x.powi(power - 1),
            power,
        )
    }

    #[inline]
    pub fn exp(self) -> Self {
        self.apply_unary_function(f64::exp, f64::exp)
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        self.apply_unary_function(f64::sqrt, |x| 0.5 * x.sqrt().recip())
    }

    #[inline]
    pub fn cos(self) -> Self {
        self.apply_unary_function(f64::cos, |x| -f64::sin(x))
    }

    #[inline]
    pub fn tan(self) -> Self {
        self.apply_unary_function(f64::tan, |x| f64::cos(x).powi(2).recip())
    }

    #[inline]
    pub fn sinh(self) -> Self {
        self.apply_unary_function(f64::sinh, f64::cosh)
    }

    #[inline]
    pub fn cosh(self) -> Self {
        self.apply_unary_function(f64::cosh, f64::sinh)
    }

    #[inline]
    pub fn tanh(self) -> Self {
        self.apply_unary_function(f64::tanh, |x| f64::cosh(x).powi(2).recip())
    }

    #[inline]
    pub fn recip(self) -> Self {
        self.apply_unary_function(f64::recip, |x| -(x * x).recip())
    }
}
