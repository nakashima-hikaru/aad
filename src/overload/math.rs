use crate::variable::Variable;

//
impl Variable<'_> {
    #[inline]
    pub fn ln(self) -> Self {
        self.apply_unary_function(f64::ln, f64::recip)
    }

    #[inline]
    pub fn log(self, base: f64) -> Self {
        self.apply_scalar_function(|x, b| x.log(b), |x, b| x.recip() / b.ln(), base)
    }
    #[inline]
    pub fn powf(self, power: f64) -> Self {
        self.apply_scalar_function(f64::powf, |x, p| p * x.powf(p - 1.0), power)
    }

    #[inline]
    pub fn powi(self, power: i32) -> Self {
        self.apply_scalar_function::<i32>(f64::powi, |x, p| f64::from(p) * x.powi(p - 1), power)
    }

    #[inline]
    pub fn exp(self) -> Self {
        self.apply_unary_function(f64::exp, f64::exp)
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        self.apply_unary_function(f64::sqrt, |x| 0.5f64 * x.sqrt().recip())
    }

    #[inline]
    pub fn cbrt(self) -> Self {
        self.apply_unary_function(f64::cbrt, |x| (1.0 / 3.0) * x.powf(-2.0 / 3.0))
    }

    #[inline]
    pub fn recip(self) -> Self {
        self.apply_unary_function(f64::recip, |x| -(x * x).recip())
    }

    #[inline]
    pub fn exp2(self) -> Self {
        self.apply_unary_function(f64::exp2, |x| f64::ln(2.0) * f64::exp2(x))
    }

    #[inline]
    pub fn log2(self) -> Self {
        self.apply_unary_function(f64::log2, |x| x.recip() * f64::ln(2.0).recip())
    }

    #[inline]
    pub fn log10(self) -> Self {
        self.apply_unary_function(f64::log10, |x| x.recip() * f64::ln(10.0).recip())
    }

    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        self.apply_binary_function(&other, f64::hypot, |x, y| {
            let denom = (x * x + y * y).sqrt();
            (x / denom, y / denom)
        })
    }

    #[inline]
    pub fn abs(self) -> Self {
        self.apply_unary_function(f64::abs, f64::signum)
    }
}

// trigonometric functions
impl Variable<'_> {
    #[inline]
    pub fn sin(self) -> Self {
        self.apply_unary_function(f64::sin, f64::cos)
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
    pub fn asin(self) -> Self {
        self.apply_unary_function(f64::asin, |x| (1.0 - x * x).sqrt().recip())
    }

    #[inline]
    pub fn acos(self) -> Self {
        self.apply_unary_function(f64::acos, |x| -(1.0 - x * x).sqrt().recip())
    }

    #[inline]
    pub fn atan(self) -> Self {
        self.apply_unary_function(f64::atan, |x| (1.0 + x * x).recip())
    }

    #[inline]
    pub fn asinh(self) -> Self {
        self.apply_unary_function(f64::asinh, |x| (x * x + 1.0).sqrt().recip())
    }

    #[inline]
    pub fn acosh(self) -> Self {
        self.apply_unary_function(f64::acosh, |x| (x * x - 1.0).sqrt().recip())
    }

    #[inline]
    pub fn atanh(self) -> Self {
        self.apply_unary_function(f64::atanh, |x| (1.0 - x * x).recip())
    }
}
