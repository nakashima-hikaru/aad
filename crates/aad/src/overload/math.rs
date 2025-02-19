use crate::{variable::Variable, FloatLike};
use num_traits::One;
use std::ops::{Div as _, Mul, Neg, Sub as _};

impl FloatLike<f64> for Variable<'_, f64> {
    #[inline]
    #[must_use]
    fn sin(self) -> Self {
        self.apply_unary_function(f64::sin, f64::cos)
    }

    #[inline]
    #[must_use]
    fn cos(self) -> Self {
        self.apply_unary_function(f64::cos, |x| -x.sin())
    }

    #[inline]
    #[must_use]
    fn tan(self) -> Self {
        self.apply_unary_function(f64::tan, |x| x.cos().powi(2).recip())
    }

    #[inline]
    #[must_use]
    fn ln(self) -> Self {
        self.apply_unary_function(f64::ln, f64::recip)
    }

    #[inline]
    #[must_use]
    fn log(self, base: f64) -> Self {
        self.apply_scalar_function(f64::log, |x, b| x.recip().mul(b.ln().recip()), base)
    }

    #[inline]
    #[must_use]
    fn powf(self, power: f64) -> Self {
        self.apply_scalar_function(f64::powf, |x, p| p.mul(x.powf(p.sub(f64::one()))), power)
    }

    #[inline]
    #[must_use]
    fn powi(self, power: i32) -> Self {
        self.apply_scalar_function(f64::powi, |x, p| f64::from(p) * x.powi(p - 1), power)
    }

    #[inline]
    #[must_use]
    fn exp(self) -> Self {
        self.apply_unary_function(f64::exp, f64::exp)
    }

    #[inline]
    #[must_use]
    fn sqrt(self) -> Self {
        self.apply_unary_function(f64::sqrt, |x| x.sqrt().recip().div(f64::one() + f64::one()))
    }

    #[inline]
    #[must_use]
    fn cbrt(self) -> Self {
        self.apply_unary_function(f64::cbrt, |x| {
            x.powf(-(f64::from(2) / f64::from(3))) / f64::from(3)
        })
    }

    #[inline]
    #[must_use]
    fn recip(self) -> Self {
        self.apply_unary_function(f64::recip, |x| x.powi(2).recip().neg())
    }

    #[inline]
    #[must_use]
    fn exp2(self) -> Self {
        self.apply_unary_function(f64::exp2, |x| {
            f64::ln(f64::one() + f64::one()) * f64::exp2(x)
        })
    }

    #[inline]
    #[must_use]
    fn log2(self) -> Self {
        self.apply_unary_function(f64::log2, |x| {
            x.recip() * f64::ln(f64::one() + f64::one()).recip()
        })
    }

    #[inline]
    #[must_use]
    fn log10(self) -> Self {
        self.apply_unary_function(f64::log10, |x| x.recip() * 10.0_f64.ln().recip())
    }

    #[inline]
    #[must_use]
    fn hypot(self, other: Self) -> Self {
        self.apply_binary_function(&other, f64::hypot, |x, y| {
            let denom = x.hypot(y);
            (x / denom, y / denom)
        })
    }

    #[inline]
    #[must_use]
    fn abs(self) -> Self {
        self.apply_unary_function(f64::abs, f64::signum)
    }

    #[inline]
    #[must_use]
    fn sinh(self) -> Self {
        self.apply_unary_function(f64::sinh, f64::cosh)
    }

    #[inline]
    #[must_use]
    fn cosh(self) -> Self {
        self.apply_unary_function(f64::cosh, f64::sinh)
    }

    #[inline]
    #[must_use]
    fn tanh(self) -> Self {
        self.apply_unary_function(f64::tanh, |x| f64::cosh(x).powi(2).recip())
    }

    #[inline]
    #[must_use]
    fn asin(self) -> Self {
        self.apply_unary_function(f64::asin, |x| (f64::one() - x * x).sqrt().recip())
    }

    #[inline]
    #[must_use]
    fn acos(self) -> Self {
        self.apply_unary_function(f64::acos, |x| -(f64::one() - x * x).sqrt().recip())
    }

    #[inline]
    #[must_use]
    fn atan(self) -> Self {
        self.apply_unary_function(f64::atan, |x| (f64::one() + x * x).recip())
    }

    #[inline]
    #[must_use]
    fn asinh(self) -> Self {
        self.apply_unary_function(f64::asinh, |x| (x * x + f64::one()).sqrt().recip())
    }

    #[inline]
    #[must_use]
    fn acosh(self) -> Self {
        self.apply_unary_function(f64::acosh, |x| (x * x - f64::one()).sqrt().recip())
    }

    #[inline]
    #[must_use]
    fn atanh(self) -> Self {
        self.apply_unary_function(f64::atanh, |x| (f64::one() - x * x).recip())
    }
}

// impl<'a, 'b> FloatLike<f64> for Variable<'a, Variable<'b, f64>> {
//     #[inline]
//     #[must_use]
//     fn sin(self) -> Self {
//         self.apply_unary_function(Variable::sin, Variable::cos)
//     }

//     #[inline]
//     #[must_use]
//     fn cos(self) -> Self {
//         self.apply_unary_function(Variable::cos, |x| -x.sin())
//     }

//     #[inline]
//     #[must_use]
//     fn tan(self) -> Self {
//         self.apply_unary_function(Variable::tan, |x| x.cos().powi(2).recip())
//     }

//     #[inline]
//     #[must_use]
//     fn ln(self) -> Self {
//         self.apply_unary_function(Variable::ln, Variable::recip)
//     }

//     #[inline]
//     #[must_use]
//     fn log(self, base: f64) -> Self {
//         self.apply_scalar_function(Variable::log, |x, b| x.recip().mul(b.ln().recip()), base)
//     }

//     #[inline]
//     #[must_use]
//     fn powf(self, power: f64) -> Self {
//         self.apply_scalar_function(
//             Variable::powf,
//             |x, p| p.mul(x.powf(p.sub(f64::one()))),
//             power,
//         )
//     }

//     #[inline]
//     #[must_use]
//     fn powi(self, power: i32) -> Self {
//         self.apply_scalar_function(Variable::powi, |x, p| f64::from(p) * x.powi(p - 1), power)
//     }

//     #[inline]
//     #[must_use]
//     fn exp(self) -> Self {
//         self.apply_unary_function(Variable::exp, Variable::exp)
//     }

//     #[inline]
//     #[must_use]
//     fn sqrt(self) -> Self {
//         self.apply_unary_function(Variable::sqrt, |x| {
//             x.sqrt().recip().div(f64::one() + f64::one())
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn cbrt(self) -> Self {
//         self.apply_unary_function(Variable::cbrt, |x| {
//             x.powf(-(f64::from(2) / f64::from(3))) / f64::from(3)
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn recip(self) -> Self {
//         self.apply_unary_function(Variable::recip, |x| x.powi(2).recip().neg())
//     }

//     #[inline]
//     #[must_use]
//     fn exp2(self) -> Self {
//         self.apply_unary_function(Variable::exp2, |x| {
//             Variable::ln(Variable::one() + Variable::one()) * Variable::exp2(x)
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn log2(self) -> Self {
//         self.apply_unary_function(Variable::log2, |x| {
//             x.recip() * Variable::ln(Variable::one() + Variable::one()).recip()
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn log10(self) -> Self {
//         self.apply_unary_function(Variable::log10, |x| {
//             x.recip() * 10.0_f64.ln().recip()
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn hypot(self, other: Self) -> Self {
//         self.apply_binary_function(&other, Variable::hypot, |x, y| {
//             let denom = x.hypot(y);
//             (x / denom, y / denom)
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn abs(self) -> Self {
//         todo!()
//         // self.apply_unary_function(Variable::abs, Variable::signum)
//     }

//     #[inline]
//     #[must_use]
//     fn sinh(self) -> Self {
//         self.apply_unary_function(Variable::sinh, Variable::cosh)
//     }

//     #[inline]
//     #[must_use]
//     fn cosh(self) -> Self {
//         self.apply_unary_function(Variable::cosh, Variable::sinh)
//     }

//     #[inline]
//     #[must_use]
//     fn tanh(self) -> Self {
//         self.apply_unary_function(Variable::tanh, |x| Variable::cosh(x).powi(2).recip())
//     }

//     #[inline]
//     #[must_use]
//     fn asin(self) -> Self {
//         self.apply_unary_function(Variable::asin, |x| (Variable::one() - x * x).sqrt().recip())
//     }

//     #[inline]
//     #[must_use]
//     fn acos(self) -> Self {
//         self.apply_unary_function(Variable::acos, |x| {
//             -(Variable::one() - x * x).sqrt().recip()
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn atan(self) -> Self {
//         self.apply_unary_function(Variable::atan, |x| (Variable::one() + x * x).recip())
//     }

//     #[inline]
//     #[must_use]
//     fn asinh(self) -> Self {
//         self.apply_unary_function(Variable::asinh, |x| {
//             (x * x + Variable::one()).sqrt().recip()
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn acosh(self) -> Self {
//         self.apply_unary_function(Variable::acosh, |x| {
//             (x * x - Variable::one()).sqrt().recip()
//         })
//     }

//     #[inline]
//     #[must_use]
//     fn atanh(self) -> Self {
//         self.apply_unary_function(Variable::atanh, |x| (Variable::one() - x * x).recip())
//     }
// }
