impl<'a, 'b> Variable<'a, Variable<'b, f64>> {
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        self.apply_unary_function(|v: Variable<f64>| v.ln(), |v: Variable<f64>| v.recip())
    }

    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        self.apply_unary_function(|v: Variable<f64>| v.sin(), |v: Variable<f64>| v.cos())
    }
} 