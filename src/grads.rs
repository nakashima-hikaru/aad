use crate::var::Var;
pub struct Grads(pub(crate) Vec<f64>);

impl Grads {
    pub fn get_one(&self, x: &Var) -> f64 {
        self.0[x.idx]
    }
    pub fn get(&self, vars: &[Var]) -> Vec<f64> {
        vars.iter().map(|var| self.get_one(var)).collect()
    }
}
