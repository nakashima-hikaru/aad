use aad::autodiff;

#[autodiff]
fn f(x: f64, y: f64) -> f64 {
    x.powi(2) * y.sin()
}

#[autodiff]
fn g(x: f64, y: f64, z: f64) -> f64 {
    x * f(x, y) + z.ln()
}

#[test]
fn main() {
    use aad::Tape;
    let tape = Tape::default();
    let [x, y, z] = tape.create_variables(&[2.0, 3.0, 4.0]);
    let w = g(x, y, z);
    let grads = w.compute_gradients();

    let [dx, dy, dz] = grads.get_gradients(&[x, y, z]);
    assert_eq!(dx, 3.0 * x.value().powi(2) * y.value().sin());
    assert_eq!(dy, x.value().powi(3) * y.value().cos());
    assert_eq!(dz, z.value().recip());
    let [x, y, z] = [2.0, 3.0, 4.0];
    assert_eq!(g(x, y, z), w.value());
}
