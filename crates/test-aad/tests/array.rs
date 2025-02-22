use aad::autodiff;

#[autodiff]
fn f(x: &[f64]) -> f64 {
    x[0]
}

#[test]
fn main() {
    use aad::Tape;
    let tape = Tape::default();
    let x = tape.create_variables(&[2.0, 3.0, 4.0]);
    let y = f(&x);
    let grads = y.compute_gradients();

    let dx = grads.get_gradients(&x).unwrap();
    assert_eq!(dx, [1.0, 0.0, 0.0]);
}
