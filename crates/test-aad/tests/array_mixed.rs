use aad::autodiff;

#[autodiff]
fn f(x: &[f64], y: f64) -> f64 {
    x.iter().sum::<f64>() + y
}

#[test]
fn main() {
    use aad::Tape;
    let tape = Tape::default();
    let x = tape.create_variables(&[2.0, 3.0, 4.0]);
    let y = tape.create_variable(6.0);
    let z = f(&x, y);
    let grads = z.compute_gradients();

    let dx = grads.get_gradients_iter(&x).collect::<Vec<_>>();
    assert_eq!(dx, [1.0, 1.0, 1.0]);
    let dy = grads.get_gradient(&y);
    assert_eq!(dy, 1.0);
}
