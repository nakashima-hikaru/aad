use aad::autodiff;

#[autodiff]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {}
