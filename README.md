# AAD

This crate provides tools for implementing adjoint(a.k.a. reverse-mode) automatic differentiation in Rust. It
enables gradient computation for scalar values through a flexible and extensible API.

- **User-Friendly Design**: Equations can be manipulated as seamlessly as primitive floating-point types.
    - This design draws heavy inspiration from the `rustograd` library.
- **High Performance**: The library is designed to be both efficient and scalable, with minimal overhead.
    - Benchmarks show it is up to **4x faster** compared to `rustograd`.
- **No Dependencies**: The library is self-contained and does not rely on any external dependencies.

## Quick Start

Here's an example of how to use the library:

```rust
use aad::tape::Tape;

fn main() {
    let tape = Tape::default();

    let x = tape.var(2.0);
    let y = tape.var(3.0);

    let z = (x + y) * x.sin();

    println!("{}", z.value());

    z.backward();

    println!("Gradient of x: {}", x.grad());
    println!("Gradient of y: {}", y.grad());
}
```

## License

This project is licensed under the [MIT License](LICENSE).

