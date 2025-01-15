use crate::core::tape::Tape;
use std::ops::AddAssign;

pub(crate) type BinaryFn<T> = fn(T, T) -> T;
pub(crate) type UnaryFn<T> = fn(T) -> T;

pub(crate) enum Operation {
    Unary(UnaryFnPayload),
    Binary(BinaryFnPayload),
    Scalar(ScalarFnPayload),
}

impl Operation {
    pub(crate) fn backward(&self, tape: &Tape) {
        match self {
            Operation::Unary(payload) => unsafe {
                let values = &*tape.values.get();
                let x = *values.get_unchecked(payload.x);
                let grads = &mut *tape.grads.get();
                let grad = *grads.get_unchecked(payload.y);

                grads
                    .get_unchecked_mut(payload.x)
                    .add_assign((payload.dfdx)(x) * grad);
            },
            Operation::Binary(payload) => unsafe {
                let values = &*tape.values.get();
                let x = *values.get_unchecked(payload.x);
                let y = *values.get_unchecked(payload.y);

                let grads = &mut *tape.grads.get();
                let grad = *grads.get_unchecked(payload.z);

                grads
                    .get_unchecked_mut(payload.x)
                    .add_assign((payload.dfdx)(x, y) * grad);
                grads
                    .get_unchecked_mut(payload.y)
                    .add_assign((payload.dfdy)(x, y) * grad);
            },
            Operation::Scalar(payload) => unsafe {
                let values = &*tape.values.get();
                let x = *values.get_unchecked(payload.x);

                let grads = &mut *tape.grads.get();
                let grad = *grads.get_unchecked(payload.y);

                grads
                    .get_unchecked_mut(payload.x)
                    .add_assign((payload.dfdx)(payload.scalar, x) * grad);
            },
        }
    }
}

pub(crate) struct UnaryFnPayload {
    pub x: usize,
    pub y: usize,
    pub dfdx: UnaryFn<f64>,
}

pub(crate) struct ScalarFnPayload {
    pub scalar: f64,
    pub x: usize,
    pub y: usize,
    pub dfdx: BinaryFn<f64>,
}

pub(crate) struct BinaryFnPayload {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub dfdx: BinaryFn<f64>,
    pub dfdy: BinaryFn<f64>,
}
