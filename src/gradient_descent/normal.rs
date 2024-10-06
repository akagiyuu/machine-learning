use nalgebra::DVector;

use crate::util::gradient;

pub fn gradient_descent<F: Fn(DVector<f64>) -> f64 + Sync>(
    start: DVector<f64>,
    f: F,
    step_size: f64,
    epsilon: f64,
    max_iter: usize
) -> DVector<f64> {
    assert!(epsilon > 0.);

    let mut current = start;

    for _ in 0..max_iter {
        let pre = current.clone();
        current -= gradient(current.clone(), &f, 1).scale(step_size);
        if (current.clone() - pre).norm() < epsilon {
            break;
        }
    }

    current
}

pub fn gradient_descent_with_gradient_func<G: Fn(DVector<f64>) -> DVector<f64>>(
    start: DVector<f64>,
    gradient_func: G,
    step_size: f64,
    epsilon: f64,
    max_iter: usize,
) -> DVector<f64> {
    assert!(epsilon > 0.);

    let mut current = start;

    for _ in 0..max_iter{
        let pre = current.clone();
        current -= gradient_func(current.clone()).scale(step_size);
        if (current.clone() - pre).norm() < epsilon {
            break;
        }
    }

    current
}
