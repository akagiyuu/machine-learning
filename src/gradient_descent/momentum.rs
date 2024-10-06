use nalgebra::DVector;

use crate::util::gradient;

pub fn gradient_descent_with_momentum<F: Fn(DVector<f64>) -> f64 + Sync>(
    start: DVector<f64>,
    f: F,
    step_size: f64,
    momentum: f64,
    epsilon: f64,
    max_iter: usize,
) -> DVector<f64> {
    assert!(epsilon > 0.);

    let mut current_change = DVector::from_vec(vec![0.; start.len()]);
    let mut current = start;

    for _ in 0..max_iter {
        current_change =
            gradient(current.clone(), &f, 1).scale(step_size) + current_change.scale(momentum);
        let pre = current.clone();
        current -= current_change.clone();
        if (current.clone() - pre).norm() < epsilon {
            break;
        }
    }

    current
}

pub fn gradient_descent_with_momentum_and_gradient_func<G: Fn(DVector<f64>) -> DVector<f64>>(
    start: DVector<f64>,
    gradient: G,
    step_size: f64,
    momentum: f64,
    epsilon: f64,
    max_iter: usize,
) -> DVector<f64> {
    assert!(epsilon > 0.);

    let mut current_change = DVector::from_vec(vec![0.; start.len()]);
    let mut current = start;

    for _ in 0..max_iter {
        current_change =
            gradient(current.clone()).scale(step_size) + current_change.scale(momentum);
        let pre = current.clone();
        current -= current_change.clone();
        if (current.clone() - pre).norm() < epsilon {
            break;
        }
    }

    current
}
