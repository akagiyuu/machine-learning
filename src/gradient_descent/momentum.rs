use nalgebra::DVector;

use crate::util::gradient;

pub fn gradient_descent_with_momentum<F: Fn(DVector<f64>) -> f64 + Sync>(
    start: DVector<f64>,
    f: F,
    step_size: f64,
    momentum: f64,
    epsilon: f64,
) -> DVector<f64> {
    assert!(step_size > 0.);
    assert!(epsilon > 0.);

    let mut current_change = DVector::from_vec(vec![0.; start.len()]);
    let mut current = start;

    loop {
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
