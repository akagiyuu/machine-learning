use nalgebra::DVector;

use crate::util::gradient;

pub fn gradient_descent<F: Fn(DVector<f64>) -> f64 + Sync>(
    start: DVector<f64>,
    f: F,
    step_size: f64,
    epsilon: f64,
) -> DVector<f64> {
    assert!(step_size > 0.);
    assert!(epsilon > 0.);

    let mut current = start;

    loop {
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
) -> DVector<f64> {
    assert!(step_size > 0.);
    assert!(epsilon > 0.);

    let mut current = start;

    loop {
        let pre = current.clone();
        current -= gradient_func(current.clone()).scale(step_size);
        println!("{}", current);
        if (current.clone() - pre).norm() < epsilon {
            break;
        }
    }

    current
}
