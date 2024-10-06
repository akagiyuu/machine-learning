use itertools::Itertools;
use nalgebra::{DMatrix, DVector, RowDVector};

use crate::gradient_descent::gradient_descent_with_momentum_and_gradient_func;

fn sigmoid(x: &RowDVector<f64>, beta: &DVector<f64>) -> f64 {
    1. / (1. + (-(x * beta).x).exp())
}

pub struct LogisticRegression {
    beta: DVector<f64>,
}

impl LogisticRegression {
    pub fn train(xs: DMatrix<f64>, ys: DVector<f64>, step_size: f64, max_iter: usize) -> LogisticRegression {
        let gradient = |beta: DVector<f64>| {
            DMatrix::from_rows(
                &xs.row_iter()
                    .zip(ys.iter())
                    .map(|(x, y)| x.scale(y - sigmoid(&x.into(), &beta)))
                    .collect_vec(),
            )
            .row_sum_tr()
        };

        let beta = gradient_descent_with_momentum_and_gradient_func(
            DVector::new_random(xs.ncols()),
            gradient,
            -step_size,
            0.5,
            f64::EPSILON,
            max_iter
        );

        LogisticRegression { beta }
    }

    pub fn predict(&self, xs: DMatrix<f64>) -> DVector<f64> {
        let predicted = xs
            .row_iter()
            .map(|x| sigmoid(&x.into(), &self.beta))
            .map(|p| if p > 0.5 { 1. } else { 0. })
            .collect_vec();

        DVector::from_vec(predicted)
    }
}
