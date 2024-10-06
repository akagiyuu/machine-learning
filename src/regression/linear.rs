use nalgebra::{DMatrix, DVector};

use crate::util;

#[derive(Debug)]
pub struct LinearRegression {
    beta: DVector<f64>,
}

impl LinearRegression {
    pub fn train(xs: DMatrix<f64>, ys: DVector<f64>) -> Option<LinearRegression> {
        let xs = util::pad(xs);
        (xs.transpose() * xs.clone())
            .try_inverse()
            .map(|inverse| inverse * xs.transpose() * ys)
            .map(|beta| LinearRegression { beta })
    }
}

impl LinearRegression {
    pub fn predict(&self, xs: DMatrix<f64>) -> DVector<f64> {
        util::pad(xs) * self.beta.clone()
    }
}
