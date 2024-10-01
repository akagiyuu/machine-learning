use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector};

use crate::{Init, Trained};

pub struct LinearRegression<State = Init> {
    beta: DVector<f64>,
    _state: PhantomData<State>,
}

impl<State> LinearRegression<State> {
    fn map_input(xs: DMatrix<f64>) -> DMatrix<f64> {
        xs.insert_column(0, 1.)
    }
}

impl LinearRegression<Init> {
    pub fn new(xs: DMatrix<f64>, ys: DVector<f64>) -> Option<LinearRegression<Trained>> {
        let xs = Self::map_input(xs);
        (xs.transpose() * xs.clone())
            .try_inverse()
            .map(|inverse| inverse * xs.transpose() * ys)
            .map(|beta| LinearRegression {
                beta,
                _state: PhantomData::<Trained>,
            })
    }
}

impl LinearRegression<Trained> {
    pub fn predict(&self, xs: DMatrix<f64>) -> DVector<f64> {
        Self::map_input(xs) * self.beta.clone()
    }
}
