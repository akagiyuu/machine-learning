use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, RowDVector};

pub struct KNN {
    pub xs: DMatrix<f64>,
    pub ys: DVector<f64>,
    pub k: usize,
}

impl KNN {
    fn _regression(&self, x: RowDVector<f64>) -> f64 {
        let mut distances = self
            .xs
            .row_iter()
            .zip(self.ys.iter())
            .map(|(pre_x, pre_y)| ((pre_x - &x).norm(), pre_y))
            .collect_vec();
        distances.sort_by(|(_, pre_y1), (_, pre_y2)| pre_y1.total_cmp(&pre_y2));
        distances
            .into_iter()
            .take(self.k)
            .map(|(_, &pre_y)| pre_y)
            .sum::<f64>()
            / self.k as f64
    }

    pub fn regression(&self, xs: DMatrix<f64>) -> DVector<f64> {
        DVector::from_iterator(
            xs.nrows(),
            xs.row_iter().map(|x| self._regression(x.into())),
        )
    }

    fn _classification(&self, x: RowDVector<f64>) -> usize {
        let mut distances = self
            .xs
            .row_iter()
            .zip(self.ys.iter())
            .map(|(pre_x, &pre_y)| ((pre_x - &x).norm(), pre_y as usize))
            .collect_vec();
        distances.sort_by(|(_, pre_y1), (_, pre_y2)| pre_y1.cmp(pre_y2));

        let mut distances_map = HashMap::new();
        for (_, y) in distances.into_iter().take(self.k) {
            *distances_map.entry(y).or_insert(0) += 1;
        }
        distances_map
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap()
            .0
    }

    pub fn classification(&self, xs: DMatrix<f64>) -> DVector<usize> {
        DVector::from_iterator(
            xs.nrows(),
            xs.row_iter().map(|x| self._classification(x.into())),
        )
    }
}
