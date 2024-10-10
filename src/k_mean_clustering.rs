use itertools::Itertools;
use nalgebra::{DMatrix, DVector, RowDVector};

pub struct KMeanClustering {
    pub means: DMatrix<f64>,
}

impl KMeanClustering {
    fn get_optimal_means(xs: &DMatrix<f64>, xs_class: &DVector<usize>, k: usize) -> DMatrix<f64> {
        let means = (0..k)
            .map(|i| {
                let filtered_xs = &xs
                    .row_iter()
                    .zip(xs_class.iter())
                    .filter_map(|(x, &class)| if class == i { Some(x) } else { None })
                    .collect_vec();
                if filtered_xs.is_empty() {
                    RowDVector::new_random(xs.ncols())
                } else {
                    DMatrix::from_rows(&filtered_xs).row_mean()
                }
            })
            .collect_vec();

        DMatrix::from_rows(&means)
    }

    fn get_optimal_classes(xs: &DMatrix<f64>, means: &DMatrix<f64>) -> DVector<usize> {
        let xs_class = xs.row_iter().map(|x| {
            means
                .row_iter()
                .enumerate()
                .map(|(i, mean)| (i, (x - mean).norm_squared()))
                .min_by(|(_, mean_a), (_, mean_b)| mean_a.total_cmp(mean_b))
                .unwrap()
                .0
        });

        DVector::from_iterator(xs.nrows(), xs_class)
    }

    pub fn train(xs: DMatrix<f64>, k: usize, max_iter: usize) -> Self {
        let mut means = DMatrix::new_random(k, xs.ncols());
        let mut x_classes;

        for _ in 0..max_iter {
            x_classes = Self::get_optimal_classes(&xs, &means);
            means = Self::get_optimal_means(&xs, &x_classes, k);
        }

        Self { means }
    }

    pub fn predict(&self, xs: DMatrix<f64>) -> DVector<usize> {
        Self::get_optimal_classes(&xs, &self.means)
    }
}
