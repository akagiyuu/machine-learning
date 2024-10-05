use nalgebra::{DMatrix, DVector};

pub struct BinaryPLA {
    beta: DVector<f64>,
}

impl BinaryPLA {
    pub fn train(xs: DMatrix<f64>, ys: DVector<f64>, step_size:f64, max_iter: usize) -> BinaryPLA {
        let mut beta = DVector::new_random(xs.ncols());
        for _ in 0..max_iter {
            for (x, &y) in xs.row_iter().zip(ys.iter()) {
                // Prediction is right
                if (x * &beta).x * y >= 0. {
                    continue; 
                }

                beta += x.scale(y * step_size).transpose();
            }
        }

        BinaryPLA { beta }
    }

    pub fn predict(&self, xs: DMatrix<f64>) -> DVector<f64> {
        (xs * &self.beta).map(|x| if x >= 0. { 1. } else { -1. })
    }
}
