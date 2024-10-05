use nalgebra::DMatrix;

pub fn pad(xs: DMatrix<f64>) -> DMatrix<f64> {
    xs.insert_column(0, 1.)
}
