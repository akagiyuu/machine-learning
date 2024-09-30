use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use statistic::regression::LinearRegression;

fn main() {
    let xs = DMatrix::from_row_slice(4, 2, &[1., 1., 1., 2., 2., 2., 2., 3.]);
    let ys = DVector::from_vec(vec![6., 8., 9., 11.]);

    let model = LinearRegression::new(xs.clone(), ys.clone()).unwrap();
    let xs_test = DMatrix::from_row_slice(2, 2, &[1., 5., 6., 7.]);
    let result = model.predict(xs_test);
    println!("result: {}", result);
}
