use nalgebra::{DMatrix, DVector};
use statistic::{pla::BinaryPLA, regression::LinearRegression};

fn main() {
    let xs = DMatrix::new_random(100, 5);
    let ys: DVector<f64> = DVector::new_random(100);

    let xs_train = xs.rows(0, 90);
    let ys_train = ys.rows(0, 90);

    let xs_test = xs.rows(90, 10);
    let ys_test = ys.rows(90, 10);
    let model = BinaryPLA::train(xs.clone(), ys.clone(), 1., 10000);
    println!("{}", ys_train);
    println!("{}", (model.predict(xs_train.into()) - ys_train).iter().filter(|&&x| x != 0.).count());
}
