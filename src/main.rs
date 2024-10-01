use nalgebra::DVector;
use statistic::gradient_descent::{gradient_descent, gradient_descent_with_momentum};

fn main() {
    let f = |x: DVector<f64>| x[0] * x[0];
    let value =
        gradient_descent_with_momentum(DVector::from_vec(vec![4.]), f, 0.1, 0.5, f64::EPSILON);
    println!("{}", value);
    // let g = |x: DVector<f64>| x.map(|x_i| 2. * x_i);
    // let mut current = DVector::from_vec(vec![4., 3.]);
    // let step = 0.1;
    //
    // loop {
    //     let pre = current.clone();
    //     current -= DVector::from_vec(vec![2. * current[0], 4. * current[1]]) * step;
    //     println!("{}", current);
    //     if (current.clone() - pre).norm() < f64::EPSILON {
    //         break;
    //     }
    // }
    // let xs = DMatrix::from_row_slice(4, 2, &[1., 1., 1., 2., 2., 2., 2., 3.]);
    // let ys = DVector::from_vec(vec![6., 8., 9., 11.]);
    //
    // let model = LinearRegression::new(xs.clone(), ys.clone()).unwrap();
    // let xs_test = DMatrix::from_row_slice(2, 2, &[1., 5., 6., 7.]);
    // let result = model.predict(xs_test);
    // println!("result: {}", result);
}
