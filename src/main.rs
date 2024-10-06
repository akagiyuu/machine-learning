use machine_learning::{regression::LogisticRegression, util::csv};

fn main() {
    let (xs, ys) = csv::load::<f64, f64>("regression.csv").unwrap();

    let split = (0.9 * xs.nrows() as f64).floor() as usize;

    let xs_train = xs.rows(0, split);
    let ys_train = ys.rows(0, split);

    let xs_test = xs.rows(split, xs.nrows() - split);
    let ys_test = ys.rows(split, xs.nrows() - split);
    let model = LogisticRegression::train(xs_train.into(), ys_train.into(), 0.01, 100000);
    println!(
        "Loss: {}/{}",
        (model.predict(xs_test.into()) - ys_test)
            .iter()
            .filter(|&&x| x != 0.)
            .count(),
        ys_train.len()
    );
    // Loss: 1/512
}
