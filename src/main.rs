use machine_learning::{pla::BinaryPLA, util::csv};

fn main() {
    let (xs, ys) = csv::load::<f64, f64>("pla.csv").unwrap();
    let ys = ys.map(|x| if x == 0. { -1. } else { 1. });

    let split = (0.9 * xs.nrows() as f64).floor() as usize;

    let xs_train = xs.rows(0, split);
    let ys_train = ys.rows(0, split);

    let xs_test = xs.rows(split, xs.nrows() - split);
    let ys_test = ys.rows(split, xs.nrows() - split);
    let model = BinaryPLA::train(xs_train.into(), ys_train.into(), 1., 1000);
    println!(
        "Loss: {}/{}",
        (model.predict(xs_test.into()) - ys_test)
            .iter()
            .filter(|&&x| x != 0.)
            .count(),
        ys_train.len()
    );
    // Loss: 31/691
}
