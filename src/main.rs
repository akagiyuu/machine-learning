use statistic::{util, GaussianNaiveBayesClassifier, NaiveBayesClassifier};

fn main() {
    let (train, test) =
        util::load_csv::<2, String>("Naive-Bayes-Classification-Data.csv", 0.95).unwrap();
    println!("{:?}", test);
    let model = NaiveBayesClassifier::<2>::from_collection(train).unwrap();
    let model = model.train();
    let (test_input, test_output): (Vec<_>, Vec<_>) = test.into_iter().unzip();
    let prediction = model.predict(test_input);
    let diff = prediction
        .iter()
        .zip(test_output.iter())
        .filter(|(x, y)| x != y)
        .count() as f64
        / prediction.len() as f64;
    println!("{:?}", diff);
    // let mut model =
    //     NaiveBayesClassifier::<2>::from_csv("Naive-Bayes-Classification-Data.csv").unwrap();
    //
    // println!(
    //     "{:?}",
    //     model.predict(vec![["45".to_string(), "82".to_string()]])
    // );
    //
    // println!(
    //     "{:?}",
    //     naive_bayes_classification(&data, &columns_labels, &[1, 1])
    // );
}
