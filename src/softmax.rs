use std::collections::HashMap;

pub fn softmax(probability: HashMap<String, f64>) -> HashMap<String, f64> {
    let denominator: f64 = probability.values().map(|p| p.exp()).sum();

    probability
        .into_iter()
        .map(|(output, p)| (output, p.exp() / denominator))
        .collect()
}
