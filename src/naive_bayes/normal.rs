use nalgebra::{DMatrix, DVector, RowDVector};
use std::collections::HashMap;


#[derive(Debug, Default)]
pub struct NaiveBayesClassifier {
    x_tokens: HashMap<String, usize>,
    y_tokens: HashMap<String, usize>,
    xs: DMatrix<usize>,
    ys: DVector<usize>,
    priors: Vec<f64>,
}

impl NaiveBayesClassifier {
    pub fn train(xs: DMatrix<String>, ys: DVector<String>) -> NaiveBayesClassifier {
        let mut x_tokens = HashMap::new();
        let xs = xs.map(|x| match x_tokens.get(&x) {
            Some(&i) => i,
            None => {
                x_tokens.insert(x, x_tokens.len());
                x_tokens.len() - 1
            }
        });

        let mut y_tokens = HashMap::new();
        let ys = ys.map(|y| match y_tokens.get(&y) {
            Some(&i) => i,
            None => {
                y_tokens.insert(y, y_tokens.len());
                y_tokens.len() - 1
            }
        });

        let mut priors = vec![0.; y_tokens.len()];
        ys.iter().for_each(|&y| priors[y] += 1.);

        NaiveBayesClassifier {
            x_tokens,
            y_tokens,
            xs,
            ys,
            priors,
        }
    }
}

impl NaiveBayesClassifier {
    pub fn probability(&self, x: RowDVector<String>) -> HashMap<String, f64> {
        let x = x.map(|xj| self.x_tokens[&xj]);

        self.priors
            .iter()
            .enumerate()
            .map(|(class, &prior)| {
                let likelihood: f64 = (0..x.len())
                    .map(|j| {
                        self.xs
                            .row_iter()
                            .zip(self.ys.iter())
                            .filter(|(data_x, &y)| y == class && data_x[j] == x[j])
                            .count() as f64
                            / self.xs.nrows() as f64
                            / prior
                    })
                    .product();

                let class_name = self
                    .y_tokens
                    .iter()
                    .find(|(_, &value)| value == class)
                    .unwrap()
                    .0
                    .clone();

                (class_name, prior * likelihood)
            })
            .collect()
    }

    pub fn predict(&self, inputs: DMatrix<String>) -> Vec<String> {
        inputs
            .row_iter()
            .map(|input| {
                self.probability(input.transpose().transpose())
                    .into_iter()
                    .max_by(|(_, p1), (_, p2)| p1.total_cmp(p2))
                    .unwrap()
                    .0
            })
            .collect()
    }
}
