use itertools::multizip;
use nalgebra::{DMatrix, DVector, RowDVector};
use std::collections::HashMap;
use std::f64;


#[derive(Debug, Default)]
pub struct GaussianNaiveBayesClassifier {
    means: Vec<RowDVector<f64>>,
    variances: Vec<RowDVector<f64>>,
    priors: Vec<f64>,
    tokens: HashMap<String, usize>,
}

impl GaussianNaiveBayesClassifier {
    pub fn train(xs: DMatrix<f64>, ys: DVector<String>) -> Self {
        let mut tokens = HashMap::new();
        let ys = ys.map(|y| match tokens.get(&y) {
            Some(&i) => i,
            None => {
                tokens.insert(y, tokens.len());
                tokens.len() - 1
            }
        });

        let row_count = xs.nrows() as f64;
        let mut priors = vec![0.; tokens.len()];
        ys.iter().for_each(|&y| priors[y] += 1.);
        priors.iter_mut().for_each(|p| *p /= row_count);

        let (means, variances): (Vec<_>, Vec<_>) = (0..priors.len())
            .map(|class| {
                let filtered_inputs = DMatrix::from_rows(
                    &xs.row_iter()
                        .zip(ys.iter())
                        .filter_map(
                            |(input, &output)| if output == class { Some(input) } else { None },
                        )
                        .collect::<Vec<_>>(),
                );
                let means = filtered_inputs.column_mean().transpose();
                let variances = filtered_inputs.row_variance();

                (means, variances)
            })
            .unzip();

        GaussianNaiveBayesClassifier {
            means,
            variances,
            priors,
            tokens,
        }
    }
}

impl GaussianNaiveBayesClassifier {
    pub fn probability(&self, input: RowDVector<f64>) -> HashMap<String, f64> {
        self.priors
            .iter()
            .enumerate()
            .map(|(class, &prior)| {
                let mean = &self.means[class];
                let variance = &self.variances[class];
                let likelihood: f64 = multizip((input.iter(), mean.iter(), variance.iter()))
                    .map(|(&x, &m, &v)| {
                        (-(x - m).powi(2) / (2. * v)).exp() / (2. * f64::consts::PI * v).sqrt()
                    })
                    .product();

                let class_name = self
                    .tokens
                    .iter()
                    .find(|(_, &i)| i == class)
                    .unwrap()
                    .0
                    .clone();

                (class_name, prior * likelihood)
            })
            .collect()
    }

    pub fn predict(&self, inputs: DMatrix<f64>) -> Vec<String> {
        inputs
            .row_iter()
            .map(|input| {
                // TODO: fix this type hack
                self.probability(input.transpose().transpose())
                    .into_iter()
                    .max_by(|(_, p1), (_, p2)| p1.total_cmp(p2))
                    .unwrap()
                    .0
            })
            .collect()
    }
}
