use anyhow::Result;
use core::f64;
use std::{collections::HashMap, fs, marker::PhantomData};

use crate::Collection;

#[derive(Default)]
pub struct Init;

#[derive(Default)]
pub struct Trained;

#[derive(Debug, Default)]
pub struct GaussianNaiveBayesClassifier<const N: usize, State = Init> {
    unique_outputs: Vec<f64>,
    inputs: Vec<[f64; N]>,
    outputs: Vec<f64>,
    input_means: Option<Vec<[f64; N]>>,
    input_variances: Option<Vec<[f64; N]>>,
    _state: PhantomData<State>,
}

impl<const N: usize> GaussianNaiveBayesClassifier<N, Init> {
    pub fn from_collection(collection: Collection<N, f64>) -> Result<Self> {
        let mut model = Self::default();

        collection.into_iter().for_each(|row| {
            model.add(row.0, row.1);
        });

        Ok(model)
    }

    pub fn add(&mut self, input: [f64; N], output: f64) {
        println!("{}: {}", output, self.unique_outputs.contains(&output));
        if !self.unique_outputs.contains(&output) {
            self.unique_outputs.push(output);
        }
        self.inputs.push(input);
        self.outputs.push(output);
    }

    pub fn train(self) -> GaussianNaiveBayesClassifier<N, Trained> {
        let (input_means, input_variances): (Vec<_>, Vec<_>) = self
            .unique_outputs
            .iter()
            .map(|&k| {
                let filtered_inputs: Vec<_> = self
                    .inputs
                    .iter()
                    .zip(self.outputs.iter())
                    .filter_map(|(&input, &output)| if output == k { Some(input) } else { None })
                    .collect();
                let row_count = filtered_inputs.len();
                let means: [f64; N] = (0..N)
                    .map(|j| {
                        (0..row_count).map(|i| filtered_inputs[i][j]).sum::<f64>() as f64
                            / row_count as f64
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let variances: [f64; N] = (0..N)
                    .map(|j| {
                        (0..row_count)
                            .map(|i| (filtered_inputs[i][j] as f64 - means[j]).powi(2))
                            .sum::<f64>()
                            / row_count as f64
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                (means, variances)
            })
            .unzip();

        GaussianNaiveBayesClassifier {
            unique_outputs: self.unique_outputs,
            inputs: self.inputs,
            outputs: self.outputs,
            input_means: Some(input_means),
            input_variances: Some(input_variances),
            _state: PhantomData::<Trained>,
        }
    }
}

impl<const N: usize> GaussianNaiveBayesClassifier<N, Trained> {
    pub fn probability(&self, input: [f64; N]) -> Vec<(f64, f64)> {
        let row_count = self.inputs.len() as f64;
        let means = self.input_means.as_ref().unwrap();
        let variances = self.input_variances.as_ref().unwrap();

        self.unique_outputs
            .iter()
            .enumerate()
            .map(|(k, &class)| {
                let prior = self.outputs.iter().filter(|x| **x == class).count() as f64 / row_count;
                let likelihood: f64 = (0..N)
                    .map(|j| {
                        (-(input[j] - means[k][j]).powi(2) / (2. * variances[k][j])).exp()
                            / (2. * f64::consts::PI * variances[k][j]).sqrt()
                    })
                    .product();

                (class, prior * likelihood)
            })
            .collect()
    }

    pub fn predict(&self, inputs: Vec<[f64; N]>) -> Vec<f64> {
        inputs
            .into_iter()
            .map(|input| {
                println!("{:?}", self.probability(input));
                self.probability(input)
                    .into_iter()
                    .max_by(|(_, p1), (_, p2)| p1.total_cmp(p2))
                    .unwrap()
                    .0
            })
            .collect()
    }
}
