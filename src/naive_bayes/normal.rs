use anyhow::Result;
use std::{collections::HashMap, marker::PhantomData};

use crate::{Collection, Init, Trained};

#[derive(Debug, Default)]
pub struct NaiveBayesClassifier<const N: usize, State = Init> {
    input_tokens: HashMap<String, usize>,
    output_tokens: HashMap<String, usize>,
    inputs: Vec<[usize; N]>,
    outputs: Vec<usize>,
    priors: Option<Vec<f64>>,
    _state: PhantomData<State>,
}

impl<const N: usize> NaiveBayesClassifier<N, Init> {
    pub fn from_collection(collection: Collection<N, String>) -> Result<Self> {
        let mut model = Self::default();

        collection.into_iter().for_each(|row| {
            model.add(row.0, row.1);
        });
        Ok(model)
    }

    fn map_add_input(&mut self, input: [String; N]) -> [usize; N] {
        input.map(|x| match self.input_tokens.get(&x) {
            Some(&value) => value,
            None => {
                self.input_tokens.insert(x, self.input_tokens.len());
                self.input_tokens.len() - 1
            }
        })
    }

    fn map_add_output(&mut self, output: String) -> usize {
        match self.output_tokens.get(&output) {
            Some(&value) => value,
            None => {
                self.output_tokens
                    .insert(output.clone(), self.output_tokens.len());
                self.output_tokens.len() - 1
            }
        }
    }

    pub fn add(&mut self, input: [String; N], output: String) {
        let input = self.map_add_input(input);
        let output = self.map_add_output(output);

        self.inputs.push(input);
        self.outputs.push(output);
    }

    pub fn train(self) -> NaiveBayesClassifier<N, Trained> {
        let row_count = self.outputs.len() as f64;
        let priors = (0..self.output_tokens.len())
            .map(|k| self.outputs.iter().filter(|x| **x == k).count() as f64 / row_count)
            .collect();
        NaiveBayesClassifier {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            inputs: self.inputs,
            outputs: self.outputs,

            priors: Some(priors),
            _state: PhantomData::<Trained>,
        }
    }
}

impl<const N: usize> NaiveBayesClassifier<N, Trained> {
    fn map_input(&self, input: [String; N]) -> [usize; N] {
        input.map(|x| self.input_tokens[&x])
    }

    pub fn probability(&self, input: [String; N]) -> HashMap<String, f64> {
        let input = self.map_input(input);
        let row_count = self.inputs.len() as f64;

        (0..self.output_tokens.len())
            .map(|k| {
                let prior = self.priors.as_ref().unwrap()[k];
                let likelihood: f64 = (0..N)
                    .map(|j| {
                        self.inputs
                            .iter()
                            .zip(self.outputs.iter())
                            .filter(|(&row_input, &row_output)| {
                                row_input[j] == input[j] && row_output == k
                            })
                            .count() as f64
                            / row_count
                            / prior
                    })
                    .product();

                let output_name = self
                    .output_tokens
                    .iter()
                    .find(|(_, &value)| value == k)
                    .unwrap()
                    .0
                    .clone();

                (output_name, prior * likelihood)
            })
            .collect()
    }

    pub fn predict(&self, inputs: Vec<[String; N]>) -> Vec<String> {
        inputs
            .into_iter()
            .map(|input| {
                self.probability(input)
                    .into_iter()
                    .max_by(|(_, p1), (_, p2)| p1.total_cmp(p2))
                    .unwrap()
                    .0
            })
            .collect()
    }
}
