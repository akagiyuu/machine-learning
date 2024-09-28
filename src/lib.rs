mod naive_bayes;
mod softmax;
pub mod util;

pub use naive_bayes::*;
pub use softmax::softmax;

#[derive(Debug, Default)]
pub struct Init;

#[derive(Debug, Default)]
pub struct Trained;

type Row<const N: usize, T> = ([T; N], T);
type Collection<const N: usize, T> = Vec<Row<N, T>>;
