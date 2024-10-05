pub mod naive_bayes;
pub mod regression;
pub mod util;
pub mod gradient_descent;
pub mod pla;

#[derive(Debug, Default)]
pub struct Init;

#[derive(Debug, Default)]
pub struct Trained;

type Row<const N: usize, T> = ([T; N], T);
type Collection<const N: usize, T> = Vec<Row<N, T>>;
