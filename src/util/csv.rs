use std::{fmt::Debug, fs, path::Path, str::FromStr};

use anyhow::Result;

use crate::Collection;

pub fn load<const N: usize, T>(
    path: impl AsRef<Path>,
    split: f64,
) -> Result<(Collection<N, T>, Collection<N, T>)>
where
    T: FromStr + Debug + Clone,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let raw = fs::read_to_string(path)?;
    let full_rows = raw
        .lines()
        .map(|row| {
            let mut input: Vec<_> = row.split(',').map(|x| x.parse::<T>().unwrap()).collect();
            let output = input.pop().unwrap();
            (input.try_into().unwrap(), output)
        })
        .collect::<Vec<_>>();
    let row_count = full_rows.len();
    let (left, right) = full_rows.split_at((row_count as f64 * split).floor() as usize);

    Ok((left.to_vec(), right.to_vec()))
}
