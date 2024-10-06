use std::{fmt::Debug, fs, path::Path, str::FromStr};

use anyhow::Result;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector, RowDVector, Scalar};

pub fn load<I, O>(path: impl AsRef<Path>) -> Result<(DMatrix<I>, DVector<O>)>
where
    I: FromStr + Debug + Clone+Scalar,
    <I as std::str::FromStr>::Err: Debug,
    O: FromStr + Debug + Clone+ Scalar,
    <O as std::str::FromStr>::Err: Debug,
{
    let raw = fs::read_to_string(path)?;
    let (xs, ys): (Vec<_>, Vec<_>) = raw
        .lines()
        .skip(1)
        .map(|row| row.split(',').collect_vec())
        .map(|row| {
            let length = row.len();
            let xs = row.iter().take(length - 1).map(|x| x.parse::<I>().unwrap());
            let xs = RowDVector::from_iterator(length - 1, xs);
            let ys = row[length - 1].parse::<O>().unwrap();

            (xs, ys)
        })
        .unzip();

    let xs = DMatrix::from_rows(xs.as_slice());
    let ys = DVector::from_vec(ys);

    Ok((xs, ys))
}
