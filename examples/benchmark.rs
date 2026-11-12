//! Time forest training and prediction on synthetic data.
//!
//! cargo run --release --example benchmark [rows] [features] [trees]

use mlib::prelude::*;
use mlib::random::Rng;
use std::time::Instant;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1).map(|arg| arg.parse::<usize>().expect("arguments are numbers"));
    let n_rows = args.next().unwrap_or(100_000);
    let n_features = args.next().unwrap_or(20);
    let n_trees = args.next().unwrap_or(100);

    // Gaussian-ish features; the class depends non-linearly on the first five.
    let mut rng = Rng::new(1, 0);
    let mut uniform = move || rng.next_u64() as f64 / u64::MAX as f64;
    let x = Array2::from_shape_fn((n_rows, n_features), |_| uniform() + uniform() + uniform() - 1.5);
    let signal = |i: usize| x[[i, 0]] * x[[i, 1]] + x[[i, 2]].sin() + (x[[i, 3]] - x[[i, 4]]).abs() - 0.6;
    let y = Array2::from_shape_fn((n_rows, 1), |(i, _)| (signal(i) + 0.3 * (uniform() - 0.5) > 0.0) as u8 as f64);
    let data = EncodedData::new(x, y)?;

    let n_train = n_rows * 4 / 5;
    let train = data.take_rows(&(0..n_train).collect::<Vec<_>>());
    let test = data.take_rows(&(n_train..n_rows).collect::<Vec<_>>());
    println!("{n_train} training rows × {n_features} features, {n_trees} trees");

    let forest = RandomForestSpec::default().n_trees(n_trees).seed(42).build()?;
    let start = Instant::now();
    let model = forest.fit(&train)?;
    println!("fit:      {:>8.2?}", start.elapsed());

    let start = Instant::now();
    let accuracy = model.score(&test)?;
    println!("predict:  {:>8.2?}  ({} rows)", start.elapsed(), test.num_rows());

    let nodes: usize = model.trees().map(|tree| tree.n_nodes()).sum();
    println!("accuracy: {accuracy:.4}   nodes per tree: {}", nodes / n_trees);
    Ok(())
}
