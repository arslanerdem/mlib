//! Time forest, boosting or linear-model training and prediction on synthetic data.
//!
//! cargo run --release --example benchmark [rows] [features] [trees] [forest|boosting|linear] [% missing]
//!
//! The last argument blanks out that share of the feature values, to
//! time the tree models on data with missing values (the linear model
//! does not take any).

use mlib::prelude::*;
use mlib::random::Rng;
use std::time::Instant;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |position: usize, default: usize| -> usize {
        args.get(position).map_or(default, |arg| arg.parse().expect("every argument but the fourth is a number"))
    };
    let (n_rows, n_features, n_trees) = (number(0, 100_000), number(1, 20), number(2, 100));
    let model_name = args.get(3).map_or("forest", |name| name.as_str());

    // Gaussian-ish features; the class depends non-linearly on the first five.
    let mut rng = Rng::new(1, 0);
    let mut uniform = move || rng.next_u64() as f64 / u64::MAX as f64;
    let x = Array2::from_shape_fn((n_rows, n_features), |_| uniform() + uniform() + uniform() - 1.5);
    let signal = |i: usize| x[[i, 0]] * x[[i, 1]] + x[[i, 2]].sin() + (x[[i, 3]] - x[[i, 4]]).abs() - 0.6;
    let y = Array2::from_shape_fn((n_rows, 1), |(i, _)| (signal(i) + 0.3 * (uniform() - 0.5) > 0.0) as u8 as f64);
    let percent_missing = number(4, 0) as f64;
    let x = x.mapv(|value| if uniform() * 100.0 < percent_missing { f64::NAN } else { value });
    let data = EncodedData::new(x, y)?;

    let n_train = n_rows * 4 / 5;
    let train = data.take_rows(&(0..n_train).collect::<Vec<_>>());
    let test = data.take_rows(&(n_train..n_rows).collect::<Vec<_>>());
    println!("{n_train} training rows × {n_features} features, {n_trees} trees");

    let start = Instant::now();
    let (model, n_nodes): (Box<dyn Predictor>, usize) = match model_name {
        "boosting" => {
            let model = GradientBoostingSpec::new().n_rounds(n_trees).build()?.fit(&train)?;
            let n_nodes = model.trees().map(|tree| tree.n_nodes()).sum();
            (Box::new(model), n_nodes)
        }
        // A linear model has no trees; the third argument is ignored.
        "linear" => (Box::new(LinearModelSpec::new().build()?.fit(&train)?), 0),
        _ => {
            let model = RandomForestSpec::default().n_trees(n_trees).seed(42).build()?.fit(&train)?;
            let n_nodes = model.trees().map(|tree| tree.n_nodes()).sum();
            (Box::new(model), n_nodes)
        }
    };
    println!("fit:      {:>8.2?}", start.elapsed());

    let start = Instant::now();
    let accuracy = model.score(&test)?;
    println!("predict:  {:>8.2?}  ({} rows)", start.elapsed(), test.num_rows());
    println!("accuracy: {accuracy:.4}   nodes per tree: {}", n_nodes / n_trees);
    Ok(())
}
