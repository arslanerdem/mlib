//! A conformance battery for estimators — call it from your own tests
//! when you implement [`Estimator`] for a new algorithm.

use crate::encoded::EncodedData;
use crate::traits::{Estimator, Predictor};
use ndarray::Array2;

/// A small, learnable dataset: two features, and a 0/1 target that
/// depends on the first. Works for classifiers and regressors alike.
fn toy_data(n_rows: usize) -> EncodedData {
    let x = Array2::from_shape_fn((n_rows, 2), |(i, j)| if j == 0 { i as f64 } else { (i * 7 % 5) as f64 });
    let y = Array2::from_shape_fn((n_rows, 1), |(i, _)| if i < n_rows / 2 { 0.0 } else { 1.0 });
    EncodedData::new(x, y).expect("toy data is well-formed")
}

/// Check that an estimator honours the library's behaviour contracts.
/// Panics with a description of the first contract it breaks.
///
/// - Junk data (no rows, no features, mismatched y, NaN) → `Err`, never a panic.
/// - Fitting twice on the same data gives identical predictions.
/// - Predictions come one per row; the wrong number of features → `Err`.
pub fn check_estimator<E: Estimator>(spec: &E) {
    let data = toy_data(40);

    let no_rows = data.take_rows(&[]);
    assert!(spec.fit(&no_rows).is_err(), "fit must fail on data with no rows");

    let no_features = EncodedData::new(Array2::zeros((40, 0)), data.y.clone()).unwrap();
    assert!(spec.fit(&no_features).is_err(), "fit must fail on data with no features");

    let mut short_y = data.clone();
    short_y.y = Array2::zeros((39, 1));
    assert!(spec.fit(&short_y).is_err(), "fit must fail when x and y have different row counts");

    let mut with_nan = data.clone();
    with_nan.x[[3, 0]] = f64::NAN;
    assert!(spec.fit(&with_nan).is_err(), "fit must fail on NaN in x");

    let first = spec.fit(&data).expect("fit must succeed on clean data");
    let second = spec.fit(&data).expect("fit must succeed a second time");
    let predictions = first.predict(&data).expect("predict must succeed on the training data");
    assert_eq!(predictions.dim(), data.y.dim(), "predict must return one row per sample, one column per target");
    assert_eq!(predictions, second.predict(&data).unwrap(), "two fits on the same data must agree");

    let one_feature = EncodedData::new(Array2::zeros((5, 1)), Array2::zeros((5, 1))).unwrap();
    assert!(first.predict(&one_feature).is_err(), "predict must fail on the wrong number of features");

    let score = first.score(&data).expect("score must succeed on the training data");
    assert!(score.is_finite(), "score must be a finite number");
}
