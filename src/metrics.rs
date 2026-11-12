//! Metrics: plain functions over slices, plus a registry to look them up
//! by name for cross-validation and grid search.
//!
//! `y_true` and `y_pred` must have the same length; a mismatch panics.

use crate::error::{Error, Result};
use crate::traits::Scorer;
use ndarray::Array2;

fn check_lengths(y_true: &[f64], n_predictions: usize) {
    assert_eq!(y_true.len(), n_predictions, "y_true and y_pred must have the same length");
}

/// Fraction of predictions that equal the truth.
pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let correct = y_true.iter().zip(y_pred).filter(|(truth, prediction)| truth == prediction).count();
    correct as f64 / y_true.len() as f64
}

pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let squares: f64 = y_true.iter().zip(y_pred).map(|(truth, prediction)| (truth - prediction).powi(2)).sum();
    squares / y_true.len() as f64
}

pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let total: f64 = y_true.iter().zip(y_pred).map(|(truth, prediction)| (truth - prediction).abs()).sum();
    total / y_true.len() as f64
}

/// Coefficient of determination: 1 is perfect, 0 is no better than
/// always predicting the mean.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let residual: f64 = y_true.iter().zip(y_pred).map(|(truth, prediction)| (truth - prediction).powi(2)).sum();
    let total: f64 = y_true.iter().map(|truth| (truth - mean).powi(2)).sum();
    if total == 0.0 {
        // A constant target: perfect if matched exactly, otherwise no credit.
        return if residual == 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - residual / total
}

/// F1 for binary classification; the positive class is `1.0`.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let pairs = || y_true.iter().zip(y_pred);
    let true_positives = pairs().filter(|(t, p)| **t == 1.0 && **p == 1.0).count() as f64;
    let predicted_positives = y_pred.iter().filter(|p| **p == 1.0).count() as f64;
    let actual_positives = y_true.iter().filter(|t| **t == 1.0).count() as f64;
    if predicted_positives + actual_positives == 0.0 {
        return 0.0;
    }
    2.0 * true_positives / (predicted_positives + actual_positives)
}

/// Mean negative log-likelihood. `y_true` holds class indices
/// (0, 1, 2, ...); `probabilities[[i, c]]` is row `i`'s probability of
/// class `c`.
pub fn log_loss(y_true: &[f64], probabilities: &Array2<f64>) -> f64 {
    check_lengths(y_true, probabilities.nrows());
    let total: f64 = y_true
        .iter()
        .zip(probabilities.rows())
        .map(|(class, row)| -row[*class as usize].clamp(1e-15, 1.0).ln())
        .sum();
    total / y_true.len() as f64
}

/// Apply a metric to each target column and average the results.
/// `y_true` and `y_pred` are (rows × targets) and must have the same shape.
pub fn mean_over_targets(
    metric: impl Fn(&[f64], &[f64]) -> f64,
    y_true: &Array2<f64>,
    y_pred: &Array2<f64>,
) -> Result<f64> {
    if y_true.dim() != y_pred.dim() || y_true.ncols() == 0 {
        return Err(Error::shape(
            format!("{} rows × {} targets (at least one)", y_pred.nrows(), y_pred.ncols()),
            format!("{} rows × {} targets", y_true.nrows(), y_true.ncols()),
        ));
    }
    let per_target = y_true.columns().into_iter().zip(y_pred.columns());
    let total: f64 = per_target.map(|(truth, prediction)| metric(&truth.to_vec(), &prediction.to_vec())).sum();
    Ok(total / y_true.ncols() as f64)
}

/// Counts of (actual, predicted) pairs: rows are actual classes, columns
/// predicted. Labels must be class indices `0..n_classes`.
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64], n_classes: usize) -> Vec<Vec<usize>> {
    check_lengths(y_true, y_pred.len());
    let mut matrix = vec![vec![0; n_classes]; n_classes];
    for (truth, prediction) in y_true.iter().zip(y_pred) {
        matrix[*truth as usize][*prediction as usize] += 1;
    }
    matrix
}

/// Any metric that compares truth with predictions.
type Metric = fn(&[f64], &[f64]) -> f64;

/// A named metric function.
struct MetricScorer {
    name: &'static str,
    metric: Metric,
    greater_is_better: bool,
    uses_probabilities: bool,
}

impl Scorer for MetricScorer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        (self.metric)(y_true, y_pred)
    }

    fn greater_is_better(&self) -> bool {
        self.greater_is_better
    }

    fn uses_probabilities(&self) -> bool {
        self.uses_probabilities
    }
}

/// Binary log loss from the probability of class 1.
fn binary_log_loss(y_true: &[f64], probability_of_one: &[f64]) -> f64 {
    let probabilities = Array2::from_shape_fn((y_true.len(), 2), |(row, class)| {
        if class == 1 { probability_of_one[row] } else { 1.0 - probability_of_one[row] }
    });
    log_loss(y_true, &probabilities)
}

/// Look a scorer up by name: "accuracy", "f1", "r2", "rmse", "mse",
/// "mae", "log_loss" (binary targets 0/1 only).
pub fn scorer(name: &str) -> Result<Box<dyn Scorer>> {
    let (name, metric, greater_is_better): (_, Metric, _) = match name {
        "accuracy" => ("accuracy", accuracy, true),
        "f1" => ("f1", f1_score, true),
        "r2" => ("r2", r2_score, true),
        "rmse" => ("rmse", root_mean_squared_error, false),
        "mse" => ("mse", mean_squared_error, false),
        "mae" => ("mae", mean_absolute_error, false),
        "log_loss" => ("log_loss", binary_log_loss, false),
        unknown => return Err(Error::InvalidSpec(format!("unknown scorer: '{unknown}'"))),
    };
    Ok(Box::new(MetricScorer { name, metric, greater_is_better, uses_probabilities: name == "log_loss" }))
}
