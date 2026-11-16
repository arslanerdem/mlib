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
///
/// ```
/// use mlib::accuracy;
///
/// assert_eq!(accuracy(&[0.0, 1.0, 1.0, 0.0], &[0.0, 1.0, 0.0, 0.0]), 0.75);
/// ```
pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let correct = y_true.iter().zip(y_pred).filter(|(truth, prediction)| truth == prediction).count();
    correct as f64 / y_true.len() as f64
}

/// The average squared difference between truth and prediction.
///
/// ```
/// use mlib::mean_squared_error;
///
/// assert_eq!(mean_squared_error(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 8.0]), 4.0);
/// ```
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let squares: f64 = y_true.iter().zip(y_pred).map(|(truth, prediction)| (truth - prediction).powi(2)).sum();
    squares / y_true.len() as f64
}

/// The square root of [`mean_squared_error`]: a typical error, in the
/// target's own units.
///
/// ```
/// use mlib::root_mean_squared_error;
///
/// assert_eq!(root_mean_squared_error(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 8.0]), 2.0);
/// ```
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// The average size of the difference between truth and prediction.
/// One large miss counts for less here than in [`mean_squared_error`].
///
/// ```
/// use mlib::mean_absolute_error;
///
/// assert_eq!(mean_absolute_error(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 8.0]), 1.0);
/// ```
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    check_lengths(y_true, y_pred.len());
    let total: f64 = y_true.iter().zip(y_pred).map(|(truth, prediction)| (truth - prediction).abs()).sum();
    total / y_true.len() as f64
}

/// Coefficient of determination: 1 is perfect, 0 is no better than
/// always predicting the mean.
///
/// ```
/// use mlib::r2_score;
///
/// let truth = [1.0, 2.0, 3.0];
/// assert_eq!(r2_score(&truth, &[1.0, 2.0, 3.0]), 1.0);
/// assert_eq!(r2_score(&truth, &[2.0, 2.0, 2.0]), 0.0);
/// ```
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

/// F1 for binary classification; the positive class is `1.0`. It is
/// high only when most predicted positives are right *and* most actual
/// positives are found.
///
/// ```
/// use mlib::f1_score;
///
/// // One of two predicted positives is right; one of two actual positives is found.
/// assert_eq!(f1_score(&[1.0, 1.0, 0.0, 0.0], &[1.0, 0.0, 1.0, 0.0]), 0.5);
/// ```
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
///
/// ```
/// use mlib::log_loss;
/// use mlib::ndarray::array;
///
/// // Confident and right costs little; undecided costs ln 2.
/// let confident = array![[0.9, 0.1], [0.1, 0.9]];
/// let undecided = array![[0.5, 0.5], [0.5, 0.5]];
/// assert!(log_loss(&[0.0, 1.0], &confident) < 0.11);
/// assert!((log_loss(&[0.0, 1.0], &undecided) - 2f64.ln()).abs() < 1e-12);
/// ```
pub fn log_loss(y_true: &[f64], probabilities: &Array2<f64>) -> f64 {
    check_lengths(y_true, probabilities.nrows());
    let total: f64 =
        y_true.iter().zip(probabilities.rows()).map(|(class, row)| -row[*class as usize].clamp(1e-15, 1.0).ln()).sum();
    total / y_true.len() as f64
}

/// Apply a metric to each target column and average the results.
/// `y_true` and `y_pred` are (rows × targets) and must have the same shape.
///
/// ```
/// use mlib::metrics::{mean_absolute_error, mean_over_targets};
/// use mlib::ndarray::array;
///
/// let truth = array![[1.0, 10.0], [2.0, 20.0]];
/// let predicted = array![[1.0, 12.0], [2.0, 22.0]];
/// // The first target is exact, the second is off by 2.
/// assert_eq!(mean_over_targets(mean_absolute_error, &truth, &predicted)?, 1.0);
/// # Ok::<(), mlib::Error>(())
/// ```
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

/// Counts of (actual, predicted) pairs as a matrix: rows are actual
/// classes, columns predicted. Labels must be class indices `0..n_classes`.
///
/// ```
/// use mlib::confusion_matrix;
/// use mlib::ndarray::array;
///
/// let matrix = confusion_matrix(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 1.0], 2);
/// assert_eq!(matrix, array![[1, 1], [0, 2]]);
/// ```
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64], n_classes: usize) -> Array2<usize> {
    check_lengths(y_true, y_pred.len());
    let mut matrix = Array2::zeros((n_classes, n_classes));
    for (truth, prediction) in y_true.iter().zip(y_pred) {
        matrix[[*truth as usize, *prediction as usize]] += 1;
    }
    matrix
}

/// How one class fared: (rows rightly predicted as the class, rows
/// predicted as it, rows that really are it).
fn counts_for_class(y_true: &[f64], y_pred: &[f64], class: f64) -> (f64, f64, f64) {
    check_lengths(y_true, y_pred.len());
    let right = y_true.iter().zip(y_pred).filter(|(truth, prediction)| **truth == class && **prediction == class);
    let predicted = y_pred.iter().filter(|prediction| **prediction == class).count();
    let actual = y_true.iter().filter(|truth| **truth == class).count();
    (right.count() as f64, predicted as f64, actual as f64)
}

/// `part / whole`, or 0 when there is nothing to divide by.
fn share(part: f64, whole: f64) -> f64 {
    if whole == 0.0 {
        0.0
    } else {
        part / whole
    }
}

/// F1 of one class against all the others.
fn f1_of_class(y_true: &[f64], y_pred: &[f64], class: f64) -> f64 {
    let (right, predicted, actual) = counts_for_class(y_true, y_pred, class);
    share(2.0 * right, predicted + actual)
}

/// The distinct values of a list, sorted.
fn distinct(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut distinct: Vec<f64> = values.collect();
    distinct.sort_unstable_by(f64::total_cmp);
    distinct.dedup();
    distinct
}

/// Precision for binary classification: of the rows predicted positive
/// (`1.0`), the share that really are. 0 if none was predicted positive.
///
/// ```
/// use mlib::precision_score;
///
/// // Two rows are predicted positive; one of them is.
/// assert_eq!(precision_score(&[1.0, 1.0, 0.0, 0.0], &[1.0, 0.0, 1.0, 0.0]), 0.5);
/// ```
pub fn precision_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let (right, predicted, _) = counts_for_class(y_true, y_pred, 1.0);
    share(right, predicted)
}

/// Recall for binary classification: of the rows that really are
/// positive (`1.0`), the share that was found. 0 if there are none.
///
/// ```
/// use mlib::recall_score;
///
/// // Four rows are positive; the model finds three of them.
/// assert_eq!(recall_score(&[1.0, 1.0, 1.0, 1.0, 0.0], &[1.0, 1.0, 1.0, 0.0, 0.0]), 0.75);
/// ```
pub fn recall_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let (right, _, actual) = counts_for_class(y_true, y_pred, 1.0);
    share(right, actual)
}

/// F1 for any number of classes: the F1 of each class against the
/// rest, averaged with every class counting the same — so a rare class
/// matters as much as a common one. The classes are the values found in
/// `y_true` or `y_pred`.
///
/// ```
/// use mlib::f1_macro;
///
/// let truth = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0];
/// let predicted = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
/// // Class 0 is perfect (1), class 1 is half right (2/3), class 2 is never found (0).
/// assert!((f1_macro(&truth, &predicted) - (1.0 + 2.0 / 3.0 + 0.0) / 3.0).abs() < 1e-12);
/// ```
pub fn f1_macro(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let classes = distinct(y_true.iter().chain(y_pred).copied());
    classes.iter().map(|&class| f1_of_class(y_true, y_pred, class)).sum::<f64>() / classes.len() as f64
}

/// F1 for any number of classes, like [`f1_macro`], but each class
/// counts by how many rows it has in `y_true` — so common classes
/// dominate.
///
/// ```
/// use mlib::f1_weighted;
///
/// let truth = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0];
/// let predicted = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
/// // Four rows of class 0 (F1 1), one of class 1 (2/3), one of class 2 (0).
/// assert!((f1_weighted(&truth, &predicted) - (4.0 + 2.0 / 3.0) / 6.0).abs() < 1e-12);
/// ```
pub fn f1_weighted(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let classes = distinct(y_true.iter().chain(y_pred).copied());
    let weighted = |&class: &f64| {
        let (_, _, actual) = counts_for_class(y_true, y_pred, class);
        actual * f1_of_class(y_true, y_pred, class)
    };
    classes.iter().map(weighted).sum::<f64>() / y_true.len() as f64
}

/// The recall of each class in `y_true`, averaged. Unlike accuracy it
/// cannot be made high by always predicting the common class: that
/// scores `1 / n_classes`.
///
/// ```
/// use mlib::{accuracy, balanced_accuracy_score};
///
/// // Nine rows of class 0 and one of class 1; the model always says 0.
/// let truth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
/// let predicted = [0.0; 10];
/// assert_eq!(accuracy(&truth, &predicted), 0.9);
/// assert_eq!(balanced_accuracy_score(&truth, &predicted), 0.5);
/// ```
pub fn balanced_accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let classes = distinct(y_true.iter().copied());
    let recall = |&class: &f64| {
        let (right, _, actual) = counts_for_class(y_true, y_pred, class);
        right / actual
    };
    classes.iter().map(recall).sum::<f64>() / classes.len() as f64
}

/// Area under the ROC curve for binary classification: the chance that
/// a positive row (`1.0`) gets a higher score than a negative one, ties
/// counting half. 1 is a perfect ranking, 0.5 is no better than chance.
/// It looks only at the order of the scores, so any score will do — a
/// probability is the usual one.
///
/// `NaN` if `y_true` holds only one class: there is nothing to rank.
///
/// ```
/// use mlib::roc_auc_score;
///
/// let truth = [0.0, 0.0, 1.0, 1.0];
/// assert_eq!(roc_auc_score(&truth, &[0.1, 0.2, 0.8, 0.9]), 1.0);
/// // One of the four positive-negative pairs is the wrong way round.
/// assert_eq!(roc_auc_score(&truth, &[0.1, 0.4, 0.35, 0.8]), 0.75);
/// ```
pub fn roc_auc_score(y_true: &[f64], y_score: &[f64]) -> f64 {
    check_lengths(y_true, y_score.len());
    let mut by_score: Vec<usize> = (0..y_score.len()).collect();
    by_score.sort_unstable_by(|&a, &b| y_score[a].total_cmp(&y_score[b]));

    // Give every row its rank by score, lowest first; rows with equal
    // scores share the average of their ranks. The positives' ranks then
    // count how many negatives each positive is above.
    let (mut positive_ranks, mut n_positive) = (0.0, 0.0);
    let mut start = 0;
    while start < by_score.len() {
        let mut end = start + 1;
        while end < by_score.len() && y_score[by_score[end]] == y_score[by_score[start]] {
            end += 1;
        }
        let shared_rank = (start + 1 + end) as f64 / 2.0;
        let positives = by_score[start..end].iter().filter(|&&row| y_true[row] == 1.0).count() as f64;
        positive_ranks += positives * shared_rank;
        n_positive += positives;
        start = end;
    }
    let n_negative = y_true.len() as f64 - n_positive;
    (positive_ranks - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative)
}

/// Any metric that compares truth with predictions.
type Metric = fn(&[f64], &[f64]) -> f64;

/// What a named scorer measures.
enum Measure {
    /// Hard predictions, by a metric function.
    Predictions(Metric),
    /// Class probabilities, by log loss.
    LogLoss,
    /// The probability of the second of two classes, by ROC AUC.
    RocAuc,
    /// Class probabilities, by the ROC AUC of each class against the
    /// rest, averaged.
    RocAucOneVsRest,
}

/// A named metric.
struct MetricScorer {
    name: &'static str,
    measure: Measure,
    greater_is_better: bool,
}

impl Scorer for MetricScorer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        match self.measure {
            Measure::Predictions(metric) => metric(y_true, y_pred),
            Measure::LogLoss => binary_log_loss(y_true, y_pred),
            Measure::RocAuc | Measure::RocAucOneVsRest => roc_auc_score(y_true, y_pred),
        }
    }

    fn greater_is_better(&self) -> bool {
        self.greater_is_better
    }

    fn uses_probabilities(&self) -> bool {
        !matches!(self.measure, Measure::Predictions(_))
    }

    fn score_probabilities(&self, y_true: &[f64], probabilities: &Array2<f64>, classes: &[f64]) -> Result<f64> {
        check_lengths(y_true, probabilities.nrows());
        let n_classes = probabilities.ncols();
        // Without labels, the classes are taken to be 0, 1, 2, ...
        let counted: Vec<f64> = (0..n_classes).map(|class| class as f64).collect();
        let classes = if classes.is_empty() { &counted } else { classes };
        if classes.len() != n_classes {
            return Err(Error::shape(format!("{} columns of probabilities", classes.len()), format!("{n_classes}")));
        }
        // Each row's class as a column number.
        let column_of = |label: &f64| {
            classes.iter().position(|class| class == label).map(|column| column as f64).ok_or_else(|| {
                Error::InvalidData(format!(
                    "the rows being scored hold class {label}, which the model was not trained on — \
                     StratifiedKFold keeps every class in every fold"
                ))
            })
        };
        let columns: Vec<f64> = y_true.iter().map(column_of).collect::<Result<_>>()?;
        let second_class: Vec<f64> = probabilities.column(n_classes - 1).to_vec();

        // The area under the curve for one class against the rest.
        let area = |class: usize, scores: &[f64]| {
            let is_class: Vec<f64> =
                columns.iter().map(|&column| if column == class as f64 { 1.0 } else { 0.0 }).collect();
            let area = roc_auc_score(&is_class, scores);
            if area.is_nan() {
                return Err(Error::InvalidData(format!(
                    "scorer '{}' needs every class among the rows being scored — \
                     StratifiedKFold keeps every class in every fold",
                    self.name
                )));
            }
            Ok(area)
        };

        match self.measure {
            Measure::LogLoss if n_classes == 2 => Ok(binary_log_loss(&columns, &second_class)),
            Measure::LogLoss => Ok(log_loss(&columns, probabilities)),
            Measure::RocAuc if n_classes == 2 => area(1, &second_class),
            Measure::RocAuc => Err(Error::InvalidSpec(format!(
                "scorer 'roc_auc' needs exactly two classes, and there are {n_classes} — use 'roc_auc_ovr'"
            ))),
            Measure::RocAucOneVsRest => {
                let areas = (0..n_classes).map(|class| area(class, &probabilities.column(class).to_vec()));
                Ok(areas.sum::<Result<f64>>()? / n_classes as f64)
            }
            Measure::Predictions(_) => {
                Err(Error::InvalidSpec(format!("scorer '{}' scores predictions, not probabilities", self.name)))
            }
        }
    }
}

/// Binary log loss from the probability of class 1.
fn binary_log_loss(y_true: &[f64], probability_of_one: &[f64]) -> f64 {
    let probabilities = Array2::from_shape_fn((y_true.len(), 2), |(row, class)| {
        if class == 1 {
            probability_of_one[row]
        } else {
            1.0 - probability_of_one[row]
        }
    });
    log_loss(y_true, &probabilities)
}

/// Look a scorer up by name.
///
/// | Name | Measures | Classes |
/// |---|---|---|
/// | "accuracy", "balanced_accuracy" | predicted classes | any number |
/// | "f1", "precision", "recall" | predicted classes | two; the positive class is 1 |
/// | "f1_macro", "f1_weighted" | predicted classes | any number |
/// | "log_loss" | class probabilities | any number |
/// | "roc_auc" | class probabilities | two; the positive class is the larger label |
/// | "roc_auc_ovr" | class probabilities | any number: each class against the rest, averaged |
/// | "r2", "rmse", "mse", "mae" | predicted numbers | — |
///
/// ```
/// use mlib::prelude::*;
///
/// let rmse = scorer("rmse")?;
/// assert_eq!(rmse.score(&[1.0, 2.0], &[1.0, 4.0]), 2f64.sqrt());
/// // Cross-validation uses this to report every score as "higher is better".
/// assert!(!rmse.greater_is_better());
/// assert!(scorer("no_such_metric").is_err());
/// # Ok::<(), Error>(())
/// ```
pub fn scorer(name: &str) -> Result<Box<dyn Scorer>> {
    use Measure::*;
    let (name, measure, greater_is_better) = match name {
        "accuracy" => ("accuracy", Predictions(accuracy), true),
        "balanced_accuracy" => ("balanced_accuracy", Predictions(balanced_accuracy_score), true),
        "f1" => ("f1", Predictions(f1_score), true),
        "f1_macro" => ("f1_macro", Predictions(f1_macro), true),
        "f1_weighted" => ("f1_weighted", Predictions(f1_weighted), true),
        "precision" => ("precision", Predictions(precision_score), true),
        "recall" => ("recall", Predictions(recall_score), true),
        "r2" => ("r2", Predictions(r2_score), true),
        "rmse" => ("rmse", Predictions(root_mean_squared_error), false),
        "mse" => ("mse", Predictions(mean_squared_error), false),
        "mae" => ("mae", Predictions(mean_absolute_error), false),
        "log_loss" => ("log_loss", LogLoss, false),
        "roc_auc" => ("roc_auc", RocAuc, true),
        "roc_auc_ovr" => ("roc_auc_ovr", RocAucOneVsRest, true),
        unknown => return Err(Error::InvalidSpec(format!("unknown scorer: '{unknown}'"))),
    };
    Ok(Box::new(MetricScorer { name, measure, greater_is_better }))
}
