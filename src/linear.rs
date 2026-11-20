//! The linear model: every feature gets one number, its coefficient, and
//! a prediction is the features times their coefficients, added up, plus
//! an intercept. It is the plain baseline a forest or boosting has to beat.
//!
//! - Regression is least squares. Several targets get their own
//!   coefficients each.
//! - Classification is logistic regression: the sum is a *score* that
//!   turns into probabilities exactly as in gradient boosting — one score
//!   for two classes, one per class otherwise.
//!
//! Features are standardised (mean 0, standard deviation 1) inside the
//! fit, so that features in different units are treated alike; the
//! coefficients are handed back in the features' own units. With
//! `l2_regularization`, large coefficients are penalised, which steadies
//! the model when features are many or strongly related.
//!
//! Least squares is solved exactly. Logistic regression has no formula
//! for its answer and is found step by step (see `minimise.rs`).
//!
//! The types follow the other models: [`LinearModelSpec`] → `build()` →
//! [`ValidLinearModelSpec`] → `fit()` → [`FittedLinearModel`].

use crate::classes::{self, ClassWeight};
use crate::encoded::{EncodedData, Feature};
use crate::error::{Error, Result};
use crate::minimise::minimise;
use crate::scores::{self, by_row, class_probabilities, probabilities, starting_score, Wanted};
use crate::traits::{Estimator, Label, ParamValue, Predictor};
use crate::tree::{self, Task};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Linear model hyper-parameters. `LinearModelSpec::default()` trains a
/// classifier (logistic regression) out of the box.
///
/// ```
/// use mlib::prelude::*;
///
/// let classifier = LinearModelSpec::default().l2_regularization(1.0).max_iterations(500).tolerance(1e-6).build()?;
/// assert_eq!(classifier.spec().max_iterations, 500);
///
/// let regressor = LinearModelSpec::regressor().build()?;
/// assert_eq!(regressor.spec().task, Task::Regression);
///
/// // A setting that makes no sense is caught here, not during the fit.
/// assert!(LinearModelSpec::default().l2_regularization(-1.0).build().is_err());
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LinearModelSpec {
    /// Default: classification.
    pub task: Task,
    /// The penalty on large coefficients, 0 or more: this number times
    /// half the sum of the squared coefficients (of the standardised
    /// features) is added to the total error. The intercept is never
    /// penalised. Default 0 (none).
    pub l2_regularization: f64,
    /// Classification only: the most steps the search for the
    /// coefficients may take. Default 200.
    pub max_iterations: usize,
    /// Classification only: the search stops when a small change in any
    /// coefficient would change the average loss per row by less than
    /// this much per unit of change. Default 1e-5.
    pub tolerance: f64,
    /// Classification only: how much the rows of each class count.
    /// Default `ClassWeight::None` (all the same).
    #[serde(default)]
    pub class_weight: ClassWeight,
}

impl Default for LinearModelSpec {
    fn default() -> LinearModelSpec {
        LinearModelSpec {
            task: Task::Classification,
            l2_regularization: 0.0,
            max_iterations: 200,
            tolerance: 1e-5,
            class_weight: ClassWeight::None,
        }
    }
}

impl LinearModelSpec {
    /// A classifier (logistic regression) with default settings.
    pub fn new() -> LinearModelSpec {
        LinearModelSpec::default()
    }

    /// A regressor (least squares) with default settings.
    pub fn regressor() -> LinearModelSpec {
        LinearModelSpec { task: Task::Regression, ..Default::default() }
    }

    /// Classification (logistic regression) or regression (least squares).
    pub fn task(mut self, task: Task) -> LinearModelSpec {
        self.task = task;
        self
    }

    /// The penalty on large coefficients, 0 or more.
    pub fn l2_regularization(mut self, l2: f64) -> LinearModelSpec {
        self.l2_regularization = l2;
        self
    }

    /// Classification only: the most steps the search may take.
    pub fn max_iterations(mut self, n: usize) -> LinearModelSpec {
        self.max_iterations = n;
        self
    }

    /// Classification only: how flat the loss must be for the search to
    /// stop. Smaller is more exact and slower.
    pub fn tolerance(mut self, tolerance: f64) -> LinearModelSpec {
        self.tolerance = tolerance;
        self
    }

    /// How much the rows of each class count (classification only).
    /// `ClassWeight::Balanced` makes a rare class count as much as a
    /// common one.
    pub fn class_weight(mut self, class_weight: ClassWeight) -> LinearModelSpec {
        self.class_weight = class_weight;
        self
    }

    /// Validate the hyper-parameters. Only a validated spec can be fitted.
    pub fn build(&self) -> Result<ValidLinearModelSpec> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        self.class_weight.check(self.task)?;
        if !(self.l2_regularization >= 0.0 && self.l2_regularization.is_finite()) {
            return invalid("l2_regularization must be 0 or more");
        }
        if self.max_iterations == 0 {
            return invalid("max_iterations must be at least 1");
        }
        if !(self.tolerance > 0.0 && self.tolerance.is_finite()) {
            return invalid("tolerance must be greater than 0");
        }
        Ok(ValidLinearModelSpec { spec: self.clone() })
    }
}

/// A `LinearModelSpec` that passed validation. Create one with
/// [`LinearModelSpec::build`].
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // 200 rows; the class is 1 when the first feature is large.
/// let x = Array2::from_shape_fn((200, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// let y = Array2::from_shape_fn((200, 1), |(row, _)| if x[[row, 0]] >= 25.0 { 1.0 } else { 0.0 });
/// let data = EncodedData::new(x, y)?;
///
/// let mut logistic = LinearModelSpec::default().l2_regularization(1.0).build()?;
/// let fitted = logistic.fit(&data)?;
/// assert!(fitted.converged());
/// assert!(fitted.score(&data)? > 0.95);
///
/// // Changing a setting by name checks it again.
/// logistic.set_param("l2_regularization", &10.0.into())?;
/// assert!(logistic.set_param("l2_regularization", &(-1.0).into()).is_err());
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidLinearModelSpec {
    spec: LinearModelSpec,
}

impl ValidLinearModelSpec {
    /// The validated hyper-parameters.
    pub fn spec(&self) -> &LinearModelSpec {
        &self.spec
    }
}

impl Estimator for ValidLinearModelSpec {
    type Fitted = FittedLinearModel;

    fn fit(&self, data: &EncodedData) -> Result<FittedLinearModel> {
        let spec = &self.spec;
        tree::check_training_data(data, false)?;
        let weights = classes::row_weights(&spec.class_weight, data)?.unwrap_or_else(|| vec![1.0; data.num_rows()]);
        let Wanted { values: wanted, groups, classes, classes_per_target } = Wanted::new(spec.task, data)?;
        // The features' scale describes the data, so it is measured with
        // the sample weights alone; class weights say what a mistake
        // costs, and only enter the fit. (This is also what scikit-learn
        // does with a `StandardScaler` before a weighted model.)
        let features = match &spec.class_weight {
            ClassWeight::None => Standardised::new(&data.x, &weights),
            _ => Standardised::new(&data.x, &data.weights.clone().unwrap_or_else(|| vec![1.0; data.num_rows()])),
        };

        let fit = match spec.task {
            Task::Regression => least_squares(&features, &wanted, &weights, spec.l2_regularization),
            // Each class target is a separate problem with its own scores.
            Task::Classification => {
                let fits =
                    groups.iter().map(|group| logistic_regression(&features, &wanted[group.clone()], &weights, spec));
                Fit::side_by_side(fits)
            }
        };

        // Back to the features' own units. A standardised feature is
        // `(x - mean) / scale`, so its coefficient is divided by the scale
        // and the means move into the intercept.
        let mut coefficients = Vec::new();
        let mut intercepts = Vec::new();
        for (standardised, intercept) in fit.coefficients.iter().zip(&fit.intercepts) {
            let own_units: Vec<f64> = standardised.iter().zip(&features.scales).map(|(c, scale)| c / scale).collect();
            intercepts.push(intercept - own_units.iter().zip(&features.means).map(|(c, mean)| c * mean).sum::<f64>());
            coefficients.push(own_units);
        }

        Ok(FittedLinearModel {
            task: spec.task,
            coefficients,
            intercepts,
            classes,
            classes_per_target,
            features: data.features.clone(),
            n_features: data.num_features(),
            n_iterations: fit.n_iterations,
            converged: fit.converged,
        })
    }

    /// Parameters: `l2_regularization`, `tolerance` (Float);
    /// `max_iterations` (Int); `class_weight` (Text "none" / "balanced").
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        let invalid = || Error::InvalidSpec(format!("invalid value for {name}: {value:?}"));
        let mut spec = self.spec.clone();
        match (name, value) {
            ("l2_regularization", ParamValue::Float(l2)) => spec.l2_regularization = *l2,
            ("tolerance", ParamValue::Float(tolerance)) => spec.tolerance = *tolerance,
            ("max_iterations", ParamValue::Int(n)) if *n >= 0 => spec.max_iterations = *n as usize,
            ("class_weight", value) => spec.class_weight = ClassWeight::from_param(value)?,
            ("l2_regularization" | "tolerance" | "max_iterations", _) => return Err(invalid()),
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        }
        *self = spec.build()?;
        Ok(())
    }
}

/// The features, each shifted and scaled to (weighted) mean 0 and
/// standard deviation 1, stored column by column.
struct Standardised {
    /// `columns[feature][row]`.
    columns: Vec<Vec<f64>>,
    means: Vec<f64>,
    /// A feature with a single value has no spread; its scale is 1 and
    /// its standardised column is all zeros, so its coefficient is 0.
    scales: Vec<f64>,
}

impl Standardised {
    fn new(x: &Array2<f64>, weights: &[f64]) -> Standardised {
        let total_weight: f64 = weights.iter().sum();
        let standardise = |feature: usize| {
            let column = x.column(feature);
            let mean = column.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() / total_weight;
            let variance = column.iter().zip(weights).map(|(x, w)| w * (x - mean).powi(2)).sum::<f64>() / total_weight;
            let scale = if variance > 0.0 { variance.sqrt() } else { 1.0 };
            (column.iter().map(|x| (x - mean) / scale).collect(), mean, scale)
        };
        // Each feature is summed up in row order, so the result does not
        // depend on how the features are shared out over the cores.
        let standardised: Vec<(Vec<f64>, f64, f64)> = (0..x.ncols()).into_par_iter().map(standardise).collect();

        let (mut columns, mut means, mut scales) = (Vec::new(), Vec::new(), Vec::new());
        for (column, mean, scale) in standardised {
            columns.push(column);
            means.push(mean);
            scales.push(scale);
        }
        Standardised { columns, means, scales }
    }
}

/// What a fit hands back, in standardised units: `coefficients[score][feature]`.
struct Fit {
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    n_iterations: usize,
    converged: bool,
}

impl Fit {
    /// Several fits as one: their scores one after another.
    fn side_by_side(fits: impl Iterator<Item = Fit>) -> Fit {
        let mut all = Fit { coefficients: Vec::new(), intercepts: Vec::new(), n_iterations: 0, converged: true };
        for fit in fits {
            all.coefficients.extend(fit.coefficients);
            all.intercepts.extend(fit.intercepts);
            all.n_iterations = all.n_iterations.max(fit.n_iterations);
            all.converged &= fit.converged;
        }
        all
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Least squares, solved exactly.
///
/// The best coefficients `b` satisfy the *normal equations*
/// `(ZᵀWZ + l2·I) b = ZᵀW y`, where `Z` holds the standardised features,
/// `W` the sample weights and `y` the target. `ZᵀWZ` has one entry per
/// pair of features — how strongly the two move together — and is the
/// same for every target, so it is built and factorised once.
///
/// Because the standardised features average to zero, the intercept is
/// simply the (weighted) mean of the target.
fn least_squares(features: &Standardised, wanted: &[Vec<f64>], weights: &[f64], l2: f64) -> Fit {
    let n_features = features.columns.len();
    let total_weight: f64 = weights.iter().sum();
    let weighted: Vec<Vec<f64>> = features
        .columns
        .par_iter()
        .map(|column| column.iter().zip(weights).map(|(z, weight)| z * weight).collect())
        .collect();

    // Row `i` holds the entries for features `0..=i`; the other half is its mirror image.
    let pair_sums: Vec<Vec<f64>> = (0..n_features)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<f64> = (0..=i).map(|j| dot(&weighted[i], &features.columns[j])).collect();
            row[i] += l2;
            row
        })
        .collect();
    // A standardised feature's own entry is the total weight, so that is
    // the yardstick for "nothing left of this feature".
    let factor = cholesky(pair_sums, 1e-10 * total_weight);

    let (mut coefficients, mut intercepts) = (Vec::new(), Vec::new());
    for target in wanted {
        let right_hand_side: Vec<f64> = weighted.par_iter().map(|column| dot(column, target)).collect();
        coefficients.push(solve(&factor, right_hand_side));
        intercepts.push(dot(weights, target) / total_weight);
    }
    Fit { coefficients, intercepts, n_iterations: 0, converged: true }
}

/// Factorise a symmetric matrix `A` (given as its lower triangle, row by
/// row) into `L·Lᵀ`, where `L` is lower-triangular. Solving `A b = r`
/// then takes two easy passes (see [`solve`]).
///
/// A feature that is an exact combination of earlier ones — a repeated
/// column, or the last category of a one-hot group next to an intercept —
/// has nothing new to add: what is left of it after accounting for the
/// earlier features (its *pivot*) is zero. Such a feature is marked with
/// a zero on the diagonal and gets coefficient 0.
fn cholesky(mut lower: Vec<Vec<f64>>, smallest_pivot: f64) -> Vec<Vec<f64>> {
    for i in 0..lower.len() {
        for j in 0..=i {
            let explained: f64 = (0..j).map(|k| lower[i][k] * lower[j][k]).sum();
            let rest = lower[i][j] - explained;
            lower[i][j] = if i == j {
                if rest > smallest_pivot {
                    rest.sqrt()
                } else {
                    0.0
                }
            } else if lower[j][j] > 0.0 {
                rest / lower[j][j]
            } else {
                0.0
            };
        }
    }
    lower
}

/// Solve `L·Lᵀ b = r` for `b`: first `L c = r` from the top down, then
/// `Lᵀ b = c` from the bottom up. Features marked as redundant get 0.
fn solve(factor: &[Vec<f64>], right_hand_side: Vec<f64>) -> Vec<f64> {
    let n = factor.len();
    let mut solution = right_hand_side;
    for i in 0..n {
        let known: f64 = (0..i).map(|k| factor[i][k] * solution[k]).sum();
        solution[i] = if factor[i][i] > 0.0 { (solution[i] - known) / factor[i][i] } else { 0.0 };
    }
    for i in (0..n).rev() {
        let known: f64 = (i + 1..n).map(|k| factor[k][i] * solution[k]).sum();
        solution[i] = if factor[i][i] > 0.0 { (solution[i] - known) / factor[i][i] } else { 0.0 };
    }
    solution
}

/// Logistic regression for one target: find the coefficients that make
/// the observed classes most probable — that is, minimise the average
/// log loss per row, plus the penalty. `wanted` holds the target's
/// scores: one for two classes, one per class otherwise.
///
/// The parameters are one flat list: for each score its coefficients,
/// then its intercept.
fn logistic_regression(features: &Standardised, wanted: &[Vec<f64>], weights: &[f64], spec: &LinearModelSpec) -> Fit {
    let (n_features, n_rows, n_scores) = (features.columns.len(), weights.len(), wanted.len());
    let per_score = n_features + 1;
    let total_weight: f64 = weights.iter().sum();
    let l2 = spec.l2_regularization;

    // Start with all coefficients at zero and each intercept at the score
    // that predicts the class's share of the data.
    let mut start = vec![0.0; n_scores * per_score];
    for (s, wanted) in wanted.iter().enumerate() {
        start[s * per_score + n_features] =
            starting_score(Task::Classification, n_scores, dot(wanted, weights) / total_weight);
    }

    let loss_and_gradient = |parameters: &[f64]| -> (f64, Vec<f64>) {
        let of_score = |s: usize| &parameters[s * per_score..][..per_score];
        let scores: Vec<Vec<f64>> =
            (0..n_scores).map(|s| linear_scores(&features.columns, of_score(s), n_rows)).collect();
        let expected = probabilities(&scores);

        // The loss of each row, then their sum in row order: the same
        // result whatever the number of cores.
        let row_losses: Vec<f64> =
            (0..n_rows).into_par_iter().map(|row| weights[row] * log_loss_of_row(&scores, wanted, row)).collect();
        let penalty: f64 = (0..n_scores).map(|s| dot(&of_score(s)[..n_features], &of_score(s)[..n_features])).sum();
        let loss = (row_losses.iter().sum::<f64>() + 0.5 * l2 * penalty) / total_weight;

        // A score's gradient: each feature times how far off the
        // predicted probabilities are, summed over the rows.
        let mut gradient = Vec::with_capacity(parameters.len());
        for s in 0..n_scores {
            let errors: Vec<f64> =
                expected[s].par_iter().zip(&wanted[s]).zip(weights).map(|((p, y), weight)| weight * (p - y)).collect();
            let coefficients = &of_score(s)[..n_features];
            let per_feature: Vec<f64> = features
                .columns
                .par_iter()
                .zip(coefficients)
                .map(|(column, coefficient)| (dot(column, &errors) + l2 * coefficient) / total_weight)
                .collect();
            gradient.extend(per_feature);
            gradient.push(errors.iter().sum::<f64>() / total_weight);
        }
        (loss, gradient)
    };

    let minimum = minimise(start, spec.tolerance, spec.max_iterations, loss_and_gradient);
    let of_score = |s: usize| &minimum.point[s * per_score..][..per_score];
    Fit {
        coefficients: (0..n_scores).map(|s| of_score(s)[..n_features].to_vec()).collect(),
        intercepts: (0..n_scores).map(|s| of_score(s)[n_features]).collect(),
        n_iterations: minimum.n_iterations,
        converged: minimum.converged,
    }
}

/// One score for every row: the intercept (the last parameter) plus each
/// feature column times its coefficient.
fn linear_scores(columns: &[Vec<f64>], parameters: &[f64], n_rows: usize) -> Vec<f64> {
    let (coefficients, intercept) = parameters.split_at(columns.len());
    let mut scores = vec![intercept[0]; n_rows];
    for (column, &coefficient) in columns.iter().zip(coefficients) {
        if coefficient != 0.0 {
            scores.par_iter_mut().zip(column).with_min_len(4096).for_each(|(score, z)| *score += coefficient * z);
        }
    }
    scores
}

/// The log loss of one row, computed from the scores rather than from
/// the probabilities, so that a very confident prediction does not round
/// to a probability of exactly 0 or 1.
fn log_loss_of_row(scores: &[Vec<f64>], wanted: &[Vec<f64>], row: usize) -> f64 {
    match scores {
        // Two classes: -ln(probability of the true class) = ln(1 + e^score) - wanted × score.
        [score] => {
            let score = score[row];
            score.max(0.0) + (-score.abs()).exp().ln_1p() - wanted[0][row] * score
        }
        // More classes: ln(sum of e^score) - the true class's score.
        _ => {
            let largest = scores.iter().map(|class| class[row]).fold(f64::MIN, f64::max);
            let total: f64 = scores.iter().map(|class| (class[row] - largest).exp()).sum();
            let own: f64 = scores.iter().zip(wanted).map(|(class, wanted)| wanted[row] * class[row]).sum();
            largest + total.ln() - own
        }
    }
}

/// A trained linear model: learned state only, no reference to the
/// training data.
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // The target is exactly 3 × the first feature − 2 × the second + 5.
/// let x = Array2::from_shape_fn((200, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// let y = Array2::from_shape_fn((200, 1), |(row, _)| 3.0 * x[[row, 0]] - 2.0 * x[[row, 1]] + 5.0);
/// let data = EncodedData::new(x, y)?;
///
/// let model = LinearModelSpec::regressor().build()?.fit(&data)?;
/// let close = |found: f64, expected: f64| (found - expected).abs() < 1e-8;
/// assert!(close(model.coefficients(0)[0], 3.0));
/// assert!(close(model.coefficients(0)[1], -2.0));
/// assert!(close(model.intercepts()[0], 5.0));
/// assert_eq!((model.n_scores(), model.n_features()), (1, 2));
///
/// for (feature, coefficient) in model.named_coefficients(0) {
///     println!("{coefficient:+.4} × {feature}");
/// }
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedLinearModel {
    task: Task,
    /// `coefficients[score][feature]`, in the features' own units.
    /// Regression has one score per target; classification has one score
    /// for two classes, and one per class otherwise.
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    classes: Vec<Label>,
    features: Vec<Feature>,
    n_features: usize,
    n_iterations: usize,
    converged: bool,
    /// How many classes each target has.
    #[serde(default)]
    classes_per_target: Vec<usize>,
}

impl FittedLinearModel {
    /// Whether this model classifies or predicts numbers.
    pub fn task(&self) -> Task {
        self.task
    }

    /// How many scores the model keeps: one per regression target; for
    /// a class target, one for two classes and one per class otherwise.
    /// With several class targets their scores follow each other.
    pub fn n_scores(&self) -> usize {
        self.coefficients.len()
    }

    /// The coefficients of one score, one per feature in the order of
    /// `features()`: how much the score changes when the feature goes up
    /// by one unit. For two classes, score 0 is the log-odds of the
    /// second class. Panics if there is no such score.
    pub fn coefficients(&self, score: usize) -> &[f64] {
        &self.coefficients[score]
    }

    /// The intercepts, one per score: the score when every feature is 0.
    pub fn intercepts(&self) -> &[f64] {
        &self.intercepts
    }

    /// One score's features with their coefficients, in feature order.
    pub fn named_coefficients(&self, score: usize) -> Vec<(String, f64)> {
        self.features.iter().map(Feature::display_name).zip(self.coefficients[score].iter().copied()).collect()
    }

    /// The class labels, sorted: what the columns of `predict_proba`
    /// stand for. With several targets, each target's classes follow the
    /// previous target's (see `target_classes` for them one by one).
    /// Empty for regression.
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// The features the model was fitted on.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// How many features the model was fitted on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// How many steps the search for a classifier's coefficients took
    /// (the most of any target, when there are several). 0 for
    /// regression, which is solved in one go.
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }

    /// Whether the search finished before `max_iterations` ran out. If
    /// not, the model is usable but more iterations (or some
    /// `l2_regularization`) would improve it. Always true for regression.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The scores for every row, `[score][row]`.
    fn scores(&self, data: &EncodedData) -> Result<Vec<Vec<f64>>> {
        tree::check_n_features(data, self.n_features)?;
        let score = |(coefficients, intercept): (&Vec<f64>, &f64)| -> Vec<f64> {
            let of_row =
                |row: usize| intercept + data.x.row(row).iter().zip(coefficients).map(|(x, c)| x * c).sum::<f64>();
            (0..data.num_rows()).into_par_iter().map(of_row).collect()
        };
        Ok(self.coefficients.iter().zip(&self.intercepts).map(score).collect())
    }
}

#[typetag::serde]
impl Predictor for FittedLinearModel {
    fn predict(&self, data: &EncodedData) -> Result<Array2<f64>> {
        match self.task {
            Task::Regression => Ok(by_row(&self.scores(data)?)),
            Task::Classification => {
                Ok(classes::most_probable(&self.predict_proba(data)?, &self.classes, &self.classes_per_target))
            }
        }
    }

    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>> {
        if self.task == Task::Regression {
            return Err(Error::InvalidSpec("predict_proba needs a classifier; this model is a regressor".to_string()));
        }
        let groups = scores::score_groups(self.task, self.coefficients.len(), &self.classes_per_target);
        Ok(class_probabilities(&self.scores(data)?, &groups))
    }

    fn score(&self, data: &EncodedData) -> Result<f64> {
        tree::default_score(self.task, &data.y, &self.predict(data)?)
    }

    fn target_classes(&self) -> Vec<Vec<Label>> {
        classes::by_target(&self.classes, &self.classes_per_target)
    }
}
