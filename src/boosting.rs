//! Gradient boosting: trees grown one after another, each one correcting
//! the errors of those before it.
//!
//! The model keeps a running *score* for every row. Each round compares
//! the scores with the targets, grows a small tree that predicts the
//! correction, and adds a fraction of it (`learning_rate`) to the scores.
//!
//! - Regression: the score is the predicted value. Several targets get
//!   one score each, and each round grows one tree per target.
//! - Two classes: the score is the log-odds of the second class.
//! - More classes: one score per class; their softmax gives the
//!   probabilities. Each round grows one tree per class.
//!
//! Two optional guards against overfitting: `subsample` grows each
//! round's trees on a random part of the rows, and
//! `early_stopping_rounds` holds some rows out and stops when the loss on
//! them stops improving.
//!
//! The types follow the forest: [`GradientBoostingSpec`] → `build()` →
//! [`ValidGradientBoostingSpec`] → `fit()` → [`FittedGradientBoosting`].

use crate::encoded::{rank_importances, EncodedData, Feature};
use crate::error::{Error, Result};
use crate::random::Rng;
use crate::traits::{Estimator, Label, ParamValue, Predictor};
use crate::tree::{self, BinnedFeatures, Targets, Task, Tree, TreeParams};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Gradient boosting hyper-parameters. `GradientBoostingSpec::default()`
/// trains a classifier out of the box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GradientBoostingSpec {
    /// Default: classification.
    pub task: Task,
    /// How many rounds of correction. Default 100.
    pub n_rounds: usize,
    /// The fraction of each tree's correction that is kept, in (0, 1].
    /// Smaller values learn more slowly but usually generalise better.
    /// Default 0.1.
    pub learning_rate: f64,
    /// Default `Some(6)`. Boosting works best with shallow trees.
    pub max_depth: Option<usize>,
    /// Default 20.
    pub min_samples_leaf: usize,
    /// The fraction of the training rows each round's trees are grown on,
    /// in (0, 1]. Below 1, every round sees a different random part of the
    /// data, which usually reduces overfitting and speeds up fitting.
    /// Default 1.0 (all rows).
    pub subsample: f64,
    /// Stop when the loss on held-out rows has not improved for this many
    /// rounds, and keep the model from the best round. `None` (the
    /// default) always runs all `n_rounds`.
    pub early_stopping_rounds: Option<usize>,
    /// The fraction of rows held out to watch the loss, in (0, 1). Used
    /// only with `early_stopping_rounds`. Default 0.1.
    pub validation_fraction: f64,
    /// Chooses the random rows for `subsample` and for the held-out set.
    /// Default 0. The same seed and data always give the same model.
    pub seed: u64,
}

impl Default for GradientBoostingSpec {
    fn default() -> GradientBoostingSpec {
        GradientBoostingSpec {
            task: Task::Classification,
            n_rounds: 100,
            learning_rate: 0.1,
            max_depth: Some(6),
            min_samples_leaf: 20,
            subsample: 1.0,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            seed: 0,
        }
    }
}

impl GradientBoostingSpec {
    /// A classifier with default settings.
    pub fn new() -> GradientBoostingSpec {
        GradientBoostingSpec::default()
    }

    /// A regressor with default settings.
    pub fn regressor() -> GradientBoostingSpec {
        GradientBoostingSpec { task: Task::Regression, ..Default::default() }
    }

    pub fn task(mut self, task: Task) -> GradientBoostingSpec {
        self.task = task;
        self
    }

    pub fn n_rounds(mut self, n: usize) -> GradientBoostingSpec {
        self.n_rounds = n;
        self
    }

    pub fn learning_rate(mut self, rate: f64) -> GradientBoostingSpec {
        self.learning_rate = rate;
        self
    }

    pub fn max_depth(mut self, depth: Option<usize>) -> GradientBoostingSpec {
        self.max_depth = depth;
        self
    }

    pub fn min_samples_leaf(mut self, n: usize) -> GradientBoostingSpec {
        self.min_samples_leaf = n;
        self
    }

    pub fn subsample(mut self, fraction: f64) -> GradientBoostingSpec {
        self.subsample = fraction;
        self
    }

    pub fn early_stopping_rounds(mut self, rounds: Option<usize>) -> GradientBoostingSpec {
        self.early_stopping_rounds = rounds;
        self
    }

    pub fn validation_fraction(mut self, fraction: f64) -> GradientBoostingSpec {
        self.validation_fraction = fraction;
        self
    }

    pub fn seed(mut self, seed: u64) -> GradientBoostingSpec {
        self.seed = seed;
        self
    }

    /// Validate the hyper-parameters. Only a validated spec can be fitted.
    pub fn build(&self) -> Result<ValidGradientBoostingSpec> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        if self.n_rounds == 0 {
            return invalid("n_rounds must be at least 1");
        }
        if !(self.learning_rate > 0.0 && self.learning_rate <= 1.0) {
            return invalid("learning_rate must be in (0, 1]");
        }
        if self.max_depth == Some(0) {
            return invalid("max_depth must be at least 1 (or None for unlimited)");
        }
        if self.min_samples_leaf == 0 {
            return invalid("min_samples_leaf must be at least 1");
        }
        if !(self.subsample > 0.0 && self.subsample <= 1.0) {
            return invalid("subsample must be in (0, 1]");
        }
        if self.early_stopping_rounds == Some(0) {
            return invalid("early_stopping_rounds must be at least 1 (or None to switch it off)");
        }
        if !(self.validation_fraction > 0.0 && self.validation_fraction < 1.0) {
            return invalid("validation_fraction must be between 0 and 1");
        }
        Ok(ValidGradientBoostingSpec { spec: self.clone() })
    }
}

/// A `GradientBoostingSpec` that passed validation. Create one with
/// [`GradientBoostingSpec::build`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidGradientBoostingSpec {
    spec: GradientBoostingSpec,
}

impl ValidGradientBoostingSpec {
    /// The validated hyper-parameters.
    pub fn spec(&self) -> &GradientBoostingSpec {
        &self.spec
    }
}

impl Estimator for ValidGradientBoostingSpec {
    type Fitted = FittedGradientBoosting;

    fn fit(&self, data: &EncodedData) -> Result<FittedGradientBoosting> {
        let spec = &self.spec;
        tree::check_training_data(data)?;
        let n_rows = data.num_rows();
        let sample_weights = data.weights.clone().unwrap_or_else(|| vec![1.0; n_rows]);

        // `wanted[s][row]` is what score `s` should end up predicting for
        // the row: the value of target `s` (regression), or 1 when the
        // row belongs to the score's class and 0 when it does not.
        let (wanted, classes): (Vec<Vec<f64>>, Vec<Label>) = match spec.task {
            Task::Regression => (data.y.columns().into_iter().map(|target| target.to_vec()).collect(), Vec::new()),
            Task::Classification => {
                let (classes, class_of_row) = tree::class_indices(&tree::class_labels(data)?);
                if classes.len() < 2 {
                    return Err(Error::InvalidData("a classifier needs at least two classes in y".to_string()));
                }
                // Two classes share a single score, for the second class.
                let scored_classes = if classes.len() == 2 { 1..2 } else { 0..classes.len() as u32 };
                let is_class = |class| class_of_row.iter().map(|&c| if c == class { 1.0 } else { 0.0 }).collect();
                (scored_classes.map(is_class).collect(), classes)
            }
        };

        // With early stopping, some rows are held out: no tree is grown on
        // them, and the loss on them tells when to stop.
        let mut rng = Rng::new(spec.seed, 0);
        let is_held_out: Vec<bool> = match spec.early_stopping_rounds {
            Some(_) => (0..n_rows).map(|_| rng.chance(spec.validation_fraction)).collect(),
            None => vec![false; n_rows],
        };
        let training_weights: Vec<f64> =
            sample_weights.iter().zip(&is_held_out).map(|(&weight, &held_out)| if held_out { 0.0 } else { weight }).collect();
        let total_weight: f64 = training_weights.iter().sum();
        let held_out_weight: f64 = sample_weights.iter().sum::<f64>() - total_weight;
        if total_weight <= 0.0 || (spec.early_stopping_rounds.is_some() && held_out_weight <= 0.0) {
            return Err(Error::InvalidData(
                "too few rows to hold some out for early stopping; lower or raise validation_fraction".to_string(),
            ));
        }

        // Every row starts from the same score: the one that fits the
        // training data best without looking at any feature.
        let initial_scores: Vec<f64> = wanted
            .iter()
            .map(|wanted| {
                let mean = wanted.iter().zip(&training_weights).map(|(w, weight)| w * weight).sum::<f64>() / total_weight;
                starting_score(spec.task, classes.len(), mean)
            })
            .collect();
        let mut scores: Vec<Vec<f64>> = initial_scores.iter().map(|&score| vec![score; n_rows]).collect();

        let binned = BinnedFeatures::new(&data.x);
        let params = TreeParams {
            max_depth: spec.max_depth,
            min_samples_leaf: spec.min_samples_leaf,
            features_per_split: data.num_features(),
            bootstrap: false,
            parallel: true,
        };

        let mut trees: Vec<Vec<Tree>> = vec![Vec::new(); wanted.len()];
        let mut tree_importances = Vec::new();
        let mut validation_losses = Vec::new();
        let mut best_round = 0;
        for round in 0..spec.n_rounds {
            let expected = expected_values(spec.task, &scores);

            // The rows this round's trees are grown on.
            let round_weights: Vec<f64> = if spec.subsample < 1.0 {
                training_weights.iter().map(|&weight| if rng.chance(spec.subsample) { weight } else { 0.0 }).collect()
            } else {
                training_weights.clone()
            };

            // One tree per score. They do not depend on each other, so
            // they are grown in parallel.
            let grown: Vec<(Tree, Vec<f64>)> = (0..wanted.len())
                .into_par_iter()
                .map(|s| {
                    let (corrections, weights) = newton_steps(spec.task, &wanted[s], &expected[s], &round_weights);
                    let targets = Targets::single(corrections);
                    // The tree itself draws nothing at random: it uses every row it is given and every feature.
                    let (mut tree, importances) =
                        tree::grow_tree(&binned, &targets, Some(&weights), &params, Rng::new(0, 0));
                    tree.leaf_values.iter_mut().for_each(|value| *value *= spec.learning_rate);
                    (tree, importances)
                })
                .collect();

            for (s, (tree, importances)) in grown.into_iter().enumerate() {
                let corrections = tree::sum_tree_predictions(std::slice::from_ref(&tree), &data.x);
                for (score, correction) in scores[s].iter_mut().zip(corrections) {
                    *score += correction;
                }
                trees[s].push(tree);
                tree_importances.push(importances);
            }

            if let Some(patience) = spec.early_stopping_rounds {
                let expected = expected_values(spec.task, &scores);
                validation_losses.push(held_out_loss(spec.task, &wanted, &expected, &sample_weights, &is_held_out));
                if validation_losses[round] < validation_losses[best_round] {
                    best_round = round;
                }
                if round - best_round >= patience {
                    break;
                }
            }
        }
        if spec.early_stopping_rounds.is_some() {
            // Keep the model as it was after its best round.
            trees.iter_mut().for_each(|trees| trees.truncate(best_round + 1));
            tree_importances.truncate((best_round + 1) * wanted.len());
        }

        Ok(FittedGradientBoosting {
            task: spec.task,
            trees,
            initial_scores,
            classes,
            feature_importances: tree::average_importances(&tree_importances, data.num_features()),
            features: data.features.clone(),
            n_features: data.num_features(),
            validation_losses,
        })
    }

    /// Parameters: `n_rounds`, `min_samples_leaf`, `seed` (Int);
    /// `learning_rate`, `subsample`, `validation_fraction` (Float);
    /// `max_depth` and `early_stopping_rounds` (Int, or Text "none").
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        let invalid = || Error::InvalidSpec(format!("invalid value for {name}: {value:?}"));
        let as_count = || match value {
            ParamValue::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(invalid()),
        };
        let as_fraction = || match value {
            ParamValue::Float(fraction) => Ok(*fraction),
            _ => Err(invalid()),
        };
        let as_optional_count = || match value {
            ParamValue::Text(text) if text == "none" => Ok(None),
            _ => as_count().map(Some),
        };
        let mut spec = self.spec.clone();
        match name {
            "n_rounds" => spec.n_rounds = as_count()?,
            "min_samples_leaf" => spec.min_samples_leaf = as_count()?,
            "seed" => spec.seed = as_count()? as u64,
            "learning_rate" => spec.learning_rate = as_fraction()?,
            "subsample" => spec.subsample = as_fraction()?,
            "validation_fraction" => spec.validation_fraction = as_fraction()?,
            "max_depth" => spec.max_depth = as_optional_count()?,
            "early_stopping_rounds" => spec.early_stopping_rounds = as_optional_count()?,
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        }
        *self = spec.build()?;
        Ok(())
    }
}

/// The score that predicts `mean` for every row, where `mean` is the
/// average of what the score should predict.
fn starting_score(task: Task, n_classes: usize, mean: f64) -> f64 {
    // Keeps the logarithms finite when a class has no weight at all.
    let share = mean.clamp(1e-12, 1.0 - 1e-12);
    match task {
        Task::Regression => mean,
        Task::Classification if n_classes == 2 => (share / (1.0 - share)).ln(),
        Task::Classification => share.ln(),
    }
}

/// What the current scores predict, laid out like the scores
/// (`[score][row]`): the value for regression, a probability otherwise.
fn expected_values(task: Task, scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
    match (task, scores.len()) {
        (Task::Regression, _) => scores.to_vec(),
        // Two classes: the sigmoid of the single score.
        (Task::Classification, 1) => {
            vec![scores[0].par_iter().map(|score| 1.0 / (1.0 + (-score).exp())).collect()]
        }
        // More classes: the softmax of each row's scores.
        (Task::Classification, _) => {
            // Per row: the largest score, and the sum of exp(score - largest).
            // Subtracting the largest score first keeps `exp` from overflowing.
            let (largest, total): (Vec<f64>, Vec<f64>) = (0..scores[0].len())
                .into_par_iter()
                .map(|row| {
                    let largest = scores.iter().map(|class| class[row]).fold(f64::MIN, f64::max);
                    (largest, scores.iter().map(|class| (class[row] - largest).exp()).sum::<f64>())
                })
                .unzip();
            let softmax = |class: &Vec<f64>| {
                let rows = class.par_iter().zip(&largest).zip(&total);
                rows.map(|((score, largest), total)| (score - largest).exp() / total).collect()
            };
            scores.iter().map(softmax).collect()
        }
    }
}

/// The correction each row asks for, and how much the row counts when a
/// tree averages the corrections in a leaf.
///
/// This is Newton's method. For regression the correction is simply the
/// residual, `wanted - expected`. For classification the residual is
/// divided by the curvature of the loss, `p (1 - p)`: where the model is
/// already confident (p near 0 or 1) a change in score moves the
/// probability little, so a larger correction is needed. Weighting each
/// row by its curvature makes a leaf's weighted average the Newton step
/// for all its rows together.
fn newton_steps(task: Task, wanted: &[f64], expected: &[f64], sample_weights: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Keeps the division safe when a probability rounds to exactly 0 or 1.
    const MIN_CURVATURE: f64 = 1e-12;
    let rows = wanted.par_iter().zip(expected).zip(sample_weights);
    rows.map(|((wanted, expected), sample_weight)| {
        let curvature = match task {
            Task::Regression => 1.0,
            Task::Classification => (expected * (1.0 - expected)).max(MIN_CURVATURE),
        };
        ((wanted - expected) / curvature, curvature * sample_weight)
    })
    .unzip()
}

/// The model's loss on the held-out rows: mean squared error for
/// regression, log loss for classification. `wanted` and `expected` are
/// laid out `[score][row]`.
fn held_out_loss(
    task: Task,
    wanted: &[Vec<f64>],
    expected: &[Vec<f64>],
    sample_weights: &[f64],
    is_held_out: &[bool],
) -> f64 {
    // Keeps the logarithm finite when a probability rounds to exactly 0.
    let log = |probability: f64| probability.max(1e-15).ln();
    let loss_of_row = |row: usize| -> f64 {
        let pairs = wanted.iter().zip(expected).map(|(wanted, expected)| (wanted[row], expected[row]));
        match (task, wanted.len()) {
            (Task::Regression, _) => pairs.map(|(wanted, expected)| (wanted - expected).powi(2)).sum(),
            // Two classes share one score: `expected` is the second class's probability.
            (Task::Classification, 1) => {
                pairs.map(|(wanted, p)| -(wanted * log(p) + (1.0 - wanted) * log(1.0 - p))).sum()
            }
            // Otherwise `wanted` is 1 for the row's own class and 0 for the others.
            (Task::Classification, _) => pairs.map(|(wanted, p)| -wanted * log(p)).sum(),
        }
    };
    let held_out = (0..is_held_out.len()).filter(|&row| is_held_out[row]);
    let (loss, weight) = held_out.fold((0.0, 0.0), |(loss, weight), row| {
        (loss + sample_weights[row] * loss_of_row(row), weight + sample_weights[row])
    });
    loss / weight
}

/// Turn `[score][row]` lists into a (rows × scores) matrix.
fn by_row(columns: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_fn((columns[0].len(), columns.len()), |(row, column)| columns[column][row])
}

/// A trained boosted model: learned state only, no reference to the
/// training data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedGradientBoosting {
    task: Task,
    /// `trees[s]` are the trees that build up score `s`, in the order
    /// they were grown. The learning rate is already folded into their
    /// leaf values. Regression has one score per target; classification
    /// has one score for two classes, and one per class otherwise.
    trees: Vec<Vec<Tree>>,
    initial_scores: Vec<f64>,
    classes: Vec<Label>,
    feature_importances: Vec<f64>,
    features: Vec<Feature>,
    n_features: usize,
    /// The held-out loss after each round, when early stopping was on.
    #[serde(default)]
    validation_losses: Vec<f64>,
}

impl FittedGradientBoosting {
    /// Whether this model classifies or predicts numbers.
    pub fn task(&self) -> Task {
        self.task
    }

    /// How many scores the model keeps: one per regression target, one
    /// for two classes, one per class otherwise.
    pub fn n_scores(&self) -> usize {
        self.trees.len()
    }

    /// How many rounds were run. Each round grew one tree per score.
    pub fn n_rounds(&self) -> usize {
        self.trees[0].len()
    }

    /// The tree grown for score `score` in round `round`, for inspection.
    /// Its leaf values already include the learning rate. Panics if
    /// there is no such tree.
    pub fn tree(&self, score: usize, round: usize) -> &Tree {
        &self.trees[score][round]
    }

    /// All the trees: every round of the first score, then the next score.
    pub fn trees(&self) -> impl Iterator<Item = &Tree> {
        self.trees.iter().flatten()
    }

    /// The loss on the held-out rows after each round that was run:
    /// mean squared error for regression, log loss for classification.
    /// The model keeps the rounds up to the lowest value. Empty unless
    /// the model was fitted with `early_stopping_rounds`.
    pub fn validation_losses(&self) -> &[f64] {
        &self.validation_losses
    }

    /// The score every row starts from, one per score.
    pub fn initial_scores(&self) -> &[f64] {
        &self.initial_scores
    }

    /// The class labels, sorted. Empty for regression.
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// How much each feature contributed: one number per feature, summing
    /// to 1, in the order of `features()`.
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// The features the model was fitted on.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The scores for every row, `[score][row]`: the starting score plus
    /// every tree's correction.
    fn scores(&self, data: &EncodedData) -> Result<Vec<Vec<f64>>> {
        tree::check_n_features(data, self.n_features)?;
        let score = |(trees, initial): (&Vec<Tree>, &f64)| {
            let mut totals = tree::sum_tree_predictions(trees, &data.x);
            totals.iter_mut().for_each(|total| *total += initial);
            totals
        };
        Ok(self.trees.iter().zip(&self.initial_scores).map(score).collect())
    }

    /// Features with their importance, most important first.
    pub fn ranked_importances(&self) -> Vec<(String, f64)> {
        rank_importances(&self.features, &self.feature_importances)
    }
}

#[typetag::serde]
impl Predictor for FittedGradientBoosting {
    fn predict(&self, data: &EncodedData) -> Result<Array2<f64>> {
        match self.task {
            Task::Regression => Ok(by_row(&self.scores(data)?)),
            Task::Classification => Ok(tree::most_probable_classes(&self.predict_proba(data)?, &self.classes)),
        }
    }

    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>> {
        if self.task == Task::Regression {
            return Err(Error::InvalidSpec(
                "predict_proba needs a classifier; this model is a regressor".to_string(),
            ));
        }
        let expected = expected_values(self.task, &self.scores(data)?);
        Ok(match expected.as_slice() {
            // Two classes: the single score gives the second class's probability.
            [second] => by_row(&[second.iter().map(|p| 1.0 - p).collect(), second.clone()]),
            classes => by_row(classes),
        })
    }

    fn score(&self, data: &EncodedData) -> Result<f64> {
        tree::default_score(self.task, &data.y, &self.predict(data)?)
    }

    fn feature_importances(&self) -> Option<Vec<f64>> {
        Some(self.feature_importances.clone())
    }
}
