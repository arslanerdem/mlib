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
//! - Several class targets: each target has its scores as above, and
//!   they are learned side by side in the same rounds.
//!
//! Four optional guards against overfitting:
//!
//! - `subsample` grows each round's trees on a random part of the rows;
//! - `feature_fraction` lets each round's trees use only a random part
//!   of the features;
//! - `l2_regularization` makes leaves with little data predict smaller
//!   corrections;
//! - `early_stopping_rounds` holds some rows out and stops when the loss
//!   on them stops improving.
//!
//! The types follow the forest: [`GradientBoostingSpec`] → `build()` →
//! [`ValidGradientBoostingSpec`] → `fit()` → [`FittedGradientBoosting`].

use crate::classes::{self, ClassWeight};
use crate::encoded::{rank_importances, EncodedData, Feature};
use crate::error::{Error, Result};
use crate::random::Rng;
use crate::scores::{self, by_row, class_probabilities, expected_values, starting_score, Wanted};
use crate::traits::{Estimator, Label, ParamValue, Predictor};
use crate::tree::{self, BinnedFeatures, Targets, Task, Tree, TreeParams};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Gradient boosting hyper-parameters. `GradientBoostingSpec::default()`
/// trains a classifier out of the box.
///
/// ```
/// use mlib::prelude::*;
///
/// let spec = GradientBoostingSpec::default()
///     .n_rounds(500)
///     .learning_rate(0.05)
///     .max_depth(Some(4))
///     .subsample(0.8)                  // each round sees a random 80% of the rows
///     .feature_fraction(0.8)           // ... and a random 80% of the features
///     .l2_regularization(1.0)          // small leaves predict smaller corrections
///     .early_stopping_rounds(Some(20)) // stop 20 rounds after the best one
///     .seed(42);
/// let boosting = spec.build()?; // checked, ready to fit
/// assert_eq!(boosting.spec().n_rounds, 500);
///
/// // A setting that makes no sense is caught here, not during the fit.
/// assert!(GradientBoostingSpec::default().learning_rate(0.0).build().is_err());
///
/// let regressor = GradientBoostingSpec::regressor().build()?;
/// assert_eq!(regressor.spec().task, Task::Regression);
/// # Ok::<(), Error>(())
/// ```
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
    /// The fraction of the features each round's trees may split on, in
    /// (0, 1]. Below 1, every round sees a different random part of the
    /// features, so no single strong feature dominates every tree; it
    /// also speeds up fitting. At least one feature is always kept.
    /// Default 1.0 (all features).
    #[serde(default = "all")]
    pub feature_fraction: f64,
    /// L2 regularisation, 0 or more. A leaf's correction is its rows'
    /// total error divided by their total weight; this number is added to
    /// that weight (and likewise when a split is scored). A leaf with few
    /// rows is then pulled towards zero, a leaf with many is barely
    /// changed. Typical values are between 0.1 and 10. Default 0 (none).
    #[serde(default)]
    pub l2_regularization: f64,
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
    /// Classification only: how much the rows of each class count.
    /// Default `ClassWeight::None` (all the same).
    #[serde(default)]
    pub class_weight: ClassWeight,
}

/// The value of a fraction that leaves nothing out: what a spec saved
/// before `feature_fraction` existed should load with.
fn all() -> f64 {
    1.0
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
            feature_fraction: 1.0,
            l2_regularization: 0.0,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            seed: 0,
            class_weight: ClassWeight::None,
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

    /// Classification or regression.
    pub fn task(mut self, task: Task) -> GradientBoostingSpec {
        self.task = task;
        self
    }

    /// How many rounds of correction to run.
    pub fn n_rounds(mut self, n: usize) -> GradientBoostingSpec {
        self.n_rounds = n;
        self
    }

    /// The fraction of each tree's correction that is kept, in (0, 1].
    /// A smaller rate usually needs more rounds.
    pub fn learning_rate(mut self, rate: f64) -> GradientBoostingSpec {
        self.learning_rate = rate;
        self
    }

    /// The most splits between the root and a leaf; `None` for no limit.
    pub fn max_depth(mut self, depth: Option<usize>) -> GradientBoostingSpec {
        self.max_depth = depth;
        self
    }

    /// The fewest rows a leaf may hold.
    pub fn min_samples_leaf(mut self, n: usize) -> GradientBoostingSpec {
        self.min_samples_leaf = n;
        self
    }

    /// The fraction of the rows each round's trees are grown on, in (0, 1].
    pub fn subsample(mut self, fraction: f64) -> GradientBoostingSpec {
        self.subsample = fraction;
        self
    }

    /// The fraction of the features each round's trees may split on, in (0, 1].
    pub fn feature_fraction(mut self, fraction: f64) -> GradientBoostingSpec {
        self.feature_fraction = fraction;
        self
    }

    /// L2 regularisation, 0 or more: leaves with little data predict
    /// smaller corrections.
    pub fn l2_regularization(mut self, l2: f64) -> GradientBoostingSpec {
        self.l2_regularization = l2;
        self
    }

    /// Stop when the loss on held-out rows has not improved for this
    /// many rounds; `None` to always run every round.
    pub fn early_stopping_rounds(mut self, rounds: Option<usize>) -> GradientBoostingSpec {
        self.early_stopping_rounds = rounds;
        self
    }

    /// The fraction of the rows held out for early stopping, in (0, 1).
    pub fn validation_fraction(mut self, fraction: f64) -> GradientBoostingSpec {
        self.validation_fraction = fraction;
        self
    }

    /// The seed for all random choices. The same seed and data always
    /// give the same model.
    pub fn seed(mut self, seed: u64) -> GradientBoostingSpec {
        self.seed = seed;
        self
    }

    /// How much the rows of each class count (classification only).
    /// `ClassWeight::Balanced` makes a rare class count as much as a
    /// common one.
    pub fn class_weight(mut self, class_weight: ClassWeight) -> GradientBoostingSpec {
        self.class_weight = class_weight;
        self
    }

    /// Validate the hyper-parameters. Only a validated spec can be fitted.
    pub fn build(&self) -> Result<ValidGradientBoostingSpec> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        self.class_weight.check(self.task)?;
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
        if !(self.feature_fraction > 0.0 && self.feature_fraction <= 1.0) {
            return invalid("feature_fraction must be in (0, 1]");
        }
        if !(self.l2_regularization >= 0.0 && self.l2_regularization.is_finite()) {
            return invalid("l2_regularization must be 0 or more");
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
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // 400 rows; the target is a curve in the first feature plus the second.
/// let x = Array2::from_shape_fn((400, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// let y = Array2::from_shape_fn((400, 1), |(row, _)| (x[[row, 0]] / 8.0).sin() + x[[row, 1]] / 50.0);
/// let data = EncodedData::new(x, y)?;
///
/// let mut boosting = GradientBoostingSpec::regressor().n_rounds(50).build()?;
/// let fitted = boosting.fit(&data)?;
/// assert!(fitted.score(&data)? > 0.9);
///
/// // Changing a setting by name checks it again.
/// boosting.set_param("learning_rate", &0.3.into())?;
/// assert!(boosting.set_param("learning_rate", &(-1.0).into()).is_err());
/// # Ok::<(), Error>(())
/// ```
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
        tree::check_training_data(data, true)?;
        let n_rows = data.num_rows();
        let sample_weights = classes::row_weights(&spec.class_weight, data)?.unwrap_or_else(|| vec![1.0; n_rows]);

        let Wanted { values: wanted, groups, classes, classes_per_target } = Wanted::new(spec.task, data)?;

        // With early stopping, some rows are held out: no tree is grown on
        // them, and the loss on them tells when to stop.
        let mut rng = Rng::new(spec.seed, 0);
        let is_held_out: Vec<bool> = match spec.early_stopping_rounds {
            Some(_) => (0..n_rows).map(|_| rng.chance(spec.validation_fraction)).collect(),
            None => vec![false; n_rows],
        };
        let training_weights: Vec<f64> = sample_weights
            .iter()
            .zip(&is_held_out)
            .map(|(&weight, &held_out)| if held_out { 0.0 } else { weight })
            .collect();
        let total_weight: f64 = training_weights.iter().sum();
        let held_out_weight: f64 = sample_weights.iter().sum::<f64>() - total_weight;
        if total_weight <= 0.0 || (spec.early_stopping_rounds.is_some() && held_out_weight <= 0.0) {
            return Err(Error::InvalidData(
                "too few rows to hold some out for early stopping; lower or raise validation_fraction".to_string(),
            ));
        }

        // Every row starts from the same score: the one that fits the
        // training data best without looking at any feature.
        let starting_score_of = |wanted: &Vec<f64>, group_size: usize| {
            let mean = wanted.iter().zip(&training_weights).map(|(w, weight)| w * weight).sum::<f64>() / total_weight;
            starting_score(spec.task, group_size, mean)
        };
        let initial_scores: Vec<f64> = groups
            .iter()
            .flat_map(|group| wanted[group.clone()].iter().map(|wanted| starting_score_of(wanted, group.len())))
            .collect();
        let mut scores: Vec<Vec<f64>> = initial_scores.iter().map(|&score| vec![score; n_rows]).collect();

        let binned = BinnedFeatures::new(&data.x);
        let all_features: Vec<usize> = (0..data.num_features()).collect();
        // Rounded up, so that a small fraction of a few features still keeps one.
        let n_features_per_round = (spec.feature_fraction * all_features.len() as f64).ceil() as usize;
        let mut params = TreeParams {
            max_depth: spec.max_depth,
            min_samples_leaf: spec.min_samples_leaf,
            features_per_split: data.num_features(),
            bootstrap: false,
            features: all_features.clone(),
            l2: spec.l2_regularization,
            parallel: true,
        };

        let mut trees: Vec<Vec<Tree>> = vec![Vec::new(); wanted.len()];
        // `gains[s]` holds, round after round, each feature's gain in score `s`'s tree.
        let mut gains: Vec<Vec<Vec<f64>>> = vec![Vec::new(); wanted.len()];
        let mut validation_losses = Vec::new();
        let mut best_round = 0;
        for round in 0..spec.n_rounds {
            let expected = expected_values(spec.task, &scores, &groups);

            // The rows this round's trees are grown on.
            let round_weights: Vec<f64> = if spec.subsample < 1.0 {
                training_weights.iter().map(|&weight| if rng.chance(spec.subsample) { weight } else { 0.0 }).collect()
            } else {
                training_weights.clone()
            };

            // The features this round's trees may split on.
            if n_features_per_round < all_features.len() {
                let mut shuffled = all_features.clone();
                rng.shuffle(&mut shuffled);
                shuffled.truncate(n_features_per_round);
                shuffled.sort_unstable();
                params.features = shuffled;
            }

            // One tree per score. They do not depend on each other, so
            // they are grown in parallel.
            let grown: Vec<(Tree, Vec<f64>)> = (0..wanted.len())
                .into_par_iter()
                .map(|s| {
                    let (corrections, weights) = newton_steps(spec.task, &wanted[s], &expected[s], &round_weights);
                    let targets = Targets::single(corrections);
                    // The tree itself draws nothing at random: it uses every row and feature it is given.
                    let (mut tree, gains) = tree::grow_tree(&binned, &targets, Some(&weights), &params, Rng::new(0, 0));
                    tree.leaf_values.iter_mut().for_each(|value| *value *= spec.learning_rate);
                    (tree, gains)
                })
                .collect();

            for (s, (tree, tree_gains)) in grown.into_iter().enumerate() {
                let corrections = tree::sum_tree_predictions(std::slice::from_ref(&tree), &data.x);
                for (score, correction) in scores[s].iter_mut().zip(corrections) {
                    *score += correction;
                }
                trees[s].push(tree);
                gains[s].push(tree_gains);
            }

            if let Some(patience) = spec.early_stopping_rounds {
                let expected = expected_values(spec.task, &scores, &groups);
                let loss = held_out_loss(spec.task, &wanted, &expected, &groups, &sample_weights, &is_held_out);
                validation_losses.push(loss);
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
            gains.iter_mut().for_each(|gains| gains.truncate(best_round + 1));
        }

        Ok(FittedGradientBoosting {
            task: spec.task,
            trees,
            initial_scores,
            classes,
            classes_per_target,
            feature_importances: importances(spec.task, &gains, data.num_features()),
            features: data.features.clone(),
            n_features: data.num_features(),
            validation_losses,
        })
    }

    /// Trees learn which side of each split missing values belong on.
    fn accepts_missing(&self) -> bool {
        true
    }

    /// Parameters: `n_rounds`, `min_samples_leaf`, `seed` (Int);
    /// `learning_rate`, `subsample`, `feature_fraction`,
    /// `l2_regularization`, `validation_fraction` (Float);
    /// `max_depth` and `early_stopping_rounds` (Int, or Text "none");
    /// `class_weight` (Text "none" / "balanced").
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
            "feature_fraction" => spec.feature_fraction = as_fraction()?,
            "l2_regularization" => spec.l2_regularization = as_fraction()?,
            "validation_fraction" => spec.validation_fraction = as_fraction()?,
            "max_depth" => spec.max_depth = as_optional_count()?,
            "early_stopping_rounds" => spec.early_stopping_rounds = as_optional_count()?,
            "class_weight" => spec.class_weight = ClassWeight::from_param(value)?,
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        }
        *self = spec.build()?;
        Ok(())
    }
}

/// A model's feature importances from its trees' gains (`[score][round][feature]`).
///
/// A tree's gain is how much its splits lowered the loss, so the early
/// trees, which do most of the work, count for most — and a late tree
/// that polishes details counts for little. A classifier's scores share
/// one loss, so all their trees are simply added up. Regression targets
/// may be in different units: each target's gains are turned into shares
/// first, and the targets' shares are averaged.
fn importances(task: Task, gains: &[Vec<Vec<f64>>], n_features: usize) -> Vec<f64> {
    let of_score = |rounds: &Vec<Vec<f64>>| tree::sum_per_feature(rounds, n_features);
    match task {
        Task::Classification => tree::as_shares(tree::sum_per_feature(gains.iter().flatten(), n_features)),
        Task::Regression => {
            let shares: Vec<Vec<f64>> = gains.iter().map(|rounds| tree::as_shares(of_score(rounds))).collect();
            tree::as_shares(tree::sum_per_feature(&shares, n_features))
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
/// regression, log loss for classification, added up over the targets.
/// `wanted` and `expected` are laid out `[score][row]`; `groups` says
/// which scores belong to which target.
fn held_out_loss(
    task: Task,
    wanted: &[Vec<f64>],
    expected: &[Vec<f64>],
    groups: &[Range<usize>],
    sample_weights: &[f64],
    is_held_out: &[bool],
) -> f64 {
    // Keeps the logarithm finite when a probability rounds to exactly 0.
    let log = |probability: f64| probability.max(1e-15).ln();
    let loss_of_target = |row: usize, group: &Range<usize>| -> f64 {
        let (wanted, expected) = (&wanted[group.clone()], &expected[group.clone()]);
        let pairs = wanted.iter().zip(expected).map(|(wanted, expected)| (wanted[row], expected[row]));
        match (task, group.len()) {
            (Task::Regression, _) => pairs.map(|(wanted, expected)| (wanted - expected).powi(2)).sum(),
            // Two classes share one score: `expected` is the second class's probability.
            (Task::Classification, 1) => {
                pairs.map(|(wanted, p)| -(wanted * log(p) + (1.0 - wanted) * log(1.0 - p))).sum()
            }
            // Otherwise `wanted` is 1 for the row's own class and 0 for the others.
            (Task::Classification, _) => pairs.map(|(wanted, p)| -wanted * log(p)).sum(),
        }
    };
    let loss_of_row = |row: usize| -> f64 { groups.iter().map(|group| loss_of_target(row, group)).sum() };
    let held_out = (0..is_held_out.len()).filter(|&row| is_held_out[row]);
    let (loss, weight) = held_out.fold((0.0, 0.0), |(loss, weight), row| {
        (loss + sample_weights[row] * loss_of_row(row), weight + sample_weights[row])
    });
    loss / weight
}

/// A trained boosted model: learned state only, no reference to the
/// training data.
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // 400 rows; the target is a curve in the first feature plus the second.
/// let x = Array2::from_shape_fn((400, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// let y = Array2::from_shape_fn((400, 1), |(row, _)| (x[[row, 0]] / 8.0).sin() + x[[row, 1]] / 50.0);
/// let data = EncodedData::new(x, y)?;
///
/// let model = GradientBoostingSpec::regressor()
///     .n_rounds(300)
///     .early_stopping_rounds(Some(10))
///     .seed(1)
///     .build()?
///     .fit(&data)?;
///
/// // The model keeps the rounds up to the best one on the held-out rows.
/// assert!(model.n_rounds() <= 300);
/// assert!(model.validation_losses().len() >= model.n_rounds());
/// assert!(model.score(&data)? > 0.9);
///
/// // One score per target; every row starts from the target's mean.
/// assert_eq!((model.n_scores(), model.n_features()), (1, 2));
/// println!("starting score {:.3}", model.initial_scores()[0]);
///
/// // Every tree can be read, and the features ranked.
/// assert!(model.tree(0, 0).depth() <= 6);
/// assert_eq!(model.trees().count(), model.n_rounds());
/// assert_eq!(model.ranked_importances()[0].0, "x0");
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedGradientBoosting {
    task: Task,
    /// `trees[s]` are the trees that build up score `s`, in the order
    /// they were grown. The learning rate is already folded into their
    /// leaf values. Regression has one score per target; a class target
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
    /// How many classes each target has. Empty in a model saved before
    /// classifiers took several targets: all classes are its one target's.
    #[serde(default)]
    classes_per_target: Vec<usize>,
}

impl FittedGradientBoosting {
    /// Whether this model classifies or predicts numbers.
    pub fn task(&self) -> Task {
        self.task
    }

    /// How many scores the model keeps: one per regression target; for
    /// a class target, one for two classes and one per class otherwise.
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

    /// The class labels, sorted: what the columns of `predict_proba`
    /// stand for. With several targets, each target's classes follow the
    /// previous target's (see `target_classes` for them one by one).
    /// Empty for regression.
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// How much each feature contributed: one number per feature, summing
    /// to 1, in the order of `features()`. It is the feature's share of
    /// the total improvement made by all the splits in all the trees.
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// The features the model was fitted on.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// How many features the model was fitted on.
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

    /// Which scores belong to which target.
    fn score_groups(&self) -> Vec<Range<usize>> {
        scores::score_groups(self.task, self.trees.len(), &self.classes_per_target)
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
            Task::Classification => {
                Ok(classes::most_probable(&self.predict_proba(data)?, &self.classes, &self.classes_per_target))
            }
        }
    }

    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>> {
        if self.task == Task::Regression {
            return Err(Error::InvalidSpec("predict_proba needs a classifier; this model is a regressor".to_string()));
        }
        Ok(class_probabilities(&self.scores(data)?, &self.score_groups()))
    }

    fn score(&self, data: &EncodedData) -> Result<f64> {
        tree::default_score(self.task, &data.y, &self.predict(data)?)
    }

    fn feature_importances(&self) -> Option<Vec<f64>> {
        Some(self.feature_importances.clone())
    }

    fn accepts_missing(&self) -> bool {
        true
    }

    fn target_classes(&self) -> Vec<Vec<Label>> {
        classes::by_target(&self.classes, &self.classes_per_target)
    }
}
