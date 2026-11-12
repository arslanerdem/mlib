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
}

impl Default for GradientBoostingSpec {
    fn default() -> GradientBoostingSpec {
        GradientBoostingSpec {
            task: Task::Classification,
            n_rounds: 100,
            learning_rate: 0.1,
            max_depth: Some(6),
            min_samples_leaf: 20,
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

        // Every row starts from the same score: the one that fits the
        // training data best without looking at any feature.
        let total_weight: f64 = sample_weights.iter().sum();
        let initial_scores: Vec<f64> = wanted
            .iter()
            .map(|wanted| {
                let mean = wanted.iter().zip(&sample_weights).map(|(w, weight)| w * weight).sum::<f64>() / total_weight;
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
        for _ in 0..spec.n_rounds {
            let expected = expected_values(spec.task, &scores);

            // One tree per score. They do not depend on each other, so
            // they are grown in parallel.
            let grown: Vec<(Tree, Vec<f64>)> = (0..wanted.len())
                .into_par_iter()
                .map(|s| {
                    let (corrections, weights) =
                        newton_steps(spec.task, &wanted[s], &expected[s], &sample_weights);
                    let targets = Targets::single(corrections);
                    // Nothing is random here: every row and every feature is used.
                    let rng = Rng::new(0, 0);
                    let (mut tree, importances) = tree::grow_tree(&binned, &targets, Some(&weights), &params, rng);
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
        }

        Ok(FittedGradientBoosting {
            task: spec.task,
            trees,
            initial_scores,
            classes,
            feature_importances: tree::average_importances(&tree_importances, data.num_features()),
            features: data.features.clone(),
            n_features: data.num_features(),
        })
    }

    /// Parameters: `n_rounds`, `min_samples_leaf` (Int); `learning_rate`
    /// (Float); `max_depth` (Int, or Text "none" for unlimited).
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        let invalid = || Error::InvalidSpec(format!("invalid value for {name}: {value:?}"));
        let as_count = || match value {
            ParamValue::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(invalid()),
        };
        let mut spec = self.spec.clone();
        match name {
            "n_rounds" => spec.n_rounds = as_count()?,
            "min_samples_leaf" => spec.min_samples_leaf = as_count()?,
            "learning_rate" => match value {
                ParamValue::Float(rate) => spec.learning_rate = *rate,
                _ => return Err(invalid()),
            },
            "max_depth" => {
                spec.max_depth = match value {
                    ParamValue::Text(text) if text == "none" => None,
                    _ => Some(as_count()?),
                }
            }
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

/// Turn `[score][row]` lists into a (rows × scores) matrix.
fn by_row(columns: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_fn((columns[0].len(), columns.len()), |(row, column)| columns[column][row])
}

/// A trained boosted model: learned state only, no reference to the
/// training data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedGradientBoosting {
    pub task: Task,
    /// `trees[s]` are the trees that build up score `s`, in the order
    /// they were grown. The learning rate is already folded into their
    /// leaf values. Regression has one score per target; classification
    /// has one score for two classes, and one per class otherwise.
    pub trees: Vec<Vec<Tree>>,
    /// The score every row starts from, one per score.
    pub initial_scores: Vec<f64>,
    /// The class labels, sorted. Empty for regression.
    pub classes: Vec<Label>,
    /// One per feature; sums to 1.
    pub feature_importances: Vec<f64>,
    /// The model-context features, aligned with `feature_importances`.
    pub features: Vec<Feature>,
    pub n_features: usize,
}

impl FittedGradientBoosting {
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
