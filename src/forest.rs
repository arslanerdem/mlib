//! Random forest: many decision trees, each grown on a random sample of
//! the rows and choosing each split from a random subset of the features.
//! Their averaged vote is the prediction.
//!
//! Three types, so the compiler keeps the stages apart:
//! [`RandomForestSpec`] (hyper-parameters) → `build()` →
//! [`ValidRandomForestSpec`] (checked) → `fit()` → [`FittedRandomForest`]
//! (learned state only).

use crate::encoded::{rank_importances, EncodedData, Feature};
use crate::error::{Error, Result};
use crate::random::Rng;
use crate::traits::{Estimator, Label, ParamValue, Predictor};
use crate::tree::{self, BinnedFeatures, Targets, Task, Tree, TreeParams};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How many features each split may choose from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
    /// A fraction of the features, in (0, 1].
    Fraction(f64),
}

impl MaxFeatures {
    fn count(&self, n_features: usize) -> usize {
        let n = n_features as f64;
        let count = match self {
            MaxFeatures::All => n,
            MaxFeatures::Sqrt => n.sqrt(),
            MaxFeatures::Log2 => n.log2(),
            MaxFeatures::Fraction(fraction) => fraction * n,
        };
        (count as usize).clamp(1, n_features)
    }
}

/// Random forest hyper-parameters. `RandomForestSpec::default()` trains
/// a classifier out of the box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RandomForestSpec {
    /// Default: classification.
    pub task: Task,
    /// Default 100.
    pub n_trees: usize,
    /// Default `None` (grow until the leaves are pure).
    pub max_depth: Option<usize>,
    /// Default `Sqrt`.
    pub max_features: MaxFeatures,
    /// Default 1.
    pub min_samples_leaf: usize,
    /// Whether each tree trains on a bootstrap sample of the rows. Default true.
    pub bootstrap: bool,
    /// Default 0. The same seed and data always give the same forest.
    pub seed: u64,
}

impl Default for RandomForestSpec {
    fn default() -> RandomForestSpec {
        RandomForestSpec {
            task: Task::Classification,
            n_trees: 100,
            max_depth: None,
            max_features: MaxFeatures::Sqrt,
            min_samples_leaf: 1,
            bootstrap: true,
            seed: 0,
        }
    }
}

impl RandomForestSpec {
    /// A classifier with default settings.
    pub fn new() -> RandomForestSpec {
        RandomForestSpec::default()
    }

    /// A regressor with default settings. Regression forests consider
    /// every feature at each split (`MaxFeatures::All`).
    pub fn regressor() -> RandomForestSpec {
        RandomForestSpec { task: Task::Regression, max_features: MaxFeatures::All, ..Default::default() }
    }

    pub fn task(mut self, task: Task) -> RandomForestSpec {
        self.task = task;
        self
    }

    pub fn n_trees(mut self, n: usize) -> RandomForestSpec {
        self.n_trees = n;
        self
    }

    pub fn max_depth(mut self, depth: Option<usize>) -> RandomForestSpec {
        self.max_depth = depth;
        self
    }

    pub fn max_features(mut self, max_features: MaxFeatures) -> RandomForestSpec {
        self.max_features = max_features;
        self
    }

    pub fn min_samples_leaf(mut self, n: usize) -> RandomForestSpec {
        self.min_samples_leaf = n;
        self
    }

    pub fn bootstrap(mut self, bootstrap: bool) -> RandomForestSpec {
        self.bootstrap = bootstrap;
        self
    }

    pub fn seed(mut self, seed: u64) -> RandomForestSpec {
        self.seed = seed;
        self
    }

    /// Validate the hyper-parameters. Only a validated spec can be fitted.
    pub fn build(&self) -> Result<ValidRandomForestSpec> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        if self.n_trees == 0 {
            return invalid("n_trees must be at least 1");
        }
        if self.min_samples_leaf == 0 {
            return invalid("min_samples_leaf must be at least 1");
        }
        if self.max_depth == Some(0) {
            return invalid("max_depth must be at least 1 (or None for unlimited)");
        }
        if let MaxFeatures::Fraction(fraction) = self.max_features {
            if !(fraction > 0.0 && fraction <= 1.0) {
                return invalid("max_features fraction must be in (0, 1]");
            }
        }
        Ok(ValidRandomForestSpec { spec: self.clone() })
    }
}

/// A `RandomForestSpec` that passed validation. Create one with
/// [`RandomForestSpec::build`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidRandomForestSpec {
    spec: RandomForestSpec,
}

impl ValidRandomForestSpec {
    /// The validated hyper-parameters.
    pub fn spec(&self) -> &RandomForestSpec {
        &self.spec
    }
}

impl Estimator for ValidRandomForestSpec {
    type Fitted = FittedRandomForest;

    fn fit(&self, data: &EncodedData) -> Result<FittedRandomForest> {
        let spec = &self.spec;
        tree::check_training_data(data)?;

        let (targets, classes) = match spec.task {
            Task::Classification => {
                let (classes, class_of_row) = tree::class_indices(&tree::class_labels(data)?);
                (Targets::Classes { class_of_row, n_classes: classes.len() }, classes)
            }
            Task::Regression => (Targets::values(&data.y), Vec::new()),
        };

        let binned = BinnedFeatures::new(&data.x);
        let params = TreeParams {
            max_depth: spec.max_depth,
            min_samples_leaf: spec.min_samples_leaf,
            features_per_split: spec.max_features.count(data.num_features()),
            bootstrap: spec.bootstrap,
            parallel: false,
        };

        // Trees are independent: grow them in parallel. Each tree has its
        // own random stream, so the result does not depend on scheduling.
        let (trees, tree_importances): (Vec<Tree>, Vec<Vec<f64>>) = (0..spec.n_trees)
            .into_par_iter()
            .map(|t| {
                let rng = Rng::new(spec.seed, t as u64);
                tree::grow_tree(&binned, &targets, data.weights.as_deref(), &params, rng)
            })
            .unzip();

        Ok(FittedRandomForest {
            task: spec.task,
            trees,
            classes,
            feature_importances: tree::average_importances(&tree_importances, data.num_features()),
            features: data.features.clone(),
            n_features: data.num_features(),
        })
    }

    /// Parameters: `n_trees`, `min_samples_leaf`, `seed` (Int);
    /// `max_depth` (Int, or Text "none" for unlimited); `bootstrap` (Bool);
    /// `max_features` (Text "all" / "sqrt" / "log2", or a Float fraction).
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        let invalid = || Error::InvalidSpec(format!("invalid value for {name}: {value:?}"));
        let as_count = || match value {
            ParamValue::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(invalid()),
        };
        let mut spec = self.spec.clone();
        match name {
            "n_trees" => spec.n_trees = as_count()?,
            "min_samples_leaf" => spec.min_samples_leaf = as_count()?,
            "seed" => spec.seed = as_count()? as u64,
            "max_depth" => {
                spec.max_depth = match value {
                    ParamValue::Text(text) if text == "none" => None,
                    _ => Some(as_count()?),
                }
            }
            "bootstrap" => match value {
                ParamValue::Bool(bootstrap) => spec.bootstrap = *bootstrap,
                _ => return Err(invalid()),
            },
            "max_features" => {
                spec.max_features = match value {
                    ParamValue::Text(text) if text == "all" => MaxFeatures::All,
                    ParamValue::Text(text) if text == "sqrt" => MaxFeatures::Sqrt,
                    ParamValue::Text(text) if text == "log2" => MaxFeatures::Log2,
                    ParamValue::Float(fraction) => MaxFeatures::Fraction(*fraction),
                    _ => return Err(invalid()),
                }
            }
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        }
        *self = spec.build()?;
        Ok(())
    }
}

/// A trained forest: learned state only, no reference to the training data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedRandomForest {
    task: Task,
    trees: Vec<Tree>,
    classes: Vec<Label>,
    feature_importances: Vec<f64>,
    features: Vec<Feature>,
    n_features: usize,
}

impl FittedRandomForest {
    /// Whether this forest classifies or predicts numbers.
    pub fn task(&self) -> Task {
        self.task
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Tree number `index`, for inspection. Panics if there is no such tree.
    pub fn tree(&self, index: usize) -> &Tree {
        &self.trees[index]
    }

    /// All the trees, in the order they were grown.
    pub fn trees(&self) -> impl Iterator<Item = &Tree> {
        self.trees.iter()
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

    /// The features the forest was fitted on.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// For every row, the average of the trees' leaf predictions: class
    /// probabilities (classification) or the mean of each target (regression).
    fn average_tree_predictions(&self, data: &EncodedData) -> Result<Array2<f64>> {
        tree::check_n_features(data, self.n_features)?;
        let mut averages = tree::sum_tree_predictions(&self.trees, &data.x);
        let scale = 1.0 / self.trees.len() as f64;
        averages.iter_mut().for_each(|total| *total *= scale);
        let n_outputs = self.trees[0].n_outputs;
        // Invariant: the sums hold `n_outputs` numbers for every row.
        Ok(Array2::from_shape_vec((data.num_rows(), n_outputs), averages).unwrap())
    }

    /// Features with their importance, most important first.
    pub fn ranked_importances(&self) -> Vec<(String, f64)> {
        rank_importances(&self.features, &self.feature_importances)
    }
}

#[typetag::serde]
impl Predictor for FittedRandomForest {
    fn predict(&self, data: &EncodedData) -> Result<Array2<f64>> {
        let averages = self.average_tree_predictions(data)?;
        Ok(match self.task {
            Task::Regression => averages,
            Task::Classification => tree::most_probable_classes(&averages, &self.classes),
        })
    }

    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>> {
        match self.task {
            Task::Classification => self.average_tree_predictions(data),
            Task::Regression => Err(Error::InvalidSpec(
                "predict_proba needs a classifier; this forest is a regressor".to_string(),
            )),
        }
    }

    fn score(&self, data: &EncodedData) -> Result<f64> {
        tree::default_score(self.task, &data.y, &self.predict(data)?)
    }

    fn feature_importances(&self) -> Option<Vec<f64>> {
        Some(self.feature_importances.clone())
    }
}
