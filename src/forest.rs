//! Random forest: many decision trees, each grown on a random sample of
//! the rows and choosing each split from a random subset of the features.
//! Their averaged vote is the prediction.
//!
//! Three types, so the compiler keeps the stages apart:
//! [`RandomForestSpec`] (hyper-parameters) → `build()` →
//! [`ValidRandomForestSpec`] (checked) → `fit()` → [`FittedRandomForest`]
//! (learned state only).

use crate::classes::{self, ClassWeight, TrainingClasses};
use crate::encoded::{rank_importances, EncodedData, Feature};
use crate::error::{Error, Result};
use crate::random::Rng;
use crate::traits::{Estimator, Label, ParamValue, Predictor};
use crate::tree::{self, BinnedFeatures, Targets, Task, Tree, TreeParams};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How many features each split may choose from.
///
/// ```
/// use mlib::prelude::*;
///
/// // With 16 features: all 16, 4, 4, or 8 of them, drawn afresh at every split.
/// for choice in [MaxFeatures::All, MaxFeatures::Sqrt, MaxFeatures::Log2, MaxFeatures::Fraction(0.5)] {
///     let forest = RandomForestSpec::default().max_features(choice).build()?;
///     assert_eq!(forest.spec().max_features, choice);
/// }
/// assert!(RandomForestSpec::default().max_features(MaxFeatures::Fraction(1.5)).build().is_err());
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MaxFeatures {
    /// Every feature.
    All,
    /// The square root of the number of features.
    Sqrt,
    /// The base-2 logarithm of the number of features.
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
///
/// ```
/// use mlib::prelude::*;
///
/// let spec = RandomForestSpec::default()
///     .n_trees(200)
///     .max_depth(Some(8))
///     .max_features(MaxFeatures::Fraction(0.5))
///     .min_samples_leaf(5)
///     .seed(42);
/// let forest = spec.build()?; // checked, ready to fit
/// assert_eq!(forest.spec().n_trees, 200);
///
/// // A setting that makes no sense is caught here, not during the fit.
/// assert!(RandomForestSpec::default().n_trees(0).build().is_err());
///
/// let regressor = RandomForestSpec::regressor().build()?;
/// assert_eq!(regressor.spec().task, Task::Regression);
/// # Ok::<(), Error>(())
/// ```
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
    /// Classification only: how much the rows of each class count.
    /// Default `ClassWeight::None` (all the same).
    #[serde(default)]
    pub class_weight: ClassWeight,
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
            class_weight: ClassWeight::None,
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

    /// Classification or regression.
    pub fn task(mut self, task: Task) -> RandomForestSpec {
        self.task = task;
        self
    }

    /// How many trees to grow. More trees give steadier predictions and
    /// take proportionally longer.
    pub fn n_trees(mut self, n: usize) -> RandomForestSpec {
        self.n_trees = n;
        self
    }

    /// The most splits between the root and a leaf; `None` for no limit.
    pub fn max_depth(mut self, depth: Option<usize>) -> RandomForestSpec {
        self.max_depth = depth;
        self
    }

    /// How many features each split may choose from.
    pub fn max_features(mut self, max_features: MaxFeatures) -> RandomForestSpec {
        self.max_features = max_features;
        self
    }

    /// The fewest rows a leaf may hold. Larger values give smoother,
    /// smaller trees.
    pub fn min_samples_leaf(mut self, n: usize) -> RandomForestSpec {
        self.min_samples_leaf = n;
        self
    }

    /// Whether each tree trains on a bootstrap sample of the rows
    /// (drawn with replacement) rather than on all of them.
    pub fn bootstrap(mut self, bootstrap: bool) -> RandomForestSpec {
        self.bootstrap = bootstrap;
        self
    }

    /// The seed for all random choices. The same seed and data always
    /// give the same forest.
    pub fn seed(mut self, seed: u64) -> RandomForestSpec {
        self.seed = seed;
        self
    }

    /// How much the rows of each class count (classification only).
    /// `ClassWeight::Balanced` makes a rare class count as much as a
    /// common one.
    pub fn class_weight(mut self, class_weight: ClassWeight) -> RandomForestSpec {
        self.class_weight = class_weight;
        self
    }

    /// Validate the hyper-parameters. Only a validated spec can be fitted.
    pub fn build(&self) -> Result<ValidRandomForestSpec> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        self.class_weight.check(self.task)?;
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
/// let mut forest = RandomForestSpec::default().n_trees(20).build()?;
/// let fitted = forest.fit(&data)?;
/// assert!(fitted.score(&data)? > 0.95);
///
/// // Changing a setting by name checks it again.
/// forest.set_param("max_depth", &2.into())?;
/// assert!(forest.set_param("n_trees", &0.into()).is_err());
/// # Ok::<(), Error>(())
/// ```
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
        let weights = classes::row_weights(&spec.class_weight, data)?;

        let (targets, classes, classes_per_target) = match spec.task {
            Task::Classification => {
                let found = TrainingClasses::new(data);
                (Targets::classes(found.index_of_row, &found.per_target), found.labels, found.per_target)
            }
            Task::Regression => (Targets::values(&data.y), Vec::new(), Vec::new()),
        };

        let binned = BinnedFeatures::new(&data.x);
        let params = TreeParams {
            max_depth: spec.max_depth,
            min_samples_leaf: spec.min_samples_leaf,
            features_per_split: spec.max_features.count(data.num_features()),
            bootstrap: spec.bootstrap,
            features: (0..data.num_features()).collect(),
            l2: 0.0,
            parallel: false,
        };

        // Trees are independent: grow them in parallel. Each tree has its
        // own random stream, so the result does not depend on scheduling.
        let (trees, tree_importances): (Vec<Tree>, Vec<Vec<f64>>) = (0..spec.n_trees)
            .into_par_iter()
            .map(|t| {
                let rng = Rng::new(spec.seed, t as u64);
                let (tree, gains) = tree::grow_tree(&binned, &targets, weights.as_deref(), &params, rng);
                (tree, tree::as_shares(gains))
            })
            .unzip();

        Ok(FittedRandomForest {
            task: spec.task,
            trees,
            classes,
            classes_per_target,
            // Every tree counts the same: the mean of the trees' shares.
            feature_importances: tree::as_shares(tree::sum_per_feature(&tree_importances, data.num_features())),
            features: data.features.clone(),
            n_features: data.num_features(),
        })
    }

    /// Parameters: `n_trees`, `min_samples_leaf`, `seed` (Int);
    /// `max_depth` (Int, or Text "none" for unlimited); `bootstrap` (Bool);
    /// `max_features` (Text "all" / "sqrt" / "log2", or a Float fraction);
    /// `class_weight` (Text "none" / "balanced").
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
            "class_weight" => spec.class_weight = ClassWeight::from_param(value)?,
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        }
        *self = spec.build()?;
        Ok(())
    }
}

/// A trained forest: learned state only, no reference to the training data.
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // 200 rows; the class is 1 when the first feature is large.
/// let x = Array2::from_shape_fn((200, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// let y = Array2::from_shape_fn((200, 1), |(row, _)| if x[[row, 0]] >= 25.0 { 1.0 } else { 0.0 });
/// let data = EncodedData::new(x, y)?;
/// let forest = RandomForestSpec::default().n_trees(20).build()?.fit(&data)?;
///
/// assert_eq!((forest.n_trees(), forest.n_features()), (20, 2));
/// assert_eq!(forest.classes(), [0.0, 1.0]);
/// let (most_important, share) = &forest.ranked_importances()[0];
/// assert_eq!(most_important, "x0");
/// assert!(*share > 0.5);
///
/// // Every tree can be read.
/// let deepest = forest.trees().map(Tree::depth).max().unwrap();
/// println!("the deepest of the {} trees has depth {deepest}", forest.n_trees());
/// # Ok::<(), Error>(())
/// ```
///
/// With several class targets, the probabilities of each sit side by side:
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// let x = Array2::from_shape_fn((200, 2), |(row, feature)| ((row * (feature + 3)) % 50) as f64);
/// // Target 0 has two classes, target 1 has three.
/// let y = Array2::from_shape_fn((200, 2), |(row, target)| {
///     if target == 0 { (x[[row, 0]] / 25.0).floor() } else { (x[[row, 1]] / 17.0).floor() }
/// });
/// let data = EncodedData::new(x, y)?;
/// let forest = RandomForestSpec::default().n_trees(20).build()?.fit(&data)?;
///
/// assert_eq!(forest.predict(&data)?.dim(), (200, 2));       // one column per target
/// assert_eq!(forest.predict_proba(&data)?.dim(), (200, 5)); // 2 classes, then 3
/// assert_eq!(forest.target_classes(), [vec![0.0, 1.0], vec![0.0, 1.0, 2.0]]);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedRandomForest {
    task: Task,
    trees: Vec<Tree>,
    /// Every target's classes, one target after another.
    classes: Vec<Label>,
    feature_importances: Vec<f64>,
    features: Vec<Feature>,
    n_features: usize,
    /// How many classes each target has. Empty in a model saved before
    /// classifiers took several targets: all classes are its one target's.
    #[serde(default)]
    classes_per_target: Vec<usize>,
}

impl FittedRandomForest {
    /// Whether this forest classifies or predicts numbers.
    pub fn task(&self) -> Task {
        self.task
    }

    /// How many trees the forest has.
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

    /// The class labels, sorted: what the columns of `predict_proba`
    /// stand for. With several targets, each target's classes follow the
    /// previous target's (see `target_classes` for them one by one).
    /// Empty for regression.
    pub fn classes(&self) -> &[Label] {
        &self.classes
    }

    /// How much each feature contributed: one number per feature, summing
    /// to 1, in the order of `features()`. Each tree's improvement is
    /// shared out among the features it split on, and the trees' shares
    /// are averaged.
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// The features the forest was fitted on.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// How many features the forest was fitted on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// For every row, the average of the trees' leaf predictions: class
    /// probabilities, the targets side by side (classification), or the
    /// mean of each target (regression).
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
            Task::Classification => classes::most_probable(&averages, &self.classes, &self.classes_per_target),
        })
    }

    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>> {
        match self.task {
            Task::Classification => self.average_tree_predictions(data),
            Task::Regression => {
                Err(Error::InvalidSpec("predict_proba needs a classifier; this forest is a regressor".to_string()))
            }
        }
    }

    fn score(&self, data: &EncodedData) -> Result<f64> {
        tree::default_score(self.task, &data.y, &self.predict(data)?)
    }

    fn feature_importances(&self) -> Option<Vec<f64>> {
        Some(self.feature_importances.clone())
    }

    fn target_classes(&self) -> Vec<Vec<Label>> {
        classes::by_target(&self.classes, &self.classes_per_target)
    }
}
