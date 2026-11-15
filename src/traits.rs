//! The contracts every algorithm speaks: `fit`, `predict`, `transform`.
//!
//! - An [`Estimator`] is a validated spec. `fit` returns a *different
//!   type* — the fitted model — so an unfitted model can never predict.
//! - A [`Predictor`] is a fitted model.
//! - A [`Transformer`] is a preprocessing step that learns statistics at
//!   `fit` and re-applies them verbatim at `transform`.
//!
//! To add your own algorithm, implement these traits; pipelines,
//! cross-validation and grid search then work with it unchanged. Mark the
//! `Predictor` / `Transformer` impl with `#[typetag::serde]` so a fitted
//! pipeline holding your type can be saved and loaded.

use crate::encoded::{EncodedData, Feature};
use crate::error::{Error, Result};
use crate::table::Table;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A class id or a regression value.
///
/// Classes are numbers: a `Categorical` target gives its dictionary
/// codes, an Int or Bool target its own values.
///
/// ```
/// use mlib::prelude::*;
///
/// let predicted: Vec<Label> = vec![0.0, 1.0, 1.0];
/// assert_eq!(accuracy(&[0.0, 1.0, 0.0], &predicted), 2.0 / 3.0);
/// ```
pub type Label = f64;

/// A hyper-parameter value set by name (grid search).
///
/// ```
/// use mlib::prelude::*;
///
/// // Plain values convert with `.into()`.
/// let depth: ParamValue = 6.into();
/// assert_eq!(depth, ParamValue::Int(6));
/// let strategy: ParamValue = "median".into();
/// assert_eq!(strategy, ParamValue::Text("median".to_string()));
///
/// let mut forest = RandomForestSpec::default().build()?;
/// forest.set_param("n_trees", &50.into())?;
/// assert_eq!(forest.spec().n_trees, 50);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ParamValue {
    /// A whole number, such as a count of trees.
    Int(i64),
    /// A number with decimals, such as a learning rate.
    Float(f64),
    /// A switch.
    Bool(bool),
    /// A named choice, such as "median".
    Text(String),
}

impl From<i64> for ParamValue {
    fn from(value: i64) -> ParamValue {
        ParamValue::Int(value)
    }
}

impl From<f64> for ParamValue {
    fn from(value: f64) -> ParamValue {
        ParamValue::Float(value)
    }
}

impl From<bool> for ParamValue {
    fn from(value: bool) -> ParamValue {
        ParamValue::Bool(value)
    }
}

impl From<&str> for ParamValue {
    fn from(value: &str) -> ParamValue {
        ParamValue::Text(value.to_string())
    }
}

/// Anything that can be fitted to encoded data and produces a predictor.
///
/// The built-in models are used like this:
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// let x = Array2::from_shape_fn((100, 1), |(row, _)| row as f64);
/// let y = x.mapv(|value| 3.0 * value + 1.0);
/// let data = EncodedData::new(x, y)?;
///
/// let model = LinearModelSpec::regressor().build()?.fit(&data)?;
/// assert!(model.score(&data)? > 0.999);
/// # Ok::<(), Error>(())
/// ```
///
/// Your own algorithm is two types: the settings, which implement
/// `Estimator`, and the fitted model, which implements [`Predictor`].
/// Here is a model that always predicts the training mean (your project
/// needs `serde` and `typetag` among its dependencies):
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::{prelude::*, Result};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Clone)]
/// struct MeanSpec;
///
/// #[derive(Debug, Serialize, Deserialize)]
/// struct FittedMean {
///     mean: f64,
///     n_features: usize,
/// }
///
/// impl Estimator for MeanSpec {
///     type Fitted = FittedMean;
///
///     fn fit(&self, data: &EncodedData) -> Result<FittedMean> {
///         let usable = data.num_rows() > 0
///             && data.num_features() > 0
///             && data.y.nrows() == data.num_rows()
///             && data.x.iter().all(|value| value.is_finite());
///         if !usable {
///             return Err(Error::InvalidData("rows, features and finite numbers are needed".to_string()));
///         }
///         let y = data.target()?;
///         Ok(FittedMean { mean: y.iter().sum::<f64>() / y.len() as f64, n_features: data.num_features() })
///     }
///
///     fn set_param(&mut self, name: &str, _value: &ParamValue) -> Result<()> {
///         Err(Error::InvalidSpec(format!("unknown parameter: {name}")))
///     }
/// }
///
/// #[typetag::serde]
/// impl Predictor for FittedMean {
///     fn predict(&self, data: &EncodedData) -> Result<Array2<f64>> {
///         if data.num_features() != self.n_features {
///             return Err(Error::SchemaMismatch(format!("the model was fitted on {} features", self.n_features)));
///         }
///         Ok(Array2::from_elem((data.num_rows(), 1), self.mean))
///     }
///
///     fn predict_proba(&self, _data: &EncodedData) -> Result<Array2<f64>> {
///         Err(Error::InvalidSpec("a regressor has no class probabilities".to_string()))
///     }
///
///     fn score(&self, data: &EncodedData) -> Result<f64> {
///         Ok(r2_score(&data.target()?, &self.predict_single(data)?))
///     }
/// }
///
/// // It honours the library's contracts, and so works in a pipeline,
/// // in cross-validation and in grid search.
/// mlib::testing::check_estimator(&MeanSpec);
/// ```
pub trait Estimator {
    /// The fitted model this estimator produces.
    type Fitted: Predictor;

    /// Learn from the data and return the fitted model. The estimator
    /// itself is left unchanged, so it can be fitted again.
    fn fit(&self, data: &EncodedData) -> Result<Self::Fitted>;

    /// Set a hyper-parameter by name. Must re-validate: a value that makes
    /// the spec invalid, or an unknown name, is `Error::InvalidSpec`.
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()>;
}

/// A type-erased estimator: the form a `Pipeline` holds. Every `Clone`
/// estimator gets this automatically — implement `Estimator`, never this
/// (the compiler will not let you).
///
/// ```
/// use mlib::prelude::*;
///
/// // Any validated spec can be boxed and handed to a pipeline.
/// let model: Box<dyn AnyEstimator> = Box::new(RandomForestSpec::default().build()?);
/// let pipe = Pipeline::new(vec![], model);
/// # let _ = pipe;
/// # Ok::<(), Error>(())
/// ```
pub trait AnyEstimator: sealed::Sealed {
    /// Like [`Estimator::fit`], with the fitted model behind a box.
    fn fit_boxed(&self, data: &EncodedData) -> Result<Box<dyn Predictor>>;
    /// Like [`Estimator::set_param`]. It has its own name so that
    /// `spec.set_param(..)` always means the estimator's.
    fn set_param_boxed(&mut self, name: &str, value: &ParamValue) -> Result<()>;
    /// A copy of the estimator, behind a box.
    fn clone_box(&self) -> Box<dyn AnyEstimator>;
}

mod sealed {
    /// Only this crate can implement `AnyEstimator`.
    pub trait Sealed {}
    impl<T: super::Estimator + Clone + 'static> Sealed for T where T::Fitted: 'static {}
}

impl<T> AnyEstimator for T
where
    T: Estimator + Clone + 'static,
    T::Fitted: 'static,
{
    fn fit_boxed(&self, data: &EncodedData) -> Result<Box<dyn Predictor>> {
        Ok(Box::new(self.fit(data)?))
    }

    fn set_param_boxed(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        self.set_param(name, value)
    }

    fn clone_box(&self) -> Box<dyn AnyEstimator> {
        Box::new(self.clone())
    }
}

/// A fitted model that can predict.
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
/// let model = RandomForestSpec::default().n_trees(20).build()?.fit(&data)?;
/// assert_eq!(model.predict(&data)?.dim(), (200, 1));       // rows × targets
/// assert_eq!(model.predict_single(&data)?.len(), 200);     // the same, as a list
/// assert_eq!(model.predict_proba(&data)?.dim(), (200, 2)); // rows × classes
/// assert_eq!(model.target_classes(), [[0.0, 1.0]]);
/// assert!(model.score(&data)? > 0.95);
/// # Ok::<(), Error>(())
/// ```
///
/// To write your own, see the example on [`Estimator`].
#[typetag::serde(tag = "type")]
pub trait Predictor: Debug {
    /// Predictions as a matrix: one row per sample, one column per
    /// target. A classifier returns each target's predicted class.
    fn predict(&self, data: &EncodedData) -> Result<Array2<f64>>;

    /// Predictions for a model with one target, as a plain list with one
    /// value per sample. Fails if the model predicts several targets.
    fn predict_single(&self, data: &EncodedData) -> Result<Vec<Label>> {
        single_column(self.predict(data)?)
    }

    /// Class probabilities as a matrix: one row per sample, one column
    /// per class. With several targets, each target's classes sit side
    /// by side in target order, and each target's probabilities sum to 1;
    /// [`Predictor::target_classes`] names the columns. Classifiers only;
    /// regressors return `Error::InvalidSpec`.
    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>>;

    /// The default score: accuracy for classifiers, R² for regressors
    /// (averaged over the targets when there are several). Higher is better.
    fn score(&self, data: &EncodedData) -> Result<f64>;

    /// The class labels of each target, sorted, one list per target.
    /// Read one list after another, they name the columns of
    /// `predict_proba`. Empty for regressors.
    fn target_classes(&self) -> Vec<Vec<Label>> {
        Vec::new()
    }

    /// How much each feature contributed, aligned with the features the
    /// model was fitted on. `None` for models that have no such notion.
    fn feature_importances(&self) -> Option<Vec<f64>> {
        None
    }
}

/// The only column of a one-target prediction matrix, as a plain list.
pub(crate) fn single_column(predictions: Array2<f64>) -> Result<Vec<Label>> {
    if predictions.ncols() != 1 {
        return Err(Error::shape("predictions for one target", format!("{} targets", predictions.ncols())));
    }
    Ok(predictions.into_raw_vec_and_offset().0)
}

/// A preprocessing step. `fit` learns statistics from a table;
/// `transform` applies them to any table with the same columns.
///
/// ```
/// use mlib::prelude::*;
///
/// let train = Table::new(vec![("age", Column::Float(vec![Some(30.0), None, Some(50.0)]))])?;
/// let new = Table::new(vec![("age", Column::Float(vec![None]))])?;
///
/// let mut imputer = Imputer::new(ImputeStrategy::Mean);
/// imputer.fit(&train)?;
/// // The new row is filled with the training mean, not its own.
/// assert_eq!(imputer.transform(&new)?.column_at(0).get_f64(0), Some(40.0));
/// # Ok::<(), Error>(())
/// ```
///
/// To write your own, see the example on
/// [`check_transformer`](crate::testing::check_transformer).
#[typetag::serde(tag = "type")]
pub trait Transformer: Debug {
    /// Learn whatever the step needs from the training table.
    fn fit(&mut self, table: &Table) -> Result<()>;

    /// Apply what was learned to a table. `Err(NotFitted)` before `fit`.
    fn transform(&self, table: &Table) -> Result<Table>;

    /// `fit`, then `transform`, on the same table.
    fn fit_transform(&mut self, table: &Table) -> Result<Table> {
        self.fit(table)?;
        self.transform(table)
    }

    /// The features this step produces from the given input features,
    /// read from fitted state. `Err(NotFitted)` before `fit`.
    fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>>;

    /// Clone behind a trait object. Implementors write
    /// `Box::new(self.clone())`.
    fn clone_box(&self) -> Box<dyn Transformer>;

    /// Set a hyper-parameter by name, for grid search. Steps with
    /// tunable parameters override this.
    fn set_param(&mut self, name: &str, _value: &ParamValue) -> Result<()> {
        Err(Error::InvalidSpec(format!("unknown parameter: {name}")))
    }
}

/// A metric that can be looked up by name for CV and grid search.
///
/// The built-in ones come from [`scorer`](crate::metrics::scorer). Your
/// own metric is a few lines:
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// /// The share of predictions within 5 of the truth.
/// struct WithinFive;
///
/// impl Scorer for WithinFive {
///     fn name(&self) -> &'static str {
///         "within_five"
///     }
///
///     fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
///         let close = y_true.iter().zip(y_pred).filter(|(truth, prediction)| (*truth - *prediction).abs() <= 5.0);
///         close.count() as f64 / y_true.len() as f64
///     }
///
///     fn greater_is_better(&self) -> bool {
///         true
///     }
/// }
///
/// let x = Array2::from_shape_fn((60, 1), |(row, _)| row as f64);
/// let y = x.mapv(|value| 2.0 * value);
/// let data = EncodedData::new(x, y)?;
/// let folds = KFold::new(3).with_shuffle(1).splits(60);
/// let cv = cross_validate(&LinearModelSpec::regressor().build()?, &data, &folds, &WithinFive)?;
/// assert_eq!(cv.mean, 1.0);
/// # Ok::<(), Error>(())
/// ```
pub trait Scorer {
    /// The name the metric is reported under.
    fn name(&self) -> &'static str;

    /// `y_pred` holds hard predictions — or, when `uses_probabilities`
    /// is true, the predicted probability of class 1.
    fn score(&self, y_true: &[f64], y_pred: &[f64]) -> f64;

    /// Whether a higher value is better. Cross-validation reports loss
    /// metrics negated, so that "higher is better" holds for every score.
    fn greater_is_better(&self) -> bool;

    /// Whether the metric scores probabilities rather than hard predictions.
    fn uses_probabilities(&self) -> bool {
        false
    }
}
