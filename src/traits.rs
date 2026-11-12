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
pub type Label = f64;

/// A hyper-parameter value set by name (grid search).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
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
pub trait Estimator {
    type Fitted: Predictor;

    fn fit(&self, data: &EncodedData) -> Result<Self::Fitted>;

    /// Set a hyper-parameter by name. Must re-validate: a value that makes
    /// the spec invalid, or an unknown name, is `Error::InvalidSpec`.
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()>;
}

/// A type-erased estimator: the form a `Pipeline` holds. Every `Clone`
/// estimator gets this automatically — implement `Estimator`, never this
/// (the compiler will not let you).
pub trait AnyEstimator: sealed::Sealed {
    fn fit_boxed(&self, data: &EncodedData) -> Result<Box<dyn Predictor>>;
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()>;
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

    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        <T as Estimator>::set_param(self, name, value)
    }

    fn clone_box(&self) -> Box<dyn AnyEstimator> {
        Box::new(self.clone())
    }
}

/// A fitted model that can predict.
#[typetag::serde(tag = "type")]
pub trait Predictor: Debug {
    /// Predictions as a matrix: one row per sample, one column per
    /// target. A classifier returns the predicted class in one column.
    fn predict(&self, data: &EncodedData) -> Result<Array2<f64>>;

    /// Predictions for a model with one target, as a plain list with one
    /// value per sample. Fails if the model predicts several targets.
    fn predict_single(&self, data: &EncodedData) -> Result<Vec<Label>> {
        single_column(self.predict(data)?)
    }

    /// Class probabilities as a matrix: one row per sample, one column
    /// per class. Classifiers only; regressors return `Error::InvalidSpec`.
    fn predict_proba(&self, data: &EncodedData) -> Result<Array2<f64>>;

    /// The default score: accuracy for classifiers, R² for regressors
    /// (averaged over the targets when there are several). Higher is better.
    fn score(&self, data: &EncodedData) -> Result<f64>;

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
#[typetag::serde(tag = "type")]
pub trait Transformer: Debug {
    fn fit(&mut self, table: &Table) -> Result<()>;

    fn transform(&self, table: &Table) -> Result<Table>;

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
pub trait Scorer {
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
