//! Conformance batteries — call them from your own tests when you
//! implement [`Estimator`] for a new algorithm or [`Transformer`] for a
//! new preprocessing step.

use crate::encoded::{EncodedData, Feature};
use crate::table::{Column, Table};
use crate::traits::{Estimator, ParamValue, Predictor, Transformer};
use ndarray::Array2;

/// A small, learnable dataset: two features, and a 0/1 target that
/// depends on the first. Works for classifiers and regressors alike.
fn toy_data(n_rows: usize) -> EncodedData {
    let x = Array2::from_shape_fn((n_rows, 2), |(i, j)| if j == 0 { i as f64 } else { (i * 7 % 5) as f64 });
    let y = Array2::from_shape_fn((n_rows, 1), |(i, _)| if i < n_rows / 2 { 0.0 } else { 1.0 });
    EncodedData::new(x, y).expect("toy data is well-formed")
}

/// Check that an estimator honours the library's behaviour contracts.
/// Panics with a description of the first contract it breaks.
///
/// - Junk data (no rows, no features, mismatched y, NaN) → `Err`, never a panic.
/// - Fitting twice on the same data gives identical predictions.
/// - Predictions come one per row; the wrong number of features → `Err`.
///
/// ```
/// use mlib::prelude::*;
/// use mlib::testing::check_estimator;
///
/// check_estimator(&RandomForestSpec::default().n_trees(10).build()?);
/// check_estimator(&LinearModelSpec::regressor().build()?);
/// # Ok::<(), Error>(())
/// ```
///
/// The example on [`Estimator`] runs it on a hand-written model.
pub fn check_estimator<E: Estimator>(spec: &E) {
    let data = toy_data(40);

    let no_rows = data.take_rows(&[]);
    assert!(spec.fit(&no_rows).is_err(), "fit must fail on data with no rows");

    let no_features = EncodedData::new(Array2::zeros((40, 0)), data.y.clone()).unwrap();
    assert!(spec.fit(&no_features).is_err(), "fit must fail on data with no features");

    let mut short_y = data.clone();
    short_y.y = Array2::zeros((39, 1));
    assert!(spec.fit(&short_y).is_err(), "fit must fail when x and y have different row counts");

    let mut with_nan = data.clone();
    with_nan.x[[3, 0]] = f64::NAN;
    assert!(spec.fit(&with_nan).is_err(), "fit must fail on NaN in x");

    let first = spec.fit(&data).expect("fit must succeed on clean data");
    let second = spec.fit(&data).expect("fit must succeed a second time");
    let predictions = first.predict(&data).expect("predict must succeed on the training data");
    assert_eq!(predictions.dim(), data.y.dim(), "predict must return one row per sample, one column per target");
    assert_eq!(predictions, second.predict(&data).unwrap(), "two fits on the same data must agree");

    let one_feature = EncodedData::new(Array2::zeros((5, 1)), Array2::zeros((5, 1))).unwrap();
    assert!(first.predict(&one_feature).is_err(), "predict must fail on the wrong number of features");

    let score = first.score(&data).expect("score must succeed on the training data");
    assert!(score.is_finite(), "score must be a finite number");
}

/// A small table with one column of each common kind and no missing values.
fn toy_table() -> Table {
    let columns = vec![
        ("height", Column::floats(&[1.5, 1.7, 1.6, 1.9, 1.8, 1.7])),
        ("rooms", Column::ints(&[1, 3, 2, 3, 1, 2])),
        ("city", Column::texts(&["Oslo", "Rome", "Oslo", "Paris", "Rome", "Oslo"])),
    ];
    Table::new(columns).expect("toy table is well-formed")
}

/// Check that a preprocessing step honours the library's behaviour
/// contracts. Pass it unfitted. Panics with a description of the first
/// contract it breaks.
///
/// - Before `fit`, `transform` and `output_columns` → `Err`.
/// - `transform` keeps every row, gives the same answer each time, and
///   does not depend on which rows it is given alongside.
/// - `output_columns` names exactly the columns `transform` produces.
/// - `fit_transform` equals `fit` then `transform`; a clone and a saved
///   and reloaded copy transform identically.
/// - A table without the fitted columns → `Err`; an unknown parameter
///   name → `Err`; a missing value → a result or an `Err`, never a panic.
///
/// The step is fitted on a table with a Float, an Int and a Text column.
///
/// Here it checks a hand-written step that takes the logarithm of every
/// decimal column (your project needs `serde` and `typetag` among its
/// dependencies):
///
/// ```
/// use mlib::{prelude::*, Result};
/// use serde::{Deserialize, Serialize};
///
/// /// Replaces x by ln(1 + x) in the Float columns seen at fit time.
/// #[derive(Debug, Clone, Default, Serialize, Deserialize)]
/// struct LogTransform {
///     /// The Float columns of the training table. `None` before `fit`.
///     columns: Option<Vec<String>>,
/// }
///
/// #[typetag::serde]
/// impl Transformer for LogTransform {
///     fn fit(&mut self, table: &Table) -> Result<()> {
///         let floats = table.named_columns().filter(|(_, column)| column.data_type() == DataType::Float);
///         self.columns = Some(floats.map(|(name, _)| name.to_string()).collect());
///         Ok(())
///     }
///
///     fn transform(&self, table: &Table) -> Result<Table> {
///         let fitted = self.columns.as_ref().ok_or(Error::NotFitted)?;
///         for name in fitted {
///             table.require_column(name)?;
///         }
///         let log = |value: &Option<f64>| value.map(f64::ln_1p);
///         let columns = table.named_columns().map(|(name, column)| match column {
///             Column::Float(values) if fitted.iter().any(|fitted| fitted == name) => {
///                 (name, Column::Float(values.iter().map(log).collect()))
///             }
///             other => (name, other.clone()),
///         });
///         Table::new(columns.collect())
///     }
///
///     fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>> {
///         self.columns.as_ref().ok_or(Error::NotFitted)?;
///         Ok(input.to_vec())
///     }
///
///     fn clone_box(&self) -> Box<dyn Transformer> {
///         Box::new(self.clone())
///     }
/// }
///
/// mlib::testing::check_transformer(&LogTransform::default());
/// ```
pub fn check_transformer<T: Transformer + Clone>(step: &T) {
    let table = toy_table();
    let input: Vec<Feature> = table.schema().names().into_iter().map(Feature::numeric).collect();

    assert!(step.transform(&table).is_err(), "transform must fail before fit");
    assert!(step.output_columns(&input).is_err(), "output_columns must fail before fit");

    let mut fitted = step.clone();
    fitted.fit(&table).expect("fit must succeed on clean data");
    let out = fitted.transform(&table).expect("transform must succeed on the training data");
    assert_eq!(out.num_rows(), table.num_rows(), "transform must keep every row");
    assert_eq!(out, fitted.transform(&table).unwrap(), "transforming twice must give the same table");

    let features = fitted.output_columns(&input).expect("output_columns must succeed after fit");
    let feature_names: Vec<String> = features.iter().map(Feature::display_name).collect();
    assert_eq!(feature_names, out.schema().names(), "output_columns must name the columns transform produces");

    // A row is transformed from fitted state alone, whatever else is in the table.
    let two_rows = fitted.transform(&table.take_rows(&[4, 1])).expect("transform must succeed on a few rows");
    assert_eq!(two_rows, out.take_rows(&[4, 1]), "a row's result must not depend on the other rows");

    let mut in_one_step = step.clone();
    assert_eq!(in_one_step.fit_transform(&table).unwrap(), out, "fit_transform must equal fit, then transform");
    assert_eq!(fitted.clone_box().transform(&table).unwrap(), out, "a clone must transform identically");

    let boxed: Box<dyn Transformer> = Box::new(fitted.clone());
    let saved = serde_json::to_string(&boxed).expect("a fitted step must be saveable");
    let loaded: Box<dyn Transformer> = serde_json::from_str(&saved).expect("a saved step must load");
    assert_eq!(loaded.transform(&table).unwrap(), out, "a saved and loaded step must transform identically");

    let other_columns = Table::new(vec![("unrelated", Column::floats(&[1.0, 2.0]))]).unwrap();
    assert!(fitted.transform(&other_columns).is_err(), "transform must fail when the fitted columns are missing");
    assert!(fitted.set_param("no_such_parameter", &ParamValue::Int(1)).is_err(), "an unknown parameter must fail");

    // Missing values may be filled, passed on or refused — but not by panicking.
    let with_missing = Table::new(vec![
        ("height", Column::Float(vec![None, Some(1.6)])),
        ("rooms", Column::Int(vec![Some(2), None])),
        ("city", Column::Text(vec![None, Some("Oslo".to_string())])),
    ]);
    let _ = fitted.transform(&with_missing.unwrap());
}
