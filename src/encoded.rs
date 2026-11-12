//! `EncodedData`: the dense numeric view a model actually trains on.
//!
//! There are two column worlds. The *input schema* is the raw table's
//! columns (`sex`, `age`). The *model-context schema* is the columns after
//! encoding (`sex=female`, `sex=male`, `age`). The model only ever sees the
//! second, and [`Feature`] is the identity of one such column.

use crate::error::{Error, Result};
use crate::table::{DataType, Table};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// The identity of one model-context feature. Structured, not a string:
/// a name like `a_b_c` cannot say whether it is column `a_b` category `c`
/// or column `a` category `b_c`. Display names are derived, never parsed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Feature {
    /// Input column this feature came from.
    pub source: String,
    /// Category, for one-hot expansions. `None` for numeric features.
    pub category: Option<String>,
}

impl Feature {
    pub fn numeric(source: &str) -> Feature {
        Feature { source: source.to_string(), category: None }
    }

    pub fn one_hot(source: &str, category: &str) -> Feature {
        Feature { source: source.to_string(), category: Some(category.to_string()) }
    }

    /// Display name for humans: "age", or "sex=female" for one-hot.
    pub fn display_name(&self) -> String {
        match &self.category {
            Some(category) => format!("{}={}", self.source, category),
            None => self.source.clone(),
        }
    }
}

/// Pair feature names with importances, most important first.
pub(crate) fn rank_importances(features: &[Feature], importances: &[f64]) -> Vec<(String, f64)> {
    let mut ranked: Vec<(String, f64)> =
        features.iter().map(Feature::display_name).zip(importances.iter().copied()).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedData {
    /// (n_samples, n_features). Dense, no nulls.
    pub x: Array2<f64>,
    /// (n_samples, n_targets). Zero columns when there are no targets
    /// (prediction time).
    pub y: Array2<f64>,
    /// Per-row sample weights, if any.
    pub weights: Option<Vec<f64>>,
    /// The model-context features, aligned with the columns of `x`.
    pub features: Vec<Feature>,
}

impl EncodedData {
    /// Wrap matrices you built yourself. Features are named `x0`, `x1`, ...
    pub fn new(x: Array2<f64>, y: Array2<f64>) -> Result<EncodedData> {
        if x.nrows() != y.nrows() {
            return Err(Error::shape(format!("{} rows in y", x.nrows()), format!("{} rows", y.nrows())));
        }
        let features = (0..x.ncols()).map(|j| Feature::numeric(&format!("x{j}"))).collect();
        Ok(EncodedData { x, y, weights: None, features })
    }

    /// Build the dense view from a fully preprocessed table: every column
    /// must be numeric (Float, Int or Bool) with no nulls, and the columns
    /// must be exactly the given features, in order.
    pub fn from_table(
        table: &Table,
        features: Vec<Feature>,
        y: Array2<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<EncodedData> {
        let expected: Vec<String> = features.iter().map(Feature::display_name).collect();
        if table.schema.names() != expected {
            return Err(Error::SchemaMismatch(format!(
                "encoded columns {:?} do not match the expected features {:?}",
                table.schema.names(),
                expected
            )));
        }
        let (n_rows, n_columns) = (table.num_rows(), table.num_columns());
        let mut values = vec![0.0; n_rows * n_columns];
        for (j, (name, column)) in table.named_columns().enumerate() {
            if matches!(column.data_type(), DataType::Text | DataType::Categorical) {
                return Err(Error::TypeMismatch {
                    column: format!("{name} (not numeric — add a OneHotEncoder for it)"),
                    expected: DataType::Float,
                    found: column.data_type(),
                });
            }
            for i in 0..n_rows {
                values[i * n_columns + j] = column.get_f64(i).ok_or_else(|| {
                    Error::InvalidData(format!(
                        "column '{name}' has a missing value in row {i} — add an Imputer or drop the row"
                    ))
                })?;
            }
        }
        // Invariant: `values` was allocated with exactly n_rows * n_columns slots.
        let x = Array2::from_shape_vec((n_rows, n_columns), values).unwrap();
        if y.nrows() != n_rows {
            return Err(Error::shape(format!("{n_rows} rows in y"), format!("{} rows", y.nrows())));
        }
        Ok(EncodedData { x, y, weights, features })
    }

    pub fn num_rows(&self) -> usize {
        self.x.nrows()
    }

    pub fn num_features(&self) -> usize {
        self.x.ncols()
    }

    /// Display names for the features, in order (e.g. "age", "sex=female").
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(Feature::display_name).collect()
    }

    /// The target as a plain list, one value per row. Fails unless
    /// there is exactly one target; with several, read `y` directly.
    pub fn target(&self) -> Result<Vec<f64>> {
        if self.y.ncols() != 1 {
            return Err(Error::shape("one target column", format!("{} targets", self.y.ncols())));
        }
        Ok(self.y.column(0).to_vec())
    }

    /// The rows at the given indices (for CV folds). Panics if an index
    /// is out of range.
    pub fn take_rows(&self, indices: &[usize]) -> EncodedData {
        EncodedData {
            x: self.x.select(Axis(0), indices),
            y: self.y.select(Axis(0), indices),
            weights: self.weights.as_ref().map(|w| indices.iter().map(|&i| w[i]).collect()),
            features: self.features.clone(),
        }
    }
}
