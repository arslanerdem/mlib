//! `Imputer`: fill missing numeric values with a statistic learned at fit.

use super::{map_fitted_columns, numeric_columns};
use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::Table;
use crate::traits::{ParamValue, Transformer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What an [`Imputer`] puts in place of a missing number.
///
/// ```
/// use mlib::prelude::*;
///
/// let table = Table::new(vec![("age", Column::Float(vec![Some(20.0), Some(20.0), Some(50.0), None]))])?;
/// let fill = |strategy| -> Result<Option<f64>, Error> {
///     Ok(Imputer::new(strategy).fit_transform(&table)?.column_at(0).get_f64(3))
/// };
/// assert_eq!(fill(ImputeStrategy::Mean)?, Some(30.0));
/// assert_eq!(fill(ImputeStrategy::Median)?, Some(20.0));
/// assert_eq!(fill(ImputeStrategy::MostFrequent)?, Some(20.0));
/// assert_eq!(fill(ImputeStrategy::Constant(-1.0))?, Some(-1.0));
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImputeStrategy {
    /// The column's average in the training data.
    Mean,
    /// The column's middle value in the training data. Not pulled about
    /// by a few extreme values, as the mean is.
    Median,
    /// The column's most common value in the training data. Ties go to
    /// the smallest.
    MostFrequent,
    /// The same number for every column.
    Constant(f64),
}

impl ImputeStrategy {
    /// The fill value for a column, from its non-missing values.
    /// `None` when there is nothing to learn from.
    fn fill_value(&self, mut values: Vec<f64>) -> Option<f64> {
        if let ImputeStrategy::Constant(value) = self {
            return Some(*value);
        }
        if values.is_empty() {
            return None;
        }
        values.sort_unstable_by(f64::total_cmp);
        Some(match self {
            ImputeStrategy::Mean => values.iter().sum::<f64>() / values.len() as f64,
            ImputeStrategy::Median => {
                let mid = values.len() / 2;
                if values.len() % 2 == 1 {
                    values[mid]
                } else {
                    (values[mid - 1] + values[mid]) / 2.0
                }
            }
            // The longest run of equal values; ties go to the smallest value.
            _ => {
                let (mut best, mut best_count, mut run_start) = (values[0], 0, 0);
                for i in 0..values.len() {
                    if values[i] != values[run_start] {
                        run_start = i;
                    }
                    if i - run_start + 1 > best_count {
                        (best, best_count) = (values[i], i - run_start + 1);
                    }
                }
                best
            }
        })
    }
}

/// Fills nulls in numeric (Float and Int) columns. Numeric columns come
/// out as Float; other columns pass through unchanged. Impute *before*
/// encoding.
///
/// ```
/// use mlib::prelude::*;
///
/// let train = Table::new(vec![
///     ("age", Column::Float(vec![Some(30.0), None, Some(50.0)])),
///     ("city", Column::Text(vec![Some("Oslo".to_string()), None, None])),
/// ])?;
/// let mut imputer = Imputer::new(ImputeStrategy::Mean);
/// let filled = imputer.fit_transform(&train)?;
/// assert_eq!(filled.column("age").unwrap().null_count(), 0);
/// assert_eq!(filled.column("city").unwrap().null_count(), 2); // text is left alone
/// assert_eq!(imputer.fill_values()["age"], 40.0);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Imputer {
    strategy: ImputeStrategy,
    /// Learned at fit: one fill value per numeric column, by name.
    fill_values: BTreeMap<String, f64>,
    fitted: bool,
}

impl Imputer {
    /// An unfitted imputer with the given strategy.
    pub fn new(strategy: ImputeStrategy) -> Imputer {
        Imputer { strategy, fill_values: BTreeMap::new(), fitted: false }
    }

    /// The value learnt for each column at fit time. Empty before `fit`.
    pub fn fill_values(&self) -> &BTreeMap<String, f64> {
        &self.fill_values
    }
}

#[typetag::serde]
impl Transformer for Imputer {
    fn fit(&mut self, table: &Table) -> Result<()> {
        let mut fill_values = BTreeMap::new();
        for (name, values) in numeric_columns(table) {
            let present = values.into_iter().flatten().collect();
            let fill = self
                .strategy
                .fill_value(present)
                .ok_or_else(|| Error::InvalidData(format!("column '{name}' has no values to impute from")))?;
            fill_values.insert(name.to_string(), fill);
        }
        self.fill_values = fill_values;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        map_fitted_columns(table, &self.fill_values, |&fill, value| Some(value.unwrap_or(fill)))
    }

    fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        Ok(input.to_vec())
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }

    /// `strategy`: "mean", "median", "most_frequent", or a number (constant fill).
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        if name != "strategy" {
            return Err(Error::InvalidSpec(format!("unknown parameter: {name}")));
        }
        self.strategy = match value {
            ParamValue::Text(text) if text == "mean" => ImputeStrategy::Mean,
            ParamValue::Text(text) if text == "median" => ImputeStrategy::Median,
            ParamValue::Text(text) if text == "most_frequent" => ImputeStrategy::MostFrequent,
            ParamValue::Float(constant) => ImputeStrategy::Constant(*constant),
            ParamValue::Int(constant) => ImputeStrategy::Constant(*constant as f64),
            other => return Err(Error::InvalidSpec(format!("invalid imputer strategy: {other:?}"))),
        };
        Ok(())
    }
}
