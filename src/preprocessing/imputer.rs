//! `Imputer`: fill missing numeric values with a statistic learned at fit.

use super::{map_fitted_columns, numeric_columns};
use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::Table;
use crate::traits::{ParamValue, Transformer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImputeStrategy {
    Mean,
    Median,
    MostFrequent,
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
                if values.len() % 2 == 1 { values[mid] } else { (values[mid - 1] + values[mid]) / 2.0 }
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Imputer {
    strategy: ImputeStrategy,
    /// Learned at fit: one fill value per numeric column, by name.
    fill_values: BTreeMap<String, f64>,
    fitted: bool,
}

impl Imputer {
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
            let fill = self.strategy.fill_value(present).ok_or_else(|| {
                Error::InvalidData(format!("column '{name}' has no values to impute from"))
            })?;
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
