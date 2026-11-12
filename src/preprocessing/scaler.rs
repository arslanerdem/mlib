//! `StandardScaler`: centre and scale numeric columns to mean 0, std 1.

use super::{map_fitted_columns, numeric_columns};
use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::Table;
use crate::traits::Transformer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Applies `z = (x - mean) / std` to numeric (Float and Int) columns,
/// which come out as Float. A column with zero variance maps to 0.
/// Nulls stay null; other columns pass through unchanged.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Learned at fit: (mean, std) per numeric column, by name.
    pub stats: BTreeMap<String, (f64, f64)>,
    pub fitted: bool,
}

impl StandardScaler {
    pub fn new() -> StandardScaler {
        StandardScaler::default()
    }
}

#[typetag::serde]
impl Transformer for StandardScaler {
    fn fit(&mut self, table: &Table) -> Result<()> {
        let mut stats = BTreeMap::new();
        for (name, values) in numeric_columns(table) {
            let present: Vec<f64> = values.into_iter().flatten().collect();
            if present.is_empty() {
                return Err(Error::InvalidData(format!("column '{name}' has no values to scale")));
            }
            let n = present.len() as f64;
            let mean = present.iter().sum::<f64>() / n;
            let variance = present.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            stats.insert(name.to_string(), (mean, variance.sqrt()));
        }
        self.stats = stats;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        map_fitted_columns(table, &self.stats, |&(mean, std), value| {
            value.map(|x| if std > 0.0 { (x - mean) / std } else { 0.0 })
        })
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
}
