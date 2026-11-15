//! `CategoricalImputer`: fill missing categories with a value learned at fit.
//!
//! The counterpart of `Imputer` for columns that hold labels rather than
//! numbers. A missing label is replaced either by the column's most
//! frequent one or by a label of your choice, such as "missing" — which
//! lets a model learn that "not recorded" itself carries information.

use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::{Column, DataType, Table};
use crate::traits::{ParamValue, Transformer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What a `CategoricalImputer` puts in place of a missing value.
///
/// ```
/// use mlib::prelude::*;
///
/// let most_frequent = CategoricalImputer::new(CategoricalFill::MostFrequent);
/// let constant = CategoricalImputer::new(CategoricalFill::Constant("missing".to_string()));
/// // The same two, by their shortcuts.
/// assert_eq!(most_frequent, CategoricalImputer::most_frequent());
/// assert_eq!(constant, CategoricalImputer::constant("missing"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CategoricalFill {
    /// The column's most frequent value in the training data. Ties go to
    /// the value that sorts first.
    MostFrequent,
    /// The same label for every column, whether or not the training data
    /// contains it.
    Constant(String),
}

/// Fills nulls in Text, Categorical and Bool columns, which keep their
/// type. Numeric columns pass through unchanged — use `Imputer` for
/// those. Impute *before* encoding.
///
/// ```
/// use mlib::prelude::*;
///
/// let city = |values: &[Option<&str>]| Column::Text(values.iter().map(|value| value.map(str::to_string)).collect());
/// let train = Table::new(vec![("city", city(&[Some("Oslo"), Some("Rome"), Some("Oslo"), None]))])?;
///
/// let mut imputer = CategoricalImputer::most_frequent();
/// let filled = imputer.fit_transform(&train)?;
/// assert_eq!(filled.column_at(0).get_string(3), Some("Oslo".to_string()));
/// assert_eq!(imputer.fill_values()["city"], "Oslo");
///
/// // Or let "not recorded" be a category of its own.
/// let filled = CategoricalImputer::constant("missing").fit_transform(&train)?;
/// assert_eq!(filled.column_at(0).get_string(3), Some("missing".to_string()));
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalImputer {
    fill: CategoricalFill,
    /// Learned at fit: one fill value per column, by name. A Bool
    /// column's value is "true" or "false".
    fill_values: BTreeMap<String, String>,
    fitted: bool,
}

impl CategoricalImputer {
    /// An unfitted imputer that fills with the given choice.
    pub fn new(fill: CategoricalFill) -> CategoricalImputer {
        CategoricalImputer { fill, fill_values: BTreeMap::new(), fitted: false }
    }

    /// Fill with each column's most frequent value.
    pub fn most_frequent() -> CategoricalImputer {
        CategoricalImputer::new(CategoricalFill::MostFrequent)
    }

    /// Fill every column with the given label.
    pub fn constant(label: &str) -> CategoricalImputer {
        CategoricalImputer::new(CategoricalFill::Constant(label.to_string()))
    }

    /// The value learnt for each column at fit time. Empty before `fit`.
    pub fn fill_values(&self) -> &BTreeMap<String, String> {
        &self.fill_values
    }

    /// The fill value for one column, from the training data.
    fn fill_value(&self, name: &str, column: &Column) -> Result<String> {
        let fill = match &self.fill {
            CategoricalFill::Constant(label) => label.clone(),
            CategoricalFill::MostFrequent => {
                let mut counts: BTreeMap<String, usize> = BTreeMap::new();
                for value in (0..column.len()).filter_map(|i| column.get_string(i)) {
                    *counts.entry(value).or_default() += 1;
                }
                // The values come in sorted order, and a later one must be
                // strictly more frequent to win: ties go to the first.
                let mut most_frequent: Option<(String, usize)> = None;
                for (value, count) in counts {
                    if most_frequent.as_ref().is_none_or(|(_, most)| count > *most) {
                        most_frequent = Some((value, count));
                    }
                }
                let (value, _) = most_frequent
                    .ok_or_else(|| Error::InvalidData(format!("column '{name}' has no values to impute from")))?;
                value
            }
        };
        if column.data_type() == DataType::Bool && fill != "true" && fill != "false" {
            return Err(Error::InvalidData(format!(
                "column '{name}' holds true/false values and cannot be filled with '{fill}'"
            )));
        }
        Ok(fill)
    }
}

/// Whether the imputer fills columns of this type.
fn is_filled(data_type: DataType) -> bool {
    matches!(data_type, DataType::Text | DataType::Categorical | DataType::Bool)
}

/// A copy of the column with every null replaced by `fill`. `None` if the
/// column is numeric.
fn filled(column: &Column, fill: &str) -> Option<Column> {
    Some(match column {
        Column::Text(values) => {
            Column::Text(values.iter().map(|value| value.clone().or_else(|| Some(fill.to_string()))).collect())
        }
        Column::Bool(values) => Column::Bool(values.iter().map(|value| value.or(Some(fill == "true"))).collect()),
        Column::Categorical { codes, categories } => {
            // A label the column has not seen becomes a new category at the end.
            let mut categories = categories.clone();
            let code = categories.iter().position(|category| category == fill).unwrap_or_else(|| {
                categories.push(fill.to_string());
                categories.len() - 1
            }) as u32;
            Column::Categorical { codes: codes.iter().map(|value| value.or(Some(code))).collect(), categories }
        }
        _ => return None,
    })
}

#[typetag::serde]
impl Transformer for CategoricalImputer {
    fn fit(&mut self, table: &Table) -> Result<()> {
        let mut fill_values = BTreeMap::new();
        for (name, column) in table.named_columns().filter(|(_, column)| is_filled(column.data_type())) {
            fill_values.insert(name.to_string(), self.fill_value(name, column)?);
        }
        self.fill_values = fill_values;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        if let Some(missing) = self.fill_values.keys().find(|name| table.column(name).is_none()) {
            return Err(Error::SchemaMismatch(format!("column '{missing}' was present at fit time")));
        }
        let (mut names, mut columns) = (Vec::new(), Vec::new());
        for (name, column) in table.named_columns() {
            names.push(name.to_string());
            columns.push(match self.fill_values.get(name) {
                None => column.clone(),
                Some(fill) => filled(column, fill).ok_or_else(|| Error::TypeMismatch {
                    column: name.to_string(),
                    expected: DataType::Text,
                    found: column.data_type(),
                })?,
            });
        }
        Table::from_parts(names, columns)
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

    /// `strategy`: "most_frequent". `fill_value`: the label to fill with.
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        self.fill = match (name, value) {
            ("strategy", ParamValue::Text(text)) if text == "most_frequent" => CategoricalFill::MostFrequent,
            ("fill_value", ParamValue::Text(label)) => CategoricalFill::Constant(label.clone()),
            ("strategy" | "fill_value", other) => {
                return Err(Error::InvalidSpec(format!("invalid value for {name}: {other:?}")))
            }
            _ => return Err(Error::InvalidSpec(format!("unknown parameter: {name}"))),
        };
        Ok(())
    }
}
