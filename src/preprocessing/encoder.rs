//! `OneHotEncoder`: expand a column into one 0/1 column per category.

use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::{Column, Table};
use crate::traits::Transformer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// What to do with a category at transform time that was not seen at fit.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum HandleUnknown {
    /// Encode as all-zeros across the column's one-hot features.
    #[default]
    Ignore,
    /// Fail with `Error::InvalidData`.
    Error,
}

/// One-hot encodes **every column it is given**, except Float columns,
/// which pass through. Text, Categorical, Bool and Int columns are all
/// treated as categories — route columns with a `ColumnTransformer` to
/// choose which ones are encoded.
///
/// The category list is learned at fit and never re-derived: an unseen
/// category (or a null) at transform time encodes as all-zeros, so the
/// output columns are always the same.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OneHotEncoder {
    pub handle_unknown: HandleUnknown,
    /// Learned at fit: the sorted category list per encoded column, by name.
    pub categories: BTreeMap<String, Vec<String>>,
    pub fitted: bool,
}

impl OneHotEncoder {
    pub fn new() -> OneHotEncoder {
        OneHotEncoder::default()
    }

    pub fn with_handle_unknown(mut self, handle_unknown: HandleUnknown) -> OneHotEncoder {
        self.handle_unknown = handle_unknown;
        self
    }
}

#[typetag::serde]
impl Transformer for OneHotEncoder {
    fn fit(&mut self, table: &Table) -> Result<()> {
        let mut categories = BTreeMap::new();
        for (name, column) in table.named_columns() {
            let Some(distinct) = distinct_values(column) else { continue };
            if distinct.is_empty() {
                return Err(Error::InvalidData(format!("column '{name}' has no values to encode")));
            }
            categories.insert(name.to_string(), distinct);
        }
        self.categories = categories;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        if let Some(missing) = self.categories.keys().find(|name| table.column(name).is_none()) {
            return Err(Error::SchemaMismatch(format!("column '{missing}' was present at fit time")));
        }
        let (mut names, mut columns) = (Vec::new(), Vec::new());
        for (name, column) in table.named_columns() {
            let Some(categories) = self.categories.get(name) else {
                names.push(name.to_string());
                columns.push(column.clone());
                continue;
            };
            let indices = category_indices(column, categories);
            let unknown = indices.iter().filter(|index| index.is_none()).count() - column.null_count();
            if unknown > 0 && self.handle_unknown == HandleUnknown::Error {
                return Err(Error::InvalidData(format!(
                    "column '{name}' has {unknown} value(s) in categories not seen at fit time"
                )));
            }
            for (k, category) in categories.iter().enumerate() {
                let hot = |index: &Option<usize>| Some(if *index == Some(k) { 1.0 } else { 0.0 });
                names.push(Feature::one_hot(name, category).display_name());
                columns.push(Column::Float(indices.iter().map(hot).collect()));
            }
        }
        Table::from_parts(names, columns)
    }

    fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        let mut output = Vec::new();
        for feature in input {
            match self.categories.get(&feature.display_name()) {
                Some(categories) => {
                    output.extend(categories.iter().map(|c| Feature::one_hot(&feature.source, c)))
                }
                None => output.push(feature.clone()),
            }
        }
        Ok(output)
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }
}

/// The distinct non-null values of a column as sorted category names
/// (ints sort numerically). `None` for Float columns, which are not encoded.
fn distinct_values(column: &Column) -> Option<Vec<String>> {
    fn sorted<T: Ord + ToString>(values: impl Iterator<Item = T>) -> Vec<String> {
        values.collect::<BTreeSet<T>>().iter().map(T::to_string).collect()
    }
    Some(match column {
        Column::Float(_) => return None,
        Column::Int(values) => sorted(values.iter().flatten()),
        Column::Bool(values) => sorted(values.iter().flatten()),
        Column::Text(values) => sorted(values.iter().flatten()),
        Column::Categorical { codes, categories } => {
            sorted(codes.iter().flatten().map(|&code| &categories[code as usize]))
        }
    })
}

/// For each row, the position of its value in the fitted category list.
/// `None` for nulls and for values that are not in the list.
fn category_indices(column: &Column, categories: &[String]) -> Vec<Option<usize>> {
    let position: HashMap<&str, usize> =
        categories.iter().enumerate().map(|(k, category)| (category.as_str(), k)).collect();
    match column {
        Column::Float(values) => vec![None; values.len()],
        Column::Int(values) => {
            // Compare as integers: far cheaper than formatting every row.
            let by_value: HashMap<i64, usize> =
                position.iter().filter_map(|(text, &k)| Some((text.parse().ok()?, k))).collect();
            values.iter().map(|value| by_value.get(value.as_ref()?).copied()).collect()
        }
        Column::Bool(values) => {
            let lookup = |value: &Option<bool>| position.get(value.as_ref()?.to_string().as_str()).copied();
            values.iter().map(lookup).collect()
        }
        Column::Text(values) => {
            values.iter().map(|value| position.get(value.as_ref()?.as_str()).copied()).collect()
        }
        Column::Categorical { codes, categories: dictionary } => {
            // Translate the dictionary once, then each row is a table lookup.
            let translated: Vec<Option<usize>> =
                dictionary.iter().map(|entry| position.get(entry.as_str()).copied()).collect();
            codes.iter().map(|code| translated[(*code)? as usize]).collect()
        }
    }
}
