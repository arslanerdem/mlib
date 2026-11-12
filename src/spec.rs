//! `ModelSpec` says which table columns are inputs, targets, and roles.
//! `ModelData` is a model's view of a table: the table plus the spec.

use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::{Column, DataType, Table};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which columns are x's, y's, and roles — by name. One table can feed
/// many specs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelSpec {
    pub x_columns: Vec<String>,
    pub y_columns: Vec<String>,
    /// Per-row sample weight.
    pub weight_column: Option<String>,
    /// Row identifier. Never a feature.
    pub id_column: Option<String>,
    /// Group id, for grouped cross-validation.
    pub group_column: Option<String>,
}

impl ModelSpec {
    pub fn new(x: &[&str], y: &[&str]) -> ModelSpec {
        ModelSpec {
            x_columns: x.iter().map(|name| name.to_string()).collect(),
            y_columns: y.iter().map(|name| name.to_string()).collect(),
            weight_column: None,
            id_column: None,
            group_column: None,
        }
    }

    pub fn with_weights(mut self, column: &str) -> ModelSpec {
        self.weight_column = Some(column.to_string());
        self
    }

    pub fn with_id(mut self, column: &str) -> ModelSpec {
        self.id_column = Some(column.to_string());
        self
    }

    pub fn with_groups(mut self, column: &str) -> ModelSpec {
        self.group_column = Some(column.to_string());
        self
    }
}

/// A view, not a copy: borrows the table and carries the spec.
#[derive(Debug)]
pub struct ModelData<'a> {
    pub(crate) table: &'a Table,
    pub(crate) spec: ModelSpec,
}

impl<'a> ModelData<'a> {
    /// Build a view. Fails with `ColumnNotFound` if a named column is
    /// absent, or `TypeMismatch` if a target is Text or a weight is not
    /// numeric. X columns may have any type — the pipeline encodes them.
    pub fn new(table: &'a Table, spec: ModelSpec) -> Result<ModelData<'a>> {
        let roles = [&spec.weight_column, &spec.id_column, &spec.group_column];
        let role_names = roles.into_iter().flatten();
        for name in spec.x_columns.iter().chain(&spec.y_columns).chain(role_names) {
            table.require_column(name)?;
        }
        for name in &spec.y_columns {
            if table.schema.get(name) == Some(DataType::Text) {
                return Err(Error::TypeMismatch {
                    column: format!("{name} (a target must be numeric or categorical)"),
                    expected: DataType::Categorical,
                    found: DataType::Text,
                });
            }
        }
        if let Some(name) = &spec.weight_column {
            let found = table.require_column(name)?.data_type();
            if !matches!(found, DataType::Float | DataType::Int) {
                return Err(Error::TypeMismatch { column: name.clone(), expected: DataType::Float, found });
            }
        }
        Ok(ModelData { table, spec })
    }

    /// A column the constructor already checked exists.
    fn checked_column(&self, name: &str) -> &Column {
        self.table.column(name).expect("ModelData::new verified this column exists")
    }

    pub fn num_rows(&self) -> usize {
        self.table.num_rows()
    }

    /// The table this view reads from.
    pub fn table(&self) -> &'a Table {
        self.table
    }

    /// Which columns play which role.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// The x columns as their own table, in spec order.
    pub fn x_table(&self) -> Table {
        let names: Vec<&str> = self.spec.x_columns.iter().map(String::as_str).collect();
        self.table.select(&names).expect("ModelData::new verified these columns exist")
    }

    /// All targets as a (n_samples, n_targets) matrix. Categorical targets
    /// give their dictionary code. A missing target is an error.
    pub fn y_matrix(&self) -> Result<Array2<f64>> {
        let (n_rows, n_targets) = (self.num_rows(), self.spec.y_columns.len());
        let mut y = Array2::zeros((n_rows, n_targets));
        for (j, name) in self.spec.y_columns.iter().enumerate() {
            let column = self.checked_column(name);
            for i in 0..n_rows {
                y[[i, j]] = column.get_f64(i).ok_or_else(|| {
                    Error::InvalidData(format!("target '{name}' is missing in row {i} — drop the row"))
                })?;
            }
        }
        Ok(y)
    }

    /// The first (usually the only) target, one value per row.
    pub fn y(&self) -> Result<Vec<f64>> {
        if self.spec.y_columns.is_empty() {
            return Err(Error::shape("at least one target column", "no targets"));
        }
        Ok(self.y_matrix()?.column(0).to_vec())
    }

    /// Per-row sample weights, if a weight column is set. A missing
    /// weight counts as zero.
    pub fn weights(&self) -> Option<Vec<f64>> {
        let column = self.checked_column(self.spec.weight_column.as_ref()?);
        Some((0..self.num_rows()).map(|i| column.get_f64(i).unwrap_or(0.0)).collect())
    }

    /// One group id per row, from the group column, for `GroupKFold`.
    /// Rows with the same value (of any type) share an id.
    pub fn groups(&self) -> Result<Vec<i64>> {
        let name = self.spec.group_column.as_ref().ok_or_else(|| {
            Error::InvalidSpec("no group column set — use ModelSpec::with_groups".to_string())
        })?;
        let column = self.checked_column(name);
        let mut ids: HashMap<Option<String>, i64> = HashMap::new();
        Ok((0..self.num_rows())
            .map(|i| {
                let next_id = ids.len() as i64;
                *ids.entry(column.get_string(i)).or_insert(next_id)
            })
            .collect())
    }

    /// The input features: one per x column, in spec order.
    pub fn input_features(&self) -> Vec<Feature> {
        self.spec.x_columns.iter().map(|name| Feature::numeric(name)).collect()
    }

    /// Row indices split into (train, test) by ratio, in row order.
    pub fn train_test_split(&self, train_ratio: f64) -> (Vec<usize>, Vec<usize>) {
        crate::model_selection::train_test_split(self.num_rows(), train_ratio)
    }
}
