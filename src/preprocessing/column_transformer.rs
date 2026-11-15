//! `ColumnTransformer`: route column subsets to different preprocessing.

use crate::encoded::Feature;
use crate::error::{Error, Result};
use crate::table::Table;
use crate::traits::{ParamValue, Transformer};
use serde::{Deserialize, Serialize};

/// What happens to columns no branch asked for.
///
/// ```
/// use mlib::prelude::*;
///
/// let table = Table::new(vec![("age", Column::floats(&[30.0, 50.0])), ("visits", Column::ints(&[2, 7]))])?;
/// let scale_age = || ColumnTransformer::empty().with_branch("num", StandardScaler::new(), &["age"]);
///
/// let dropped = scale_age().fit_transform(&table)?;
/// assert_eq!(dropped.schema().names(), ["age"]);
///
/// let kept = scale_age().with_remainder(Remainder::Passthrough).fit_transform(&table)?;
/// assert_eq!(kept.schema().names(), ["age", "visits"]);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Remainder {
    /// Append them unchanged, in table order.
    Passthrough,
    /// Leave them out.
    #[default]
    Drop,
}

/// Sends named column subsets through different branches and joins the
/// results side by side, in branch order.
///
/// ```
/// use mlib::prelude::*;
///
/// let table = Table::new(vec![
///     ("age", Column::Float(vec![Some(30.0), None, Some(50.0)])),
///     ("city", Column::texts(&["Oslo", "Rome", "Oslo"])),
/// ])?;
/// let mut pre = ColumnTransformer::empty()
///     .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age"])
///     .with_branch("cat", OneHotEncoder::new(), &["city"]);
/// let out = pre.fit_transform(&table)?;
/// assert_eq!(out.schema().names(), ["age", "city=Oslo", "city=Rome"]);
/// assert_eq!(out.column("age").unwrap().get_f64(1), Some(40.0));
///
/// // A branch's settings are reached by its name.
/// pre.set_param("num__strategy", &"mean".into())?;
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct ColumnTransformer {
    /// (branch name, step, the columns routed to it)
    branches: Vec<(String, Box<dyn Transformer>, Vec<String>)>,
    remainder: Remainder,
}

impl ColumnTransformer {
    /// A transformer with no branches yet; add them with `with_branch`.
    pub fn empty() -> ColumnTransformer {
        ColumnTransformer { branches: Vec::new(), remainder: Remainder::Drop }
    }

    /// Add a branch: send the named columns through `step`.
    pub fn with_branch(mut self, name: &str, step: impl Transformer + 'static, columns: &[&str]) -> ColumnTransformer {
        let columns = columns.iter().map(|column| column.to_string()).collect();
        self.branches.push((name.to_string(), Box::new(step), columns));
        self
    }

    /// Choose what happens to columns no branch asked for (dropped unless
    /// you say otherwise).
    pub fn with_remainder(mut self, remainder: Remainder) -> ColumnTransformer {
        self.remainder = remainder;
        self
    }

    fn is_routed(&self, name: &str) -> bool {
        self.branches.iter().any(|(_, _, columns)| columns.iter().any(|column| column == name))
    }

    /// Join the branches' outputs side by side, then any passthrough columns.
    fn join(&self, table: &Table, outputs: Vec<Table>) -> Result<Table> {
        let (mut names, mut columns) = (Vec::new(), Vec::new());
        for output in outputs {
            names.extend(output.schema.fields.into_iter().map(|(name, _)| name));
            columns.extend(output.columns);
        }
        if self.remainder == Remainder::Passthrough {
            for (name, column) in table.named_columns().filter(|(name, _)| !self.is_routed(name)) {
                names.push(name.to_string());
                columns.push(column.clone());
            }
        }
        Table::from_parts(names, columns)
    }
}

/// The branch's own columns as a table. A missing column is a schema mismatch.
fn branch_input(table: &Table, columns: &[String]) -> Result<Table> {
    let names: Vec<&str> = columns.iter().map(String::as_str).collect();
    table.select(&names).map_err(|error| match error {
        Error::ColumnNotFound(name) => Error::SchemaMismatch(format!("missing column '{name}'")),
        other => other,
    })
}

impl Clone for ColumnTransformer {
    fn clone(&self) -> ColumnTransformer {
        let clone_branch = |(name, step, columns): &(String, Box<dyn Transformer>, Vec<String>)| {
            (name.clone(), step.clone_box(), columns.clone())
        };
        ColumnTransformer { branches: self.branches.iter().map(clone_branch).collect(), remainder: self.remainder }
    }
}

#[typetag::serde]
impl Transformer for ColumnTransformer {
    fn fit(&mut self, table: &Table) -> Result<()> {
        self.fit_transform(table).map(|_| ())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        let outputs = self
            .branches
            .iter()
            .map(|(_, step, columns)| step.transform(&branch_input(table, columns)?))
            .collect::<Result<Vec<Table>>>()?;
        self.join(table, outputs)
    }

    fn fit_transform(&mut self, table: &Table) -> Result<Table> {
        let outputs = self
            .branches
            .iter_mut()
            .map(|(_, step, columns)| step.fit_transform(&branch_input(table, columns)?))
            .collect::<Result<Vec<Table>>>()?;
        self.join(table, outputs)
    }

    fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>> {
        let mut output = Vec::new();
        for (_, step, columns) in &self.branches {
            let routed = columns
                .iter()
                .map(|column| {
                    input
                        .iter()
                        .find(|feature| &feature.display_name() == column)
                        .cloned()
                        .ok_or_else(|| Error::SchemaMismatch(format!("missing column '{column}'")))
                })
                .collect::<Result<Vec<Feature>>>()?;
            output.extend(step.output_columns(&routed)?);
        }
        if self.remainder == Remainder::Passthrough {
            output.extend(input.iter().filter(|feature| !self.is_routed(&feature.display_name())).cloned());
        }
        Ok(output)
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }

    /// Routes `branch__param` to the named branch (e.g. `num__strategy`).
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        let unknown = || Error::InvalidSpec(format!("unknown parameter: {name}"));
        let (branch_name, param) = name.split_once("__").ok_or_else(unknown)?;
        let (_, step, _) = self.branches.iter_mut().find(|(name, _, _)| name == branch_name).ok_or_else(unknown)?;
        step.set_param(param, value)
    }
}
