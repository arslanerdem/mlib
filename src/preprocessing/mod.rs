//! Preprocessing steps: fitted objects that travel with the model.
//!
//! Each step learns its statistics from the training table at `fit` and
//! re-applies them verbatim at `transform` — the defence against
//! train/predict skew.

mod chain;
mod column_transformer;
mod encoder;
mod imputer;
mod scaler;

pub use chain::Chain;
pub(crate) use chain::set_param_on_first;
pub use column_transformer::{ColumnTransformer, Remainder};
pub use encoder::{HandleUnknown, OneHotEncoder};
pub use imputer::{ImputeStrategy, Imputer};
pub use scaler::StandardScaler;

use crate::error::{Error, Result};
use crate::table::{Column, DataType, Table};
use std::collections::BTreeMap;

/// The numeric (Float and Int) columns of a table, as (name, values).
fn numeric_columns(table: &Table) -> impl Iterator<Item = (&str, Vec<Option<f64>>)> {
    table.named_columns().filter_map(|(name, column)| Some((name, column.to_floats()?)))
}

/// Rebuild a table, passing every value of each fitted numeric column
/// through `apply` (with that column's fitted state). Fitted columns come
/// out as Float; all other columns are passed through unchanged.
///
/// Every fitted column must be present and numeric.
fn map_fitted_columns<S>(
    table: &Table,
    fitted: &BTreeMap<String, S>,
    apply: impl Fn(&S, Option<f64>) -> Option<f64>,
) -> Result<Table> {
    if let Some(missing) = fitted.keys().find(|name| table.column(name).is_none()) {
        return Err(Error::SchemaMismatch(format!("column '{missing}' was present at fit time")));
    }
    let mut out = table.clone();
    for (column, (name, _)) in out.columns.iter_mut().zip(&table.schema.fields) {
        let Some(state) = fitted.get(name) else { continue };
        let values = column.to_floats().ok_or_else(|| Error::TypeMismatch {
            column: name.clone(),
            expected: DataType::Float,
            found: column.data_type(),
        })?;
        *column = Column::Float(values.into_iter().map(|value| apply(state, value)).collect());
    }
    for (field, column) in out.schema.fields.iter_mut().zip(&out.columns) {
        field.1 = column.data_type();
    }
    Ok(out)
}
