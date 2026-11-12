//! `Table`: the boundary type. A schema plus equal-length typed columns.
//!
//! A table is a plain data container. It knows nothing about features,
//! targets, or any algorithm — a [`ModelSpec`](crate::ModelSpec) says which
//! columns play which role. Rows are positional; there is no row index.

use crate::error::{Error, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DataType {
    Float,
    Int,
    Bool,
    Text,
    Categorical,
}

/// Column names and types, in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    pub(crate) fields: Vec<(String, DataType)>,
}

impl Schema {
    /// Index of a column by name. `None` if absent.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|(field, _)| field == name)
    }

    /// Type of a column by name. `None` if absent.
    pub fn get(&self, name: &str) -> Option<DataType> {
        self.index_of(name).map(|i| self.fields[i].1)
    }

    /// Every column's name and type, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, DataType)> {
        self.fields.iter().map(|(name, data_type)| (name.as_str(), *data_type))
    }

    /// Names of all columns, in order.
    pub fn names(&self) -> Vec<&str> {
        self.fields.iter().map(|(name, _)| name.as_str()).collect()
    }
}

/// One typed column. A missing value is `None` — null is a property of
/// the slot, never a sentinel like `NaN` or `"NA"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Column {
    Float(Vec<Option<f64>>),
    Int(Vec<Option<i64>>),
    Bool(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
    /// Dictionary-encoded text: `codes[i]` indexes into `categories`.
    Categorical { codes: Vec<Option<u32>>, categories: Vec<String> },
}

impl Column {
    /// A float column with no nulls.
    pub fn floats(values: &[f64]) -> Column {
        Column::Float(values.iter().map(|&v| Some(v)).collect())
    }

    /// An integer column with no nulls.
    pub fn ints(values: &[i64]) -> Column {
        Column::Int(values.iter().map(|&v| Some(v)).collect())
    }

    /// A text column with no nulls.
    pub fn texts(values: &[&str]) -> Column {
        Column::Text(values.iter().map(|v| Some(v.to_string())).collect())
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Int(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Text(v) => v.len(),
            Column::Categorical { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Column::Float(_) => DataType::Float,
            Column::Int(_) => DataType::Int,
            Column::Bool(_) => DataType::Bool,
            Column::Text(_) => DataType::Text,
            Column::Categorical { .. } => DataType::Categorical,
        }
    }

    /// Whether row `i` is missing.
    pub fn is_null(&self, i: usize) -> bool {
        match self {
            Column::Float(v) => v[i].is_none(),
            Column::Int(v) => v[i].is_none(),
            Column::Bool(v) => v[i].is_none(),
            Column::Text(v) => v[i].is_none(),
            Column::Categorical { codes, .. } => codes[i].is_none(),
        }
    }

    /// Number of null slots.
    pub fn null_count(&self) -> usize {
        fn nulls<T>(values: &[Option<T>]) -> usize {
            values.iter().filter(|v| v.is_none()).count()
        }
        match self {
            Column::Float(v) => nulls(v),
            Column::Int(v) => nulls(v),
            Column::Bool(v) => nulls(v),
            Column::Text(v) => nulls(v),
            Column::Categorical { codes, .. } => nulls(codes),
        }
    }

    /// Row `i` as a number: floats and ints as themselves, bools as 0/1,
    /// categoricals as their dictionary code. `None` for nulls and text.
    pub fn get_f64(&self, i: usize) -> Option<f64> {
        match self {
            Column::Float(v) => v[i],
            Column::Int(v) => v[i].map(|x| x as f64),
            Column::Bool(v) => v[i].map(|x| if x { 1.0 } else { 0.0 }),
            Column::Text(_) => None,
            Column::Categorical { codes, .. } => codes[i].map(|code| code as f64),
        }
    }

    /// Row `i` as display text. `None` for nulls.
    pub fn get_string(&self, i: usize) -> Option<String> {
        match self {
            Column::Float(v) => v[i].map(|x| x.to_string()),
            Column::Int(v) => v[i].map(|x| x.to_string()),
            Column::Bool(v) => v[i].map(|x| x.to_string()),
            Column::Text(v) => v[i].clone(),
            Column::Categorical { codes, categories } => {
                codes[i].map(|code| categories[code as usize].clone())
            }
        }
    }

    /// The values of a numeric (Float or Int) column as floats.
    /// `None` if the column is not numeric.
    pub fn to_floats(&self) -> Option<Vec<Option<f64>>> {
        match self {
            Column::Float(v) => Some(v.clone()),
            Column::Int(v) => Some(v.iter().map(|x| x.map(|x| x as f64)).collect()),
            _ => None,
        }
    }

    /// The rows at the given indices, in the given order.
    fn take(&self, indices: &[usize]) -> Column {
        fn pick<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| values[i].clone()).collect()
        }
        match self {
            Column::Float(v) => Column::Float(pick(v, indices)),
            Column::Int(v) => Column::Int(pick(v, indices)),
            Column::Bool(v) => Column::Bool(pick(v, indices)),
            Column::Text(v) => Column::Text(pick(v, indices)),
            Column::Categorical { codes, categories } => {
                Column::Categorical { codes: pick(codes, indices), categories: categories.clone() }
            }
        }
    }

    /// Dictionary-encode this column: the categories are its distinct
    /// values, sorted.
    fn to_categorical(&self) -> Column {
        let values: Vec<Option<String>> = (0..self.len()).map(|i| self.get_string(i)).collect();
        let distinct: BTreeSet<&String> = values.iter().flatten().collect();
        let categories: Vec<String> = distinct.into_iter().cloned().collect();
        let codes = values
            .iter()
            .map(|value| {
                // Invariant: every value was put into `categories` above.
                value.as_ref().map(|v| categories.binary_search(v).unwrap() as u32)
            })
            .collect();
        Column::Categorical { codes, categories }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Table {
    pub(crate) schema: Schema,
    pub(crate) columns: Vec<Column>,
}

impl Table {
    /// Build a table from named columns. All columns must have the same
    /// length and distinct names.
    pub fn new(columns: Vec<(&str, Column)>) -> Result<Table> {
        let (names, columns) = columns.into_iter().map(|(name, col)| (name.to_string(), col)).unzip();
        Table::from_parts(names, columns)
    }

    pub(crate) fn from_parts(names: Vec<String>, columns: Vec<Column>) -> Result<Table> {
        let mut seen = HashSet::new();
        if let Some(duplicate) = names.iter().find(|name| !seen.insert(name.as_str())) {
            return Err(Error::InvalidData(format!("duplicate column name '{duplicate}'")));
        }
        let num_rows = columns.first().map_or(0, Column::len);
        if let Some(i) = columns.iter().position(|column| column.len() != num_rows) {
            return Err(Error::shape(
                format!("{num_rows} rows in every column"),
                format!("{} rows in column '{}'", columns[i].len(), names[i]),
            ));
        }
        let fields = names.into_iter().zip(columns.iter().map(Column::data_type)).collect();
        Ok(Table { schema: Schema { fields }, columns })
    }

    /// Load from a CSV file. The first row is the header.
    ///
    /// Each column gets the narrowest type that fits every one of its
    /// values, tried in this order: Int, Float, Bool, Text. Empty cells are
    /// null. Text columns stay Text — use [`Table::with_categorical`] to
    /// dictionary-encode them.
    pub fn from_csv(path: &str) -> Result<Table> {
        Table::from_csv_reader(std::fs::File::open(path)?)
    }

    /// Load from any CSV reader (for tests and in-memory data).
    pub fn from_csv_reader<R: std::io::Read>(reader: R) -> Result<Table> {
        let mut reader = csv::ReaderBuilder::new().trim(csv::Trim::All).from_reader(reader);
        let names: Vec<String> = reader.headers()?.iter().map(str::to_string).collect();

        // Read the text column-wise, then parse every column in parallel.
        let mut raw_columns: Vec<Vec<String>> = vec![Vec::new(); names.len()];
        let mut record = csv::StringRecord::new();
        while reader.read_record(&mut record)? {
            for (raw, field) in raw_columns.iter_mut().zip(record.iter()) {
                raw.push(field.to_string());
            }
        }
        let columns = raw_columns.par_iter().map(|raw| parse_column(raw)).collect();
        Table::from_parts(names, columns)
    }

    /// The column names and types, in order.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Number of rows. All columns have this length.
    pub fn num_rows(&self) -> usize {
        self.columns.first().map_or(0, Column::len)
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// A column by name. `None` if absent.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.schema.index_of(name).map(|i| &self.columns[i])
    }

    /// A column by name, or `Error::ColumnNotFound`.
    pub fn require_column(&self, name: &str) -> Result<&Column> {
        self.column(name).ok_or_else(|| Error::ColumnNotFound(name.to_string()))
    }

    /// A column by position. Panics if `i` is out of range.
    pub fn column_at(&self, i: usize) -> &Column {
        &self.columns[i]
    }

    /// Column names paired with their columns, in order.
    pub fn named_columns(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.schema.fields.iter().map(|(name, _)| name.as_str()).zip(&self.columns)
    }

    /// A new table with only the named columns, in the given order.
    pub fn select(&self, names: &[&str]) -> Result<Table> {
        let columns = names
            .iter()
            .map(|name| self.require_column(name).cloned())
            .collect::<Result<Vec<Column>>>()?;
        Table::from_parts(names.iter().map(|name| name.to_string()).collect(), columns)
    }

    /// A new table of the rows at the given indices, in the given order.
    /// Panics if an index is out of range. Splits use this; the table
    /// itself is never reordered.
    pub fn take_rows(&self, indices: &[usize]) -> Table {
        let columns = self.columns.par_iter().map(|column| column.take(indices)).collect();
        Table { schema: self.schema.clone(), columns }
    }

    /// A new table without the rows that are null in any of the named columns.
    pub fn drop_nulls(&self, names: &[&str]) -> Result<Table> {
        let checked = names.iter().map(|name| self.require_column(name)).collect::<Result<Vec<_>>>()?;
        let keep: Vec<usize> =
            (0..self.num_rows()).filter(|&i| checked.iter().all(|column| !column.is_null(i))).collect();
        Ok(self.take_rows(&keep))
    }

    /// A new table with the named columns dictionary-encoded as
    /// `Categorical`. The categories are each column's distinct values,
    /// sorted. This is the explicit override CSV loading never does on its own.
    pub fn with_categorical(&self, names: &[&str]) -> Result<Table> {
        let mut table = self.clone();
        for name in names {
            let i = self.schema.index_of(name).ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
            table.columns[i] = self.columns[i].to_categorical();
            table.schema.fields[i].1 = DataType::Categorical;
        }
        Ok(table)
    }
}

/// Parse one CSV column into the narrowest type that fits all its values.
fn parse_column(raw: &[String]) -> Column {
    /// Parse every non-empty cell, or give up (`None`) at the first that fails.
    fn parse_all<T>(raw: &[String], parse: impl Fn(&str) -> Option<T>) -> Option<Vec<Option<T>>> {
        raw.iter().map(|cell| if cell.is_empty() { Some(None) } else { parse(cell).map(Some) }).collect()
    }
    fn parse_bool(cell: &str) -> Option<bool> {
        match cell.to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }

    let has_values = raw.iter().any(|cell| !cell.is_empty());
    if !has_values {
        return Column::Float(vec![None; raw.len()]);
    }
    if let Some(values) = parse_all(raw, |cell| cell.parse::<i64>().ok()) {
        return Column::Int(values);
    }
    if let Some(values) = parse_all(raw, |cell| cell.parse::<f64>().ok()) {
        // A literal "NaN" cell is a missing value, not a number.
        return Column::Float(values.into_iter().map(|v| v.filter(|x| !x.is_nan())).collect());
    }
    if let Some(values) = parse_all(raw, parse_bool) {
        return Column::Bool(values);
    }
    Column::Text(raw.iter().map(|cell| if cell.is_empty() { None } else { Some(cell.clone()) }).collect())
}

/// Prints the shape and the first rows, for a quick look at the data.
impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const HEAD: usize = 5;
        const MAX_WIDTH: usize = 16;
        let clip = |text: String| -> String {
            if text.chars().count() > MAX_WIDTH {
                text.chars().take(MAX_WIDTH - 1).chain(std::iter::once('…')).collect()
            } else {
                text
            }
        };
        let shown_rows = self.num_rows().min(HEAD);
        writeln!(f, "Table: {} rows × {} columns", self.num_rows(), self.num_columns())?;

        // One vector of cell texts per column: name, type, then values.
        let cells: Vec<Vec<String>> = self
            .named_columns()
            .map(|(name, column)| {
                let mut texts = vec![clip(name.to_string()), format!("{:?}", column.data_type())];
                texts.extend((0..shown_rows).map(|i| clip(column.get_string(i).unwrap_or_default())));
                texts
            })
            .collect();
        for line in 0..shown_rows + 2 {
            for texts in &cells {
                let width = texts.iter().map(|text| text.chars().count()).max().unwrap_or(0);
                write!(f, "{:<width$}  ", texts[line])?;
            }
            writeln!(f)?;
        }
        if self.num_rows() > shown_rows {
            writeln!(f, "… {} more rows", self.num_rows() - shown_rows)?;
        }
        Ok(())
    }
}
