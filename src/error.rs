//! The one error type every fallible operation in the library returns.

use crate::table::DataType;
use std::fmt;

/// Shorthand for `std::result::Result<T, mlib::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A named column does not exist in the table's schema.
    ColumnNotFound(String),

    /// A column has the wrong type for the operation.
    TypeMismatch { column: String, expected: DataType, found: DataType },

    /// A spec failed validation (bad hyper-parameter or parameter name).
    InvalidSpec(String),

    /// The data itself cannot be used: nulls where values are required,
    /// non-finite numbers, a column with nothing to learn from.
    InvalidData(String),

    /// predict/transform called on an unfitted object.
    NotFitted,

    /// The data has a shape the operation cannot consume
    /// (zero rows, zero features, row counts that disagree).
    ShapeMismatch { expected: String, found: String },

    /// A saved artifact's format_version is unknown or unsupported.
    UnsupportedVersion { version: u32, max_supported: u32 },

    /// The input does not match what the model was fitted on.
    SchemaMismatch(String),

    /// I/O and serialisation errors, wrapped.
    Io(std::io::Error),
    /// A CSV file could not be read. The underlying error is kept as
    /// this error's `source()`.
    Csv(Box<dyn std::error::Error + Send + Sync>),
    /// A saved model could not be written or read.
    Json(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub(crate) fn shape(expected: impl Into<String>, found: impl Into<String>) -> Error {
        Error::ShapeMismatch { expected: expected.into(), found: found.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::ColumnNotFound(name) => write!(f, "column not found: '{name}'"),
            Error::TypeMismatch { column, expected, found } => {
                write!(f, "column '{column}' has type {found:?}, expected {expected:?}")
            }
            Error::InvalidSpec(message) => write!(f, "invalid spec: {message}"),
            Error::InvalidData(message) => write!(f, "invalid data: {message}"),
            Error::NotFitted => write!(f, "not fitted: call fit first"),
            Error::ShapeMismatch { expected, found } => {
                write!(f, "shape mismatch: expected {expected}, found {found}")
            }
            Error::UnsupportedVersion { version, max_supported } => write!(
                f,
                "unsupported model format version {version} (this library reads up to {max_supported})"
            ),
            Error::SchemaMismatch(message) => write!(f, "schema mismatch: {message}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Csv(e) => write!(f, "CSV error: {e}"),
            Error::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Csv(e) => Some(e.as_ref()),
            Error::Json(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

impl From<csv::Error> for Error {
    fn from(e: csv::Error) -> Error {
        Error::Csv(Box::new(e))
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Error {
        Error::Json(Box::new(e))
    }
}
