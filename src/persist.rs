//! Saving and loading fitted pipelines.
//!
//! The saved file is one self-describing JSON object: the fitted steps,
//! the fitted model, the column spec, the input schema and the versions —
//! never a bare weight blob. Loading runs no code from the file.

use crate::error::{Error, Result};
use crate::pipeline::FittedPipeline;
use crate::spec::ModelSpec;
use crate::table::Schema;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};

/// Bumped on any breaking change to the saved layout.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct SavedModel {
    pub format_version: u32,
    pub library_version: String,
    /// Which columns were x's, y's and roles.
    pub spec: ModelSpec,
    /// The input (x) schema at fit time.
    pub schema: Schema,
    /// The fitted transformers and the fitted model.
    pub fitted: FittedPipeline,
}

/// `SavedModel` with borrowed fields, so saving does not copy the model.
/// The field names and order must match `SavedModel`.
#[derive(Serialize)]
struct SavedModelRef<'a> {
    format_version: u32,
    library_version: &'a str,
    spec: &'a ModelSpec,
    schema: &'a Schema,
    fitted: &'a FittedPipeline,
}

/// Just the version field, readable from any format version.
#[derive(Deserialize)]
struct VersionHeader {
    format_version: u32,
}

/// Save a fitted pipeline to a JSON file.
pub fn save(pipeline: &FittedPipeline, spec: &ModelSpec, path: &str) -> Result<()> {
    let saved = SavedModelRef {
        format_version: FORMAT_VERSION,
        library_version: env!("CARGO_PKG_VERSION"),
        spec,
        schema: pipeline.schema(),
        fitted: pipeline,
    };
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer(&mut writer, &saved)?;
    writer.flush()?;
    Ok(())
}

/// Load a saved pipeline. A file with an unknown `format_version` is
/// rejected with `Error::UnsupportedVersion`.
pub fn load(path: &str) -> Result<SavedModel> {
    let bytes = std::fs::read(path)?;
    // Check the version first: a newer layout may not even parse as ours.
    let header: VersionHeader = serde_json::from_slice(&bytes)?;
    if header.format_version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            version: header.format_version,
            max_supported: FORMAT_VERSION,
        });
    }
    Ok(serde_json::from_slice(&bytes)?)
}
