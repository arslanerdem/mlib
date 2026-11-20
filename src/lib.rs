//! # mlib
//!
//! Machine learning for tabular data, built to be fast and to read
//! plainly. A study is a handful of steps:
//!
//! ```
//! use mlib::prelude::*;
//!
//! fn main() -> Result<(), Error> {
//!     // 1. Load a table and say which columns play which role.
//!     // The data lives in the example, so it runs as written.
//!     let csv = "\
//! age,income,city,churned
//! 34,52000,Oslo,0
//! 45,88000,Rome,0
//! 23,31000,Paris,1
//! 52,95000,Oslo,0
//! 29,42000,Rome,1
//! 41,76000,Paris,0
//! 36,,Oslo,0
//! 27,38000,Paris,1
//! 48,91000,Rome,0
//! 31,47000,Oslo,1
//! 55,102000,Paris,0
//! 24,29000,Rome,1
//! 39,68000,Oslo,0
//! 43,81000,Paris,0
//! 26,35000,Oslo,1
//! 58,115000,Rome,0
//! 33,54000,Paris,0
//! ,61000,Oslo,0
//! 28,40000,Rome,1
//! 50,98000,Paris,0
//! ";
//!     let table = Table::from_csv_reader(csv.as_bytes())?;
//!     let spec = ModelSpec::new(&["age", "income", "city"], &["churned"]);
//!     let data = ModelData::new(&table, spec.clone())?;
//!
//!     // 2. Describe the preprocessing and the model.
//!     let pre = ColumnTransformer::empty()
//!         .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age", "income"])
//!         .with_branch("cat", OneHotEncoder::new(), &["city"]);
//!     let forest = RandomForestSpec::default().n_trees(200).seed(42).build()?;
//!     let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(forest));
//!
//!     // 3. Cross-validate, fit, save.
//!     let folds = KFold::new(5).with_shuffle(42).splits(table.num_rows());
//!     let cv = cross_validate_pipeline(&pipe, &data, &folds, &*scorer("accuracy")?)?;
//!     println!("accuracy: {:.3} ± {:.3}", cv.mean, cv.std);
//!
//!     let fitted = pipe.fit(&data)?;
//!     // The test saves to a temp file so it leaves the checkout clean.
//!     let path = std::env::temp_dir().join("mlib_quickstart_model.json");
//!     let path = path.to_str().expect("temp path is valid UTF-8");
//!     save(&fitted, &spec, path)?;
//!     Ok(())
//! }
//! ```
//!
//! ## How the pieces fit
//!
//! - [`Table`] holds raw, typed columns with nulls. It knows nothing about models.
//! - [`ModelSpec`] names the x and y columns; [`ModelData`] pairs it with a table.
//! - Transformers ([`Imputer`], [`CategoricalImputer`], [`OneHotEncoder`],
//!   [`StandardScaler`]) learn from the training table and replay the
//!   same steps at predict time.
//! - A [`Pipeline`] chains transformers and a model. Fitting it gives a
//!   [`FittedPipeline`] — a different type, so nothing unfitted can predict.
//! - Models train on [`EncodedData`]: a dense numeric matrix. There are
//!   three: a random forest ([`RandomForestSpec`]), gradient boosting
//!   ([`GradientBoostingSpec`]) and a linear model ([`LinearModelSpec`]),
//!   the plain baseline. All do classification and regression. The two
//!   tree models learn from missing values; the linear model needs them
//!   filled in first.

// Every public item carries a doc comment; CI fails on one without.
#![warn(missing_docs)]

mod boosting;
mod classes;
mod encoded;
mod error;
mod forest;
mod linear;
pub mod metrics;
mod minimise;
pub mod model_selection;
mod persist;
mod pipeline;
pub mod preprocessing;
pub mod random;
mod scores;
mod spec;
mod table;
pub mod testing;
mod traits;
mod tree;

/// The matrix library mlib's API uses (`Array2` for predictions and
/// encoded data). Use this re-export rather than adding `ndarray` to your
/// own project, so the versions always match.
pub use ndarray;

pub use boosting::{FittedGradientBoosting, GradientBoostingSpec, ValidGradientBoostingSpec};
pub use classes::ClassWeight;
pub use encoded::{EncodedData, Feature};
pub use error::{Error, Result};
pub use forest::{FittedRandomForest, MaxFeatures, RandomForestSpec, ValidRandomForestSpec};
pub use linear::{FittedLinearModel, LinearModelSpec, ValidLinearModelSpec};
pub use metrics::{
    accuracy, balanced_accuracy_score, confusion_matrix, f1_macro, f1_score, f1_weighted, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    root_mean_squared_error, scorer,
};
pub use model_selection::{
    cross_validate, cross_validate_pipeline, grid_search, grid_search_pipeline, random_search, random_search_pipeline,
    train_test_split, train_test_split_shuffled, CrossValidationResult, Folds, GridSearchResult, GroupKFold, KFold,
    ParamDistribution, ParamDistributions, ParamGrid, Params, StratifiedKFold, TimeSeriesSplit,
};
pub use persist::{load, save, SavedModel, FORMAT_VERSION};
pub use pipeline::{FittedPipeline, Pipeline};
pub use preprocessing::{
    CategoricalFill, CategoricalImputer, Chain, ColumnTransformer, HandleUnknown, ImputeStrategy, Imputer,
    OneHotEncoder, Remainder, StandardScaler,
};
pub use spec::{ModelData, ModelSpec};
pub use table::{Column, DataType, Schema, Table};
pub use traits::{AnyEstimator, Estimator, Label, ParamValue, Predictor, Scorer, Transformer};
pub use tree::{Node, Task, Tree};

/// Everything a study needs: `use mlib::prelude::*;`
///
/// `Result` is deliberately left out so that `Result<(), Error>` in your
/// own `main` keeps its usual meaning.
pub mod prelude {
    pub use crate::ndarray::Array2;
    pub use crate::{
        accuracy, balanced_accuracy_score, confusion_matrix, cross_validate, cross_validate_pipeline, f1_macro,
        f1_score, f1_weighted, grid_search, grid_search_pipeline, load, log_loss, mean_absolute_error,
        mean_squared_error, precision_score, r2_score, random_search, random_search_pipeline, recall_score,
        roc_auc_score, root_mean_squared_error, save, scorer, train_test_split, train_test_split_shuffled,
        AnyEstimator, CategoricalFill, CategoricalImputer, Chain, ClassWeight, Column, ColumnTransformer,
        CrossValidationResult, DataType, EncodedData, Error, Estimator, Feature, FittedGradientBoosting,
        FittedLinearModel, FittedPipeline, FittedRandomForest, Folds, GradientBoostingSpec, GridSearchResult,
        GroupKFold, HandleUnknown, ImputeStrategy, Imputer, KFold, Label, LinearModelSpec, MaxFeatures, ModelData,
        ModelSpec, Node, OneHotEncoder, ParamDistribution, ParamDistributions, ParamGrid, ParamValue, Params, Pipeline,
        Predictor, RandomForestSpec, Remainder, SavedModel, Scorer, StandardScaler, StratifiedKFold, Table, Task,
        TimeSeriesSplit, Transformer, Tree, ValidGradientBoostingSpec, ValidLinearModelSpec, ValidRandomForestSpec,
    };
}
