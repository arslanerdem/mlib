//! # mlib
//!
//! Machine learning for tabular data, built to be fast and to read
//! plainly. A study is a handful of steps:
//!
//! ```no_run
//! use mlib::prelude::*;
//!
//! fn main() -> Result<(), Error> {
//!     // 1. Load a table and say which columns play which role.
//!     let table = Table::from_csv("data.csv")?;
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
//!     save(&fitted, &spec, "model.json")?;
//!     Ok(())
//! }
//! ```
//!
//! ## How the pieces fit
//!
//! - [`Table`] holds raw, typed columns with nulls. It knows nothing about models.
//! - [`ModelSpec`] names the x and y columns; [`ModelData`] pairs it with a table.
//! - Transformers ([`Imputer`], [`OneHotEncoder`], [`StandardScaler`]) learn
//!   from the training table and replay the same steps at predict time.
//! - A [`Pipeline`] chains transformers and a model. Fitting it gives a
//!   [`FittedPipeline`] — a different type, so nothing unfitted can predict.
//! - Models train on [`EncodedData`]: a dense numeric matrix. There are
//!   two: a random forest ([`RandomForestSpec`]) and gradient boosting
//!   ([`GradientBoostingSpec`]). Both do classification and regression.

pub mod boosting;
pub mod encoded;
pub mod error;
pub mod forest;
pub mod metrics;
pub mod model_selection;
pub mod persist;
pub mod pipeline;
pub mod preprocessing;
pub mod random;
pub mod spec;
pub mod table;
pub mod testing;
pub mod traits;
pub mod tree;

pub use boosting::{FittedGradientBoosting, GradientBoostingSpec, ValidGradientBoostingSpec};
pub use encoded::{EncodedData, Feature};
pub use error::{Error, Result};
pub use forest::{FittedRandomForest, MaxFeatures, RandomForestSpec, ValidRandomForestSpec};
pub use metrics::{
    accuracy, confusion_matrix, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
    root_mean_squared_error, scorer,
};
pub use model_selection::{
    cross_validate, cross_validate_pipeline, grid_search, grid_search_pipeline, train_test_split,
    train_test_split_shuffled, CrossValidationResult, Folds, GridSearchResult, GroupKFold, KFold, ParamGrid,
    Params, StratifiedKFold,
};
pub use persist::{load, save, SavedModel, FORMAT_VERSION};
pub use pipeline::{FittedPipeline, Pipeline};
pub use preprocessing::{
    Chain, ColumnTransformer, HandleUnknown, ImputeStrategy, Imputer, OneHotEncoder, Remainder, StandardScaler,
};
pub use spec::{ModelData, ModelSpec};
pub use table::{Column, DataType, Schema, Table};
pub use traits::{AnyEstimator, Estimator, Label, ParamValue, Predictor, Scorer, Transformer};
pub use tree::{Task, Tree, TreeNode};

/// Everything a study needs: `use mlib::prelude::*;`
///
/// `Result` is deliberately left out so that `Result<(), Error>` in your
/// own `main` keeps its usual meaning.
pub mod prelude {
    pub use crate::{
        accuracy, confusion_matrix, cross_validate, cross_validate_pipeline, f1_score, grid_search,
        grid_search_pipeline, load, log_loss, mean_absolute_error, mean_squared_error, r2_score,
        root_mean_squared_error, save, scorer, train_test_split, train_test_split_shuffled, AnyEstimator, Chain,
        Column, ColumnTransformer, CrossValidationResult, DataType, EncodedData, Error, Estimator, Feature,
        FittedGradientBoosting, FittedPipeline, FittedRandomForest, Folds, GradientBoostingSpec, GridSearchResult, GroupKFold, HandleUnknown, ImputeStrategy, Imputer,
        KFold, Label, MaxFeatures, ModelData, ModelSpec, OneHotEncoder, ParamGrid, ParamValue, Params, Pipeline,
        Predictor, RandomForestSpec, Remainder, SavedModel, Scorer, StandardScaler, StratifiedKFold, Table, Task,
        Transformer, ValidGradientBoostingSpec, ValidRandomForestSpec,
    };
}
