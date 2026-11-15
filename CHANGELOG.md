# Changelog

What changed in each release, written for someone using the library.
The record starts at 0.1.0; versions 0.0.1 to 0.0.3 were early
snapshots published while the design settled.

## 0.1.0 — 2026-11-15

The first release with promises. From here on:

- **The public surface is stable within 0.1.x.** Code written against
  0.1.0 keeps compiling with every later 0.1.x.
- **Saved models keep loading.** A file saved by any 0.1.x release loads
  and predicts the same in every later 0.1.x.
- **Results are reproducible.** The same data, settings and seed give a
  byte-identical saved model, on any number of cores.

What the release contains:

- **Data**: `Table` (read from CSV), `ModelSpec` and `ModelData` to name
  features, targets, weights and groups.
- **Preprocessing**: `Imputer`, `CategoricalImputer`, `StandardScaler`,
  `OneHotEncoder`, combined with `Chain` and `ColumnTransformer`.
- **Models**: random forest, gradient boosting and a linear model, for
  classification and regression. Both take several targets at once.
  Boosting has row and feature subsampling, L2 regularisation and early
  stopping on held-out rows.
- **Pipelines**: `Pipeline` fits the preprocessing and the model
  together, so nothing leaks from the rows held out.
- **Model selection**: train/test split, `KFold`, `StratifiedKFold`,
  `GroupKFold`, cross-validation and grid search.
- **Metrics** for classification and regression, by name through
  `scorer`.
- **Saving and loading** fitted pipelines as JSON.
- **Reading a fitted model**: every tree can be walked node by node.

New since 0.0.3:

- **Linear model** (`LinearModelSpec`): least squares for regression,
  logistic regression for classification, with optional L2
  regularisation. Coefficients are reported in the features' own units
  and by name. It is the plain baseline to compare the tree models with.
- **Classifiers take several targets** (multi-label): name several
  class columns as targets and `predict` returns one column of labels
  per target. `predict_proba` puts each target's class probabilities
  side by side, and `target_classes()` says which columns are which.
- **`CategoricalImputer`**: fills missing labels in text, categorical
  and true/false columns with the most frequent value or a label of
  your choice, such as "missing".
- **Gradient boosting** gained `feature_fraction` (each round uses a
  random part of the features) and `l2_regularization` (leaves with
  little data predict smaller corrections). Both are off by default, so
  existing settings give the same model as before.
- **Gradient boosting's feature importances changed**: each split now
  counts by how much it improved the fit, so the early trees, which do
  most of the work, count for most. Before, every tree counted the
  same, which made the importances too even. Predictions are unaffected.
- **`testing::check_transformer`**: the conformance checks for your own
  preprocessing steps, as `check_estimator` is for models.
- A second complete example, `examples/housing.rs`: a regression study
  that starts from the linear baseline.
- **`spec.set_param(..)` now compiles with the prelude imported.** It
  was ambiguous between two traits; the internal one's method is now
  `AnyEstimator::set_param_boxed`.
- **Documentation**: every public item is documented, and every type
  and function has an example that is run as a test.
- The code is formatted with `rustfmt`, and every change is tested on
  Linux and macOS and on the minimum supported Rust version, 1.85.
