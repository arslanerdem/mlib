# mlib

Machine learning for tabular data in Rust: fast, and written to be read.

Add it to a project with `cargo add mlib`. Then a study is a handful of
steps:

```rust
use mlib::prelude::*;

fn main() -> Result<(), Error> {
    let table = Table::from_csv("data.csv")?;
    let spec = ModelSpec::new(&["age", "income", "city"], &["churned"]);
    let data = ModelData::new(&table, spec.clone())?;

    let pre = ColumnTransformer::empty()
        .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age", "income"])
        .with_branch("cat", OneHotEncoder::new(), &["city"]);
    let forest = RandomForestSpec::default().n_trees(200).seed(42).build()?;
    let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(forest));

    let folds = KFold::new(5).with_shuffle(42).splits(table.num_rows());
    let cv = cross_validate_pipeline(&pipe, &data, &folds, &*scorer("accuracy")?)?;
    println!("accuracy: {:.3} ± {:.3}", cv.mean, cv.std);

    let fitted = pipe.fit(&data)?;
    save(&fitted, &spec, "model.json")?;
    Ok(())
}
```

Gradient boosting is a drop-in replacement for the forest. It can hold
some rows out and stop when they stop improving:

```rust
let boosting = GradientBoostingSpec::default()
    .n_rounds(500)
    .learning_rate(0.05)
    .subsample(0.8)                    // each round sees a random 80% of the rows
    .feature_fraction(0.8)             // ... and a random 80% of the features
    .l2_regularization(1.0)            // small leaves predict smaller corrections
    .early_stopping_rounds(Some(20))   // stop 20 rounds after the best one
    .build()?;
let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(boosting));
```

A linear model is the plain baseline to compare them with — least
squares for regression, logistic regression for classification — and
its coefficients can be read off by name:

```rust
let baseline = LinearModelSpec::regressor().build()?.fit(&encoded)?;
for (feature, coefficient) in baseline.named_coefficients(0) {
    println!("{coefficient:+.4} × {feature}");
}
```

A saved model keeps working: a file saved by one 0.1.x release loads
and predicts the same in every later one. Code written against 0.1.0
keeps compiling with every later 0.1.x, too.

Every type and function in the [documentation](https://docs.rs/mlib)
comes with an example that is run as a test.

When one class is rare, make it count, and judge the model by how it
ranks rows, not by accuracy:

```rust
let boosting = GradientBoostingSpec::default().class_weight(ClassWeight::Balanced).build()?;
let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(boosting));
let folds = StratifiedKFold::new(5).with_shuffle(42).splits(&data.y()?, data.num_rows());
let cv = cross_validate_pipeline(&pipe, &data, &folds, &*scorer("roc_auc")?)?;
```

Settings can be searched at random instead of on a grid, and data in
time order has a split that never trains on the future:

```rust
let space = ParamDistributions::from([
    ("model__max_depth".to_string(), ParamDistribution::IntRange(2, 8)),
    ("model__learning_rate".to_string(), ParamDistribution::LogUniform(0.01, 0.3)),
]);
let folds = TimeSeriesSplit::new(5).splits(data.num_rows());
let found = random_search_pipeline(&pipe, &space, 30, 42, &data, &folds, &*scorer("roc_auc")?)?;
println!("best: {:?} ({:.3})", found.best_params, found.best_score);
```

Several numeric targets can be predicted together — name them all in the
`ModelSpec`, and `predict` returns one column per target:

```rust
let spec = ModelSpec::new(&["income", "rooms", "age"], &["latitude", "longitude"]);
let predictions = fitted.predict(&table)?;         // rows × 2 matrix
let latitude = predictions.column(0);
```

Classifiers take several targets the same way. `predict_proba` then
puts each target's class probabilities side by side, and
`target_classes()` says which columns belong to which target.

With a single target, `predict_single` returns a plain list. Matrices
are `ndarray` arrays; use `mlib::ndarray` rather than adding the crate
yourself, so the versions always match.

A fitted model can be read, not just used:

```rust
let tree = forest.tree(0);
println!("{} nodes, depth {}", tree.n_nodes(), tree.depth());
if let Node::Split { feature, threshold, .. } = tree.node(0) {
    println!("the first question: is feature {feature} <= {threshold}?");
}
```

## Try it

```sh
cargo run --release --example titanic     # a complete classification study
cargo run --release --example housing     # a complete regression study, baseline first
cargo run --release --example benchmark   # fitting speed on synthetic data
cargo test
```

## What is in this version

| Area | Contents |
|---|---|
| Data | `Table` (typed columns with nulls, CSV loading), `ModelSpec`, `ModelData`, `EncodedData` |
| Preprocessing | `Imputer`, `CategoricalImputer`, `StandardScaler`, `OneHotEncoder`, `Chain`, `ColumnTransformer` |
| Models | Random forest, gradient boosting and a linear model — classification and regression, with class weights |
| Evaluation | accuracy, balanced accuracy, precision, recall, F1 (also macro and weighted), ROC AUC, log loss, confusion matrix, MSE, RMSE, MAE, R² |
| Model selection | train/test split, k-fold, stratified and grouped k-fold, time-series split, cross-validation, grid search, random search |
| Persistence | `save` / `load` of a whole fitted pipeline as versioned JSON |

## Reading the code

Each file is one idea and opens with a comment that explains it.

```text
src/
  table.rs            Table, Column, Schema; CSV loading
  spec.rs             ModelSpec (which columns play which role), ModelData
  encoded.rs          EncodedData (the dense matrix models train on), Feature
  traits.rs           Estimator, Predictor, Transformer, Scorer
  preprocessing/      imputer, categorical_imputer, scaler, encoder, chain, column_transformer
  pipeline.rs         Pipeline, FittedPipeline
  forest.rs           random forest: the spec, fitting, prediction
  boosting.rs         gradient boosting: the spec, fitting, prediction
  linear.rs           the linear model: least squares and logistic regression
  minimise.rs         finding the lowest point of a function (used by logistic regression)
  scores.rs           from scores to predictions, shared by boosting and the linear model
  classes.rs          class labels per target, where they sit among the probabilities, class weights
  tree/               what the two tree models share
    mod.rs            Tree, TreeNode, prediction
    binning.rs        features → at most 256 bins each
    histogram.rs      split finding (the hot loop)
    grow.rs           growing one tree
  metrics.rs          metric functions and the scorer registry
  model_selection.rs  splits, cross-validation, grid search, random search
  persist.rs          save / load
  random.rs           the seedable random number generator
  testing.rs          check_estimator and check_transformer, for your own algorithms
```

## Speed

Measured against scikit-learn 1.9.1 on a 10-core machine, with 100 trees
or boosting rounds and the same data and settings for both. The numbers
are how many times faster mlib is; 1.0 means the same speed.

| Dataset (train rows × features) | Forest fit | Forest predict | Boosting fit | Boosting predict |
|---|---|---|---|---|
| California housing (16,512 × 8) | 4.3× | 4.0× | 7.7× | 8.2× |
| Friedman #1 (80,000 × 10) | 4.7× | 3.0× | 3.0× | 4.6× |
| Synthetic (80,000 × 20) | 13.0× | 2.5× | 2.7× | 4.8× |
| Covertype (464,809 × 54, 7 classes) | 2.8× | 1.6× | 1.6–2.2× | 5.0× |
| Synthetic (800,000 × 20) | 13.6× | 2.3× | 1.0–1.7× | 2.9× |

Forest scores are within 0.003 of scikit-learn's and boosting scores
within 0.001 or better. Boosting is compared with scikit-learn's
histogram gradient boosting. Where a range is shown, scikit-learn's
timing varied between runs; the low end is against its best run.
The linear model is not timed here; its coefficients are tested to
match scikit-learn's.

## Licence

MIT. See `LICENSE`.
