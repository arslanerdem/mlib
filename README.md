# mlib

Machine learning for tabular data in Rust: fast, and written to be read.

> **Early development.** This is version 0.0.3: it works and is tested,
> but the API will change before 0.1.0.

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
    .early_stopping_rounds(Some(20))   // stop 20 rounds after the best one
    .build()?;
let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(boosting));
```

A saved model keeps working: from 0.1.0 on, a file saved by one 0.1.x
release loads and predicts the same in every later one.

Several numeric targets can be predicted together — name them all in the
`ModelSpec`, and `predict` returns one column per target:

```rust
let spec = ModelSpec::new(&["income", "rooms", "age"], &["latitude", "longitude"]);
let predictions = fitted.predict(&table)?;         // rows × 2 matrix
let latitude = predictions.column(0);
```

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
cargo run --release --example titanic     # a complete study, start to finish
cargo run --release --example benchmark   # forest speed on synthetic data
cargo test
```

## What is in this version

| Area | Contents |
|---|---|
| Data | `Table` (typed columns with nulls, CSV loading), `ModelSpec`, `ModelData`, `EncodedData` |
| Preprocessing | `Imputer`, `StandardScaler`, `OneHotEncoder`, `Chain`, `ColumnTransformer` |
| Models | Random forest and gradient boosting — classification and regression |
| Evaluation | accuracy, F1, log loss, confusion matrix, MSE, RMSE, MAE, R² |
| Model selection | train/test split, k-fold, stratified and grouped k-fold, cross-validation, grid search |
| Persistence | `save` / `load` of a whole fitted pipeline as versioned JSON |

## Reading the code

Each file is one idea and opens with a comment that explains it.

```text
src/
  table.rs            Table, Column, Schema; CSV loading
  spec.rs             ModelSpec (which columns play which role), ModelData
  encoded.rs          EncodedData (the dense matrix models train on), Feature
  traits.rs           Estimator, Predictor, Transformer, Scorer
  preprocessing/      imputer, scaler, encoder, chain, column_transformer
  pipeline.rs         Pipeline, FittedPipeline
  forest.rs           random forest: the spec, fitting, prediction
  boosting.rs         gradient boosting: the spec, fitting, prediction
  tree/               what the two tree models share
    mod.rs            Tree, TreeNode, prediction
    binning.rs        features → at most 256 bins each
    histogram.rs      split finding (the hot loop)
    grow.rs           growing one tree
  metrics.rs          metric functions and the scorer registry
  model_selection.rs  splits, cross-validation, grid search
  persist.rs          save / load
  random.rs           the seedable random number generator
  testing.rs          check_estimator, for your own algorithms
```

## Speed

Measured against scikit-learn 1.9.1 on a 10-core machine, with 100 trees
or boosting rounds and the same data and settings for both. The numbers
are how many times faster mlib is; 1.0 means the same speed.

| Dataset (train rows × features) | Forest fit | Forest predict | Boosting fit | Boosting predict |
|---|---|---|---|---|
| California housing (16,512 × 8) | 4.2× | 4.8× | 7.9× | 7.6× |
| Friedman #1 (80,000 × 10) | 4.7× | 3.1× | 3.6× | 4.7× |
| Synthetic (80,000 × 20) | 10.1× | 2.4× | 2.4× | 3.9× |
| Covertype (464,809 × 54, 7 classes) | 3.0× | 1.8× | 1.6–2.2× | 6.6× |
| Synthetic (800,000 × 20) | 13.8× | 2.3× | 1.0–1.7× | 3.6× |

Forest scores are within 0.003 of scikit-learn's and boosting scores
within 0.001 or better. Boosting is compared with scikit-learn's
histogram gradient boosting. Where a range is shown, scikit-learn's
timing varied between runs; the low end is against its best run.

## Licence

MIT. See `LICENSE`.
