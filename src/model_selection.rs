//! Model selection: splits, cross-validation, grid search.
//!
//! Splits are lists of row indices — the table itself is never reordered.
//! When a pipeline does any preprocessing, use the `*_pipeline` variants:
//! they refit every step on each fold's training rows, so no statistic
//! from a fold's test rows leaks into its training.

use crate::encoded::EncodedData;
use crate::error::{Error, Result};
use crate::metrics;
use crate::pipeline::Pipeline;
use crate::random::Rng;
use crate::spec::ModelData;
use crate::traits::{Estimator, Predictor, Scorer};
use std::collections::{BTreeMap, HashMap};

pub use crate::traits::ParamValue;

/// One (train_indices, test_indices) pair per fold.
pub type Folds = Vec<(Vec<usize>, Vec<usize>)>;

/// Split row indices into (train, test) by ratio, in row order.
pub fn train_test_split(n_rows: usize, train_ratio: f64) -> (Vec<usize>, Vec<usize>) {
    let n_train = ((n_rows as f64 * train_ratio.clamp(0.0, 1.0)).round() as usize).min(n_rows);
    ((0..n_train).collect(), (n_train..n_rows).collect())
}

/// Like `train_test_split`, but rows are assigned at random (by seed).
pub fn train_test_split_shuffled(n_rows: usize, train_ratio: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..n_rows).collect();
    Rng::new(seed, 0).shuffle(&mut indices);
    let (train, test) = train_test_split(n_rows, train_ratio);
    (train.iter().map(|&i| indices[i]).collect(), test.iter().map(|&i| indices[i]).collect())
}

/// Turn "row i belongs to fold f" into (train, test) index lists.
fn folds_from_assignment(fold_of_row: &[usize], n_splits: usize) -> Folds {
    (0..n_splits)
        .map(|fold| {
            let rows = 0..fold_of_row.len();
            let train = rows.clone().filter(|&row| fold_of_row[row] != fold).collect();
            let test = rows.filter(|&row| fold_of_row[row] == fold).collect();
            (train, test)
        })
        .collect()
}

/// The row order to deal folds from: as-is, or shuffled by seed.
fn row_order(n_rows: usize, shuffle: bool, seed: Option<u64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_rows).collect();
    if shuffle {
        Rng::new(seed.unwrap_or(0), 0).shuffle(&mut order);
    }
    order
}

/// K-fold: split the rows into `n_splits` consecutive blocks; each block
/// is the test set once. `KFold::new(n_rows)` is leave-one-out.
///
/// `splits` returns no folds if `n_splits` is below 2 or above the row
/// count (cross-validation then reports the problem).
#[derive(Debug, Clone, PartialEq)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    seed: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize) -> KFold {
        KFold { n_splits, shuffle: false, seed: None }
    }

    /// Shuffle the rows (reproducibly, by seed) before splitting.
    pub fn with_shuffle(mut self, seed: u64) -> KFold {
        self.shuffle = true;
        self.seed = Some(seed);
        self
    }

    pub fn splits(&self, n_rows: usize) -> Folds {
        if self.n_splits < 2 || self.n_splits > n_rows {
            return Vec::new();
        }
        let mut fold_of_row = vec![0; n_rows];
        for (position, row) in row_order(n_rows, self.shuffle, self.seed).into_iter().enumerate() {
            fold_of_row[row] = position * self.n_splits / n_rows;
        }
        folds_from_assignment(&fold_of_row, self.n_splits)
    }
}

/// Stratified k-fold: every fold keeps the class proportions of `y`.
#[derive(Debug, Clone, PartialEq)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    seed: Option<u64>,
}

impl StratifiedKFold {
    pub fn new(n_splits: usize) -> StratifiedKFold {
        StratifiedKFold { n_splits, shuffle: false, seed: None }
    }

    pub fn with_shuffle(mut self, seed: u64) -> StratifiedKFold {
        self.shuffle = true;
        self.seed = Some(seed);
        self
    }

    pub fn splits(&self, y: &[f64], n_rows: usize) -> Folds {
        if self.n_splits < 2 || self.n_splits > n_rows || y.len() != n_rows {
            return Vec::new();
        }
        // Group the rows by class, then deal each class out across the folds.
        let mut rows_of_class: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for row in row_order(n_rows, self.shuffle, self.seed) {
            rows_of_class.entry(y[row].to_bits()).or_default().push(row);
        }
        let mut fold_of_row = vec![0; n_rows];
        for (dealt, row) in rows_of_class.into_values().flatten().enumerate() {
            fold_of_row[row] = dealt % self.n_splits;
        }
        folds_from_assignment(&fold_of_row, self.n_splits)
    }
}

/// Grouped k-fold: rows that share a group id always land in the same
/// fold. Get the ids from `ModelData::groups()`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    pub fn new(n_splits: usize) -> GroupKFold {
        GroupKFold { n_splits }
    }

    pub fn splits(&self, groups: &[i64], n_rows: usize) -> Folds {
        let mut size_of_group: BTreeMap<i64, usize> = BTreeMap::new();
        for group in groups {
            *size_of_group.entry(*group).or_default() += 1;
        }
        if self.n_splits < 2 || self.n_splits > size_of_group.len() || groups.len() != n_rows {
            return Vec::new();
        }
        // Largest groups first, each into the fold that is smallest so far.
        let mut by_size: Vec<(i64, usize)> = size_of_group.into_iter().collect();
        by_size.sort_by_key(|&(group, size)| (std::cmp::Reverse(size), group));
        let mut fold_sizes = vec![0; self.n_splits];
        let mut fold_of_group = HashMap::new();
        for (group, size) in by_size {
            // Invariant: n_splits >= 2, so there is always a smallest fold.
            let smallest = (0..self.n_splits).min_by_key(|&fold| fold_sizes[fold]).unwrap();
            fold_sizes[smallest] += size;
            fold_of_group.insert(group, smallest);
        }
        let fold_of_row: Vec<usize> = groups.iter().map(|group| fold_of_group[group]).collect();
        folds_from_assignment(&fold_of_row, self.n_splits)
    }
}

/// Scores from cross-validation. Loss metrics (rmse, mse, mae, log_loss)
/// are negated, so higher is always better.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CrossValidationResult {
    /// One score per fold, in fold order. The mean alone hides the spread.
    pub scores: Vec<f64>,
    pub mean: f64,
    pub std: f64,
}

impl CrossValidationResult {
    fn from_scores(scores: Vec<f64>) -> CrossValidationResult {
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|score| (score - mean).powi(2)).sum::<f64>() / n;
        CrossValidationResult { scores, mean, std: variance.sqrt() }
    }
}

/// Reject folds that are empty or point outside the data.
fn check_folds(folds: &[(Vec<usize>, Vec<usize>)], n_rows: usize) -> Result<()> {
    if folds.is_empty() {
        return Err(Error::InvalidSpec(
            "no folds — n_splits must be at least 2 and at most the number of rows (or groups)".to_string(),
        ));
    }
    let in_range = |indices: &Vec<usize>| indices.iter().all(|&i| i < n_rows);
    if !folds.iter().all(|(train, test)| in_range(train) && in_range(test)) {
        return Err(Error::shape(format!("fold indices below {n_rows}"), "an index out of range"));
    }
    Ok(())
}

/// Score a fitted model on encoded test data. Higher is better. With
/// several targets the scorer is applied to each and the results averaged.
fn score_model(model: &dyn Predictor, test: &EncodedData, scorer: &dyn Scorer) -> Result<f64> {
    let score = if scorer.uses_probabilities() {
        let probabilities = model.predict_proba(test)?;
        if probabilities.ncols() != 2 {
            return Err(Error::InvalidSpec(format!("scorer '{}' needs exactly two classes", scorer.name())));
        }
        scorer.score(&test.target()?, &probabilities.column(1).to_vec())
    } else {
        let metric = |y_true: &[f64], y_pred: &[f64]| scorer.score(y_true, y_pred);
        metrics::mean_over_targets(metric, &test.y, &model.predict(test)?)?
    };
    Ok(if scorer.greater_is_better() { score } else { -score })
}

/// Cross-validate an estimator on already-encoded data: fit on each
/// train fold, score on each test fold.
///
/// If the data needs preprocessing, use [`cross_validate_pipeline`].
pub fn cross_validate<E: Estimator>(
    spec: &E,
    data: &EncodedData,
    folds: &[(Vec<usize>, Vec<usize>)],
    scorer: &dyn Scorer,
) -> Result<CrossValidationResult> {
    check_folds(folds, data.num_rows())?;
    let scores = folds
        .iter()
        .map(|(train, test)| {
            let model = spec.fit(&data.take_rows(train))?;
            score_model(&model, &data.take_rows(test), scorer)
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok(CrossValidationResult::from_scores(scores))
}

/// Leakage-safe cross-validation: for each fold the whole pipeline —
/// every preprocessing step and the model — is refitted on that fold's
/// training rows only, then scored on its test rows.
pub fn cross_validate_pipeline(
    pipe: &Pipeline,
    data: &ModelData,
    folds: &[(Vec<usize>, Vec<usize>)],
    scorer: &dyn Scorer,
) -> Result<CrossValidationResult> {
    check_folds(folds, data.num_rows())?;
    let scores = folds
        .iter()
        .map(|(train, test)| {
            let train_table = data.table.take_rows(train);
            let test_table = data.table.take_rows(test);
            let fitted = pipe.fit(&ModelData::new(&train_table, data.spec.clone())?)?;
            let test_data = fitted.encode_with_targets(&ModelData::new(&test_table, data.spec.clone())?)?;
            score_model(fitted.model(), &test_data, scorer)
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok(CrossValidationResult::from_scores(scores))
}

/// Parameter name → candidate values. For pipelines, `model__n_trees`
/// reaches the model and `branch__param` a preprocessing branch.
pub type ParamGrid = HashMap<String, Vec<ParamValue>>;

/// One choice of value for every parameter in a grid.
pub type Params = HashMap<String, ParamValue>;

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GridSearchResult {
    pub best_params: Params,
    /// Mean cross-validated score of the best combination (higher is better).
    pub best_score: f64,
    /// Every combination with its mean score, in search order.
    pub all_scores: Vec<(Params, f64)>,
}

/// Every combination of the grid's values, in a fixed (sorted-name) order.
fn combinations(grid: &ParamGrid) -> Vec<Params> {
    let mut names: Vec<&String> = grid.keys().collect();
    names.sort();
    let mut combos = vec![Params::new()];
    for name in names {
        combos = combos
            .iter()
            .flat_map(|combo| {
                grid[name].iter().map(move |value| {
                    let mut extended = combo.clone();
                    extended.insert(name.clone(), value.clone());
                    extended
                })
            })
            .collect();
    }
    combos
}

/// Score every combination and keep the best (the first, on ties).
fn search(grid: &ParamGrid, evaluate: impl Fn(&Params) -> Result<f64>) -> Result<GridSearchResult> {
    let mut all_scores = Vec::new();
    for params in combinations(grid) {
        let score = evaluate(&params)?;
        all_scores.push((params, score));
    }
    let mut best = &all_scores[0];
    for candidate in &all_scores {
        if candidate.1 > best.1 {
            best = candidate;
        }
    }
    Ok(GridSearchResult { best_params: best.0.clone(), best_score: best.1, all_scores: all_scores.clone() })
}

/// Grid search on already-encoded data: cross-validate every parameter
/// combination and keep the best.
///
/// If the data needs preprocessing, use [`grid_search_pipeline`].
pub fn grid_search<E: Estimator + Clone>(
    spec: &E,
    param_grid: &ParamGrid,
    data: &EncodedData,
    folds: &[(Vec<usize>, Vec<usize>)],
    scorer: &dyn Scorer,
) -> Result<GridSearchResult> {
    search(param_grid, |params| {
        let mut candidate = spec.clone();
        for (name, value) in params {
            candidate.set_param(name, value)?;
        }
        Ok(cross_validate(&candidate, data, folds, scorer)?.mean)
    })
}

/// Leakage-safe grid search: every parameter combination is scored by
/// [`cross_validate_pipeline`].
pub fn grid_search_pipeline(
    pipe: &Pipeline,
    param_grid: &ParamGrid,
    data: &ModelData,
    folds: &[(Vec<usize>, Vec<usize>)],
    scorer: &dyn Scorer,
) -> Result<GridSearchResult> {
    search(param_grid, |params| {
        let mut candidate = pipe.clone();
        for (name, value) in params {
            candidate.set_param(name, value)?;
        }
        Ok(cross_validate_pipeline(&candidate, data, folds, scorer)?.mean)
    })
}
