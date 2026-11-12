//! Decision trees: the tree itself and the code that grows one. Shared by
//! the random forest and gradient boosting.

mod binning;
mod grow;
mod histogram;

pub(crate) use binning::BinnedFeatures;
pub(crate) use grow::{grow_tree, TreeParams};
pub(crate) use histogram::Targets;

use crate::encoded::EncodedData;
use crate::error::{Error, Result};
use crate::metrics;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// What a model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    /// Predict a class. The target's distinct values are the classes.
    Classification,
    /// Predict a number.
    Regression,
}

/// A decision tree, stored as a flat node array. Node 0 is the root.
///
/// The layout is built for fast prediction: nodes are 16 bytes each, so
/// many fit in the CPU cache, and the leaves' predictions sit together in
/// one array instead of one allocation per leaf.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<TreeNode>,
    /// Every leaf's prediction, one leaf after another, `n_outputs`
    /// numbers each: class probabilities (classification) or the mean
    /// target (regression).
    pub leaf_values: Vec<f64>,
    /// How many numbers one leaf predicts.
    pub n_outputs: usize,
}

/// Marks a leaf in `TreeNode::feature`.
pub const LEAF: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    /// Rows with `value <= threshold` go left, the rest go right.
    pub threshold: f64,
    /// Feature index to split on, or `LEAF`.
    pub feature: u32,
    /// Split node: the index of the left child. The right child is always
    /// the node after it.
    /// Leaf: which leaf this is, counting from 0 — its position in
    /// `Tree::leaf_values`.
    pub left_or_leaf: u32,
}

impl TreeNode {
    pub fn is_leaf(&self) -> bool {
        self.feature == LEAF
    }

    /// The index of the child a row of features moves to.
    #[inline]
    fn child_for(&self, row: &[f64]) -> usize {
        let goes_left = row[self.feature as usize] <= self.threshold;
        self.left_or_leaf as usize + if goes_left { 0 } else { 1 }
    }
}

/// How many rows walk down a tree side by side (see `add_predictions`).
const ROWS_IN_STEP: usize = 8;

impl Tree {
    /// The prediction stored in a leaf node.
    pub fn leaf_prediction(&self, leaf: &TreeNode) -> &[f64] {
        &self.leaf_values[leaf.left_or_leaf as usize * self.n_outputs..][..self.n_outputs]
    }

    /// Walk one row of features down to its leaf and return the leaf's prediction.
    pub fn predict_row(&self, row: &[f64]) -> &[f64] {
        let mut node = &self.nodes[0];
        while !node.is_leaf() {
            node = &self.nodes[node.child_for(row)];
        }
        self.leaf_prediction(node)
    }

    /// Add this tree's prediction for every row to `totals`.
    ///
    /// `rows` holds the feature values row after row (`n_features` each);
    /// `totals` holds `n_outputs` numbers per row.
    ///
    /// The result is the same as calling `predict_row` on each row, but
    /// several rows walk down the tree side by side, one level at a time.
    /// Each step of a walk waits on memory for the next node; with several
    /// walks in flight the CPU fetches their nodes at the same time instead
    /// of one after another, which makes prediction two to three times faster.
    pub fn add_predictions(&self, rows: &[f64], n_features: usize, totals: &mut [f64]) {
        let row_groups = rows.chunks(n_features * ROWS_IN_STEP);
        let total_groups = totals.chunks_mut(self.n_outputs * ROWS_IN_STEP);
        for (rows, totals) in row_groups.zip(total_groups) {
            // Where each row's walk currently is. All start at the root.
            let mut position = [0; ROWS_IN_STEP];
            let mut any_moved = true;
            while any_moved {
                any_moved = false;
                for (position, row) in position.iter_mut().zip(rows.chunks(n_features)) {
                    let node = &self.nodes[*position];
                    if !node.is_leaf() {
                        *position = node.child_for(row);
                        any_moved = true;
                    }
                }
            }
            for (position, row_totals) in position.iter().zip(totals.chunks_mut(self.n_outputs)) {
                let prediction = self.leaf_prediction(&self.nodes[*position]);
                for (total, value) in row_totals.iter_mut().zip(prediction) {
                    *total += value;
                }
            }
        }
    }
}

/// For every row of `x`, the sum of the trees' predictions: one flat run
/// of `n_outputs` numbers per row.
pub(crate) fn sum_tree_predictions(trees: &[Tree], x: &Array2<f64>) -> Vec<f64> {
    let (n_features, n_outputs) = (x.ncols(), trees[0].n_outputs);
    let x = x.as_standard_layout();
    // Invariant: a standard-layout array is one contiguous row-major slice.
    let values = x.as_slice().unwrap();

    // Rows are handled in blocks, in parallel. Within a block each tree
    // visits all the rows before the next tree starts, which keeps the
    // tree's nodes in the CPU cache. Blocks are large enough for that
    // to pay off, and small enough to keep every core busy.
    let block_rows = (x.nrows() / (2 * rayon::current_num_threads())).clamp(256, 16_384);
    let blocks: Vec<Vec<f64>> = values
        .par_chunks(n_features * block_rows)
        .map(|block| {
            let mut totals = vec![0.0; block.len() / n_features * n_outputs];
            for tree in trees {
                tree.add_predictions(block, n_features, &mut totals);
            }
            totals
        })
        .collect();
    blocks.concat()
}

/// Reject data a tree model cannot learn from.
pub(crate) fn check_training_data(data: &EncodedData) -> Result<()> {
    let n_rows = data.num_rows();
    if n_rows == 0 || data.num_features() == 0 {
        return Err(Error::shape(
            "at least one row and one feature",
            format!("{n_rows} rows × {} features", data.num_features()),
        ));
    }
    if data.y.nrows() != n_rows || data.y.ncols() == 0 {
        return Err(Error::shape(
            format!("y with {n_rows} rows and at least one target"),
            format!("{} rows × {} targets", data.y.nrows(), data.y.ncols()),
        ));
    }
    if data.x.iter().chain(data.y.iter()).any(|value| !value.is_finite()) {
        return Err(Error::InvalidData("x and y must not contain NaN or infinite values".to_string()));
    }
    if let Some(weights) = &data.weights {
        if weights.len() != n_rows {
            return Err(Error::shape(format!("{n_rows} weights"), format!("{}", weights.len())));
        }
        let usable = weights.iter().all(|w| w.is_finite() && *w >= 0.0) && weights.iter().any(|w| *w > 0.0);
        if !usable {
            return Err(Error::InvalidData(
                "sample weights must be finite, non-negative, and not all zero".to_string(),
            ));
        }
    }
    Ok(())
}

/// The class labels of the rows. Classification handles one target;
/// regression can take several.
pub(crate) fn class_labels(data: &EncodedData) -> Result<Vec<f64>> {
    if data.y.ncols() != 1 {
        return Err(Error::shape("one target for classification", format!("{} targets", data.y.ncols())));
    }
    Ok(data.y.column(0).to_vec())
}

/// A model's default score. Classification: accuracy. Regression: R²,
/// averaged over the targets when there are several.
pub(crate) fn default_score(task: Task, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Result<f64> {
    let metric = match task {
        Task::Classification => metrics::accuracy,
        Task::Regression => metrics::r2_score,
    };
    metrics::mean_over_targets(metric, y_true, y_pred)
}

/// Check that data to predict on has the features the model was fitted on.
pub(crate) fn check_n_features(data: &EncodedData, n_features: usize) -> Result<()> {
    if data.num_features() != n_features {
        return Err(Error::shape(format!("{n_features} features"), format!("{}", data.num_features())));
    }
    Ok(())
}

/// The distinct target values, sorted, and each row's index into them.
pub(crate) fn class_indices(y: &[f64]) -> (Vec<f64>, Vec<u32>) {
    let mut classes = y.to_vec();
    classes.sort_unstable_by(f64::total_cmp);
    classes.dedup();
    // Invariant: every y value is in `classes`, which was built from y.
    let class_of = |value: &f64| classes.binary_search_by(|class| class.total_cmp(value)).unwrap() as u32;
    let class_of_row = y.iter().map(class_of).collect();
    (classes, class_of_row)
}

/// A model's importances: the mean of its trees', rescaled to sum to 1.
pub(crate) fn average_importances(per_tree: &[Vec<f64>], n_features: usize) -> Vec<f64> {
    let mut importances = vec![0.0; n_features];
    for tree in per_tree {
        for (total, importance) in importances.iter_mut().zip(tree) {
            *total += importance;
        }
    }
    let sum: f64 = importances.iter().sum();
    if sum > 0.0 {
        importances.iter_mut().for_each(|importance| *importance /= sum);
    }
    importances
}

/// Each row's most probable class label, as a one-column matrix. Ties
/// go to the first class.
pub(crate) fn most_probable_classes(probabilities: &Array2<f64>, classes: &[f64]) -> Array2<f64> {
    let label_of = |row: ndarray::ArrayView1<f64>| {
        let mut best = 0;
        for (class, probability) in row.iter().enumerate() {
            if *probability > row[best] {
                best = class;
            }
        }
        classes[best]
    };
    let labels: Vec<f64> = probabilities.rows().into_iter().map(label_of).collect();
    Array2::from_shape_vec((labels.len(), 1), labels).expect("one label per row")
}
