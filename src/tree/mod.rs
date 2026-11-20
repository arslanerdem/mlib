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
///
/// ```
/// use mlib::prelude::*;
///
/// let classifier = GradientBoostingSpec::default().build()?;
/// assert_eq!(classifier.spec().task, Task::Classification);
/// let regressor = GradientBoostingSpec::default().task(Task::Regression).build()?;
/// assert_eq!(regressor.spec(), GradientBoostingSpec::regressor().build()?.spec());
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
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
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // One feature; the class is 1 from 50 upwards. One tree on all the rows.
/// let x = Array2::from_shape_fn((100, 1), |(row, _)| row as f64);
/// let y = x.mapv(|value| if value >= 50.0 { 1.0 } else { 0.0 });
/// let data = EncodedData::new(x, y)?;
/// let forest = RandomForestSpec::default().n_trees(1).bootstrap(false).build()?.fit(&data)?;
///
/// let tree = forest.tree(0);
/// assert_eq!((tree.n_nodes(), tree.n_leaves(), tree.depth()), (3, 2, 1));
///
/// // A leaf holds the probability of each class.
/// assert_eq!(tree.predict_row(&[20.0]), [1.0, 0.0]);
/// assert_eq!(tree.predict_row(&[80.0]), [0.0, 1.0]);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "SavedTree")]
pub struct Tree {
    pub(crate) nodes: Vec<TreeNode>,
    /// Every leaf's prediction, one leaf after another, `n_outputs`
    /// numbers each: class probabilities (classification) or the mean
    /// target (regression).
    pub(crate) leaf_values: Vec<f64>,
    /// How many numbers one leaf predicts.
    pub(crate) n_outputs: usize,
    /// Whether any split sends missing values left. Most trees have no
    /// such split, and predicting with them can then skip the question.
    /// Not saved: it is read off the nodes when a model is loaded.
    #[serde(skip_serializing)]
    sends_missing_left: bool,
}

/// A tree as it is read from a saved model.
#[derive(Deserialize)]
struct SavedTree {
    nodes: Vec<TreeNode>,
    leaf_values: Vec<f64>,
    n_outputs: usize,
}

impl From<SavedTree> for Tree {
    fn from(saved: SavedTree) -> Tree {
        Tree::new(saved.nodes, saved.leaf_values, saved.n_outputs)
    }
}

/// One node of a tree, as [`Tree::node`] describes it.
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // One feature; the class is 1 from 50 upwards. One tree on all the rows.
/// let x = Array2::from_shape_fn((100, 1), |(row, _)| row as f64);
/// let y = x.mapv(|value| if value >= 50.0 { 1.0 } else { 0.0 });
/// let data = EncodedData::new(x, y)?;
/// let forest = RandomForestSpec::default().n_trees(1).bootstrap(false).build()?.fit(&data)?;
/// let tree = forest.tree(0);
///
/// // The root asks one question and sends the row left or right.
/// let Node::Split { feature, threshold, left, right } = tree.node(0) else {
///     panic!("the root of this tree is a split");
/// };
/// assert_eq!(feature, 0);
/// assert!((49.0..50.0).contains(&threshold));
/// assert_eq!(tree.node(left), Node::Leaf { prediction: &[1.0, 0.0] });
/// assert_eq!(tree.node(right), Node::Leaf { prediction: &[0.0, 1.0] });
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Node<'a> {
    /// Rows whose `feature` value is at most `threshold` continue at node
    /// `left`; the rest continue at node `right`. A row whose value is
    /// missing goes right, unless [`Tree::missing_goes_left`] says
    /// otherwise for this node.
    Split {
        /// Which feature is looked at, by its position among the model's features.
        feature: usize,
        /// The value the feature is compared with.
        threshold: f64,
        /// The node for rows at or below the threshold.
        left: usize,
        /// The node for rows above it.
        right: usize,
    },
    /// The end of a path: class probabilities (classification) or the
    /// mean of each target (regression).
    Leaf {
        /// One number per class, or one per regression target.
        prediction: &'a [f64],
    },
}

/// Marks a leaf in `TreeNode::feature`.
const LEAF: u32 = u32::MAX;

/// The bit of `TreeNode::feature` that says "missing values go left".
const MISSING_LEFT: u32 = 1 << 31;

/// How a node is stored. Kept small (16 bytes) because prediction speed
/// depends on how many nodes fit in the CPU cache; [`Node`] is the
/// readable form, and [`SavedNode`] the form written to a file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "SavedNode", into = "SavedNode")]
pub(crate) struct TreeNode {
    /// Rows with `value <= threshold` go left, the rest go right.
    threshold: f64,
    /// Feature index to split on, or `LEAF`. A split node's highest bit
    /// is `MISSING_LEFT`: there was no room for it anywhere else.
    feature: u32,
    /// Split node: the index of the left child. The right child is always
    /// the node after it.
    /// Leaf: which leaf this is, counting from 0 — its position in
    /// `Tree::leaf_values`.
    left_or_leaf: u32,
}

/// A node as it is written to a saved model: the same numbers, with
/// "missing values go left" spelled out. It is left out when false, so a
/// tree without such nodes is saved exactly as before they existed.
#[derive(Serialize, Deserialize)]
struct SavedNode {
    threshold: f64,
    feature: u32,
    left_or_leaf: u32,
    #[serde(default, skip_serializing_if = "is_false")]
    missing_left: bool,
}

fn is_false(flag: &bool) -> bool {
    !flag
}

impl From<SavedNode> for TreeNode {
    fn from(saved: SavedNode) -> TreeNode {
        let flag = if saved.missing_left && saved.feature != LEAF { MISSING_LEFT } else { 0 };
        TreeNode { threshold: saved.threshold, feature: saved.feature | flag, left_or_leaf: saved.left_or_leaf }
    }
}

impl From<TreeNode> for SavedNode {
    fn from(node: TreeNode) -> SavedNode {
        let feature = if node.is_leaf() { LEAF } else { node.feature & !MISSING_LEFT };
        SavedNode {
            threshold: node.threshold,
            feature,
            left_or_leaf: node.left_or_leaf,
            missing_left: node.missing_goes_left(),
        }
    }
}

impl TreeNode {
    /// A leaf: number `leaf_number` among the tree's leaves.
    pub(crate) fn leaf(leaf_number: usize) -> TreeNode {
        TreeNode { threshold: 0.0, feature: LEAF, left_or_leaf: leaf_number as u32 }
    }

    /// A split on `feature` whose children are nodes `left` and `left + 1`.
    pub(crate) fn split(feature: usize, threshold: f64, left: usize, missing_left: bool) -> TreeNode {
        let flag = if missing_left { MISSING_LEFT } else { 0 };
        TreeNode { threshold, feature: feature as u32 | flag, left_or_leaf: left as u32 }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.feature == LEAF
    }

    /// Whether a split node sends rows with a missing value left.
    fn missing_goes_left(&self) -> bool {
        !self.is_leaf() && self.feature & MISSING_LEFT != 0
    }

    /// The index of the child a row of features moves to.
    #[inline]
    fn child_for(&self, row: &[f64]) -> usize {
        let value = row[(self.feature & !MISSING_LEFT) as usize];
        // A missing value (NaN) is not `<=` anything, so by itself it goes right.
        let goes_left = (value <= self.threshold) | (value.is_nan() & (self.feature & MISSING_LEFT != 0));
        self.left_or_leaf as usize + if goes_left { 0 } else { 1 }
    }

    /// The same, for a tree in which no split sends missing values left:
    /// a missing value goes right by itself, and there is nothing to look up.
    #[inline]
    fn child_for_missing_right(&self, row: &[f64]) -> usize {
        let goes_left = row[self.feature as usize] <= self.threshold;
        self.left_or_leaf as usize + if goes_left { 0 } else { 1 }
    }
}

/// How many rows walk down a tree side by side (see `add_predictions`).
const ROWS_IN_STEP: usize = 8;

impl Tree {
    /// A tree from its nodes and its leaves' predictions.
    pub(crate) fn new(nodes: Vec<TreeNode>, leaf_values: Vec<f64>, n_outputs: usize) -> Tree {
        let sends_missing_left = nodes.iter().any(TreeNode::missing_goes_left);
        Tree { nodes, leaf_values, n_outputs, sends_missing_left }
    }

    /// How many nodes the tree has, splits and leaves together.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// How many leaves the tree has.
    pub fn n_leaves(&self) -> usize {
        self.leaf_values.len() / self.n_outputs
    }

    /// The number of splits on the longest path from the root to a leaf.
    /// A tree that is a single leaf has depth 0.
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        let mut to_visit = vec![(0, 0)];
        while let Some((index, depth)) = to_visit.pop() {
            deepest = deepest.max(depth);
            if let Node::Split { left, right, .. } = self.node(index) {
                to_visit.extend([(left, depth + 1), (right, depth + 1)]);
            }
        }
        deepest
    }

    /// Node number `index`, in readable form. Node 0 is the root. Panics
    /// if there is no such node.
    pub fn node(&self, index: usize) -> Node<'_> {
        let node = &self.nodes[index];
        if node.is_leaf() {
            return Node::Leaf { prediction: self.leaf_prediction(node) };
        }
        let left = node.left_or_leaf as usize;
        let feature = (node.feature & !MISSING_LEFT) as usize;
        Node::Split { feature, threshold: node.threshold, left, right: left + 1 }
    }

    /// Whether node number `index` sends a row left when its value for
    /// the node's feature is missing. A split learns this from the
    /// training rows: missing values go to the side where they fit best.
    /// False for a leaf, and for a split whose training rows had nothing
    /// missing — there a missing value goes right, with the large values.
    /// Panics if there is no such node.
    ///
    /// ```
    /// use mlib::ndarray::Array2;
    /// use mlib::prelude::*;
    ///
    /// // The feature is missing in every third row, and those rows are
    /// // like the low values: their class is 0, as it is below 50.
    /// let x = Array2::from_shape_fn((99, 1), |(row, _)| if row % 3 == 0 { f64::NAN } else { row as f64 });
    /// let y = x.mapv(|value| if value >= 50.0 { 1.0 } else { 0.0 });
    /// let data = EncodedData::new(x, y)?;
    /// let forest = RandomForestSpec::default().n_trees(1).bootstrap(false).build()?.fit(&data)?;
    ///
    /// // One split is enough: low and missing to the left, high to the right.
    /// let tree = forest.tree(0);
    /// assert_eq!(tree.n_nodes(), 3);
    /// assert!(tree.missing_goes_left(0));
    /// assert_eq!(tree.predict_row(&[f64::NAN]), [1.0, 0.0]); // class 0
    /// assert_eq!(tree.predict_row(&[80.0]), [0.0, 1.0]); // class 1
    /// # Ok::<(), Error>(())
    /// ```
    pub fn missing_goes_left(&self, index: usize) -> bool {
        self.nodes[index].missing_goes_left()
    }

    /// The prediction stored in a leaf node.
    pub(crate) fn leaf_prediction(&self, leaf: &TreeNode) -> &[f64] {
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
    pub(crate) fn add_predictions(&self, rows: &[f64], n_features: usize, totals: &mut [f64]) {
        if self.sends_missing_left {
            self.walk(rows, n_features, totals, TreeNode::child_for)
        } else {
            self.walk(rows, n_features, totals, TreeNode::child_for_missing_right)
        }
    }

    /// `add_predictions`, with the way a node picks a child given, so
    /// that each way gets its own copy of the loop.
    #[inline(always)]
    fn walk(
        &self,
        rows: &[f64],
        n_features: usize,
        totals: &mut [f64],
        child_for: impl Fn(&TreeNode, &[f64]) -> usize,
    ) {
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
                        *position = child_for(node, row);
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

/// Reject data a model cannot learn from. Tree models take missing
/// values in `x` (NaN); the linear model does not.
pub(crate) fn check_training_data(data: &EncodedData, missing_allowed: bool) -> Result<()> {
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
    if data.y.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidData("y must not contain NaN or infinite values".to_string()));
    }
    if data.x.iter().any(|value| value.is_infinite()) {
        return Err(Error::InvalidData("x must not contain infinite values".to_string()));
    }
    if !missing_allowed && data.x.iter().any(|value| value.is_nan()) {
        return Err(Error::InvalidData(
            "x has missing values (NaN), which this model cannot use — add an Imputer or drop the rows".to_string(),
        ));
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

/// Rescale gains to shares that sum to 1 (all zeros stay zeros).
pub(crate) fn as_shares(mut gains: Vec<f64>) -> Vec<f64> {
    let total: f64 = gains.iter().sum();
    if total > 0.0 {
        gains.iter_mut().for_each(|gain| *gain /= total);
    }
    gains
}

/// Add up several lists of per-feature numbers, feature by feature.
pub(crate) fn sum_per_feature<'a>(lists: impl IntoIterator<Item = &'a Vec<f64>>, n_features: usize) -> Vec<f64> {
    let mut totals = vec![0.0; n_features];
    for list in lists {
        for (total, value) in totals.iter_mut().zip(list) {
            *total += value;
        }
    }
    totals
}
