//! Class labels: which classes each target has, and where they sit in a
//! matrix of probabilities.
//!
//! A classifier may have several targets, each with its own classes
//! (say, "colour" with three and "size" with two). Everything about the
//! classes is kept in target order, side by side:
//!
//! ```text
//! targets:           colour            size
//! classes:           [0, 1, 2,         0, 1]
//! classes_per_target [3,               2]
//! predict_proba:     three columns,    two columns     (each block sums to 1)
//! ```

use crate::encoded::EncodedData;
use crate::error::{Error, Result};
use crate::traits::Label;
use ndarray::Array2;
use std::ops::Range;

/// The classes found in training data.
pub(crate) struct TrainingClasses {
    /// Every target's sorted class labels, one target after another.
    pub labels: Vec<Label>,
    /// How many classes each target has.
    pub per_target: Vec<usize>,
    /// `index_of_row[target][row]`: the row's class, as an index into
    /// that target's own classes.
    pub index_of_row: Vec<Vec<u32>>,
}

impl TrainingClasses {
    /// Read the classes off the target columns: each target's classes
    /// are its distinct values, sorted.
    pub fn new(data: &EncodedData) -> TrainingClasses {
        let mut classes = TrainingClasses { labels: Vec::new(), per_target: Vec::new(), index_of_row: Vec::new() };
        for target in data.y.columns() {
            let mut labels = target.to_vec();
            labels.sort_unstable_by(f64::total_cmp);
            labels.dedup();
            // Invariant: every value of the target is in `labels`, which was built from it.
            let index_of = |value: &f64| labels.binary_search_by(|label| label.total_cmp(value)).unwrap() as u32;
            classes.index_of_row.push(target.iter().map(index_of).collect());
            classes.per_target.push(labels.len());
            classes.labels.extend(labels);
        }
        classes
    }

    /// Fail unless every target has at least two classes to tell apart.
    pub fn need_two_each(&self) -> Result<()> {
        if self.per_target.iter().any(|&n_classes| n_classes < 2) {
            return Err(Error::InvalidData("a classifier needs at least two classes in every target".to_string()));
        }
        Ok(())
    }
}

/// Where each target's classes sit among all the classes: one range per
/// target. A model saved before classifiers took several targets has no
/// `per_target` list; all its classes belong to its one target.
pub(crate) fn blocks(n_classes: usize, per_target: &[usize]) -> Vec<Range<usize>> {
    if n_classes == 0 {
        return Vec::new();
    }
    let one_target = [n_classes];
    let per_target = if per_target.is_empty() { &one_target } else { per_target };
    let mut start = 0;
    let block = |&size: &usize| {
        start += size;
        start - size..start
    };
    per_target.iter().map(block).collect()
}

/// Each target's class labels, as a list per target.
pub(crate) fn by_target(labels: &[Label], per_target: &[usize]) -> Vec<Vec<Label>> {
    blocks(labels.len(), per_target).into_iter().map(|block| labels[block].to_vec()).collect()
}

/// Each row's most probable class label for every target, as a
/// (rows × targets) matrix. Ties go to the first class.
pub(crate) fn most_probable(probabilities: &Array2<f64>, labels: &[Label], per_target: &[usize]) -> Array2<f64> {
    let blocks = blocks(labels.len(), per_target);
    Array2::from_shape_fn((probabilities.nrows(), blocks.len()), |(row, target)| {
        let block = blocks[target].clone();
        let mut best = block.start;
        for class in block {
            if probabilities[[row, class]] > probabilities[[row, best]] {
                best = class;
            }
        }
        labels[best]
    })
}
