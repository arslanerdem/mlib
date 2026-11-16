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
use crate::traits::{Label, ParamValue};
use crate::tree::Task;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// How much the rows of each class count while a classifier is fitted.
///
/// When one class is rare — fraud, churn, a default — a model can score
/// well by ignoring it. Giving its rows more weight makes mistakes on
/// them cost more. A row's class weight multiplies its sample weight,
/// if it has one.
///
/// ```
/// use mlib::ndarray::Array2;
/// use mlib::prelude::*;
///
/// // 300 rows; one in ten is of class 1, and the classes overlap.
/// let x = Array2::from_shape_fn((300, 1), |(row, _)| ((row * 7) % 30) as f64);
/// let y = Array2::from_shape_fn((300, 1), |(row, _)| if row % 10 == 0 { 1.0 } else { 0.0 });
/// let data = EncodedData::new(x, y)?;
/// let found = |model: &FittedLinearModel| -> Result<f64, Error> {
///     Ok(recall_score(&data.target()?, &model.predict_single(&data)?))
/// };
///
/// // Left alone, the model never predicts the rare class ...
/// let plain = LinearModelSpec::default().build()?.fit(&data)?;
/// assert_eq!(found(&plain)?, 0.0);
///
/// // ... balanced, each class counts for the same in total.
/// let balanced = LinearModelSpec::default().class_weight(ClassWeight::Balanced).build()?.fit(&data)?;
/// assert!(found(&balanced)? > 0.3);
///
/// // Or say what each class counts for; a class left out counts 1.
/// let costly = ClassWeight::PerClass(vec![(1.0, 20.0)]);
/// let weighted = LinearModelSpec::default().class_weight(costly).build()?.fit(&data)?;
/// assert!(found(&weighted)? > 0.3);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ClassWeight {
    /// Every row counts the same. The default.
    #[default]
    None,
    /// Every class counts the same in total: a row of a class weighs
    /// `n_rows / (n_classes × rows in that class)`, as in scikit-learn.
    Balanced,
    /// A weight for each named class, as (class label, weight) pairs. A
    /// class that is not named weighs 1.
    PerClass(Vec<(Label, f64)>),
}

impl ClassWeight {
    /// Fail if the choice cannot be used for this task: class weights
    /// are for classifiers, and a weight is a number, 0 or more.
    pub(crate) fn check(&self, task: Task) -> Result<()> {
        let invalid = |message: &str| Err(Error::InvalidSpec(message.to_string()));
        if *self != ClassWeight::None && task == Task::Regression {
            return invalid("class_weight is for classification; a regressor has no classes");
        }
        if let ClassWeight::PerClass(weights) = self {
            if weights.iter().any(|(_, weight)| !(weight.is_finite() && *weight >= 0.0)) {
                return invalid("every class weight must be a number, 0 or more");
            }
            let repeated =
                |(i, (label, _)): (usize, &(Label, f64))| weights[..i].iter().any(|(earlier, _)| earlier == label);
            if weights.iter().enumerate().any(repeated) {
                return invalid("class_weight names a class twice");
            }
        }
        Ok(())
    }

    /// The choice named by a parameter value: Text "none" or "balanced".
    pub(crate) fn from_param(value: &ParamValue) -> Result<ClassWeight> {
        match value {
            ParamValue::Text(text) if text == "none" => Ok(ClassWeight::None),
            ParamValue::Text(text) if text == "balanced" => Ok(ClassWeight::Balanced),
            other => Err(Error::InvalidSpec(format!("invalid value for class_weight: {other:?}"))),
        }
    }
}

/// The weight of every row during a fit: its sample weight (1 if there
/// are none) times the weight of its class. `None` when no row is
/// weighted at all.
pub(crate) fn row_weights(class_weight: &ClassWeight, data: &EncodedData) -> Result<Option<Vec<f64>>> {
    if *class_weight == ClassWeight::None {
        return Ok(data.weights.clone());
    }
    if data.y.ncols() != 1 {
        return Err(Error::InvalidSpec(format!(
            "class_weight needs exactly one class target, and there are {}",
            data.y.ncols()
        )));
    }
    let found = TrainingClasses::new(data);
    let (labels, class_of_row) = (&found.labels, &found.index_of_row[0]);

    let weight_of_class: Vec<f64> = match class_weight {
        ClassWeight::Balanced => {
            let mut rows_in_class = vec![0usize; labels.len()];
            class_of_row.iter().for_each(|&class| rows_in_class[class as usize] += 1);
            let (n_rows, n_classes) = (data.num_rows() as f64, labels.len() as f64);
            rows_in_class.iter().map(|&rows| n_rows / (n_classes * rows as f64)).collect()
        }
        ClassWeight::PerClass(named) => {
            if let Some((unknown, _)) = named.iter().find(|(label, _)| !labels.contains(label)) {
                return Err(Error::InvalidData(format!(
                    "class_weight names class {unknown}, which the target does not have"
                )));
            }
            let weight_of =
                |label: &Label| named.iter().find(|(named, _)| named == label).map_or(1.0, |(_, weight)| *weight);
            labels.iter().map(weight_of).collect()
        }
        ClassWeight::None => unreachable!("handled above"),
    };

    let sample_weight = |row: usize| data.weights.as_ref().map_or(1.0, |weights| weights[row]);
    let weights: Vec<f64> = class_of_row
        .iter()
        .enumerate()
        .map(|(row, &class)| sample_weight(row) * weight_of_class[class as usize])
        .collect();
    if !weights.iter().any(|weight| *weight > 0.0) {
        return Err(Error::InvalidData("with these class weights no row has any weight".to_string()));
    }
    Ok(Some(weights))
}

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
