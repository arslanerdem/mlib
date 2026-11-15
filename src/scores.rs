//! From scores to predictions: what gradient boosting and the linear
//! model share.
//!
//! Both models compute *scores* for every row and turn them into
//! predictions the same way:
//!
//! - Regression: one score per target, which is the predicted value.
//! - A target with two classes: one score, the log-odds of the second
//!   class. Its sigmoid is that class's probability.
//! - A target with more classes: one score per class; their softmax
//!   gives the probabilities.
//!
//! Scores are laid out `[score][row]`. The scores of one target form a
//! *group*: with several targets the groups follow each other in target
//! order, and each is turned into probabilities on its own.

use crate::classes::TrainingClasses;
use crate::encoded::EncodedData;
use crate::error::Result;
use crate::traits::Label;
use crate::tree::Task;
use ndarray::Array2;
use rayon::prelude::*;
use std::ops::Range;

/// What a model is asked to learn from its training data.
pub(crate) struct Wanted {
    /// What each score should end up predicting, `[score][row]`: the
    /// target's value (regression), or 1 when the row belongs to the
    /// score's class and 0 when it does not (classification).
    pub values: Vec<Vec<f64>>,
    /// Which scores belong to which target.
    pub groups: Vec<Range<usize>>,
    /// The class labels, one target after another. Empty for regression.
    pub classes: Vec<Label>,
    /// How many classes each target has. Empty for regression.
    pub classes_per_target: Vec<usize>,
}

impl Wanted {
    pub fn new(task: Task, data: &EncodedData) -> Result<Wanted> {
        if task == Task::Regression {
            let values: Vec<Vec<f64>> = data.y.columns().into_iter().map(|target| target.to_vec()).collect();
            let groups = score_groups(task, values.len(), &[]);
            return Ok(Wanted { values, groups, classes: Vec::new(), classes_per_target: Vec::new() });
        }
        let found = TrainingClasses::new(data);
        found.need_two_each()?;
        let mut values = Vec::new();
        for (index_of_row, &n_classes) in found.index_of_row.iter().zip(&found.per_target) {
            // Two classes share a single score, for the second class.
            let scored_classes = if n_classes == 2 { 1..2 } else { 0..n_classes as u32 };
            let is_class = |class| index_of_row.iter().map(|&c| if c == class { 1.0 } else { 0.0 }).collect();
            values.extend(scored_classes.map(is_class));
        }
        let groups = score_groups(task, values.len(), &found.per_target);
        Ok(Wanted { values, groups, classes: found.labels, classes_per_target: found.per_target })
    }
}

/// Which scores belong to which target: one range of scores per target.
/// A model saved before classifiers took several targets has no
/// `classes_per_target`; all its scores belong to its one target.
pub(crate) fn score_groups(task: Task, n_scores: usize, classes_per_target: &[usize]) -> Vec<Range<usize>> {
    if task == Task::Regression {
        return (0..n_scores).map(|score| score..score + 1).collect();
    }
    // One target: two classes share a single score, more have one each.
    let one_target = [if n_scores == 1 { 2 } else { n_scores }];
    let classes_per_target = if classes_per_target.is_empty() { &one_target } else { classes_per_target };
    let mut start = 0;
    let group = |&n_classes: &usize| {
        let n_scores = if n_classes == 2 { 1 } else { n_classes };
        start += n_scores;
        start - n_scores..start
    };
    classes_per_target.iter().map(group).collect()
}

/// The score that predicts `mean` for every row, where `mean` is the
/// average of what the score should predict and `group_size` the number
/// of scores its target has.
pub(crate) fn starting_score(task: Task, group_size: usize, mean: f64) -> f64 {
    // Keeps the logarithms finite when a class has no weight at all.
    let share = mean.clamp(1e-12, 1.0 - 1e-12);
    match task {
        Task::Regression => mean,
        Task::Classification if group_size == 1 => (share / (1.0 - share)).ln(),
        Task::Classification => share.ln(),
    }
}

/// What the current scores predict, laid out like the scores
/// (`[score][row]`): the value for regression, a probability otherwise.
pub(crate) fn expected_values(task: Task, scores: &[Vec<f64>], groups: &[Range<usize>]) -> Vec<Vec<f64>> {
    match task {
        Task::Regression => scores.to_vec(),
        Task::Classification => groups.iter().flat_map(|group| probabilities(&scores[group.clone()])).collect(),
    }
}

/// One target's scores as probabilities, laid out like the scores.
pub(crate) fn probabilities(scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
    match scores {
        // Two classes: the sigmoid of the single score.
        [score] => vec![score.par_iter().map(|score| 1.0 / (1.0 + (-score).exp())).collect()],
        // More classes: the softmax of each row's scores.
        _ => {
            // Per row: the largest score, and the sum of exp(score - largest).
            // Subtracting the largest score first keeps `exp` from overflowing.
            let (largest, total): (Vec<f64>, Vec<f64>) = (0..scores[0].len())
                .into_par_iter()
                .map(|row| {
                    let largest = scores.iter().map(|class| class[row]).fold(f64::MIN, f64::max);
                    (largest, scores.iter().map(|class| (class[row] - largest).exp()).sum::<f64>())
                })
                .unzip();
            let softmax = |class: &Vec<f64>| {
                let rows = class.par_iter().zip(&largest).zip(&total);
                rows.map(|((score, largest), total)| (score - largest).exp() / total).collect()
            };
            scores.iter().map(softmax).collect()
        }
    }
}

/// Turn `[score][row]` lists into a (rows × scores) matrix.
pub(crate) fn by_row(columns: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_fn((columns[0].len(), columns.len()), |(row, column)| columns[column][row])
}

/// A classifier's scores as a matrix of probabilities: one row per
/// sample, one column per class, the targets side by side.
pub(crate) fn class_probabilities(scores: &[Vec<f64>], groups: &[Range<usize>]) -> Array2<f64> {
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for group in groups {
        match probabilities(&scores[group.clone()]).as_slice() {
            // Two classes: the single score gives the second class's probability.
            [second] => columns.extend([second.iter().map(|p| 1.0 - p).collect(), second.clone()]),
            classes => columns.extend(classes.iter().cloned()),
        }
    }
    by_row(&columns)
}
