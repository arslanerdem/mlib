//! Split finding by histogram: the one dense, regular piece of tree
//! building.
//!
//! This file is deliberately a small, self-contained boundary — it is the
//! seam where a GPU implementation could slot in later.
//!
//! One formula serves both tasks. Give every sample a target vector
//! (one-hot class for classification, the value for regression); then the
//! quality of a group of samples is `sum_k (S_k)^2 / W`, where `S_k` is
//! the weighted sum of channel `k` and `W` the total weight. Maximising
//! that score over the two sides of a split minimises Gini impurity
//! (classification) or variance (regression).

use super::binning::MAX_BINS;
use ndarray::{Array2, ArrayView1};

/// What a tree learns to predict, per row.
pub enum Targets {
    /// Classification: the class index of each row.
    Classes { class_of_row: Vec<u32>, n_classes: usize },
    /// Regression on one or several targets.
    Values {
        /// Row after row, `n_targets` numbers each: `(value - mean) / scale`.
        /// Subtracting the mean keeps the sums numerically well-behaved;
        /// dividing by the scale lets targets with different units count
        /// equally when a split is chosen.
        values: Vec<f64>,
        n_targets: usize,
        /// Per target.
        means: Vec<f64>,
        scales: Vec<f64>,
    },
}

impl Targets {
    /// Regression targets from a (rows × targets) matrix. Several targets
    /// are each scaled by their standard deviation; a single target needs
    /// no scaling, because it has nothing to be balanced against.
    pub fn values(y: &Array2<f64>) -> Targets {
        let n_targets = y.ncols();
        let means: Vec<f64> =
            y.columns().into_iter().map(|column| column.iter().sum::<f64>() / column.len() as f64).collect();
        let scale_of = |(column, mean): (ArrayView1<f64>, &f64)| {
            let variance = column.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / column.len() as f64;
            // A constant target has no spread to scale by.
            if n_targets == 1 || variance == 0.0 { 1.0 } else { variance.sqrt() }
        };
        let scales: Vec<f64> = y.columns().into_iter().zip(&means).map(scale_of).collect();
        let standardised = y.rows().into_iter().flat_map(|row| {
            let (means, scales) = (&means, &scales);
            (0..n_targets).map(move |t| (row[t] - means[t]) / scales[t])
        });
        Targets::Values { values: standardised.collect(), n_targets, means, scales }
    }

    /// A single regression target, used exactly as given.
    pub fn single(values: Vec<f64>) -> Targets {
        Targets::Values { values, n_targets: 1, means: vec![0.0], scales: vec![1.0] }
    }

    pub fn n_channels(&self) -> usize {
        match self {
            Targets::Classes { n_classes, .. } => *n_classes,
            Targets::Values { n_targets, .. } => *n_targets,
        }
    }

    /// What a [`Sample`] carries of its row's target: the class index, or
    /// the value of a single regression target. With several regression
    /// targets the values stay in `Targets` and are looked up by row.
    pub fn of_row(&self, row: usize) -> f64 {
        match self {
            Targets::Classes { class_of_row, .. } => class_of_row[row] as f64,
            Targets::Values { values, n_targets: 1, .. } => values[row],
            Targets::Values { .. } => 0.0,
        }
    }

    /// A row's values, when there are several regression targets.
    fn values_of<'a>(values: &'a [f64], n_targets: usize, sample: &Sample) -> &'a [f64] {
        &values[sample.row as usize * n_targets..][..n_targets]
    }

    /// Add a sample's weighted targets to per-target sums, when there are
    /// several regression targets.
    fn add_to(&self, sample: &Sample, sums: &mut [f64]) {
        if let Targets::Values { values, n_targets, .. } = self {
            for (sum, value) in sums.iter_mut().zip(Targets::values_of(values, *n_targets, sample)) {
                *sum += sample.weight * value;
            }
        }
    }

    /// Whether two samples have the same target.
    pub fn same(&self, a: &Sample, b: &Sample) -> bool {
        match self {
            Targets::Classes { .. } | Targets::Values { n_targets: 1, .. } => a.target == b.target,
            Targets::Values { values, n_targets, .. } => {
                Targets::values_of(values, *n_targets, a) == Targets::values_of(values, *n_targets, b)
            }
        }
    }

    /// Totals over a group of samples. Like [`Histogram::fill`], this
    /// runs over every sample of a node, so each kind of target has its
    /// own loop.
    pub fn totals(&self, samples: &[Sample]) -> Totals {
        fn sum_up(samples: &[Sample], n_channels: usize, add_target: impl Fn(&Sample, &mut [f64])) -> Totals {
            let mut totals = Totals::zero(n_channels);
            for sample in samples {
                totals.count += sample.count as f64;
                totals.weight += sample.weight;
                add_target(sample, &mut totals.sums);
            }
            totals
        }
        let n_channels = self.n_channels();
        match self {
            Targets::Classes { .. } => {
                sum_up(samples, n_channels, |sample, sums| sums[sample.target as usize] += sample.weight)
            }
            Targets::Values { n_targets: 1, .. } => {
                sum_up(samples, n_channels, |sample, sums| sums[0] += sample.weight * sample.target)
            }
            Targets::Values { .. } => sum_up(samples, n_channels, |sample, sums| self.add_to(sample, sums)),
        }
    }

    /// What a leaf holding these samples predicts: class probabilities,
    /// or the mean of each target in its original units.
    pub fn leaf_prediction(&self, totals: &Totals) -> Vec<f64> {
        let averages = totals.sums.iter().map(|sum| sum / totals.weight);
        match self {
            Targets::Classes { .. } => averages.collect(),
            Targets::Values { means, scales, .. } => {
                averages.zip(means).zip(scales).map(|((average, mean), scale)| average * scale + mean).collect()
            }
        }
    }
}

/// One training row as a tree sees it.
#[derive(Clone, Copy)]
pub struct Sample {
    /// The row's total weight: its sample weight times `count`.
    pub weight: f64,
    /// See [`Targets::of_row`].
    pub target: f64,
    /// Which row of the training data this is.
    pub row: u32,
    /// How many times the bootstrap drew the row (1 without bootstrap).
    pub count: u32,
}

/// Totals over a group of samples.
#[derive(Clone)]
pub struct Totals {
    /// Number of samples (bootstrap repeats counted).
    pub count: f64,
    /// Total sample weight.
    pub weight: f64,
    /// Weighted target sum, per channel.
    pub sums: Vec<f64>,
}

impl Totals {
    pub fn zero(n_channels: usize) -> Totals {
        Totals { count: 0.0, weight: 0.0, sums: vec![0.0; n_channels] }
    }

    /// The quality score of this group: `sum_k (S_k)^2 / W`.
    pub fn score(&self) -> f64 {
        self.sums.iter().map(|sum| sum * sum).sum::<f64>() / self.weight
    }

    /// The totals of what is left after taking `part` out of this group.
    pub fn without(&self, part: &Totals) -> Totals {
        Totals {
            count: self.count - part.count,
            weight: self.weight - part.weight,
            sums: self.sums.iter().zip(&part.sums).map(|(all, part)| all - part).collect(),
        }
    }
}

/// The best place to cut one feature.
pub struct Cut {
    /// Samples with bin code <= `bin` go left.
    pub bin: u8,
    /// Score of the left group plus score of the right group.
    pub score: f64,
}

/// Per-bin totals for one feature over one node's samples.
///
/// Each bin is a run of `2 + n_channels` numbers: count, weight, then the
/// per-channel sums. The histogram also notes which bins it has used, so
/// a node with few samples costs time proportional to its size, not to 256.
pub struct Histogram {
    stride: usize,
    cells: Vec<f64>,
    /// One bit per bin, set when the bin holds a sample: bin `b` is bit
    /// `b % 64` of word `b / 64`.
    occupied: [u64; MAX_BINS / 64],
}

impl Histogram {
    pub fn new(n_channels: usize) -> Histogram {
        let stride = 2 + n_channels;
        Histogram { stride, cells: vec![0.0; MAX_BINS * stride], occupied: [0; MAX_BINS / 64] }
    }

    /// Add a node's samples. `bins` holds this feature's bin code for
    /// every row of the training data. The histogram must be empty (see
    /// [`Histogram::clear`]).
    ///
    /// This is the hottest loop in the library, so each kind of target
    /// gets its own copy of it, with nothing left to decide per sample.
    pub fn fill(&mut self, bins: &[u8], samples: &[Sample], targets: &Targets) {
        match targets {
            Targets::Classes { .. } => {
                self.accumulate(bins, samples, |sample, sums| sums[sample.target as usize] += sample.weight)
            }
            Targets::Values { n_targets: 1, .. } => {
                self.accumulate(bins, samples, |sample, sums| sums[0] += sample.weight * sample.target)
            }
            Targets::Values { .. } => self.accumulate(bins, samples, |sample, sums| targets.add_to(sample, sums)),
        }
    }

    /// `add_target(sample, sums)` adds a sample's weighted target to a bin's sums.
    #[inline(always)]
    fn accumulate(&mut self, bins: &[u8], samples: &[Sample], add_target: impl Fn(&Sample, &mut [f64])) {
        for sample in samples {
            let bin = bins[sample.row as usize] as usize;
            self.occupied[bin / 64] |= 1 << (bin % 64);
            let cell = &mut self.cells[bin * self.stride..][..self.stride];
            cell[0] += sample.count as f64;
            cell[1] += sample.weight;
            add_target(sample, &mut cell[2..]);
        }
    }

    /// The bins that hold at least one sample, in increasing order.
    fn occupied_bins(&self) -> impl Iterator<Item = usize> {
        let occupied = self.occupied;
        occupied.into_iter().enumerate().flat_map(|(word_index, word)| {
            let mut remaining = word;
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1; // clears the lowest set bit
                Some(word_index * 64 + bit)
            })
        })
    }

    /// How many bins hold at least one sample. One means the feature has a
    /// single value across the samples and cannot split them.
    pub fn n_occupied_bins(&self) -> usize {
        self.occupied.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// Scan the bins left to right and return the best cut that leaves at
    /// least `min_samples_leaf` samples on each side and scores above
    /// `score_to_beat`.
    pub fn best_cut(&self, parent: &Totals, min_samples_leaf: f64, score_to_beat: f64) -> Option<Cut> {
        let mut best = None;
        let mut best_score = score_to_beat;
        let mut left = Totals::zero(parent.sums.len());

        // Cutting after the last occupied bin would leave the right side empty.
        let n_candidates = self.n_occupied_bins().saturating_sub(1);
        for bin in self.occupied_bins().take(n_candidates) {
            let cell = &self.cells[bin * self.stride..][..self.stride];
            left.count += cell[0];
            left.weight += cell[1];
            for (sum, add) in left.sums.iter_mut().zip(&cell[2..]) {
                *sum += add;
            }

            let right_count = parent.count - left.count;
            let right_weight = parent.weight - left.weight;
            if left.count < min_samples_leaf || right_count < min_samples_leaf || right_weight <= 0.0 {
                continue;
            }
            let right_squares: f64 =
                parent.sums.iter().zip(&left.sums).map(|(all, left)| (all - left) * (all - left)).sum();
            let score = left.score() + right_squares / right_weight;
            if score > best_score {
                best_score = score;
                best = Some(Cut { bin: bin as u8, score });
            }
        }
        best
    }

    /// Reset to empty, touching only the bins that were used.
    pub fn clear(&mut self) {
        for bin in self.occupied_bins() {
            self.cells[bin * self.stride..][..self.stride].fill(0.0);
        }
        self.occupied = [0; MAX_BINS / 64];
    }
}
