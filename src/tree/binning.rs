//! Feature binning: each feature's values are replaced by small integer
//! bin codes (at most 256 bins), computed once per forest.
//!
//! Trees then search for splits over bins instead of raw values, which
//! turns split finding into cheap histogram sums. A feature with at most
//! 256 distinct values gets one bin per value, so nothing is lost; a
//! feature with more gets quantile bins.

use ndarray::Array2;
use rayon::prelude::*;

pub const MAX_BINS: usize = 256;

pub struct BinnedFeatures {
    /// `bins[feature][row]`: the bin code of that value.
    bins: Vec<Vec<u8>>,
    /// `thresholds[feature][b]`: the upper edge of bin `b`. A value falls
    /// in bin `b` or lower exactly when `value <= thresholds[feature][b]`.
    thresholds: Vec<Vec<f64>>,
}

impl BinnedFeatures {
    pub fn new(x: &Array2<f64>) -> BinnedFeatures {
        let (bins, thresholds) = (0..x.ncols())
            .into_par_iter()
            .map(|feature| {
                let values = x.column(feature).to_vec();
                let thresholds = cut_points(&values);
                let bin_of = |&value: &f64| thresholds.partition_point(|&t| t < value) as u8;
                (values.iter().map(bin_of).collect::<Vec<u8>>(), thresholds)
            })
            .unzip();
        BinnedFeatures { bins, thresholds }
    }

    pub fn n_features(&self) -> usize {
        self.bins.len()
    }

    /// The bin code of every row, for one feature.
    pub fn bins(&self, feature: usize) -> &[u8] {
        &self.bins[feature]
    }

    /// The raw-value threshold equivalent to "bin code <= `bin`".
    pub fn threshold(&self, feature: usize, bin: u8) -> f64 {
        self.thresholds[feature][bin as usize]
    }
}

/// The increasing cut points that separate a feature's values into bins.
fn cut_points(values: &[f64]) -> Vec<f64> {
    // A point between two distinct neighbours: lo <= midpoint < hi.
    fn midpoint(lo: f64, hi: f64) -> f64 {
        let mid = lo + (hi - lo) / 2.0;
        if mid < hi { mid } else { lo }
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    // Each distinct value with the number of rows at or below it.
    let mut distinct: Vec<(f64, usize)> = Vec::new();
    for (i, &value) in sorted.iter().enumerate() {
        match distinct.last_mut() {
            Some(last) if last.0 == value => last.1 = i + 1,
            _ => distinct.push((value, i + 1)),
        }
    }

    if distinct.len() <= MAX_BINS {
        // One bin per distinct value: exact.
        return distinct.windows(2).map(|pair| midpoint(pair[0].0, pair[1].0)).collect();
    }

    // Quantile bins: walk the distinct values and close a bin once it holds
    // its share of the rows. A cut always falls between two distinct values,
    // and the share is recomputed after every cut, so a heavily repeated
    // value takes one bin without starving the values after it.
    let n_rows = sorted.len();
    let mut cuts = Vec::with_capacity(MAX_BINS - 1);
    let mut bin_end = n_rows / MAX_BINS;
    for pair in distinct.windows(2) {
        let ((value, rows_so_far), (next_value, _)) = (pair[0], pair[1]);
        if rows_so_far >= bin_end {
            cuts.push(midpoint(value, next_value));
            let bins_left = MAX_BINS - cuts.len();
            if bins_left == 1 {
                break;
            }
            bin_end = rows_so_far + (n_rows - rows_so_far) / bins_left;
        }
    }
    cuts
}
