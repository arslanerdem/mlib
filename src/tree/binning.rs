//! Feature binning: each feature's values are replaced by small integer
//! bin codes (at most 256 bins), computed once per forest.
//!
//! Trees then search for splits over bins instead of raw values, which
//! turns split finding into cheap histogram sums. A feature with at most
//! 256 distinct values gets one bin per value, so nothing is lost; a
//! feature with more gets quantile bins.
//!
//! A missing value (NaN) belongs to no range of values, so a feature
//! that has any gets one more bin, after all the others, just for them:
//!
//! ```text
//! bins:   0 | 1 | 2 | ... | last value bin | missing
//! ```
//!
//! The split search can then try the missing rows on either side.

use ndarray::Array2;
use rayon::prelude::*;

pub const MAX_BINS: usize = 256;

pub struct BinnedFeatures {
    /// `bins[feature][row]`: the bin code of that value.
    bins: Vec<Vec<u8>>,
    /// `thresholds[feature][b]`: the upper edge of bin `b`. A value falls
    /// in bin `b` or lower exactly when `value <= thresholds[feature][b]`.
    thresholds: Vec<Vec<f64>>,
    /// `missing_bin[feature]`: the bin that holds the feature's missing
    /// values — the one after its last value bin. `None` if no value of
    /// the feature is missing.
    missing_bin: Vec<Option<u8>>,
}

/// One feature's bin codes, thresholds and missing bin.
type BinnedFeature = (Vec<u8>, Vec<f64>, Option<u8>);

impl BinnedFeatures {
    pub fn new(x: &Array2<f64>) -> BinnedFeatures {
        let binned: Vec<BinnedFeature> = (0..x.ncols())
            .into_par_iter()
            .map(|feature| {
                let values = x.column(feature).to_vec();
                if !values.iter().any(|value| value.is_nan()) {
                    let thresholds = cut_points(&values, MAX_BINS);
                    let bin_of = |&value: &f64| thresholds.partition_point(|&t| t < value) as u8;
                    return (values.iter().map(bin_of).collect(), thresholds, None);
                }
                // One bin is kept for the missing values.
                let present: Vec<f64> = values.iter().copied().filter(|value| !value.is_nan()).collect();
                let thresholds = cut_points(&present, MAX_BINS - 1);
                let missing_bin = thresholds.len() as u8 + 1;
                let bin_of = |&value: &f64| {
                    if value.is_nan() {
                        missing_bin
                    } else {
                        thresholds.partition_point(|&t| t < value) as u8
                    }
                };
                (values.iter().map(bin_of).collect(), thresholds, Some(missing_bin))
            })
            .collect();
        let mut features = BinnedFeatures { bins: Vec::new(), thresholds: Vec::new(), missing_bin: Vec::new() };
        for (bins, thresholds, missing_bin) in binned {
            features.bins.push(bins);
            features.thresholds.push(thresholds);
            features.missing_bin.push(missing_bin);
        }
        features
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
        // A cut after the last value bin has every value on its left and
        // only the missing ones on its right: the threshold is the
        // largest number there is. (Not infinity, which a saved model
        // could not hold.)
        self.thresholds[feature].get(bin as usize).copied().unwrap_or(f64::MAX)
    }

    /// The bin that holds the feature's missing values, if it has any.
    pub fn missing_bin(&self, feature: usize) -> Option<u8> {
        self.missing_bin[feature]
    }
}

/// The increasing cut points that separate a feature's values into at
/// most `max_bins` bins.
fn cut_points(values: &[f64], max_bins: usize) -> Vec<f64> {
    // A point between two distinct neighbours: lo <= midpoint < hi.
    fn midpoint(lo: f64, hi: f64) -> f64 {
        let mid = lo + (hi - lo) / 2.0;
        if mid < hi {
            mid
        } else {
            lo
        }
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

    if distinct.len() <= max_bins {
        // One bin per distinct value: exact.
        return distinct.windows(2).map(|pair| midpoint(pair[0].0, pair[1].0)).collect();
    }

    // Quantile bins: walk the distinct values and close a bin once it holds
    // its share of the rows. A cut always falls between two distinct values,
    // and the share is recomputed after every cut, so a heavily repeated
    // value takes one bin without starving the values after it.
    let n_rows = sorted.len();
    let mut cuts = Vec::with_capacity(max_bins - 1);
    let mut bin_end = n_rows / max_bins;
    for pair in distinct.windows(2) {
        let ((value, rows_so_far), (next_value, _)) = (pair[0], pair[1]);
        if rows_so_far >= bin_end {
            cuts.push(midpoint(value, next_value));
            let bins_left = max_bins - cuts.len();
            if bins_left == 1 {
                break;
            }
            bin_end = rows_so_far + (n_rows - rows_so_far) / bins_left;
        }
    }
    cuts
}
