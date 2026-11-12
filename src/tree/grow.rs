//! Growing one decision tree.
//!
//! The tree's training rows are kept as one array of [`Sample`]s, and
//! every node owns a contiguous range of it. Splitting a node rearranges
//! its range so the samples going left come first, keeping their original
//! order — rows are then always visited in increasing order, which is what
//! memory likes best.

use super::binning::BinnedFeatures;
use super::histogram::{Cut, Histogram, Sample, Targets, Totals};
use super::{Tree, TreeNode, LEAF};
use crate::random::Rng;
use rayon::prelude::*;

/// The hyper-parameters one tree needs.
pub(crate) struct TreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    /// How many randomly chosen features to try at each split.
    pub features_per_split: usize,
    pub bootstrap: bool,
    /// Whether the tree may spread its own work over the cores. Turn it
    /// off when many trees are grown at the same time.
    pub parallel: bool,
}

/// Grow one tree. Returns the tree and each feature's importance (its
/// share of the tree's total impurity reduction).
pub(crate) fn grow_tree(
    binned: &BinnedFeatures,
    targets: &Targets,
    sample_weights: Option<&[f64]>,
    params: &TreeParams,
    rng: Rng,
) -> (Tree, Vec<f64>) {
    let mut builder = TreeBuilder::new(binned, targets, sample_weights, params, rng);
    builder.grow();

    let mut importances = builder.importances;
    let total: f64 = importances.iter().sum();
    if total > 0.0 {
        importances.iter_mut().for_each(|importance| *importance /= total);
    }
    let tree = Tree { nodes: builder.nodes, leaf_values: builder.leaf_values, n_outputs: targets.n_channels() };
    (tree, importances)
}

/// A split must beat the unsplit node's score by more than rounding noise.
const MIN_IMPROVEMENT: f64 = 1.0 + 1e-10;

/// When a node has at least this many (samples × features) to examine,
/// its features are examined in parallel.
const PARALLEL_WORK: usize = 60_000;

/// A node with at least this many samples is rearranged in parallel.
const PARALLEL_PARTITION: usize = 50_000;

/// A node with at least this many samples keeps its histograms so that
/// its children can reuse them (see `TreeBuilder::child_histograms`).
/// Below this, filling a fresh histogram is cheaper than keeping one.
const KEEP_HISTOGRAMS: usize = 1024;

/// A node waiting to be built: it owns `samples[start..end]`.
struct PendingNode {
    index: usize,
    start: usize,
    end: usize,
    depth: usize,
    /// Totals over the node's samples.
    totals: Totals,
    /// The first `n_constant` entries of `feature_order` are features
    /// already known to have a single value across this node's samples.
    n_constant: usize,
    /// One filled histogram per feature (indexed by feature), when they
    /// were prepared ahead; only the features not known to be constant
    /// are filled. `None` means: fill them when the node is examined.
    histograms: Option<Vec<Histogram>>,
}

struct Split {
    feature: usize,
    bin: u8,
    score: f64,
}

impl Split {
    fn new(feature: usize, cut: Cut) -> Split {
        Split { feature, bin: cut.bin, score: cut.score }
    }
}

struct TreeBuilder<'a> {
    binned: &'a BinnedFeatures,
    targets: &'a Targets,
    params: &'a TreeParams,
    rng: Rng,
    /// The rows in this tree's sample, in row order within every node.
    samples: Vec<Sample>,
    /// Working space for `partition`, as long as `samples`.
    scratch: Vec<Sample>,
    nodes: Vec<TreeNode>,
    leaf_values: Vec<f64>,
    importances: Vec<f64>,
    histogram: Histogram,
    /// Emptied histogram sets waiting to be reused, so that large nodes
    /// do not allocate new ones.
    spare_histograms: Vec<Vec<Histogram>>,
    /// All feature indices, reordered as the tree grows. A feature with a
    /// single value across a node's samples cannot split that node or any
    /// node below it, so such features are moved to the front and each
    /// node records how many there are (`PendingNode::n_constant`).
    feature_order: Vec<usize>,
}

impl<'a> TreeBuilder<'a> {
    fn new(
        binned: &'a BinnedFeatures,
        targets: &'a Targets,
        sample_weights: Option<&[f64]>,
        params: &'a TreeParams,
        mut rng: Rng,
    ) -> TreeBuilder<'a> {
        let n_rows = binned.bins(0).len();
        let weight_of = |row: usize| sample_weights.map_or(1.0, |weights| weights[row]);

        // Rows with zero weight cannot influence the tree; leave them out.
        let usable: Vec<u32> = (0..n_rows).filter(|&row| weight_of(row) > 0.0).map(|row| row as u32).collect();
        // Per row: how many times it is in this tree's sample.
        let mut count = vec![0.0; n_rows];
        if params.bootstrap {
            for _ in 0..usable.len() {
                count[usable[rng.below(usable.len())] as usize] += 1.0;
            }
        } else {
            usable.iter().for_each(|&row| count[row as usize] = 1.0);
        }
        let in_sample: Vec<u32> = usable.into_iter().filter(|&row| count[row as usize] > 0.0).collect();
        let sample_of = |&row: &u32| {
            let count = count[row as usize];
            let (weight, target) = (count * weight_of(row as usize), targets.of_row(row as usize));
            Sample { weight, target, row, count: count as u32 }
        };
        let samples: Vec<Sample> = in_sample.par_iter().map(sample_of).collect();

        TreeBuilder {
            binned,
            targets,
            params,
            rng,
            scratch: samples.clone(),
            samples,
            nodes: Vec::new(),
            leaf_values: Vec::new(),
            importances: vec![0.0; binned.n_features()],
            histogram: Histogram::new(targets.n_channels()),
            spare_histograms: Vec::new(),
            feature_order: (0..binned.n_features()).collect(),
        }
    }

    fn grow(&mut self) {
        // Every node is reserved as a placeholder and filled in when built.
        let placeholder = TreeNode { threshold: 0.0, feature: LEAF, left_or_leaf: 0 };
        self.nodes.push(placeholder.clone());
        let root = PendingNode {
            index: 0,
            start: 0,
            end: self.samples.len(),
            depth: 0,
            totals: self.targets.totals(&self.samples),
            n_constant: 0,
            histograms: None,
        };
        let mut pending = vec![root];

        while let Some(mut node) = pending.pop() {
            let (split, n_constant) = if self.may_split(node.depth, &node.totals) && !self.is_pure(&node) {
                self.best_split(&mut node)
            } else {
                (None, node.n_constant)
            };
            let Some(split) = split else {
                let leaf_number = self.leaf_values.len() / self.targets.n_channels();
                self.leaf_values.extend(self.targets.leaf_prediction(&node.totals));
                self.nodes[node.index] = TreeNode { threshold: 0.0, feature: LEAF, left_or_leaf: leaf_number as u32 };
                if let Some(histograms) = node.histograms.take() {
                    self.recycle(histograms);
                }
                continue;
            };

            self.importances[split.feature] += split.score - node.totals.score();
            let middle = node.start + self.partition(&node, &split);
            // The two children sit next to each other: left, then right.
            let left = self.nodes.len();
            self.nodes.push(placeholder.clone());
            self.nodes.push(placeholder.clone());
            self.nodes[node.index] = TreeNode {
                threshold: self.binned.threshold(split.feature, split.bin),
                feature: split.feature as u32,
                left_or_leaf: left as u32,
            };
            // Add up one side; the other is what remains of the node's totals.
            let left_totals = self.targets.totals(&self.samples[node.start..middle]);
            let right_totals = node.totals.without(&left_totals);
            let depth = node.depth + 1;
            let mut left_node = PendingNode {
                index: left,
                start: node.start,
                end: middle,
                depth,
                totals: left_totals,
                n_constant,
                histograms: None,
            };
            let mut right_node = PendingNode {
                index: left + 1,
                start: middle,
                end: node.end,
                depth,
                totals: right_totals,
                n_constant,
                histograms: None,
            };
            if let Some(histograms) = node.histograms.take() {
                self.child_histograms(histograms, &mut left_node, &mut right_node);
            }
            pending.push(right_node);
            pending.push(left_node);
        }
    }

    /// Whether a node at this depth with these totals is allowed to split.
    fn may_split(&self, depth: usize, totals: &Totals) -> bool {
        let depth_allows = self.params.max_depth.is_none_or(|max_depth| depth < max_depth);
        depth_allows && totals.count >= 2.0 * self.params.min_samples_leaf as f64
    }

    /// Whether a node is large enough to keep its histograms for its children.
    fn keeps_histograms(&self, node: &PendingNode) -> bool {
        node.end - node.start >= KEEP_HISTOGRAMS && self.may_split(node.depth, &node.totals)
    }

    /// Fill one histogram per feature the node could still split on.
    fn fill_histograms(&self, histograms: &mut [Histogram], node: &PendingNode) {
        let samples = &self.samples[node.start..node.end];
        let mut is_candidate = vec![false; histograms.len()];
        self.feature_order[node.n_constant..].iter().for_each(|&feature| is_candidate[feature] = true);
        let fill = |(feature, histogram): (usize, &mut Histogram)| {
            if is_candidate[feature] {
                histogram.fill(self.binned.bins(feature), samples, self.targets);
            }
        };
        if self.params.parallel && samples.len() * histograms.len() >= PARALLEL_WORK {
            histograms.par_iter_mut().enumerate().for_each(fill);
        } else {
            histograms.iter_mut().enumerate().for_each(fill);
        }
    }

    /// An empty set of histograms, one per feature.
    fn empty_histograms(&mut self) -> Vec<Histogram> {
        let n_channels = self.targets.n_channels();
        let new = || (0..self.binned.n_features()).map(|_| Histogram::new(n_channels)).collect();
        self.spare_histograms.pop().unwrap_or_else(new)
    }

    /// Empty a set of histograms and keep it for reuse.
    fn recycle(&mut self, mut histograms: Vec<Histogram>) {
        histograms.iter_mut().for_each(Histogram::clear);
        self.spare_histograms.push(histograms);
    }

    /// Hand a split node's histograms on to its children.
    ///
    /// Every sample of the parent went to exactly one child, so
    /// `parent = left + right` in every bin. Only the smaller child's
    /// histograms are filled by visiting its samples; the larger child's
    /// are what is left after taking those out of the parent's. That at
    /// least halves the work of the most expensive step in growing a tree.
    fn child_histograms(&mut self, parent: Vec<Histogram>, left: &mut PendingNode, right: &mut PendingNode) {
        let (smaller, larger) = if left.end - left.start <= right.end - right.start { (left, right) } else { (right, left) };
        let (smaller_keeps, larger_keeps) = (self.keeps_histograms(smaller), self.keeps_histograms(larger));
        if !larger_keeps {
            // The smaller child is then too small to keep any either.
            return self.recycle(parent);
        }
        let mut of_smaller = self.empty_histograms();
        self.fill_histograms(&mut of_smaller, smaller);
        let mut of_larger = parent;
        for (remaining, part) in of_larger.iter_mut().zip(&of_smaller) {
            remaining.subtract(part);
        }
        larger.histograms = Some(of_larger);
        if smaller_keeps {
            smaller.histograms = Some(of_smaller);
        } else {
            self.recycle(of_smaller);
        }
    }

    /// Whether all of a node's samples share one target, so that no split
    /// could improve it.
    fn is_pure(&self, node: &PendingNode) -> bool {
        let samples = &self.samples[node.start..node.end];
        samples.iter().all(|sample| self.targets.same(sample, &samples[0]))
    }

    /// The best split of a node, together with the number of features
    /// now known to be constant across its samples.
    fn best_split(&mut self, node: &mut PendingNode) -> (Option<Split>, usize) {
        if self.params.features_per_split < self.feature_order.len() {
            return self.best_split_over_random_features(node);
        }
        if node.histograms.is_none() && self.keeps_histograms(node) {
            // Only the root arrives here: every other large node was
            // given its histograms when its parent was split.
            let mut histograms = self.empty_histograms();
            self.fill_histograms(&mut histograms, node);
            node.histograms = Some(histograms);
        }
        match &node.histograms {
            Some(histograms) => self.best_split_from_histograms(node, histograms),
            None => self.best_split_over_all_features(node),
        }
    }

    /// The best split over every feature, read from histograms that are
    /// already filled.
    fn best_split_from_histograms(&mut self, node: &PendingNode, histograms: &[Histogram]) -> (Option<Split>, usize) {
        let min_samples_leaf = self.params.min_samples_leaf as f64;
        let mut best: Option<Split> = None;
        let mut score_to_beat = node.totals.score() * MIN_IMPROVEMENT;
        let mut n_constant = node.n_constant;
        for position in node.n_constant..self.feature_order.len() {
            let feature = self.feature_order[position];
            let histogram = &histograms[feature];
            if histogram.n_occupied_bins() == 1 {
                self.feature_order.swap(position, n_constant);
                n_constant += 1;
            } else if let Some(cut) = histogram.best_cut(&node.totals, min_samples_leaf, score_to_beat) {
                // On equal scores the first feature examined wins.
                score_to_beat = cut.score;
                best = Some(Split::new(feature, cut));
            }
        }
        (best, n_constant)
    }

    /// Examine every feature that could split a small node, filling a
    /// histogram for each in turn. For a large node the features are
    /// examined in parallel.
    fn best_split_over_all_features(&mut self, node: &PendingNode) -> (Option<Split>, usize) {
        let samples = &self.samples[node.start..node.end];
        let (binned, targets) = (self.binned, self.targets);
        let min_samples_leaf = self.params.min_samples_leaf as f64;
        let score_to_beat = node.totals.score() * MIN_IMPROVEMENT;

        // `None`: the feature is constant here. Otherwise its best cut, if it has one.
        let examine = |histogram: &mut Histogram, feature: &usize| {
            histogram.fill(binned.bins(*feature), samples, targets);
            let varies = histogram.n_occupied_bins() > 1;
            let found = varies.then(|| histogram.best_cut(&node.totals, min_samples_leaf, score_to_beat));
            histogram.clear();
            found
        };
        let candidates = &self.feature_order[node.n_constant..];
        let examined: Vec<Option<Option<Cut>>> = if self.params.parallel && samples.len() * candidates.len() >= PARALLEL_WORK {
            candidates.par_iter().map_init(|| Histogram::new(targets.n_channels()), examine).collect()
        } else {
            candidates.iter().map(|feature| examine(&mut self.histogram, feature)).collect()
        };

        let mut best: Option<Split> = None;
        let mut n_constant = node.n_constant;
        for (position, found) in (node.n_constant..).zip(examined) {
            match found {
                None => {
                    self.feature_order.swap(position, n_constant);
                    n_constant += 1;
                }
                // On equal scores the first feature examined wins.
                Some(Some(cut)) if best.as_ref().is_none_or(|best| cut.score > best.score) => {
                    best = Some(Split::new(self.feature_order[position], cut));
                }
                Some(_) => {}
            }
        }
        (best, n_constant)
    }

    /// Try a random subset of features and return the best split found.
    ///
    /// Features are drawn without replacement, and every draw counts
    /// towards `features_per_split`. If no valid split has been found by
    /// then, drawing continues until one is or the features run out (the
    /// same rule scikit-learn uses).
    ///
    /// During the search `feature_order` has three parts:
    ///
    /// ```text
    /// [ constant here | not yet examined | examined, not constant ]
    ///   0..n_constant   n_constant..end    end..
    /// ```
    fn best_split_over_random_features(&mut self, node: &PendingNode) -> (Option<Split>, usize) {
        let min_samples_leaf = self.params.min_samples_leaf as f64;
        let mut best: Option<Split> = None;
        let mut score_to_beat = node.totals.score() * MIN_IMPROVEMENT;

        let mut n_constant = node.n_constant;
        let mut end = self.feature_order.len();
        // Features the node inherited as constant and that no draw has hit yet.
        let mut undrawn_constants = node.n_constant;
        let mut draws = 0;

        while end > n_constant && (draws < self.params.features_per_split || best.is_none()) {
            draws += 1;
            let pick = self.rng.below(undrawn_constants + end - n_constant);
            if pick < undrawn_constants {
                // Drew a feature already known to be constant: nothing to examine.
                undrawn_constants -= 1;
                continue;
            }
            let position = n_constant + pick - undrawn_constants;
            let feature = self.feature_order[position];

            let samples = &self.samples[node.start..node.end];
            self.histogram.fill(self.binned.bins(feature), samples, self.targets);
            if self.histogram.n_occupied_bins() == 1 {
                self.feature_order.swap(position, n_constant);
                n_constant += 1;
            } else {
                end -= 1;
                self.feature_order.swap(position, end);
                if let Some(cut) = self.histogram.best_cut(&node.totals, min_samples_leaf, score_to_beat) {
                    score_to_beat = cut.score;
                    best = Some(Split::new(feature, cut));
                }
            }
            self.histogram.clear();
        }
        (best, n_constant)
    }

    /// Rearrange a node's samples so those going left come first, keeping
    /// the order within each side. Returns how many go left.
    ///
    /// The samples are copied to their new places in `scratch` and then
    /// copied back. A large node is handled as several chunks in
    /// parallel: once it is known how many samples of each chunk go left,
    /// every chunk has its own two stretches of `scratch` to write to.
    fn partition(&mut self, node: &PendingNode, split: &Split) -> usize {
        let bins = self.binned.bins(split.feature);
        let goes_left = |sample: &Sample| bins[sample.row as usize] <= split.bin;
        // Copy a run of samples into the space reserved for each side.
        let move_apart = |samples: &[Sample], left: &mut [Sample], right: &mut [Sample]| {
            let (mut left, mut right) = (left.iter_mut(), right.iter_mut());
            for sample in samples {
                // Invariant: each side has exactly as much space as samples going there.
                *(if goes_left(sample) { left.next() } else { right.next() }).unwrap() = *sample;
            }
        };

        let samples = &mut self.samples[node.start..node.end];
        let scratch = &mut self.scratch[node.start..node.end];
        if !self.params.parallel || samples.len() < PARALLEL_PARTITION {
            let n_left = samples.iter().filter(|sample| goes_left(sample)).count();
            let (left, right) = scratch.split_at_mut(n_left);
            move_apart(samples, left, right);
            samples.copy_from_slice(scratch);
            return n_left;
        }

        let chunk = samples.len().div_ceil(rayon::current_num_threads());
        let n_left_per_chunk: Vec<usize> =
            samples.par_chunks(chunk).map(|chunk| chunk.iter().filter(|sample| goes_left(sample)).count()).collect();
        let n_left = n_left_per_chunk.iter().sum();

        // Hand every chunk its own stretch of each side.
        let (mut left_space, mut right_space) = scratch.split_at_mut(n_left);
        let mut jobs = Vec::new();
        for (chunk, n_left) in samples.chunks(chunk).zip(n_left_per_chunk) {
            let (left, rest) = left_space.split_at_mut(n_left);
            left_space = rest;
            let (right, rest) = right_space.split_at_mut(chunk.len() - n_left);
            right_space = rest;
            jobs.push((chunk, left, right));
        }
        jobs.into_par_iter().for_each(|(chunk, left, right)| move_apart(chunk, left, right));
        samples.par_chunks_mut(chunk).zip(scratch.par_chunks(chunk)).for_each(|(to, from)| to.copy_from_slice(from));
        n_left
    }
}
