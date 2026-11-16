//! Splits and searches: the time-series split and random search.

use mlib::prelude::*;
use mlib::random::Rng;

/// A noisy line in the first of two features.
fn line(n: usize) -> EncodedData {
    let mut rng = Rng::new(7, 0);
    let x = Array2::from_shape_fn((n, 2), |_| rng.next_u64() as f64 / u64::MAX as f64);
    let y = Array2::from_shape_fn((n, 1), |(row, _)| 3.0 * x[[row, 0]] + 0.1 * x[[row, 1]]);
    EncodedData::new(x, y).unwrap()
}

fn range(from: usize, to: usize) -> Vec<usize> {
    (from..to).collect()
}

#[test]
fn a_time_series_split_matches_scikit_learn() {
    // Reference: scikit-learn 1.9.1, TimeSeriesSplit.
    let folds = TimeSeriesSplit::new(3).splits(10);
    assert_eq!(folds, [(range(0, 4), range(4, 6)), (range(0, 6), range(6, 8)), (range(0, 8), range(8, 10))]);

    // 11 rows: the test blocks still hold 11 / 4 = 2 rows and end at the last row.
    let folds = TimeSeriesSplit::new(3).splits(11);
    assert_eq!(folds, [(range(0, 5), range(5, 7)), (range(0, 7), range(7, 9)), (range(0, 9), range(9, 11))]);

    let folds = TimeSeriesSplit::new(2).with_test_size(3).with_gap(2).splits(12);
    assert_eq!(folds, [(range(0, 4), range(6, 9)), (range(0, 7), range(9, 12))]);

    let folds = TimeSeriesSplit::new(3).with_max_train_size(4).splits(12);
    assert_eq!(folds, [(range(0, 3), range(3, 6)), (range(2, 6), range(6, 9)), (range(5, 9), range(9, 12))]);
}

#[test]
fn a_time_series_split_never_trains_on_the_future() {
    for (n_splits, n_rows, gap) in [(2, 7, 0), (5, 103, 0), (4, 50, 3), (10, 1000, 7)] {
        let folds = TimeSeriesSplit::new(n_splits).with_gap(gap).splits(n_rows);
        assert_eq!(folds.len(), n_splits);
        for (train, test) in &folds {
            assert!(!train.is_empty() && !test.is_empty());
            assert!(train.last().unwrap() + gap < test[0]);
        }
        // The last fold is tested on the latest rows.
        assert_eq!(*folds[n_splits - 1].1.last().unwrap(), n_rows - 1);
    }
}

#[test]
fn a_time_series_split_that_cannot_be_made_gives_no_folds() {
    assert!(TimeSeriesSplit::new(1).splits(100).is_empty());
    assert!(TimeSeriesSplit::new(5).splits(5).is_empty());
    assert!(TimeSeriesSplit::new(2).with_test_size(5).splits(10).is_empty());
    assert!(TimeSeriesSplit::new(2).with_test_size(0).splits(10).is_empty());
    assert!(TimeSeriesSplit::new(2).with_gap(8).splits(10).is_empty());
    assert!(TimeSeriesSplit::new(2).with_max_train_size(0).splits(10).is_empty());
    // Cross-validation then says so.
    let model = LinearModelSpec::regressor().build().unwrap();
    let no_folds = TimeSeriesSplit::new(5).splits(5);
    assert!(cross_validate(&model, &line(5), &no_folds, &*scorer("r2").unwrap()).is_err());
}

#[test]
fn a_time_series_split_drives_cross_validation() {
    let data = line(200);
    let model = LinearModelSpec::regressor().build().unwrap();
    let folds = TimeSeriesSplit::new(4).splits(200);
    let cv = cross_validate(&model, &data, &folds, &*scorer("r2").unwrap()).unwrap();
    assert_eq!(cv.scores.len(), 4);
    assert!(cv.mean > 0.999);
}

#[test]
fn a_random_search_draws_within_the_ranges_and_is_reproducible() {
    let data = line(120);
    let model = LinearModelSpec::regressor().build().unwrap();
    let space = ParamDistributions::from([
        ("l2_regularization".to_string(), ParamDistribution::LogUniform(0.001, 1000.0)),
        ("max_iterations".to_string(), ParamDistribution::IntRange(1, 3)),
        ("tolerance".to_string(), ParamDistribution::Uniform(1e-6, 1e-3)),
    ]);
    let folds = KFold::new(3).with_shuffle(1).splits(120);
    let r2 = scorer("r2").unwrap();
    let result = random_search(&model, &space, 60, 11, &data, &folds, &*r2).unwrap();
    assert_eq!(result.all_scores.len(), 60);

    let number = |params: &Params, name: &str| match &params[name] {
        ParamValue::Float(value) => *value,
        ParamValue::Int(value) => *value as f64,
        other => panic!("{name} is {other:?}"),
    };
    let drawn = |name: &str| result.all_scores.iter().map(|(params, _)| number(params, name)).collect::<Vec<f64>>();
    assert!(drawn("l2_regularization").iter().all(|l2| (0.001..=1000.0).contains(l2)));
    assert!(drawn("tolerance").iter().all(|tolerance| (1e-6..=1e-3).contains(tolerance)));
    // A whole-number range includes both its ends.
    for value in [1.0, 2.0, 3.0] {
        assert!(drawn("max_iterations").contains(&value));
    }
    assert!(drawn("max_iterations").iter().all(|n| [1.0, 2.0, 3.0].contains(n)));
    // Drawn on a log scale, about half the penalties fall below the middle of the scale (1).
    let below_one = drawn("l2_regularization").iter().filter(|l2| **l2 < 1.0).count();
    assert!((20..=40).contains(&below_one), "{below_one} of 60");

    // The best is the candidate with the highest score, and less penalty fits this line better.
    let highest = result.all_scores.iter().map(|(_, score)| *score).fold(f64::MIN, f64::max);
    assert_eq!(result.best_score, highest);
    assert!(number(&result.best_params, "l2_regularization") < 1.0);

    // The same seed gives the same search; another seed, other candidates.
    assert_eq!(result, random_search(&model, &space, 60, 11, &data, &folds, &*r2).unwrap());
    assert_ne!(result.all_scores, random_search(&model, &space, 60, 12, &data, &folds, &*r2).unwrap().all_scores);
    // Asking for more candidates keeps the earlier ones.
    let fewer = random_search(&model, &space, 5, 11, &data, &folds, &*r2).unwrap();
    assert_eq!(fewer.all_scores[..], result.all_scores[..5]);
}

#[test]
fn a_random_search_chooses_among_listed_values() {
    let x = Array2::from_shape_fn((200, 1), |(row, _)| ((row * 7) % 40) as f64);
    let y = Array2::from_shape_fn((200, 1), |(row, _)| if row % 8 == 0 { 1.0 } else { 0.0 });
    let data = EncodedData::new(x, y).unwrap();
    let model = RandomForestSpec::default().n_trees(5).build().unwrap();
    let space = ParamDistributions::from([
        ("class_weight".to_string(), ParamDistribution::Choice(vec!["none".into(), "balanced".into()])),
        ("max_depth".to_string(), ParamDistribution::Choice(vec![2.into(), "none".into()])),
    ]);
    let folds = StratifiedKFold::new(2).splits(&data.target().unwrap(), 200);
    let result = random_search(&model, &space, 20, 3, &data, &folds, &*scorer("balanced_accuracy").unwrap()).unwrap();
    for value in ["none", "balanced"] {
        assert!(result.all_scores.iter().any(|(params, _)| params["class_weight"] == value.into()));
    }
}

#[test]
fn a_search_that_cannot_run_is_an_error() {
    let data = line(60);
    let model = LinearModelSpec::regressor().build().unwrap();
    let folds = KFold::new(3).splits(60);
    let r2 = scorer("r2").unwrap();
    let search = |name: &str, distribution: ParamDistribution, n_candidates: usize| {
        let space = ParamDistributions::from([(name.to_string(), distribution)]);
        random_search(&model, &space, n_candidates, 1, &data, &folds, &*r2)
    };
    assert!(search("l2_regularization", ParamDistribution::Uniform(0.0, 1.0), 3).is_ok());
    // No candidates; nothing to draw from; a range the wrong way round; a log scale from zero.
    assert!(search("l2_regularization", ParamDistribution::Uniform(0.0, 1.0), 0).is_err());
    assert!(search("l2_regularization", ParamDistribution::Choice(vec![]), 3).is_err());
    assert!(search("l2_regularization", ParamDistribution::Uniform(2.0, 1.0), 3).is_err());
    assert!(search("l2_regularization", ParamDistribution::LogUniform(0.0, 1.0), 3).is_err());
    assert!(search("max_iterations", ParamDistribution::IntRange(5, 1), 3).is_err());
    // A parameter the model does not have, and a value it rejects.
    assert!(search("no_such_parameter", ParamDistribution::Uniform(0.0, 1.0), 3).is_err());
    assert!(search("l2_regularization", ParamDistribution::Uniform(-2.0, -1.0), 3).is_err());

    // A grid with a parameter that has no values used to panic.
    let empty = ParamGrid::from([("l2_regularization".to_string(), vec![])]);
    assert!(grid_search(&model, &empty, &data, &folds, &*r2).is_err());
}

#[test]
fn a_random_search_reaches_into_a_pipeline() {
    let ages: Vec<f64> = (0..120).map(|row| 20.0 + ((row * 37) % 60) as f64).collect();
    let churned: Vec<i64> = ages.iter().map(|&age| (age >= 60.0) as i64).collect();
    let table = Table::new(vec![("age", Column::floats(&ages)), ("churned", Column::ints(&churned))]).unwrap();
    let data = ModelData::new(&table, ModelSpec::new(&["age"], &["churned"])).unwrap();
    let pre = ColumnTransformer::empty().with_branch("num", Imputer::new(ImputeStrategy::Mean), &["age"]);
    let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(RandomForestSpec::default().n_trees(5).build().unwrap()));

    let space = ParamDistributions::from([
        ("model__max_depth".to_string(), ParamDistribution::IntRange(1, 3)),
        ("num__strategy".to_string(), ParamDistribution::Choice(vec!["mean".into(), "median".into()])),
    ]);
    let folds = TimeSeriesSplit::new(3).splits(120);
    let result = random_search_pipeline(&pipe, &space, 6, 2, &data, &folds, &*scorer("roc_auc").unwrap()).unwrap();
    assert_eq!(result.all_scores.len(), 6);
    assert!(result.best_score > 0.99);
    assert!(result.best_params.contains_key("model__max_depth") && result.best_params.contains_key("num__strategy"));
}
