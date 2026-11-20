//! Missing values inside the trees: the forest and gradient boosting
//! learn, at every split, which side rows with a missing value belong on.

use mlib::prelude::*;
use mlib::random::Rng;
use mlib::testing::check_estimator;

const NAN: f64 = f64::NAN;

const HOUSING_INPUTS: [&str; 8] =
    ["median_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude"];

fn housing() -> EncodedData {
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let study = ModelData::new(&table, ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"])).unwrap();
    EncodedData::from_table(&study.x_table(), study.input_features(), study.y_matrix().unwrap(), None).unwrap()
}

/// California housing with holes in three features: income and rooms at
/// random, and the age of the houses whenever it is above 40 — so that
/// "missing" itself says something.
fn housing_with_holes() -> EncodedData {
    let mut data = housing();
    for row in 0..data.num_rows() {
        if row % 7 == 3 {
            data.x[[row, 0]] = NAN;
        }
        if data.x[[row, 1]] > 40.0 {
            data.x[[row, 1]] = NAN;
        }
        if row % 5 == 0 {
            data.x[[row, 2]] = NAN;
        }
    }
    data
}

/// The first 8,000 rows and the last 2,000.
fn split(data: &EncodedData) -> (EncodedData, EncodedData) {
    (data.take_rows(&(0..8000).collect::<Vec<_>>()), data.take_rows(&(8000..10000).collect::<Vec<_>>()))
}

/// Every missing value replaced by its feature's median among the training rows.
fn median_imputed(train: &EncodedData, data: &EncodedData) -> EncodedData {
    let mut filled = data.clone();
    for feature in 0..data.num_features() {
        let mut present: Vec<f64> = train.x.column(feature).iter().copied().filter(|value| !value.is_nan()).collect();
        present.sort_unstable_by(f64::total_cmp);
        let median = (present[(present.len() - 1) / 2] + present[present.len() / 2]) / 2.0;
        filled.x.column_mut(feature).mapv_inplace(|value| if value.is_nan() { median } else { value });
    }
    filled
}

fn one_feature(x: &[f64], y: &[f64]) -> EncodedData {
    let x = Array2::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
    EncodedData::new(x, Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap()).unwrap()
}

/// One tree on all the rows, with one split.
fn stump(data: &EncodedData) -> FittedRandomForest {
    RandomForestSpec::regressor().n_trees(1).bootstrap(false).max_depth(Some(1)).build().unwrap().fit(data).unwrap()
}

fn squared_error(model: &dyn Predictor, data: &EncodedData) -> f64 {
    let predicted = model.predict_single(data).unwrap();
    data.target().unwrap().iter().zip(&predicted).map(|(truth, prediction)| (truth - prediction).powi(2)).sum()
}

/// The lowest squared error any one split can reach, trying every cut
/// between two values with the missing rows on either side.
fn best_possible_squared_error(x: &[f64], y: &[f64]) -> f64 {
    let error_of = |group: &[f64]| {
        let mean = group.iter().sum::<f64>() / group.len() as f64;
        group.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
    };
    let mut values: Vec<f64> = x.iter().copied().filter(|value| !value.is_nan()).collect();
    values.sort_unstable_by(f64::total_cmp);
    // Cutting at the largest value puts every value on the left: "missing against the rest".
    let mut best = error_of(y);
    for &cut in &values {
        for missing_left in [false, true] {
            let goes_left = |value: f64| if value.is_nan() { missing_left } else { value <= cut };
            let side = |left: bool| -> Vec<f64> {
                x.iter().zip(y).filter(|(value, _)| goes_left(**value) == left).map(|(_, target)| *target).collect()
            };
            let (left, right) = (side(true), side(false));
            if !left.is_empty() && !right.is_empty() {
                best = best.min(error_of(&left) + error_of(&right));
            }
        }
    }
    best
}

#[test]
fn the_tree_models_still_pass_the_conformance_battery() {
    check_estimator(&RandomForestSpec::default().n_trees(10).build().unwrap());
    check_estimator(&RandomForestSpec::regressor().n_trees(10).build().unwrap());
    check_estimator(&GradientBoostingSpec::default().n_rounds(10).min_samples_leaf(2).build().unwrap());
    check_estimator(&GradientBoostingSpec::regressor().n_rounds(10).min_samples_leaf(2).build().unwrap());
    // The linear model does not take missing values, and the battery checks that instead.
    check_estimator(&LinearModelSpec::default().build().unwrap());
    assert!(RandomForestSpec::default().build().unwrap().accepts_missing());
    assert!(GradientBoostingSpec::default().build().unwrap().accepts_missing());
    assert!(!LinearModelSpec::default().build().unwrap().accepts_missing());
}

#[test]
fn missing_values_go_to_the_side_where_they_fit() {
    // The rows with a missing value are like the low values (target 0) ...
    let x: Vec<f64> = (0..90).map(|row| if row % 3 == 0 { NAN } else { row as f64 }).collect();
    let low: Vec<f64> = x.iter().map(|&value| if value >= 45.0 { 1.0 } else { 0.0 }).collect();
    let model = stump(&one_feature(&x, &low));
    assert!(model.tree(0).missing_goes_left(0));
    assert!(model.tree(0).predict_row(&[NAN])[0].abs() < 1e-12);
    assert!(squared_error(&model, &one_feature(&x, &low)) < 1e-12);

    // ... or like the high ones (target 1).
    let high: Vec<f64> = x.iter().map(|&value| if value < 45.0 { 0.0 } else { 1.0 }).collect();
    let model = stump(&one_feature(&x, &high));
    assert!(!model.tree(0).missing_goes_left(0));
    assert!((model.tree(0).predict_row(&[NAN])[0] - 1.0).abs() < 1e-12);
    assert!(squared_error(&model, &one_feature(&x, &high)) < 1e-12);
}

#[test]
fn being_missing_can_be_the_whole_signal() {
    // The value says nothing; whether there is one says everything.
    let mut rng = Rng::new(3, 0);
    let x: Vec<f64> = (0..200).map(|row| if row % 4 == 0 { NAN } else { (rng.below(1000)) as f64 }).collect();
    let y: Vec<f64> = x.iter().map(|value| if value.is_nan() { 5.0 } else { -1.0 }).collect();
    let data = one_feature(&x, &y);
    let model = stump(&data);
    assert!(squared_error(&model, &data) < 1e-12);

    // The split sits above every number there is: all values left, the missing ones right.
    let Node::Split { threshold, .. } = model.tree(0).node(0) else { panic!("the root is a split") };
    assert_eq!(threshold, f64::MAX);
    assert!(!model.tree(0).missing_goes_left(0));
    // Even a value larger than any seen in training is a value, not a hole.
    assert!((model.tree(0).predict_row(&[1e300])[0] + 1.0).abs() < 1e-12);
    assert!((model.tree(0).predict_row(&[NAN])[0] - 5.0).abs() < 1e-12);
}

#[test]
fn one_split_is_the_best_there_is_whatever_the_data() {
    for seed in 0..40 {
        let mut rng = Rng::new(seed, 0);
        let n = 20 + rng.below(60);
        let n_values = 2 + rng.below(12);
        let x: Vec<f64> = (0..n).map(|_| if rng.chance(0.3) { NAN } else { rng.below(n_values) as f64 }).collect();
        let lean = (rng.below(3) as f64 - 1.0) * 4.0;
        let y: Vec<f64> = x
            .iter()
            .map(|&value| {
                let noise = rng.below(100) as f64 / 25.0;
                if value.is_nan() {
                    lean + noise
                } else {
                    value - n_values as f64 / 2.0 + noise
                }
            })
            .collect();
        let data = one_feature(&x, &y);
        let (found, best) = (squared_error(&stump(&data), &data), best_possible_squared_error(&x, &y));
        assert!((found - best).abs() <= 1e-9 * best.max(1.0), "seed {seed}: found {found}, best possible {best}");
    }
}

#[test]
fn a_feature_never_missing_in_training_sends_missing_values_right() {
    let x: Vec<f64> = (0..100).map(|row| row as f64).collect();
    let y: Vec<f64> = x.iter().map(|&value| if value >= 50.0 { 1.0 } else { 0.0 }).collect();
    let model = stump(&one_feature(&x, &y));
    assert!(!model.tree(0).missing_goes_left(0));
    // With the large values.
    assert_eq!(model.tree(0).predict_row(&[NAN]), model.tree(0).predict_row(&[99.0]));
}

#[test]
fn real_data_with_holes_scores_like_scikit_learn() {
    // Reference: scikit-learn 1.9.1 on the same rows with the same holes,
    // first 8,000 rows to fit and last 2,000 to score.
    //   HistGradientBoostingRegressor (100 rounds, depth 6, 20 per leaf):
    //     R² 0.6311 with the holes left in, 0.6187 after median imputing.
    //   RandomForestRegressor (100 trees), six seeds: R² 0.611 to 0.619
    //     with the holes left in, 0.603 to 0.612 after median imputing.
    // This split is by position, so the last rows are districts unlike
    // the first, and on it mlib's forest scores about 0.01 below
    // scikit-learn's with or without holes (0.660 against 0.669 on the
    // complete data). What is checked here is that using the holes gains
    // as much as it does there.
    let (train, test) = split(&housing_with_holes());
    let (train_filled, test_filled) = (median_imputed(&train, &train), median_imputed(&train, &test));

    let boosting = GradientBoostingSpec::regressor().build().unwrap();
    let native = boosting.fit(&train).unwrap().score(&test).unwrap();
    let imputed = boosting.fit(&train_filled).unwrap().score(&test_filled).unwrap();
    assert!((native - 0.6311).abs() < 0.015, "boosting with holes: {native}");
    assert!((imputed - 0.6187).abs() < 0.015, "boosting, imputed: {imputed}");
    assert!(native > imputed, "boosting: {native} with holes, {imputed} imputed");

    let forest = RandomForestSpec::regressor().seed(1).build().unwrap();
    let native = forest.fit(&train).unwrap().score(&test).unwrap();
    let imputed = forest.fit(&train_filled).unwrap().score(&test_filled).unwrap();
    assert!((native - 0.615).abs() < 0.02, "forest with holes: {native}");
    assert!((imputed - 0.607).abs() < 0.02, "forest, imputed: {imputed}");
    assert!(native > imputed + 0.003, "forest: {native} with holes, {imputed} imputed");

    // A classifier: is the district dearer than the median one? (scikit-learn: 0.840)
    let median = {
        let mut values = housing().y.column(0).to_vec();
        values.sort_unstable_by(f64::total_cmp);
        (values[4999] + values[5000]) / 2.0
    };
    let (mut train, mut test) = (train, test);
    train.y.mapv_inplace(|value| (value > median) as u8 as f64);
    test.y.mapv_inplace(|value| (value > median) as u8 as f64);
    let accuracy = GradientBoostingSpec::default().build().unwrap().fit(&train).unwrap().score(&test).unwrap();
    assert!((accuracy - 0.840).abs() < 0.015, "boosting classifier with holes: {accuracy}");
}

#[test]
fn a_model_with_holes_is_the_same_on_any_number_of_cores() {
    let data = housing_with_holes().take_rows(&(0..3000).collect::<Vec<_>>());
    let forest = RandomForestSpec::regressor().n_trees(8).seed(5).build().unwrap();
    let boosting = GradientBoostingSpec::regressor().n_rounds(15).subsample(0.8).seed(5).build().unwrap();
    let fit_both = || (forest.fit(&data).unwrap(), boosting.fit(&data).unwrap());
    let on_all_cores = fit_both();
    let on_one_core = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(fit_both);
    assert_eq!(on_all_cores, on_one_core);
}

#[test]
fn a_saved_model_keeps_where_missing_values_go() {
    let data = housing_with_holes().take_rows(&(0..3000).collect::<Vec<_>>());
    let model = GradientBoostingSpec::regressor().n_rounds(15).build().unwrap().fit(&data).unwrap();
    let n_missing_left =
        model.trees().map(|tree| (0..tree.n_nodes()).filter(|&node| tree.missing_goes_left(node)).count());
    assert!(n_missing_left.sum::<usize>() > 0, "some split should send missing values left on this data");

    let saved = serde_json::to_string(&model).unwrap();
    assert!(saved.contains(r#""missing_left":true"#));
    let loaded: FittedGradientBoosting = serde_json::from_str(&saved).unwrap();
    assert_eq!(loaded, model);
    assert_eq!(loaded.predict(&data).unwrap(), model.predict(&data).unwrap());

    // A model fitted on complete data is saved without a word about missing values,
    // exactly as 0.1.1 saved it.
    let complete = housing().take_rows(&(0..3000).collect::<Vec<_>>());
    let model = GradientBoostingSpec::regressor().n_rounds(15).build().unwrap().fit(&complete).unwrap();
    assert!(!serde_json::to_string(&model).unwrap().contains("missing_left"));
}

#[test]
fn data_that_still_cannot_be_used_is_an_error() {
    let x: Vec<f64> = (0..60).map(|row| row as f64).collect();
    let y: Vec<f64> = x.iter().map(|&value| (value >= 30.0) as u8 as f64).collect();
    let forest = RandomForestSpec::default().n_trees(3).build().unwrap();
    let boosting = GradientBoostingSpec::default().n_rounds(3).min_samples_leaf(2).build().unwrap();
    let linear = LinearModelSpec::default().build().unwrap();

    // Infinity is not a missing value.
    let mut infinite = one_feature(&x, &y);
    infinite.x[[4, 0]] = f64::INFINITY;
    assert!(forest.fit(&infinite).is_err() && boosting.fit(&infinite).is_err() && linear.fit(&infinite).is_err());

    // A target cannot be missing.
    let mut no_target = one_feature(&x, &y);
    no_target.y[[4, 0]] = NAN;
    assert!(forest.fit(&no_target).is_err() && boosting.fit(&no_target).is_err() && linear.fit(&no_target).is_err());

    // The linear model has no use for a missing value.
    let mut hole = one_feature(&x, &y);
    hole.x[[4, 0]] = NAN;
    assert!(forest.fit(&hole).is_ok() && boosting.fit(&hole).is_ok());
    let error = linear.fit(&hole).unwrap_err().to_string();
    assert!(error.contains("Imputer"), "{error}");
}

#[test]
fn a_feature_with_nothing_but_holes_is_ignored() {
    let data = housing().take_rows(&(0..2000).collect::<Vec<_>>());
    let mut with_empty = data.clone();
    with_empty.x.column_mut(4).fill(NAN);
    let forest = RandomForestSpec::regressor().n_trees(5).seed(2).build().unwrap().fit(&with_empty).unwrap();
    assert_eq!(forest.feature_importances()[4], 0.0);
    assert!(forest.score(&with_empty).unwrap() > 0.8);
    let boosting = GradientBoostingSpec::regressor().n_rounds(20).build().unwrap().fit(&with_empty).unwrap();
    assert_eq!(boosting.feature_importances()[4], 0.0);
}

#[test]
fn a_feature_with_many_values_and_holes_still_bins_correctly() {
    // More than 256 distinct values, so the feature is cut into quantile
    // bins — one fewer than usual, the last being kept for the holes.
    let mut rng = Rng::new(9, 0);
    let x: Vec<f64> =
        (0..5000).map(|_| if rng.chance(0.1) { NAN } else { rng.next_u64() as f64 / u64::MAX as f64 }).collect();
    let y: Vec<f64> = x.iter().map(|&value| if value.is_nan() { 3.0 } else { (value * 4.0).floor() }).collect();
    let data = one_feature(&x, &y);
    let model = GradientBoostingSpec::regressor().n_rounds(200).learning_rate(0.3).build().unwrap().fit(&data).unwrap();
    assert!(model.score(&data).unwrap() > 0.99);
    // The holes are told apart from every value.
    let at_hole = model.predict_single(&one_feature(&[NAN], &[0.0])).unwrap()[0];
    assert!((at_hole - 3.0).abs() < 0.05, "{at_hole}");
}

#[test]
fn early_stopping_subsampling_and_class_weights_work_with_holes() {
    let (mut train, mut test) = split(&housing_with_holes());
    let cut = 3.5;
    train.y.mapv_inplace(|value| (value > cut) as u8 as f64);
    test.y.mapv_inplace(|value| (value > cut) as u8 as f64);
    let spec = GradientBoostingSpec::default()
        .n_rounds(300)
        .subsample(0.8)
        .feature_fraction(0.8)
        .early_stopping_rounds(Some(10))
        .class_weight(ClassWeight::Balanced)
        .seed(4);
    let model = spec.clone().build().unwrap().fit(&train).unwrap();
    assert!(model.n_rounds() < 300);
    assert!(model.validation_losses().iter().all(|loss| loss.is_finite()));
    let recall_of =
        |model: &FittedGradientBoosting| recall_score(&test.target().unwrap(), &model.predict_single(&test).unwrap());
    let unweighted = spec.class_weight(ClassWeight::None).build().unwrap().fit(&train).unwrap();
    assert!(recall_of(&model) > recall_of(&unweighted) + 0.1, "{} vs {}", recall_of(&model), recall_of(&unweighted));
    assert!(model.predict_proba(&test).unwrap().iter().all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn a_pipeline_hands_missing_values_to_a_tree_model_and_not_to_a_linear_one() {
    let ages: Vec<Option<f64>> =
        (0..120).map(|row| if row % 4 == 0 { None } else { Some(20.0 + (row % 50) as f64) }).collect();
    let cities: Vec<Option<String>> =
        (0..120).map(|row| if row % 9 == 0 { None } else { Some(["Oslo", "Rome"][row % 2].to_string()) }).collect();
    // The customers whose age is not on file churned.
    let churned: Vec<i64> = (0..120).map(|row| (row % 4 == 0) as i64).collect();
    let table = Table::new(vec![
        ("age", Column::Float(ages)),
        ("city", Column::Text(cities)),
        ("churned", Column::ints(&churned)),
    ])
    .unwrap();
    let data = ModelData::new(&table, ModelSpec::new(&["age", "city"], &["churned"])).unwrap();
    // No imputer: the age passes through as it is.
    let pre = || {
        ColumnTransformer::empty()
            .with_branch("cat", OneHotEncoder::new(), &["city"])
            .with_remainder(Remainder::Passthrough)
    };

    let forest =
        Pipeline::new(vec![Box::new(pre())], Box::new(RandomForestSpec::default().n_trees(20).build().unwrap()));
    let fitted = forest.fit(&data).unwrap();
    assert_eq!(fitted.score(&data).unwrap(), 1.0);
    let new = Table::new(vec![
        ("age", Column::Float(vec![None, Some(33.0)])),
        ("city", Column::Text(vec![Some("Rome".to_string()), None])),
    ])
    .unwrap();
    assert_eq!(fitted.predict_single(&new).unwrap(), [1.0, 0.0]);
    assert!(fitted.encode(&new).unwrap().x[[0, 2]].is_nan());

    // Cross-validation and saving work as with any pipeline.
    let folds = StratifiedKFold::new(3).with_shuffle(1).splits(&data.y().unwrap(), 120);
    let cv = cross_validate_pipeline(&forest, &data, &folds, &*scorer("accuracy").unwrap()).unwrap();
    assert_eq!(cv.mean, 1.0);
    let path = std::env::temp_dir().join("mlib-test-missing-values.json");
    save(&fitted, data.spec(), path.to_str().unwrap()).unwrap();
    let loaded = load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.fitted.predict_single(&new).unwrap(), [1.0, 0.0]);
    std::fs::remove_file(path).unwrap();

    // The same pipeline with a linear model stops at the first hole, at fit ...
    let linear = Pipeline::new(vec![Box::new(pre())], Box::new(LinearModelSpec::default().build().unwrap()));
    let error = linear.fit(&data).unwrap_err().to_string();
    assert!(error.contains("age") && error.contains("Imputer"), "{error}");
    // ... and, fitted on complete rows, at predict.
    let complete = Table::new(vec![
        ("age", Column::floats(&[25.0, 31.0, 47.0, 52.0, 38.0, 60.0])),
        ("city", Column::texts(&["Oslo", "Rome", "Oslo", "Rome", "Rome", "Oslo"])),
        ("churned", Column::ints(&[0, 0, 1, 1, 0, 1])),
    ])
    .unwrap();
    let fitted = linear.fit(&ModelData::new(&complete, data.spec().clone()).unwrap()).unwrap();
    assert!(fitted.predict(&complete).is_ok());
    assert!(fitted.predict(&new).is_err());
}
