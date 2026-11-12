use mlib::prelude::*;
use mlib::random::Rng;
use mlib::testing::check_estimator;
use ndarray::Array2;

/// `n` rows, 5 uniform features in [0, 1). Only the first two matter.
fn features(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = Rng::new(seed, 0);
    Array2::from_shape_fn((n, 5), |_| rng.next_u64() as f64 / u64::MAX as f64)
}

/// Class 1 inside a quadrant-shaped region — not linearly separable.
fn two_classes(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| ((x[[i, 0]] > 0.5) ^ (x[[i, 1]] > 0.5)) as u8 as f64);
    EncodedData::new(x, y).unwrap()
}

/// Three classes: which third of [0, 1) the first feature falls in,
/// shifted by the second feature.
fn three_classes(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| ((x[[i, 0]] + 0.3 * x[[i, 1]]) * 3.0 / 1.3).floor().min(2.0));
    EncodedData::new(x, y).unwrap()
}

fn regression(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| 1000.0 + 10.0 * x[[i, 0]] + (6.0 * x[[i, 1]]).sin());
    EncodedData::new(x, y).unwrap()
}

#[test]
fn boosting_passes_the_conformance_battery() {
    check_estimator(&GradientBoostingSpec::new().n_rounds(10).build().unwrap());
    check_estimator(&GradientBoostingSpec::regressor().n_rounds(10).build().unwrap());
}

#[test]
fn invalid_specs_are_rejected_before_any_data_is_seen() {
    assert!(GradientBoostingSpec::new().n_rounds(0).build().is_err());
    assert!(GradientBoostingSpec::new().learning_rate(0.0).build().is_err());
    assert!(GradientBoostingSpec::new().learning_rate(1.5).build().is_err());
    assert!(GradientBoostingSpec::new().min_samples_leaf(0).build().is_err());
    assert!(GradientBoostingSpec::new().max_depth(Some(0)).build().is_err());

    let mut spec = GradientBoostingSpec::new().build().unwrap();
    assert!(Estimator::set_param(&mut spec, "learning_rate", &ParamValue::Float(-1.0)).is_err());
    assert!(Estimator::set_param(&mut spec, "no_such_param", &ParamValue::Int(1)).is_err());
    Estimator::set_param(&mut spec, "max_depth", &ParamValue::Int(3)).unwrap();
    Estimator::set_param(&mut spec, "learning_rate", &ParamValue::Float(0.3)).unwrap();
    assert_eq!((spec.spec().max_depth, spec.spec().learning_rate), (Some(3), 0.3));
}

#[test]
fn a_classifier_needs_two_classes() {
    let mut data = two_classes(100, 1);
    data.y.fill(1.0);
    assert!(GradientBoostingSpec::new().build().unwrap().fit(&data).is_err());
}

#[test]
fn two_class_classifier_learns_a_non_linear_boundary() {
    let (train, test) = (two_classes(2000, 1), two_classes(1000, 2));
    let model = GradientBoostingSpec::new().build().unwrap().fit(&train).unwrap();
    let accuracy = model.score(&test).unwrap();
    assert!(accuracy > 0.95, "accuracy {accuracy}");

    // Probabilities are a distribution over the two classes.
    let probabilities = model.predict_proba(&test).unwrap();
    assert_eq!(probabilities.dim(), (1000, 2));
    assert!(probabilities.rows().into_iter().all(|row| (row.sum() - 1.0).abs() < 1e-9));

    // The two informative features carry nearly all the importance.
    let importances = &model.feature_importances;
    assert!((importances.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(importances[0] + importances[1] > 0.8, "importances {importances:?}");
}

#[test]
fn several_classes_get_one_score_each_and_keep_their_labels() {
    let (mut train, mut test) = (three_classes(3000, 1), three_classes(1000, 2));
    // Relabel 0, 1, 2 as 10, 20, 30.
    train.y.mapv_inplace(|class| 10.0 * (class + 1.0));
    test.y.mapv_inplace(|class| 10.0 * (class + 1.0));

    let model = GradientBoostingSpec::new().build().unwrap().fit(&train).unwrap();
    assert_eq!(model.classes, [10.0, 20.0, 30.0]);
    assert_eq!(model.trees.len(), 3);
    let accuracy = model.score(&test).unwrap();
    assert!(accuracy > 0.95, "accuracy {accuracy}");

    let probabilities = model.predict_proba(&test).unwrap();
    assert_eq!(probabilities.dim(), (1000, 3));
    assert!(probabilities.rows().into_iter().all(|row| (row.sum() - 1.0).abs() < 1e-9));
}

#[test]
fn regressor_fits_a_smooth_function() {
    let (train, test) = (regression(2000, 1), regression(1000, 2));
    let model = GradientBoostingSpec::regressor().build().unwrap().fit(&train).unwrap();
    let r2 = model.score(&test).unwrap();
    assert!(r2 > 0.98, "r2 {r2}");
    assert!(model.predict_proba(&test).is_err());
}

#[test]
fn more_rounds_fit_the_training_data_more_closely() {
    let data = regression(1000, 1);
    let error_after = |n_rounds| {
        let model = GradientBoostingSpec::regressor().n_rounds(n_rounds).build().unwrap().fit(&data).unwrap();
        root_mean_squared_error(&data.target().unwrap(), &model.predict_single(&data).unwrap())
    };
    let (few, some, many) = (error_after(5), error_after(20), error_after(80));
    assert!(few > some && some > many, "errors {few} {some} {many}");
}

#[test]
fn fitting_twice_gives_a_byte_identical_model() {
    let data = three_classes(1000, 1);
    let spec = GradientBoostingSpec::new().n_rounds(20).build().unwrap();
    let first = serde_json::to_string(&spec.fit(&data).unwrap()).unwrap();
    let second = serde_json::to_string(&spec.fit(&data).unwrap()).unwrap();
    assert_eq!(first, second);
}

#[test]
fn large_data_gives_the_same_model_as_one_thread_would() {
    // Big enough for the tree builder to spread work over the cores; the
    // result must not depend on how the work was divided.
    let data = two_classes(120_000, 1);
    let spec = GradientBoostingSpec::new().n_rounds(3).build().unwrap();
    let many_threads = spec.fit(&data).unwrap();
    let one_thread = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| spec.fit(&data).unwrap());
    assert_eq!(many_threads, one_thread);
}

#[test]
fn zero_weight_rows_are_ignored() {
    // Flip the labels of half the rows, then give those rows zero weight.
    let mut data = two_classes(2000, 1);
    for i in 0..1000 {
        data.y[[i, 0]] = 1.0 - data.y[[i, 0]];
    }
    data.weights = Some((0..2000).map(|i| if i < 1000 { 0.0 } else { 1.0 }).collect());
    let model = GradientBoostingSpec::new().build().unwrap().fit(&data).unwrap();
    assert!(model.score(&two_classes(500, 9)).unwrap() > 0.9);
}

const HOUSING_INPUTS: [&str; 8] = [
    "median_income", "house_age", "avg_rooms", "avg_bedrooms",
    "population", "avg_occupancy", "latitude", "longitude",
];

#[test]
fn real_data_reaches_the_expected_quality() {
    // The same California housing sample the forest is tested on.
    // scikit-learn's histogram gradient boosting scores about the same.
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let study = ModelData::new(&table, ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"])).unwrap();
    let data =
        EncodedData::from_table(&study.x_table(), study.input_features(), study.y_matrix().unwrap(), None).unwrap();

    let (train, test) = train_test_split_shuffled(data.num_rows(), 0.8, 0);
    let model = GradientBoostingSpec::regressor().build().unwrap().fit(&data.take_rows(&train)).unwrap();
    let r2 = model.score(&data.take_rows(&test)).unwrap();
    assert!(r2 > 0.78, "r2 {r2}");
}

#[test]
fn a_boosted_pipeline_survives_save_and_load() {
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap().take_rows(&(0..500).collect::<Vec<_>>());
    let spec = ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"]);
    let model = GradientBoostingSpec::regressor().n_rounds(10).build().unwrap();
    let fitted = Pipeline::new(vec![], Box::new(model)).fit(&ModelData::new(&table, spec.clone()).unwrap()).unwrap();

    let path = std::env::temp_dir().join("mlib_boosting_round_trip_test.json");
    let path = path.to_str().unwrap();
    save(&fitted, &spec, path).unwrap();
    let saved = load(path).unwrap();
    assert_eq!(saved.fitted.predict(&table).unwrap(), fitted.predict(&table).unwrap());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn several_targets_are_predicted_together() {
    // Two targets that depend on different features.
    let two_targets = |n, seed| {
        let x = features(n, seed);
        let y = Array2::from_shape_fn((n, 2), |(i, target)| match target {
            0 => 10.0 * x[[i, 0]] + (6.0 * x[[i, 1]]).sin(),
            _ => 1e6 * (x[[i, 2]] - 0.5).abs(),
        });
        EncodedData::new(x, y).unwrap()
    };
    let (train, test) = (two_targets(3000, 1), two_targets(1000, 2));
    let model = GradientBoostingSpec::regressor().build().unwrap().fit(&train).unwrap();

    let predictions = model.predict(&test).unwrap();
    assert_eq!(predictions.dim(), (1000, 2));
    assert!(model.predict_single(&test).is_err());
    for target in 0..2 {
        let (truth, predicted) = (test.y.column(target).to_vec(), predictions.column(target).to_vec());
        let r2 = r2_score(&truth, &predicted);
        assert!(r2 > 0.95, "target {target}: r2 {r2}");
    }

    // A classifier takes one target only.
    assert!(GradientBoostingSpec::new().build().unwrap().fit(&train).is_err());
}
