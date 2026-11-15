use mlib::prelude::*;
use mlib::random::Rng;
use mlib::testing::check_estimator;

/// `n` rows, 4 uniform features in [0, 1).
fn features(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = Rng::new(seed, 0);
    Array2::from_shape_fn((n, 4), |_| rng.next_u64() as f64 / u64::MAX as f64)
}

/// Uniform noise in [-0.5, 0.5), one value per row.
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed, 99);
    (0..n).map(|_| rng.next_u64() as f64 / u64::MAX as f64 - 0.5).collect()
}

/// An exactly linear target: 3 + 2·x0 − x1 + 0.5·x3 (x2 does not matter).
fn exact_line(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| 3.0 + 2.0 * x[[i, 0]] - x[[i, 1]] + 0.5 * x[[i, 3]]);
    EncodedData::new(x, y).unwrap()
}

/// Two classes, more likely 1 the larger x0 − x1 is, with noise.
fn two_classes(n: usize, seed: u64) -> EncodedData {
    let (x, noise) = (features(n, seed), noise(n, seed));
    let y = Array2::from_shape_fn((n, 1), |(i, _)| (x[[i, 0]] - x[[i, 1]] + 0.4 * noise[i] > 0.1) as u8 as f64);
    EncodedData::new(x, y).unwrap()
}

/// Three classes labelled 10, 20, 30: which third x0 + 0.3·x1 falls in.
fn three_classes(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let third = |i: usize| ((x[[i, 0]] + 0.3 * x[[i, 1]]) * 3.0 / 1.3).floor().min(2.0);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| 10.0 * (third(i) + 1.0));
    EncodedData::new(x, y).unwrap()
}

/// Two class targets: a "colour" with three classes (10, 20, 30), set by
/// the first feature, and a yes/no "size", set by the second.
fn colour_and_size(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 2), |(i, target)| match target {
        0 => 10.0 * ((x[[i, 0]] * 3.0).floor().min(2.0) + 1.0),
        _ => (x[[i, 1]] > 0.6) as u8 as f64,
    });
    EncodedData::new(x, y).unwrap()
}

/// The checks every classifier with several targets must pass.
fn check_colour_and_size(model: &dyn Predictor, test: &EncodedData) {
    // One column of predicted labels per target, each from its own classes.
    let predictions = model.predict(test).unwrap();
    assert_eq!(predictions.dim(), (test.num_rows(), 2));
    assert!(predictions.column(0).iter().all(|label| [10.0, 20.0, 30.0].contains(label)));
    assert!(predictions.column(1).iter().all(|label| [0.0, 1.0].contains(label)));
    assert!(model.predict_single(test).is_err());
    for target in 0..2 {
        let (truth, predicted) = (test.y.column(target).to_vec(), predictions.column(target).to_vec());
        assert!(accuracy(&truth, &predicted) > 0.9, "target {target}: {}", accuracy(&truth, &predicted));
    }
    // The default score is the accuracy averaged over the targets.
    assert!(model.score(test).unwrap() > 0.9);

    // Probabilities: the three colours, then the two sizes, side by side.
    assert_eq!(model.target_classes(), vec![vec![10.0, 20.0, 30.0], vec![0.0, 1.0]]);
    let probabilities = model.predict_proba(test).unwrap();
    assert_eq!(probabilities.dim(), (test.num_rows(), 5));
    for row in probabilities.rows() {
        let (colour, size) = (row.iter().take(3).sum::<f64>(), row.iter().skip(3).sum::<f64>());
        assert!((colour - 1.0).abs() < 1e-9 && (size - 1.0).abs() < 1e-9);
    }
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

#[test]
fn the_linear_model_passes_the_conformance_battery() {
    check_estimator(&LinearModelSpec::new().build().unwrap());
    check_estimator(&LinearModelSpec::regressor().build().unwrap());
}

#[test]
fn invalid_specs_are_rejected_before_any_data_is_seen() {
    assert!(LinearModelSpec::new().l2_regularization(-1.0).build().is_err());
    assert!(LinearModelSpec::new().max_iterations(0).build().is_err());
    assert!(LinearModelSpec::new().tolerance(0.0).build().is_err());

    let mut spec = LinearModelSpec::new().build().unwrap();
    assert!(spec.set_param("l2_regularization", &ParamValue::Float(-1.0)).is_err());
    assert!(spec.set_param("no_such_param", &ParamValue::Int(1)).is_err());
    spec.set_param("l2_regularization", &ParamValue::Float(2.0)).unwrap();
    spec.set_param("max_iterations", &ParamValue::Int(50)).unwrap();
    assert_eq!((spec.spec().l2_regularization, spec.spec().max_iterations), (2.0, 50));
}

#[test]
fn regression_recovers_an_exact_line() {
    let model = LinearModelSpec::regressor().build().unwrap().fit(&exact_line(500, 1)).unwrap();
    let expected = [2.0, -1.0, 0.0, 0.5];
    for (found, expected) in model.coefficients(0).iter().zip(expected) {
        assert!(close(*found, expected, 1e-9), "coefficient {found}, expected {expected}");
    }
    assert!(close(model.intercepts()[0], 3.0, 1e-9));
    assert!(close(model.score(&exact_line(200, 2)).unwrap(), 1.0, 1e-12));

    // Coefficients come with their feature names, and regression needs no search.
    assert_eq!(model.named_coefficients(0)[1].0, "x1");
    assert_eq!((model.n_iterations(), model.converged(), model.n_scores()), (0, true, 1));
    assert!(model.predict_proba(&exact_line(10, 3)).is_err());
}

#[test]
fn features_in_any_units_and_repeated_features_are_handled() {
    // The same line with x0 in units a million times smaller, a feature
    // that never changes, and x1 given twice.
    let plain = exact_line(300, 1);
    let x = Array2::from_shape_fn((300, 6), |(i, j)| match j {
        0 => plain.x[[i, 0]] * 1e6,
        4 => 7.0,
        5 => plain.x[[i, 1]],
        _ => plain.x[[i, j]],
    });
    let model =
        LinearModelSpec::regressor().build().unwrap().fit(&EncodedData::new(x.clone(), plain.y.clone()).unwrap());
    let model = model.unwrap();

    assert!(close(model.coefficients(0)[0], 2e-6, 1e-14));
    // The constant feature and the second copy of x1 add nothing: both get 0.
    assert_eq!((model.coefficients(0)[4], model.coefficients(0)[5]), (0.0, 0.0));
    assert!(close(model.coefficients(0)[1], -1.0, 1e-8));
    assert!(close(model.score(&EncodedData::new(x, plain.y).unwrap()).unwrap(), 1.0, 1e-12));
}

#[test]
fn several_targets_get_their_own_coefficients() {
    let x = features(400, 1);
    let y = Array2::from_shape_fn((400, 2), |(i, t)| if t == 0 { 1.0 + x[[i, 0]] } else { 5.0 - 3.0 * x[[i, 2]] });
    let data = EncodedData::new(x, y).unwrap();
    let model = LinearModelSpec::regressor().build().unwrap().fit(&data).unwrap();

    assert_eq!((model.n_scores(), model.predict(&data).unwrap().dim()), (2, (400, 2)));
    assert!(close(model.coefficients(0)[0], 1.0, 1e-9) && close(model.coefficients(1)[2], -3.0, 1e-9));
    assert!(close(model.intercepts()[1], 5.0, 1e-9));
}

#[test]
fn a_classifier_predicts_several_targets_together() {
    let (train, test) = (colour_and_size(3000, 1), colour_and_size(1000, 2));
    let spec = LinearModelSpec::new().l2_regularization(0.01).build().unwrap();
    let model = spec.fit(&train).unwrap();
    check_colour_and_size(&model, &test);
    // Three scores for the colour, one for the yes/no size.
    assert_eq!((model.n_scores(), model.classes()), (4, &[10.0, 20.0, 30.0, 0.0, 1.0][..]));

    // Each target is fitted exactly as it would be on its own.
    let mut size_only = train.clone();
    size_only.y = train.y.slice(ndarray::s![.., 1..2]).to_owned();
    let alone = spec.fit(&size_only).unwrap();
    assert_eq!((model.coefficients(3), model.intercepts()[3]), (alone.coefficients(0), alone.intercepts()[0]));
}

#[test]
fn l2_regularization_shrinks_the_coefficients() {
    let data = exact_line(200, 1);
    let size = |l2| {
        let model = LinearModelSpec::regressor().l2_regularization(l2).build().unwrap().fit(&data).unwrap();
        model.coefficients(0).iter().map(|c| c * c).sum::<f64>()
    };
    assert!(size(10.0) < size(0.0) && size(1000.0) < size(10.0));

    let data = two_classes(500, 1);
    let size = |l2| {
        let model = LinearModelSpec::new().l2_regularization(l2).build().unwrap().fit(&data).unwrap();
        model.coefficients(0).iter().map(|c| c * c).sum::<f64>()
    };
    assert!(size(10.0) < size(0.0) && size(1000.0) < size(10.0));
}

#[test]
fn zero_weight_rows_are_ignored_and_weights_count_like_repeated_rows() {
    // Spoil the target of the first 100 rows, then give them zero weight.
    let mut data = exact_line(300, 1);
    (0..100).for_each(|i| data.y[[i, 0]] = -50.0);
    data.weights = Some((0..300).map(|i| if i < 100 { 0.0 } else { 1.0 }).collect());
    let model = LinearModelSpec::regressor().build().unwrap().fit(&data).unwrap();
    assert!(close(model.coefficients(0)[0], 2.0, 1e-9));

    // A weight of 2 is the same as having the row twice.
    let data = two_classes(200, 1);
    let rows: Vec<usize> = (0..200).chain(0..50).collect();
    let mut weighted = data.clone();
    weighted.weights = Some((0..200).map(|i| if i < 50 { 2.0 } else { 1.0 }).collect());
    let spec = LinearModelSpec::new().tolerance(1e-9).max_iterations(2000).build().unwrap();
    let (repeated, weighted) = (spec.fit(&data.take_rows(&rows)).unwrap(), spec.fit(&weighted).unwrap());
    for (a, b) in repeated.coefficients(0).iter().zip(weighted.coefficients(0)) {
        assert!(close(*a, *b, 1e-5), "{a} vs {b}");
    }
}

#[test]
fn two_class_classifier_finds_the_direction_that_separates() {
    let (train, test) = (two_classes(3000, 1), two_classes(1000, 2));
    let model = LinearModelSpec::new().build().unwrap().fit(&train).unwrap();
    assert!(model.converged() && model.n_iterations() > 0);
    let accuracy = model.score(&test).unwrap();
    assert!(accuracy > 0.9, "accuracy {accuracy}");

    // One score for two classes; x0 pushes towards class 1, x1 away, x2 hardly matters.
    let coefficients = model.coefficients(0);
    assert_eq!(model.n_scores(), 1);
    assert!(coefficients[0] > 5.0 && coefficients[1] < -5.0 && coefficients[2].abs() < 1.0, "{coefficients:?}");

    let probabilities = model.predict_proba(&train).unwrap();
    assert_eq!(probabilities.dim(), (3000, 2));
    assert!(probabilities.rows().into_iter().all(|row| close(row.sum(), 1.0, 1e-12)));
    // At the best fit, the predicted probabilities average to the share of class 1.
    let share = train.y.sum() / 3000.0;
    assert!(close(probabilities.column(1).sum() / 3000.0, share, 1e-4));
}

#[test]
fn several_classes_get_one_score_each_and_keep_their_labels() {
    let (train, test) = (three_classes(3000, 1), three_classes(1000, 2));
    let model = LinearModelSpec::new().build().unwrap().fit(&train).unwrap();
    assert_eq!((model.n_scores(), model.classes()), (3, &[10.0, 20.0, 30.0][..]));
    let accuracy = model.score(&test).unwrap();
    assert!(accuracy > 0.95, "accuracy {accuracy}");

    let probabilities = model.predict_proba(&test).unwrap();
    assert_eq!(probabilities.dim(), (1000, 3));
    assert!(probabilities.rows().into_iter().all(|row| close(row.sum(), 1.0, 1e-12)));
    assert!(model.predict_single(&test).unwrap().iter().all(|label| [10.0, 20.0, 30.0].contains(label)));
}

#[test]
fn the_same_data_gives_a_byte_identical_model_on_any_number_of_cores() {
    for (spec, data) in [
        (LinearModelSpec::new(), three_classes(50_000, 1)),
        (LinearModelSpec::new(), two_classes(50_000, 1)),
        (LinearModelSpec::regressor(), exact_line(50_000, 1)),
    ] {
        let spec = spec.build().unwrap();
        let many_threads = serde_json::to_string(&spec.fit(&data).unwrap()).unwrap();
        let one_thread =
            rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| spec.fit(&data).unwrap());
        assert_eq!(many_threads, serde_json::to_string(&one_thread).unwrap());
    }
}

const HOUSING_INPUTS: [&str; 8] =
    ["median_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude"];

fn housing() -> EncodedData {
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let study = ModelData::new(&table, ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"])).unwrap();
    EncodedData::from_table(&study.x_table(), study.input_features(), study.y_matrix().unwrap(), None).unwrap()
}

#[test]
fn a_linear_pipeline_survives_save_and_load() {
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap().take_rows(&(0..500).collect::<Vec<_>>());
    let spec = ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"]);
    let model = LinearModelSpec::regressor().l2_regularization(1.0).build().unwrap();
    let fitted = Pipeline::new(vec![Box::new(StandardScaler::new())], Box::new(model))
        .fit(&ModelData::new(&table, spec.clone()).unwrap())
        .unwrap();

    let path = std::env::temp_dir().join("mlib_linear_round_trip.json");
    save(&fitted, &spec, path.to_str().unwrap()).unwrap();
    let loaded = load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.fitted.predict(&table).unwrap(), fitted.predict(&table).unwrap());
    // A linear model has coefficients, not importances.
    assert!(fitted.feature_importances().is_none());
}

/// A model's coefficients for one score, per standard deviation of each
/// feature — the units scikit-learn reports after a `StandardScaler`.
fn standardised_coefficients(model: &FittedLinearModel, score: usize, data: &EncodedData) -> Vec<f64> {
    let spread = |column: ndarray::ArrayView1<f64>| {
        let mean = column.sum() / column.len() as f64;
        (column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64).sqrt()
    };
    model.coefficients(score).iter().zip(data.x.columns()).map(|(c, column)| c * spread(column)).collect()
}

fn assert_all_close(found: &[f64], expected: &[f64], tolerance: f64) {
    for (found, expected) in found.iter().zip(expected) {
        assert!(close(*found, *expected, tolerance), "found {found}, expected {expected}");
    }
}

#[test]
fn real_data_regression_matches_scikit_learn() {
    // Reference: scikit-learn 1.9.1, LinearRegression and Ridge(alpha=100)
    // after StandardScaler, fitted on all 10,000 rows.
    let data = housing();
    let model = LinearModelSpec::regressor().build().unwrap().fit(&data).unwrap();
    let expected = [
        4.2786604756e-01,
        9.8873996062e-03,
        -9.1000470058e-02,
        5.6125527674e-01,
        -5.3288046081e-06,
        -3.8948320444e-03,
        -4.3305356524e-01,
        -4.4001911414e-01,
    ];
    assert_all_close(model.coefficients(0), &expected, 1e-8);
    assert!(close(model.intercepts()[0], -37.15982084292784, 1e-6));
    assert!(close(model.score(&data).unwrap(), 0.6102620774940406, 1e-10));

    let ridge = LinearModelSpec::regressor().l2_regularization(100.0).build().unwrap().fit(&data).unwrap();
    let expected = [
        0.8149338967,
        0.1355561703,
        -0.2143876217,
        0.2585728393,
        -0.0022535885,
        -0.0493570584,
        -0.8077409149,
        -0.764678831,
    ];
    assert_all_close(&standardised_coefficients(&ridge, 0, &data), &expected, 1e-8);
}

#[test]
fn real_data_classification_matches_scikit_learn() {
    // Reference: scikit-learn 1.9.1, LogisticRegression(C=1/50) after
    // StandardScaler, on all 10,000 rows. Its C is 1 / l2_regularization.
    let mut data = housing();
    let mut values = data.y.column(0).to_vec();
    values.sort_unstable_by(f64::total_cmp);
    let (median, n) = ((values[4999] + values[5000]) / 2.0, values.len() as f64);
    let spec = LinearModelSpec::new().l2_regularization(50.0).tolerance(1e-8).max_iterations(1000).build().unwrap();

    // Three classes: the cheapest, middle and dearest third of the districts.
    let mut thirds = data.clone();
    let cut = |q: f64| {
        let (position, lower) = (q * (n - 1.0), (q * (n - 1.0)).floor() as usize);
        values[lower] + (position - lower as f64) * (values[lower + 1] - values[lower])
    };
    let (low, high) = (cut(1.0 / 3.0), cut(2.0 / 3.0));
    thirds.y.mapv_inplace(|value| (value >= low) as u8 as f64 + (value >= high) as u8 as f64);
    let model = spec.fit(&thirds).unwrap();
    assert!(model.converged());
    let expected = [
        [
            -1.7692687517,
            -0.1882381401,
            0.4383842396,
            -0.5444351324,
            -0.0371867413,
            0.5112099923,
            1.6497074626,
            1.478837125,
        ],
        [
            0.1134124363,
            -0.1077009546,
            -0.106611325,
            0.1116692217,
            0.0068163264,
            0.4184538201,
            -0.2264189809,
            -0.1226189749,
        ],
        [
            1.6558563154,
            0.2959390947,
            -0.3317729146,
            0.4327659107,
            0.0303704149,
            -0.9296638123,
            -1.4232884817,
            -1.3562181501,
        ],
    ];
    for (score, expected) in expected.iter().enumerate() {
        assert_all_close(&standardised_coefficients(&model, score, &thirds), expected, 1e-4);
    }
    assert!(close(model.score(&thirds).unwrap(), 0.6978, 1e-9));

    // Two classes: is the district dearer than the median one?
    data.y.mapv_inplace(|value| (value > median) as u8 as f64);
    let model = spec.fit(&data).unwrap();
    assert!(model.converged());
    let expected = [
        2.0442652241,
        0.2944812015,
        -0.5027472494,
        0.577159525,
        0.0514311175,
        -0.7466527431,
        -1.9252022669,
        -1.7865973386,
    ];
    assert_all_close(&standardised_coefficients(&model, 0, &data), &expected, 1e-4);

    // With default settings and no penalty the search converges and
    // reaches scikit-learn's accuracy (0.8387).
    let plain = LinearModelSpec::new().build().unwrap().fit(&data).unwrap();
    assert!(close(plain.score(&data).unwrap(), 0.8387, 0.002));
}
