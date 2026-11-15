use mlib::prelude::*;
use mlib::random::Rng;
use mlib::testing::check_estimator;

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
    assert!(spec.set_param("learning_rate", &ParamValue::Float(-1.0)).is_err());
    assert!(spec.set_param("no_such_param", &ParamValue::Int(1)).is_err());
    spec.set_param("max_depth", &ParamValue::Int(3)).unwrap();
    spec.set_param("learning_rate", &ParamValue::Float(0.3)).unwrap();
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
    let importances = model.feature_importances();
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
    assert_eq!(model.classes(), [10.0, 20.0, 30.0]);
    assert_eq!((model.n_scores(), model.n_rounds()), (3, 100));
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
    let one_thread =
        rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| spec.fit(&data).unwrap());
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

const HOUSING_INPUTS: [&str; 8] =
    ["median_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude"];

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

    // Income is by far the strongest feature, as it is for the forest:
    // importances follow how much each split improved the fit.
    let (top_feature, top_importance) = model.ranked_importances()[0].clone();
    assert_eq!(top_feature, "median_income");
    assert!(top_importance > 0.4, "importance {top_importance}");
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
}

#[test]
fn a_classifier_predicts_several_targets_together() {
    let (train, test) = (colour_and_size(3000, 1), colour_and_size(1000, 2));
    let spec = GradientBoostingSpec::new().n_rounds(40).build().unwrap();
    let model = spec.fit(&train).unwrap();
    check_colour_and_size(&model, &test);
    // Three scores for the colour, one for the yes/no size.
    assert_eq!((model.n_scores(), model.classes()), (4, &[10.0, 20.0, 30.0, 0.0, 1.0][..]));

    // Each target is learned exactly as it would be on its own.
    let mut colour_only = train.clone();
    colour_only.y = train.y.slice(ndarray::s![.., 0..1]).to_owned();
    let alone = spec.fit(&colour_only).unwrap();
    let (together, alone) = (model.predict_proba(&test).unwrap(), alone.predict_proba(&test).unwrap());
    assert_eq!(together.slice(ndarray::s![.., 0..3]), alone);

    // Early stopping watches the loss of both targets together.
    let stopped = GradientBoostingSpec::new().n_rounds(300).early_stopping_rounds(Some(5)).build().unwrap();
    let stopped = stopped.fit(&train).unwrap();
    assert!(stopped.n_rounds() < 300 && stopped.score(&test).unwrap() > 0.9);
    // A target with a single class cannot be learned.
    let mut one_class = train.clone();
    one_class.y.column_mut(1).fill(1.0);
    assert!(spec.fit(&one_class).is_err());
}

/// A regression target with a lot of noise: easy to overfit.
fn noisy_regression(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let mut noise = Rng::new(seed, 99);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| {
        10.0 * x[[i, 0]] + 4.0 * (noise.next_u64() as f64 / u64::MAX as f64 - 0.5)
    });
    EncodedData::new(x, y).unwrap()
}

#[test]
fn early_stopping_stops_before_the_model_overfits() {
    let (train, test) = (noisy_regression(2000, 1), noisy_regression(2000, 2));
    let all_rounds = GradientBoostingSpec::regressor().n_rounds(400).learning_rate(0.3);
    let stopped = all_rounds.clone().early_stopping_rounds(Some(10)).build().unwrap().fit(&train).unwrap();
    let overfitted = all_rounds.build().unwrap().fit(&train).unwrap();

    // It stopped long before 400 rounds, and kept the best round's model.
    let losses = stopped.validation_losses();
    assert!(stopped.n_rounds() < 100, "ran {} rounds", stopped.n_rounds());
    assert_eq!(losses.len(), stopped.n_rounds() + 10);
    let best = losses.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(losses[stopped.n_rounds() - 1], best);

    // Stopping early generalises better than running every round.
    assert!(stopped.score(&test).unwrap() > overfitted.score(&test).unwrap());
    // Without early stopping nothing is held out and nothing is recorded.
    assert_eq!((overfitted.n_rounds(), overfitted.validation_losses().len()), (400, 0));
}

#[test]
fn early_stopping_works_for_classifiers() {
    for data in [two_classes(2000, 1), three_classes(2000, 1)] {
        let spec = GradientBoostingSpec::new().n_rounds(300).early_stopping_rounds(Some(5)).build().unwrap();
        let model = spec.fit(&data).unwrap();
        assert!(model.n_rounds() < 300);
        assert!(model.validation_losses().iter().all(|loss| loss.is_finite() && *loss >= 0.0));
        assert!(model.score(&data).unwrap() > 0.9);
    }
}

#[test]
fn subsampling_is_random_but_repeatable() {
    let (train, test) = (regression(3000, 1), regression(1000, 2));
    let fit = |seed| GradientBoostingSpec::regressor().subsample(0.5).seed(seed).build().unwrap().fit(&train).unwrap();
    // Same seed, same model; another seed, another model.
    assert_eq!(fit(1), fit(1));
    assert_ne!(fit(1), fit(2));
    // Half the rows per round is still plenty to learn this function.
    assert!(fit(1).score(&test).unwrap() > 0.97);
}

#[test]
fn the_new_settings_are_validated_and_reachable_by_name() {
    assert!(GradientBoostingSpec::new().subsample(0.0).build().is_err());
    assert!(GradientBoostingSpec::new().subsample(1.1).build().is_err());
    assert!(GradientBoostingSpec::new().early_stopping_rounds(Some(0)).build().is_err());
    assert!(GradientBoostingSpec::new().validation_fraction(1.0).build().is_err());

    let mut spec = GradientBoostingSpec::new().build().unwrap();
    spec.set_param("subsample", &ParamValue::Float(0.8)).unwrap();
    spec.set_param("early_stopping_rounds", &ParamValue::Int(7)).unwrap();
    assert_eq!((spec.spec().subsample, spec.spec().early_stopping_rounds), (0.8, Some(7)));
    spec.set_param("early_stopping_rounds", &ParamValue::Text("none".to_string())).unwrap();
    assert_eq!(spec.spec().early_stopping_rounds, None);

    // Too few rows to hold any out.
    let tiny = regression(3, 1);
    let cannot_hold_out = GradientBoostingSpec::regressor().early_stopping_rounds(Some(5)).validation_fraction(0.01);
    assert!(cannot_hold_out.build().unwrap().fit(&tiny).is_err());
}

#[test]
fn feature_sampling_is_random_but_repeatable() {
    let (train, test) = (regression(3000, 1), regression(1000, 2));
    let spec = |seed| GradientBoostingSpec::regressor().feature_fraction(0.6).seed(seed).build().unwrap();
    // Same seed, same model; another seed, another model.
    assert_eq!(spec(1).fit(&train).unwrap(), spec(1).fit(&train).unwrap());
    assert_ne!(spec(1).fit(&train).unwrap(), spec(2).fit(&train).unwrap());
    // Three of the five features per round is still plenty to learn this function.
    assert!(spec(1).fit(&train).unwrap().score(&test).unwrap() > 0.97);

    // Each round's tree uses at most the three features it was given.
    let model = spec(1).fit(&train).unwrap();
    for tree in model.trees() {
        let mut used: Vec<usize> = (0..tree.n_nodes())
            .filter_map(|index| match tree.node(index) {
                Node::Split { feature, .. } => Some(feature),
                _ => None,
            })
            .collect();
        used.sort_unstable();
        used.dedup();
        assert!(used.len() <= 3, "a tree split on {used:?}");
    }

    // However small the fraction, one feature is kept.
    let one_feature = GradientBoostingSpec::regressor().feature_fraction(0.01).build().unwrap();
    assert!(one_feature.fit(&train).unwrap().score(&train).unwrap() > 0.5);
}

#[test]
fn l2_regularization_shrinks_the_corrections() {
    let (train, test) = (noisy_regression(300, 1), noisy_regression(2000, 2));
    let spec = |l2| GradientBoostingSpec::regressor().min_samples_leaf(1).l2_regularization(l2).build().unwrap();
    let largest_leaf = |model: &FittedGradientBoosting| {
        let leaves = (0..model.tree(0, 0).n_nodes()).filter_map(|index| match model.tree(0, 0).node(index) {
            Node::Leaf { prediction } => Some(prediction[0].abs()),
            _ => None,
        });
        leaves.fold(0.0, f64::max)
    };
    let (plain, regularised) = (spec(0.0).fit(&train).unwrap(), spec(20.0).fit(&train).unwrap());

    // The first tree's corrections are smaller, the training fit is
    // looser, and on these few noisy rows the test score is better.
    assert!(largest_leaf(&regularised) < largest_leaf(&plain));
    assert!(regularised.score(&train).unwrap() < plain.score(&train).unwrap());
    assert!(regularised.score(&test).unwrap() > plain.score(&test).unwrap());
}

#[test]
fn feature_fraction_and_l2_are_validated_and_reachable_by_name() {
    assert!(GradientBoostingSpec::new().feature_fraction(0.0).build().is_err());
    assert!(GradientBoostingSpec::new().feature_fraction(1.1).build().is_err());
    assert!(GradientBoostingSpec::new().l2_regularization(-1.0).build().is_err());
    assert!(GradientBoostingSpec::new().l2_regularization(f64::NAN).build().is_err());

    let mut spec = GradientBoostingSpec::new().build().unwrap();
    spec.set_param("feature_fraction", &ParamValue::Float(0.5)).unwrap();
    spec.set_param("l2_regularization", &ParamValue::Float(2.0)).unwrap();
    assert_eq!((spec.spec().feature_fraction, spec.spec().l2_regularization), (0.5, 2.0));

    // Both work for classifiers too.
    let data = three_classes(2000, 1);
    let model = GradientBoostingSpec::new().feature_fraction(0.6).l2_regularization(1.0).build().unwrap();
    assert!(model.fit(&data).unwrap().score(&data).unwrap() > 0.9);
}
