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
fn classification(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| ((x[[i, 0]] > 0.5) ^ (x[[i, 1]] > 0.5)) as u8 as f64);
    EncodedData::new(x, y).unwrap()
}

fn regression(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| 1000.0 + 10.0 * x[[i, 0]] + (6.0 * x[[i, 1]]).sin());
    EncodedData::new(x, y).unwrap()
}

#[test]
fn forests_pass_the_conformance_battery() {
    check_estimator(&RandomForestSpec::new().n_trees(10).build().unwrap());
    check_estimator(&RandomForestSpec::regressor().n_trees(10).build().unwrap());
}

#[test]
fn invalid_specs_are_rejected_before_any_data_is_seen() {
    assert!(RandomForestSpec::new().n_trees(0).build().is_err());
    assert!(RandomForestSpec::new().min_samples_leaf(0).build().is_err());
    assert!(RandomForestSpec::new().max_features(MaxFeatures::Fraction(1.5)).build().is_err());

    let mut spec = RandomForestSpec::new().build().unwrap();
    assert!(Estimator::set_param(&mut spec, "n_trees", &ParamValue::Int(0)).is_err());
    assert!(Estimator::set_param(&mut spec, "no_such_param", &ParamValue::Int(1)).is_err());
    Estimator::set_param(&mut spec, "max_depth", &ParamValue::Int(3)).unwrap();
    assert_eq!(spec.spec().max_depth, Some(3));
}

#[test]
fn classifier_learns_a_non_linear_boundary() {
    let (train, test) = (classification(2000, 1), classification(1000, 2));
    let model = RandomForestSpec::new().seed(7).build().unwrap().fit(&train).unwrap();
    let accuracy = model.score(&test).unwrap();
    assert!(accuracy > 0.93, "accuracy {accuracy}");

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
fn classes_keep_their_original_labels() {
    let mut data = classification(500, 3);
    data.y.mapv_inplace(|class| if class == 1.0 { 9.0 } else { -4.0 });
    let model = RandomForestSpec::new().n_trees(20).build().unwrap().fit(&data).unwrap();
    assert_eq!(model.classes, [-4.0, 9.0]);
    assert!(model.predict(&data).unwrap().iter().all(|label| *label == 9.0 || *label == -4.0));
}

#[test]
fn regressor_fits_a_smooth_function() {
    let (train, test) = (regression(2000, 1), regression(1000, 2));
    let model = RandomForestSpec::regressor().seed(1).build().unwrap().fit(&train).unwrap();
    let r2 = model.score(&test).unwrap();
    assert!(r2 > 0.97, "r2 {r2}");
    assert!(model.predict_proba(&test).is_err());
}

#[test]
fn same_seed_gives_a_byte_identical_model() {
    let data = classification(500, 1);
    let spec = RandomForestSpec::new().n_trees(20).seed(11).build().unwrap();
    let first = serde_json::to_string(&spec.fit(&data).unwrap()).unwrap();
    let second = serde_json::to_string(&spec.fit(&data).unwrap()).unwrap();
    assert_eq!(first, second);

    let other_seed = RandomForestSpec::new().n_trees(20).seed(12).build().unwrap();
    assert_ne!(first, serde_json::to_string(&other_seed.fit(&data).unwrap()).unwrap());
}

#[test]
fn depth_and_leaf_size_limits_are_respected() {
    let data = classification(1000, 1);
    let stump = RandomForestSpec::new().n_trees(5).max_depth(Some(1)).build().unwrap().fit(&data).unwrap();
    assert!(stump.trees.iter().all(|tree| tree.nodes.len() <= 3));

    let unlimited = RandomForestSpec::new().n_trees(5).build().unwrap().fit(&data).unwrap();
    let coarse = RandomForestSpec::new().n_trees(5).min_samples_leaf(50).build().unwrap().fit(&data).unwrap();
    assert!(coarse.trees[0].nodes.len() < unlimited.trees[0].nodes.len() / 4);
}

#[test]
fn zero_weight_rows_are_ignored() {
    // Flip the labels of half the rows, then give those rows zero weight.
    let mut data = classification(1000, 1);
    for i in 0..500 {
        data.y[[i, 0]] = 1.0 - data.y[[i, 0]];
    }
    data.weights = Some((0..1000).map(|i| if i < 500 { 0.0 } else { 1.0 }).collect());
    let model = RandomForestSpec::new().seed(1).build().unwrap().fit(&data).unwrap();
    assert!(model.score(&classification(500, 9)).unwrap() > 0.85);
}

#[test]
fn many_distinct_values_are_binned_without_losing_accuracy() {
    // 20,000 distinct values per feature: far more than the 256 bins.
    let (train, test) = (classification(20_000, 4), classification(2000, 5));
    let model = RandomForestSpec::new().n_trees(30).build().unwrap().fit(&train).unwrap();
    assert!(model.score(&test).unwrap() > 0.97);
}

#[test]
fn heavily_repeated_values_still_get_fine_bins() {
    // One feature with 2,000 distinct values, each repeated 10 times, and
    // a target that needs a precise threshold on it. Quantile marks nearly
    // always land inside a run of ties; the bins must not collapse.
    let n = 20_000;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| (i / 10) as f64);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| ((i / 10) % 200 < 100) as u8 as f64);
    let data = EncodedData::new(x, y).unwrap();
    let model = RandomForestSpec::new().n_trees(10).build().unwrap().fit(&data).unwrap();
    assert!(model.score(&data).unwrap() > 0.95);
}

#[test]
fn real_data_reaches_the_expected_quality() {
    // A 10,000-row sample of the California housing data. Real measurements
    // repeat values in ways synthetic data does not (latitude has thousands
    // of rows but only a few hundred distinct values), which once broke the
    // binning: this test scored 0.71 then, and scores 0.77 now.
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let inputs = [
        "median_income", "house_age", "avg_rooms", "avg_bedrooms",
        "population", "avg_occupancy", "latitude", "longitude",
    ];
    let study = ModelData::new(&table, ModelSpec::new(&inputs, &["median_house_value"])).unwrap();
    let data =
        EncodedData::from_table(&study.x_table(), study.input_features(), study.y_matrix().unwrap(), None).unwrap();

    let (train, test) = train_test_split_shuffled(data.num_rows(), 0.8, 0);
    let model = RandomForestSpec::regressor().seed(1).build().unwrap().fit(&data.take_rows(&train)).unwrap();
    let r2 = model.score(&data.take_rows(&test)).unwrap();
    assert!(r2 > 0.75, "r2 {r2}");
}

#[test]
fn batch_prediction_matches_walking_each_row_alone() {
    // Batches walk several rows down a tree side by side. 1,003 rows leave
    // an incomplete last group, which must be handled like the rest.
    let (train, test) = (classification(2000, 1), classification(1003, 2));
    let model = RandomForestSpec::new().n_trees(15).build().unwrap().fit(&train).unwrap();
    let batch = model.predict_proba(&test).unwrap();

    for (row, probabilities) in test.x.rows().into_iter().zip(batch.rows()) {
        let row = row.to_vec();
        let mut expected = vec![0.0; 2];
        for tree in &model.trees {
            for (total, value) in expected.iter_mut().zip(tree.predict_row(&row)) {
                *total += value;
            }
        }
        let scale = 1.0 / model.trees.len() as f64;
        expected.iter_mut().for_each(|total| *total *= scale);
        assert_eq!(probabilities.to_vec(), expected);
    }
}

/// Two targets that depend on the features in different ways. The second
/// is measured in units a million times larger than the first.
fn two_targets(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 2), |(i, target)| match target {
        0 => 10.0 * x[[i, 0]] + (6.0 * x[[i, 1]]).sin(),
        _ => 1e6 * (x[[i, 2]] - 0.5).abs(),
    });
    EncodedData::new(x, y).unwrap()
}

#[test]
fn several_targets_are_predicted_together() {
    let (train, test) = (two_targets(3000, 1), two_targets(1000, 2));
    let model = RandomForestSpec::regressor().seed(3).build().unwrap().fit(&train).unwrap();

    // One row per sample, one column per target.
    let predictions = model.predict(&test).unwrap();
    assert_eq!(predictions.dim(), (1000, 2));
    // A plain list makes no sense for two targets.
    assert!(model.predict_single(&test).is_err());

    // Both targets are learnt well, although the second one's numbers
    // are a million times larger: neither drowns out the other.
    for target in 0..2 {
        let (truth, predicted) = (test.y.column(target).to_vec(), predictions.column(target).to_vec());
        let r2 = r2_score(&truth, &predicted);
        assert!(r2 > 0.9, "target {target}: r2 {r2}");
    }
    // The model's score is the mean R² over the targets.
    assert!(model.score(&test).unwrap() > 0.9);
}

#[test]
fn a_classifier_takes_one_target_only() {
    let mut data = classification(200, 1);
    data.y = Array2::zeros((200, 2));
    assert!(RandomForestSpec::new().build().unwrap().fit(&data).is_err());
}
