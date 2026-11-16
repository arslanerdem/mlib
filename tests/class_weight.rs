//! Class weights: making a rare class count.

use mlib::prelude::*;

const HOUSING_INPUTS: [&str; 8] =
    ["median_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude"];

/// California housing with a rare class: 1 for the dearest tenth of the
/// districts (1,000 of 10,000), 0 for the rest.
fn dearest_tenth() -> EncodedData {
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let study = ModelData::new(&table, ModelSpec::new(&HOUSING_INPUTS, &["median_house_value"])).unwrap();
    let mut data =
        EncodedData::from_table(&study.x_table(), study.input_features(), study.y_matrix().unwrap(), None).unwrap();
    let mut values = data.y.column(0).to_vec();
    values.sort_unstable_by(f64::total_cmp);
    // The 90th percentile, interpolated as numpy does.
    let position = 0.9 * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let cut = values[lower] + (position - lower as f64) * (values[lower + 1] - values[lower]);
    data.y.mapv_inplace(|value| (value >= cut) as u8 as f64);
    assert_eq!(data.y.sum(), 1000.0);
    data
}

/// The first 8,000 rows and the last 2,000.
fn split(data: &EncodedData) -> (EncodedData, EncodedData) {
    (data.take_rows(&(0..8000).collect::<Vec<_>>()), data.take_rows(&(8000..10000).collect::<Vec<_>>()))
}

fn recall_on(model: &dyn Predictor, data: &EncodedData) -> f64 {
    recall_score(&data.target().unwrap(), &model.predict_single(data).unwrap())
}

/// The weight scikit-learn's "balanced" gives each row.
fn balanced_weights(data: &EncodedData) -> Vec<f64> {
    let y = data.target().unwrap();
    let n_ones = y.iter().sum::<f64>();
    let n = y.len() as f64;
    y.iter().map(|&class| if class == 1.0 { n / (2.0 * n_ones) } else { n / (2.0 * (n - n_ones)) }).collect()
}

/// A model's coefficients for standardised features, as scikit-learn
/// reports them after a `StandardScaler`.
fn standardised_coefficients(model: &FittedLinearModel, data: &EncodedData) -> Vec<f64> {
    let n = data.num_rows() as f64;
    let spread = |feature: usize| {
        let column = data.x.column(feature);
        let mean = column.sum() / n;
        (column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
    };
    model.coefficients(0).iter().enumerate().map(|(feature, coefficient)| coefficient * spread(feature)).collect()
}

fn assert_all_close(found: &[f64], expected: &[f64], tolerance: f64) {
    for (found, expected) in found.iter().zip(expected) {
        assert!((found - expected).abs() < tolerance, "{found} is not within {tolerance} of {expected}");
    }
}

#[test]
fn balanced_logistic_regression_matches_scikit_learn() {
    // Reference: scikit-learn 1.9.1, LogisticRegression(C=1/50,
    // class_weight="balanced") after StandardScaler, on all 10,000 rows.
    let data = dearest_tenth();
    let spec = LinearModelSpec::new().l2_regularization(50.0).tolerance(1e-8).max_iterations(1000);

    let balanced = spec.clone().class_weight(ClassWeight::Balanced).build().unwrap().fit(&data).unwrap();
    assert!(balanced.converged());
    let expected = [
        1.8484117471,
        0.6172860293,
        -0.8319832292,
        1.0788125471,
        -0.132940461,
        -1.0565040735,
        -1.3341209723,
        -1.1306796976,
    ];
    assert_all_close(&standardised_coefficients(&balanced, &data), &expected, 1e-4);
    assert!((recall_on(&balanced, &data) - 0.82).abs() < 1e-9);
    assert!((balanced.score(&data).unwrap() - 0.8391).abs() < 1e-9);

    // class_weight={1: 5.0}: the class that is not named weighs 1.
    let five_to_one = ClassWeight::PerClass(vec![(1.0, 5.0)]);
    let weighted = spec.clone().class_weight(five_to_one).build().unwrap().fit(&data).unwrap();
    let expected = [
        1.9281541843,
        0.6540886515,
        -0.8885388209,
        1.1362772404,
        -0.1042725612,
        -1.2610183921,
        -1.4655077715,
        -1.2523494514,
    ];
    assert_all_close(&standardised_coefficients(&weighted, &data), &expected, 1e-4);
    assert!((recall_on(&weighted, &data) - 0.706).abs() < 1e-9);

    // Without weights most of the rare class is missed (scikit-learn: 0.386).
    let plain = spec.build().unwrap().fit(&data).unwrap();
    assert!((recall_on(&plain, &data) - 0.386).abs() < 1e-9);
}

#[test]
fn balanced_is_the_same_as_weighting_the_rows_by_hand() {
    let data = dearest_tenth().take_rows(&(0..3000).collect::<Vec<_>>());
    let mut by_hand = data.clone();
    by_hand.weights = Some(balanced_weights(&data));

    let forest = RandomForestSpec::default().n_trees(10).max_depth(Some(6)).seed(3);
    let balanced = forest.clone().class_weight(ClassWeight::Balanced).build().unwrap().fit(&data).unwrap();
    assert_eq!(balanced, forest.build().unwrap().fit(&by_hand).unwrap());

    let boosting = GradientBoostingSpec::default().n_rounds(20).subsample(0.8).seed(3);
    let balanced = boosting.clone().class_weight(ClassWeight::Balanced).build().unwrap().fit(&data).unwrap();
    assert_eq!(balanced, boosting.build().unwrap().fit(&by_hand).unwrap());

    // The linear model agrees too, though not to the last digit: by hand,
    // the weights also enter the scale it measures the features on.
    let linear = LinearModelSpec::default().tolerance(1e-7).max_iterations(2000);
    let balanced = linear.clone().class_weight(ClassWeight::Balanced).build().unwrap().fit(&data).unwrap();
    let by_hand = linear.build().unwrap().fit(&by_hand).unwrap();
    for (balanced, by_hand) in balanced.coefficients(0).iter().zip(by_hand.coefficients(0)) {
        assert!((balanced - by_hand).abs() < 1e-4 * by_hand.abs().max(1e-3), "{balanced} vs {by_hand}");
    }
}

#[test]
fn class_weights_multiply_sample_weights() {
    let mut data = dearest_tenth().take_rows(&(0..3000).collect::<Vec<_>>());
    let class_weights = balanced_weights(&data);
    let sample_weights: Vec<f64> = (0..3000).map(|row| 1.0 + (row % 3) as f64).collect();
    let mut by_hand = data.clone();
    by_hand.weights = Some(sample_weights.iter().zip(&class_weights).map(|(sample, class)| sample * class).collect());
    data.weights = Some(sample_weights);

    let boosting = GradientBoostingSpec::default().n_rounds(20).seed(3);
    let balanced = boosting.clone().class_weight(ClassWeight::Balanced).build().unwrap().fit(&data).unwrap();
    assert_eq!(balanced, boosting.build().unwrap().fit(&by_hand).unwrap());
}

#[test]
fn a_balanced_model_finds_more_of_the_rare_class_on_new_rows() {
    let (train, test) = split(&dearest_tenth());

    let forest = RandomForestSpec::default().n_trees(50).max_depth(Some(6)).seed(1);
    let plain = forest.clone().build().unwrap().fit(&train).unwrap();
    let balanced = forest.class_weight(ClassWeight::Balanced).build().unwrap().fit(&train).unwrap();
    let (plain, balanced) = (recall_on(&plain, &test), recall_on(&balanced, &test));
    assert!(balanced > plain + 0.2, "forest: balanced {balanced}, plain {plain}");

    let boosting = GradientBoostingSpec::default().n_rounds(50).seed(1);
    let plain = boosting.clone().build().unwrap().fit(&train).unwrap();
    let balanced = boosting.class_weight(ClassWeight::Balanced).build().unwrap().fit(&train).unwrap();
    let (plain, balanced) = (recall_on(&plain, &test), recall_on(&balanced, &test));
    assert!(balanced > plain + 0.1, "boosting: balanced {balanced}, plain {plain}");
}

#[test]
fn a_class_with_zero_weight_is_never_predicted() {
    let data = dearest_tenth().take_rows(&(0..3000).collect::<Vec<_>>());
    let ignore_ones = ClassWeight::PerClass(vec![(1.0, 0.0)]);
    let forest = RandomForestSpec::default().n_trees(5).class_weight(ignore_ones).build().unwrap().fit(&data).unwrap();
    assert!(forest.predict_single(&data).unwrap().iter().all(|&class| class == 0.0));
    // The class is still known to the model.
    assert_eq!(forest.classes(), [0.0, 1.0]);
}

#[test]
fn class_weights_that_cannot_be_used_are_errors() {
    let data = dearest_tenth().take_rows(&(0..500).collect::<Vec<_>>());

    // A regressor has no classes.
    assert!(RandomForestSpec::regressor().class_weight(ClassWeight::Balanced).build().is_err());
    assert!(GradientBoostingSpec::regressor().class_weight(ClassWeight::Balanced).build().is_err());
    assert!(LinearModelSpec::regressor().class_weight(ClassWeight::Balanced).build().is_err());

    // A weight is a number, 0 or more, and a class is named once.
    let build =
        |weights: Vec<(f64, f64)>| LinearModelSpec::default().class_weight(ClassWeight::PerClass(weights)).build();
    assert!(build(vec![(1.0, -1.0)]).is_err());
    assert!(build(vec![(1.0, f64::NAN)]).is_err());
    assert!(build(vec![(1.0, 2.0), (1.0, 3.0)]).is_err());
    assert!(build(vec![(0.0, 1.0), (1.0, 3.0)]).is_ok());

    // A class the target does not have; and no weight left on any row.
    assert!(build(vec![(7.0, 2.0)]).unwrap().fit(&data).is_err());
    assert!(build(vec![(0.0, 0.0), (1.0, 0.0)]).unwrap().fit(&data).is_err());

    // Several class targets: each would need its own weights.
    let mut two_targets = data.clone();
    two_targets.y =
        Array2::from_shape_fn((500, 2), |(row, target)| if target == 0 { data.y[[row, 0]] } else { (row % 2) as f64 });
    let balanced = GradientBoostingSpec::default().n_rounds(5).class_weight(ClassWeight::Balanced).build().unwrap();
    assert!(balanced.fit(&two_targets).is_err());
    assert!(GradientBoostingSpec::default().n_rounds(5).build().unwrap().fit(&two_targets).is_ok());
}

#[test]
fn class_weight_can_be_set_by_name() {
    let mut forest = RandomForestSpec::default().build().unwrap();
    forest.set_param("class_weight", &"balanced".into()).unwrap();
    assert_eq!(forest.spec().class_weight, ClassWeight::Balanced);
    forest.set_param("class_weight", &"none".into()).unwrap();
    assert_eq!(forest.spec().class_weight, ClassWeight::None);
    assert!(forest.set_param("class_weight", &"heavy".into()).is_err());
    assert!(forest.set_param("class_weight", &2.into()).is_err());

    let mut boosting = GradientBoostingSpec::default().build().unwrap();
    boosting.set_param("class_weight", &"balanced".into()).unwrap();
    assert_eq!(boosting.spec().class_weight, ClassWeight::Balanced);

    let mut linear = LinearModelSpec::default().build().unwrap();
    linear.set_param("class_weight", &"balanced".into()).unwrap();
    assert_eq!(linear.spec().class_weight, ClassWeight::Balanced);

    // A regressor cannot be switched to balanced.
    let mut regressor = LinearModelSpec::regressor().build().unwrap();
    assert!(regressor.set_param("class_weight", &"balanced".into()).is_err());
}

#[test]
fn settings_saved_before_class_weights_existed_still_load() {
    // A spec as 0.1.0 wrote it has no `class_weight`.
    let old = r#"{"task":"Classification","l2_regularization":0.0,"max_iterations":200,"tolerance":1e-5}"#;
    let spec: LinearModelSpec = serde_json::from_str(old).unwrap();
    assert_eq!(spec, LinearModelSpec::default());
}
