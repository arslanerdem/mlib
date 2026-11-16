//! Metrics and the scorers cross-validation looks up by name.

use mlib::ndarray::array;
use mlib::prelude::*;
use mlib::random::Rng;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// `n` rows, 3 uniform features in [0, 1).
fn features(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = Rng::new(seed, 0);
    Array2::from_shape_fn((n, 3), |_| rng.next_u64() as f64 / u64::MAX as f64)
}

/// Three classes labelled 10, 20, 30, mostly set by the first feature.
fn three_classes(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(row, _)| 10.0 * ((x[[row, 0]] * 2.0 + x[[row, 1]]).floor() + 1.0));
    EncodedData::new(x, y).unwrap()
}

/// Two classes labelled 1 and 2 (not 0 and 1), mostly set by the first feature.
fn ones_and_twos(n: usize, seed: u64) -> EncodedData {
    let x = features(n, seed);
    let y = Array2::from_shape_fn((n, 1), |(row, _)| if x[[row, 0]] + 0.3 * x[[row, 1]] > 0.6 { 2.0 } else { 1.0 });
    EncodedData::new(x, y).unwrap()
}

/// The position of each label among the sorted classes.
fn as_columns(labels: &[f64], classes: &[f64]) -> Vec<f64> {
    labels.iter().map(|label| classes.iter().position(|class| class == label).unwrap() as f64).collect()
}

#[test]
fn classification_metrics_match_scikit_learn() {
    // Reference: scikit-learn 1.9.1.
    let truth = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0];
    let predicted = [0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
    assert!(close(f1_macro(&truth, &predicted), 0.6666666666666666));
    assert!(close(f1_weighted(&truth, &predicted), 0.6666666666666666));
    assert!(close(balanced_accuracy_score(&truth, &predicted), 0.6722222222222222));

    let truth = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let predicted = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    assert!(close(precision_score(&truth, &predicted), 0.6));
    assert!(close(recall_score(&truth, &predicted), 0.75));
    // Scores with ties, which count half.
    let scores = [0.1, 0.6, 0.8, 0.4, 0.4, 0.9, 0.2, 0.6, 0.6, 0.3];
    assert!(close(roc_auc_score(&truth, &scores), 0.8541666666666666));
}

#[test]
fn metrics_have_sensible_answers_at_the_edges() {
    // Nothing predicted positive, nothing actually positive.
    assert_eq!(precision_score(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    assert_eq!(recall_score(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    // One class only: there is no ranking to judge.
    assert!(roc_auc_score(&[1.0, 1.0], &[0.2, 0.9]).is_nan());
    // A reversed ranking, and a constant score.
    assert_eq!(roc_auc_score(&[0.0, 1.0], &[0.9, 0.2]), 0.0);
    assert_eq!(roc_auc_score(&[0.0, 1.0, 0.0, 1.0], &[0.5; 4]), 0.5);
    // With two classes that are both found equally, the F1 averages agree with each other.
    let (truth, predicted) = ([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]);
    assert_eq!(f1_macro(&truth, &predicted), f1_weighted(&truth, &predicted));
    // A class that is only ever predicted counts in the macro average, as a zero.
    assert!(close(f1_macro(&[0.0, 0.0], &[0.0, 5.0]), (2.0 / 3.0 + 0.0) / 2.0));
}

#[test]
fn every_scorer_is_found_by_name() {
    let higher_is_better =
        ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted", "precision", "recall", "r2"];
    for name in higher_is_better {
        let scorer = scorer(name).unwrap();
        assert_eq!(scorer.name(), name);
        assert!(scorer.greater_is_better() && !scorer.uses_probabilities(), "{name}");
    }
    for name in ["rmse", "mse", "mae"] {
        let scorer = scorer(name).unwrap();
        assert!(!scorer.greater_is_better() && !scorer.uses_probabilities(), "{name}");
    }
    for name in ["roc_auc", "roc_auc_ovr"] {
        let scorer = scorer(name).unwrap();
        assert!(scorer.greater_is_better() && scorer.uses_probabilities(), "{name}");
    }
    let log_loss = scorer("log_loss").unwrap();
    assert!(!log_loss.greater_is_better() && log_loss.uses_probabilities());
    assert!(scorer("no_such_metric").is_err());
}

#[test]
fn probability_scorers_match_scikit_learn() {
    // Reference: scikit-learn 1.9.1, log_loss and roc_auc_score(multi_class="ovr").
    let truth = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0];
    let probabilities = array![
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.5, 0.2, 0.3],
        [0.2, 0.2, 0.6],
        [0.6, 0.3, 0.1],
        [0.2, 0.3, 0.5],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.3, 0.3, 0.4]
    ];
    let classes = [0.0, 1.0, 2.0];
    let score = |name: &str| scorer(name).unwrap().score_probabilities(&truth, &probabilities, &classes);
    assert!(close(score("log_loss").unwrap(), 0.6442623858684842));
    assert!(close(score("roc_auc_ovr").unwrap(), 0.9354276895943562));

    // Three classes are one too many for the plain ROC AUC, which says what to use.
    let error = score("roc_auc").unwrap_err().to_string();
    assert!(error.contains("roc_auc_ovr"), "{error}");
    // A scorer of predictions has nothing to say about probabilities.
    assert!(score("accuracy").is_err());

    // The labels can be anything; without them the columns are classes 0, 1, 2.
    let labelled: Vec<f64> = truth.iter().map(|class| 10.0 * (class + 1.0)).collect();
    let log_loss = scorer("log_loss").unwrap();
    let with_labels = log_loss.score_probabilities(&labelled, &probabilities, &[10.0, 20.0, 30.0]).unwrap();
    assert_eq!(with_labels, score("log_loss").unwrap());
    assert_eq!(log_loss.score_probabilities(&truth, &probabilities, &[]).unwrap(), with_labels);
    // A class the columns do not have.
    assert!(log_loss.score_probabilities(&[0.0, 7.0], &array![[0.5, 0.5], [0.5, 0.5]], &[0.0, 1.0]).is_err());
}

#[test]
fn cross_validation_scores_probabilities_for_any_number_of_classes() {
    let data = three_classes(600, 1);
    let model = LinearModelSpec::default().l2_regularization(1.0).build().unwrap();
    let folds = StratifiedKFold::new(3).with_shuffle(5).splits(&data.target().unwrap(), 600);

    // By hand: fit on each fold's training rows, score its test rows.
    let (mut losses, mut areas) = (Vec::new(), Vec::new());
    for (train, test) in &folds {
        let (fitted, test) = (model.fit(&data.take_rows(train)).unwrap(), data.take_rows(test));
        let probabilities = fitted.predict_proba(&test).unwrap();
        let columns = as_columns(&test.target().unwrap(), fitted.classes());
        losses.push(log_loss(&columns, &probabilities));
        let area_of = |class: usize| {
            let is_class: Vec<f64> = columns.iter().map(|&column| (column == class as f64) as u8 as f64).collect();
            roc_auc_score(&is_class, &probabilities.column(class).to_vec())
        };
        areas.push((0..3).map(area_of).sum::<f64>() / 3.0);
    }

    let cv = cross_validate(&model, &data, &folds, &*scorer("log_loss").unwrap()).unwrap();
    // A loss is reported negated.
    assert_eq!(cv.scores, losses.iter().map(|loss| -loss).collect::<Vec<_>>());
    let cv = cross_validate(&model, &data, &folds, &*scorer("roc_auc_ovr").unwrap()).unwrap();
    assert_eq!(cv.scores, areas);
    assert!(cv.mean > 0.9);

    assert!(cross_validate(&model, &data, &folds, &*scorer("roc_auc").unwrap()).is_err());
    let cv = cross_validate(&model, &data, &folds, &*scorer("f1_macro").unwrap()).unwrap();
    assert!(cv.mean > 0.7 && cv.mean <= 1.0);
}

#[test]
fn two_class_scorers_work_whatever_the_labels_are() {
    // 0.1.0 could score probabilities only for classes labelled 0 and 1.
    let data = ones_and_twos(400, 2);
    let model = GradientBoostingSpec::default().n_rounds(20).build().unwrap();
    let folds = StratifiedKFold::new(4).with_shuffle(5).splits(&data.target().unwrap(), 400);

    let (mut losses, mut areas) = (Vec::new(), Vec::new());
    for (train, test) in &folds {
        let (fitted, test) = (model.fit(&data.take_rows(train)).unwrap(), data.take_rows(test));
        let probabilities = fitted.predict_proba(&test).unwrap();
        let is_two = as_columns(&test.target().unwrap(), &[1.0, 2.0]);
        losses.push(-log_loss(&is_two, &probabilities));
        areas.push(roc_auc_score(&is_two, &probabilities.column(1).to_vec()));
    }
    let loss = cross_validate(&model, &data, &folds, &*scorer("log_loss").unwrap()).unwrap();
    for (found, expected) in loss.scores.iter().zip(&losses) {
        assert!((found - expected).abs() < 1e-12);
    }
    let area = cross_validate(&model, &data, &folds, &*scorer("roc_auc").unwrap()).unwrap();
    assert_eq!(area.scores, areas);
    assert!(area.mean > 0.95);
    // With two classes, one-against-the-rest is the same thing.
    let one_vs_rest = cross_validate(&model, &data, &folds, &*scorer("roc_auc_ovr").unwrap()).unwrap();
    assert!((one_vs_rest.mean - area.mean).abs() < 1e-12);
}

#[test]
fn several_class_targets_are_scored_one_by_one_and_averaged() {
    let (colour, size) = (three_classes(600, 3), ones_and_twos(600, 3));
    let mut both = colour.clone();
    both.y = Array2::from_shape_fn(
        (600, 2),
        |(row, target)| if target == 0 { colour.y[[row, 0]] } else { size.y[[row, 0]] },
    );

    // The linear model learns each target exactly as it would alone.
    let model = LinearModelSpec::default().l2_regularization(1.0).build().unwrap();
    let folds = KFold::new(3).with_shuffle(9).splits(600);
    let log_loss = scorer("log_loss").unwrap();
    let together = cross_validate(&model, &both, &folds, &*log_loss).unwrap();
    let alone_colour = cross_validate(&model, &colour, &folds, &*log_loss).unwrap();
    let alone_size = cross_validate(&model, &size, &folds, &*log_loss).unwrap();
    for fold in 0..3 {
        let expected = (alone_colour.scores[fold] + alone_size.scores[fold]) / 2.0;
        assert!((together.scores[fold] - expected).abs() < 1e-9);
    }
}

#[test]
fn a_hand_written_probability_scorer_still_gets_the_second_class() {
    /// The mean squared difference between the truth and the probability of class 1.
    struct Brier;
    impl Scorer for Brier {
        fn name(&self) -> &'static str {
            "brier"
        }
        fn score(&self, y_true: &[f64], probability_of_one: &[f64]) -> f64 {
            mean_squared_error(y_true, probability_of_one)
        }
        fn greater_is_better(&self) -> bool {
            false
        }
        fn uses_probabilities(&self) -> bool {
            true
        }
    }

    let mut data = ones_and_twos(300, 4);
    data.y.mapv_inplace(|label| label - 1.0);
    let model = RandomForestSpec::default().n_trees(20).build().unwrap();
    let folds = KFold::new(3).with_shuffle(1).splits(300);
    let cv = cross_validate(&model, &data, &folds, &Brier).unwrap();
    assert!(cv.mean <= 0.0 && cv.mean > -0.1);

    // It was written for two classes, and says so when given three.
    let error = cross_validate(&model, &three_classes(300, 4), &folds, &Brier).unwrap_err().to_string();
    assert!(error.contains("exactly two classes"), "{error}");
}

#[test]
fn a_fold_without_one_of_the_classes_is_an_error_not_a_wrong_number() {
    // Rows in class order and unshuffled folds: each fold's test rows are one class.
    let x = Array2::from_shape_fn((90, 1), |(row, _)| row as f64);
    let y = Array2::from_shape_fn((90, 1), |(row, _)| (row / 30) as f64);
    let data = EncodedData::new(x, y).unwrap();
    let model = RandomForestSpec::default().n_trees(5).build().unwrap();
    let folds = KFold::new(3).splits(90);
    for name in ["log_loss", "roc_auc_ovr"] {
        let error = cross_validate(&model, &data, &folds, &*scorer(name).unwrap()).unwrap_err().to_string();
        assert!(error.contains("StratifiedKFold"), "{name}: {error}");
    }
}
