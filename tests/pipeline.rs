use mlib::prelude::*;

/// 120 rows: survival depends on sex and (weakly) on age; `age` has
/// nulls, `port` is a text category.
fn passengers() -> Table {
    let mut csv = String::from("id,sex,age,port,fare,survived\n");
    for i in 0..120 {
        let sex = if i % 2 == 0 { "female" } else { "male" };
        let age = if i % 7 == 0 { String::new() } else { format!("{}", 18 + (i * 13) % 50) };
        let port = ["S", "C", "Q"][i % 3];
        let fare = 5.0 + (i * 17 % 90) as f64 / 2.0;
        let survived = (i % 2 == 0 && i % 10 != 0 || i % 15 == 1) as u8;
        csv.push_str(&format!("{i},{sex},{age},{port},{fare},{survived}\n"));
    }
    Table::from_csv_reader(csv.as_bytes()).unwrap()
}

fn spec() -> ModelSpec {
    ModelSpec::new(&["sex", "age", "port", "fare"], &["survived"]).with_id("id")
}

fn pipeline() -> Pipeline {
    let pre = ColumnTransformer::empty()
        .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age", "fare"])
        .with_branch("cat", OneHotEncoder::new(), &["sex", "port"]);
    let forest = RandomForestSpec::default().n_trees(30).seed(42).build().unwrap();
    Pipeline::new(vec![Box::new(pre)], Box::new(forest))
}

#[test]
fn fit_predict_and_score_end_to_end() {
    let table = passengers();
    let data = ModelData::new(&table, spec()).unwrap();
    let fitted = pipeline().fit(&data).unwrap();

    let names: Vec<String> = fitted.features().iter().map(Feature::display_name).collect();
    assert_eq!(names, ["age", "fare", "sex=female", "sex=male", "port=C", "port=Q", "port=S"]);
    assert_eq!(fitted.schema().names(), ["sex", "age", "port", "fare"]);
    assert!(fitted.score(&data).unwrap() > 0.9);
    assert_eq!(fitted.predict(&table).unwrap().dim(), (120, 1));
    assert_eq!(fitted.predict_single(&table).unwrap().len(), 120);

    let importances = fitted.feature_importances().unwrap();
    assert!(importances[0].0.starts_with("sex="), "most important: {importances:?}");
}

#[test]
fn new_data_may_have_missing_values_unseen_categories_and_no_target() {
    let table = passengers();
    let fitted = pipeline().fit(&ModelData::new(&table, spec()).unwrap()).unwrap();

    // No id, no target; a missing age; port "X" never seen; whole-number fares.
    let new = Table::from_csv_reader("sex,age,port,fare\nfemale,,X,10\nmale,40,S,20\n".as_bytes()).unwrap();
    assert_eq!(fitted.predict(&new).unwrap().nrows(), 2);
    assert_eq!(fitted.predict_proba(&new).unwrap().dim(), (2, 2));
}

#[test]
fn schema_mismatches_are_errors() {
    let table = passengers();
    let fitted = pipeline().fit(&ModelData::new(&table, spec()).unwrap()).unwrap();

    let missing_column = table.select(&["sex", "age", "fare"]).unwrap();
    assert!(matches!(fitted.predict(&missing_column), Err(Error::SchemaMismatch(_))));

    let wrong_type = Table::from_csv_reader("sex,age,port,fare\nfemale,old,S,10\n".as_bytes()).unwrap();
    assert!(matches!(fitted.predict(&wrong_type), Err(Error::SchemaMismatch(_))));
}

#[test]
fn unencoded_text_and_unfilled_nulls_are_reported_clearly() {
    let table = passengers();
    let forest = || Box::new(RandomForestSpec::default().n_trees(5).build().unwrap());

    let no_encoder = Pipeline::new(vec![], forest());
    let result = no_encoder.fit(&ModelData::new(&table, ModelSpec::new(&["sex"], &["survived"])).unwrap());
    assert!(matches!(result, Err(Error::TypeMismatch { .. })));

    let no_imputer = Pipeline::new(vec![], forest());
    let result = no_imputer.fit(&ModelData::new(&table, ModelSpec::new(&["age"], &["survived"])).unwrap());
    assert!(matches!(result, Err(Error::InvalidData(_))));
}

#[test]
fn save_and_load_round_trip_predicts_identically() {
    let table = passengers();
    let fitted = pipeline().fit(&ModelData::new(&table, spec()).unwrap()).unwrap();
    let path = std::env::temp_dir().join("mlib_round_trip_test.json");
    let path = path.to_str().unwrap();

    save(&fitted, &spec(), path).unwrap();
    let saved = load(path).unwrap();
    assert_eq!(saved.spec, spec());
    assert_eq!(&saved.schema, fitted.schema());
    assert_eq!(saved.fitted.predict_proba(&table).unwrap(), fitted.predict_proba(&table).unwrap());

    // Same pipeline, same data → byte-identical file.
    let refitted = pipeline().fit(&ModelData::new(&table, spec()).unwrap()).unwrap();
    let second_path = format!("{path}.second");
    save(&refitted, &spec(), &second_path).unwrap();
    assert_eq!(std::fs::read(path).unwrap(), std::fs::read(&second_path).unwrap());

    // An unknown format version is rejected outright.
    let text = std::fs::read_to_string(path).unwrap();
    std::fs::write(path, text.replacen("\"format_version\":1", "\"format_version\":99", 1)).unwrap();
    assert!(matches!(load(path), Err(Error::UnsupportedVersion { version: 99, .. })));

    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(second_path).unwrap();
}

#[test]
fn cross_validation_reports_one_score_per_fold() {
    let table = passengers();
    let data = ModelData::new(&table, spec()).unwrap();
    let folds = KFold::new(5).with_shuffle(1).splits(table.num_rows());
    let cv = cross_validate_pipeline(&pipeline(), &data, &folds, &*scorer("accuracy").unwrap()).unwrap();
    assert_eq!(cv.scores.len(), 5);
    assert!(cv.mean > 0.75, "mean {}", cv.mean); // the target is ~13% unlearnable noise

    // Loss metrics come back negated, so higher is still better.
    let loss = cross_validate_pipeline(&pipeline(), &data, &folds, &*scorer("log_loss").unwrap()).unwrap();
    assert!(loss.mean < 0.0);

    let no_folds = KFold::new(1).splits(table.num_rows());
    assert!(cross_validate_pipeline(&pipeline(), &data, &no_folds, &*scorer("accuracy").unwrap()).is_err());
    assert!(scorer("nonsense").is_err());
}

#[test]
fn grid_search_tries_every_combination() {
    let table = passengers();
    let data = ModelData::new(&table, spec()).unwrap();
    let folds = StratifiedKFold::new(3).splits(&data.y().unwrap(), table.num_rows());

    let grid = ParamGrid::from([
        ("model__max_depth".to_string(), vec![1.into(), 4.into()]),
        ("num__strategy".to_string(), vec!["mean".into(), "median".into()]),
    ]);
    let result = grid_search_pipeline(&pipeline(), &grid, &data, &folds, &*scorer("accuracy").unwrap()).unwrap();
    assert_eq!(result.all_scores.len(), 4);
    assert!(result.all_scores.iter().all(|(_, score)| *score <= result.best_score));

    let bad_grid = ParamGrid::from([("model__nope".to_string(), vec![1.into()])]);
    assert!(grid_search_pipeline(&pipeline(), &bad_grid, &data, &folds, &*scorer("accuracy").unwrap()).is_err());
}

#[test]
fn splitters_cover_every_row_exactly_once() {
    let covers_all = |folds: &Folds, n: usize| {
        let mut tested: Vec<usize> = folds.iter().flat_map(|(_, test)| test.clone()).collect();
        tested.sort();
        tested == (0..n).collect::<Vec<_>>() && folds.iter().all(|(train, test)| train.len() + test.len() == n)
    };
    assert!(covers_all(&KFold::new(4).splits(10), 10));
    assert!(covers_all(&KFold::new(3).with_shuffle(5).splits(10), 10));

    // Stratified: each fold gets the same share of each class.
    let y: Vec<f64> = (0..30).map(|i| (i % 3 == 0) as u8 as f64).collect();
    let folds = StratifiedKFold::new(5).splits(&y, 30);
    assert!(covers_all(&folds, 30));
    assert!(folds.iter().all(|(_, test)| test.iter().filter(|&&i| y[i] == 1.0).count() == 2));

    // Grouped: no group appears on both sides of a fold.
    let groups: Vec<i64> = (0..30).map(|i| i / 4).collect();
    let folds = GroupKFold::new(3).splits(&groups, 30);
    assert!(covers_all(&folds, 30));
    assert!(folds.iter().all(|(train, test)| test.iter().all(|&i| train.iter().all(|&j| groups[i] != groups[j]))));

    assert_eq!(train_test_split(10, 0.8), ((0..8).collect(), vec![8, 9]));
    let (train, test) = train_test_split_shuffled(10, 0.8, 3);
    assert_eq!((train.len(), test.len()), (8, 2));
}

#[test]
fn a_pipeline_predicts_two_targets_together() {
    // Predict a district's median income and house value from everything
    // else. scikit-learn's forest scores a mean R² of 0.75 on this.
    let table = Table::from_csv("tests/data/california_housing.csv").unwrap();
    let inputs = ["house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude"];
    let spec = ModelSpec::new(&inputs, &["median_income", "median_house_value"]);
    let data = ModelData::new(&table, spec.clone()).unwrap();
    let forest = RandomForestSpec::regressor().n_trees(30).seed(1).build().unwrap();
    let pipe = Pipeline::new(vec![], Box::new(forest));

    // Cross-validation averages the score over the two targets.
    let folds = KFold::new(3).with_shuffle(1).splits(table.num_rows());
    let cv = cross_validate_pipeline(&pipe, &data, &folds, &*scorer("r2").unwrap()).unwrap();
    assert!(cv.mean > 0.70, "mean r2 {}", cv.mean);

    // One row per table row, one column per target, in the spec's order.
    let fitted = pipe.fit(&data).unwrap();
    let predictions = fitted.predict(&table).unwrap();
    assert_eq!(predictions.dim(), (table.num_rows(), 2));
    assert!(fitted.predict_single(&table).is_err());

    // The saved model keeps both targets.
    let path = std::env::temp_dir().join("mlib_two_targets_test.json");
    let path = path.to_str().unwrap();
    save(&fitted, &spec, path).unwrap();
    assert_eq!(load(path).unwrap().fitted.predict(&table).unwrap(), predictions);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_plain_target_list_needs_exactly_one_target() {
    let x = Array2::zeros((4, 2));
    assert!(EncodedData::new(x.clone(), Array2::zeros((4, 1))).unwrap().target().is_ok());
    assert!(EncodedData::new(x.clone(), Array2::zeros((4, 2))).unwrap().target().is_err());
    assert!(EncodedData::new(x, Array2::zeros((4, 0))).unwrap().target().is_err());
}

#[test]
fn the_confusion_matrix_counts_actual_against_predicted() {
    let matrix = confusion_matrix(&[0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 1.0, 0.0], 2);
    assert_eq!(matrix, mlib::ndarray::array![[1, 1], [1, 2]]);
}
