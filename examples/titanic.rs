//! A complete study: predict Titanic survival.
//!
//! Run from the crate root:  cargo run --release --example titanic

use mlib::prelude::*;

fn main() -> Result<(), Error> {
    // 1. Load and inspect.
    let table = Table::from_csv("examples/data/titanic.csv")?;
    println!("{table}");
    for (name, column) in table.named_columns() {
        if column.null_count() > 0 {
            println!("{name:<10} {} nulls", column.null_count());
        }
    }

    // 2. Clean up: cabin is mostly missing; name and ticket carry no
    //    signal here. Two rows have no port of embarkation — drop them.
    let table = table
        .select(&["passenger_id", "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"])?
        .drop_nulls(&["embarked"])?;
    println!("\nrows after cleanup: {}", table.num_rows());

    // 3. Say which columns are inputs and which is the target.
    let spec = ModelSpec::new(&["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"], &["survived"])
        .with_id("passenger_id");

    // 4. Hold out 20% of the rows for the final test.
    let (train_rows, test_rows) = train_test_split_shuffled(table.num_rows(), 0.8, 42);
    let train_table = table.take_rows(&train_rows);
    let test_table = table.take_rows(&test_rows);
    let train = ModelData::new(&train_table, spec.clone())?;
    let test = ModelData::new(&test_table, spec.clone())?;

    // 5. The pipeline: fill missing numbers with the median, one-hot
    //    encode the categories (pclass is a category, not a quantity),
    //    then a random forest.
    let pre = ColumnTransformer::empty()
        .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age", "sibsp", "parch", "fare"])
        .with_branch("cat", OneHotEncoder::new(), &["pclass", "sex", "embarked"]);
    let forest = RandomForestSpec::default().n_trees(200).seed(42).build()?;
    let pipe = Pipeline::new(vec![Box::new(pre)], Box::new(forest));

    // 6. Cross-validate on the training rows. Every fold refits the
    //    imputer, the encoder and the forest on its own rows.
    let folds = StratifiedKFold::new(5).with_shuffle(42).splits(&train.y()?, train.num_rows());
    let cv = cross_validate_pipeline(&pipe, &train, &folds, &*scorer("accuracy")?)?;
    println!("\nCV accuracy per fold: {:.3?}", cv.scores);
    println!("CV accuracy: {:.3} ± {:.3}", cv.mean, cv.std);

    // 7. Tune two forest parameters, judged by the same cross-validation.
    let grid = ParamGrid::from([
        ("model__n_trees".to_string(), vec![100.into(), 300.into()]),
        ("model__max_depth".to_string(), vec!["none".into(), 4.into(), 8.into()]),
    ]);
    let search = grid_search_pipeline(&pipe, &grid, &train, &folds, &*scorer("accuracy")?)?;
    println!("\nbest CV accuracy: {:.3} with {:?}", search.best_score, search.best_params);

    // 8. Refit the best settings on all training rows; test once.
    let mut best_pipe = pipe.clone();
    for (name, value) in &search.best_params {
        best_pipe.set_param(name, value)?;
    }
    let fitted = best_pipe.fit(&train)?;

    let y_true = test.y()?;
    let y_pred = fitted.predict_single(&test_table)?;
    println!("\ntest accuracy: {:.3}", accuracy(&y_true, &y_pred));
    println!("confusion matrix (rows = actual, columns = predicted):");
    for row in confusion_matrix(&y_true, &y_pred, 2).rows() {
        println!("  {row}");
    }
    println!("\nfeature importances:");
    for (name, importance) in fitted.feature_importances().unwrap_or_default() {
        println!("  {name:<12} {importance:.3}");
    }

    // 9. Predict for new passengers: one has no age, and one boarded at
    //    a port the model has never seen.
    let new_passengers = "\
passenger_id,pclass,sex,age,sibsp,parch,fare,embarked
892,3,male,34.5,0,0,7.8292,Q
893,3,female,,1,0,15.0,X
894,1,female,45,0,1,120.0,S
";
    let new_table = Table::from_csv_reader(new_passengers.as_bytes())?;
    let predictions = fitted.predict_single(&new_table)?;
    println!();
    for (id, prediction) in (892..).zip(&predictions) {
        println!("passenger {id}: predicted survived = {prediction}");
    }

    // 10. Save, load, and check the loaded model predicts the same.
    let path = std::env::temp_dir().join("titanic_model.json");
    let path = path.to_str().expect("temp path is valid UTF-8");
    save(&fitted, &spec, path)?;
    let saved = load(path)?;
    assert_eq!(saved.fitted.predict_single(&new_table)?, predictions);
    println!("\nsaved to {path} and reloaded: predictions identical");
    Ok(())
}
