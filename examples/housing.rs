//! A complete regression study: predict house values in California
//! districts, starting from a plain baseline and working up.
//!
//! Run from the crate root:  cargo run --release --example housing

use mlib::prelude::*;

fn main() -> Result<(), Error> {
    // 1. Load and inspect. Each row is a district; the target is its
    //    median house value, in units of $100,000.
    let table = Table::from_csv("tests/data/california_housing.csv")?;
    println!("{table}");

    // 2. Say which columns are inputs and which is the target.
    let inputs = [
        "median_income",
        "house_age",
        "avg_rooms",
        "avg_bedrooms",
        "population",
        "avg_occupancy",
        "latitude",
        "longitude",
    ];
    let spec = ModelSpec::new(&inputs, &["median_house_value"]);

    // 3. Hold out 20% of the rows for the final test.
    let (train_rows, test_rows) = train_test_split_shuffled(table.num_rows(), 0.8, 42);
    let train_table = table.take_rows(&train_rows);
    let test_table = table.take_rows(&test_rows);
    let train = ModelData::new(&train_table, spec.clone())?;
    let test = ModelData::new(&test_table, spec.clone())?;

    // 4. Three models, simplest first. Every column is a number with
    //    nothing missing, so no preprocessing is needed.
    let linear = LinearModelSpec::regressor().build()?;
    let forest = RandomForestSpec::regressor().n_trees(200).seed(42).build()?;
    let boosting = GradientBoostingSpec::regressor().n_rounds(300).build()?;
    let candidates: Vec<(&str, Pipeline)> = vec![
        ("linear model", Pipeline::new(vec![], Box::new(linear.clone()))),
        ("random forest", Pipeline::new(vec![], Box::new(forest))),
        ("gradient boosting", Pipeline::new(vec![], Box::new(boosting))),
    ];

    // 5. Compare them by cross-validation on the training rows. R² is the
    //    share of the variation in house values the model explains.
    let folds = KFold::new(5).with_shuffle(42).splits(train.num_rows());
    println!("\ncross-validated R²:");
    for (name, pipe) in &candidates {
        let cv = cross_validate_pipeline(pipe, &train, &folds, &*scorer("r2")?)?;
        println!("  {name:<18} {:.3} ± {:.3}   folds: {:.2?}", cv.mean, cv.std, cv.scores);
    }
    // Look at the folds, not just the mean: the linear model explains
    // about 60% of the variation on four folds and fails badly on one.
    // That fold holds a district reporting 1,243 people per household,
    // and a straight line follows such a value far off the chart. Trees
    // only ask "is it above some threshold?", so they are not thrown.

    // 6. The linear model is the baseline, and the easiest to read: each
    //    coefficient is the change in predicted value per unit of the feature.
    let encoded = candidates[0].1.fit(&train)?.encode_with_targets(&train)?;
    let baseline = linear.fit(&encoded)?;
    println!("\nlinear model: value = {:.3}", baseline.intercepts()[0]);
    for (name, coefficient) in baseline.named_coefficients(0) {
        println!("  {coefficient:+.5} × {name}");
    }

    // 7. Boosting did best. Tune how much it is held back, judged by the
    //    same cross-validation.
    let pipe = &candidates[2].1;
    let grid = ParamGrid::from([
        ("model__learning_rate".to_string(), vec![0.05.into(), 0.1.into()]),
        ("model__feature_fraction".to_string(), vec![0.6.into(), 1.0.into()]),
        ("model__l2_regularization".to_string(), vec![0.0.into(), 5.0.into()]),
    ]);
    let search = grid_search_pipeline(pipe, &grid, &train, &folds, &*scorer("r2")?)?;
    println!("\nbest CV R²: {:.3} with {:?}", search.best_score, search.best_params);

    // 8. Refit the best settings on all training rows; test once.
    let mut best_pipe = pipe.clone();
    for (name, value) in &search.best_params {
        best_pipe.set_param(name, value)?;
    }
    let fitted = best_pipe.fit(&train)?;

    let y_true = test.y()?;
    let y_pred = fitted.predict_single(&test_table)?;
    println!("\ntest R²:   {:.3}", r2_score(&y_true, &y_pred));
    println!("test RMSE: {:.3}  (a typical error of about ${:.0})", root_mean_squared_error(&y_true, &y_pred), {
        root_mean_squared_error(&y_true, &y_pred) * 100_000.0
    });
    println!("\nfeature importances:");
    for (name, importance) in fitted.feature_importances().unwrap_or_default() {
        println!("  {name:<14} {importance:.3}");
    }

    // 9. Predict for two new districts.
    let new_districts = "\
median_income,house_age,avg_rooms,avg_bedrooms,population,avg_occupancy,latitude,longitude
8.0,30,6.5,1.0,900,2.8,37.80,-122.25
2.5,15,4.8,1.1,1800,3.2,36.75,-119.80
";
    let new_table = Table::from_csv_reader(new_districts.as_bytes())?;
    let predictions = fitted.predict_single(&new_table)?;
    println!();
    for (district, prediction) in (1..).zip(&predictions) {
        println!("district {district}: predicted median house value = ${:.0}", prediction * 100_000.0);
    }

    // 10. Save, load, and check the loaded model predicts the same.
    let path = std::env::temp_dir().join("housing_model.json");
    let path = path.to_str().expect("temp path is valid UTF-8");
    save(&fitted, &spec, path)?;
    let saved = load(path)?;
    assert_eq!(saved.fitted.predict_single(&new_table)?, predictions);
    println!("\nsaved to {path} and reloaded: predictions identical");
    Ok(())
}
