//! The saved-model promise: a file written by an earlier release still
//! loads and still predicts exactly what it predicted then.
//!
//! `tests/data/saved_models/` holds, for every release that changed
//! anything saved, a model file and the predictions it made when it was
//! written. To add the current release's files, run
//!
//! ```sh
//! cargo test --test saved_models -- --ignored
//! ```
//!
//! and commit the new files. Never edit or delete the old ones: they are
//! what users have on disk.

use mlib::prelude::*;

const FOLDER: &str = "tests/data/saved_models";

fn passengers() -> Table {
    let table = Table::from_csv("examples/data/titanic.csv").unwrap();
    table.drop_nulls(&["embarked"]).unwrap().take_rows(&(0..200).collect::<Vec<_>>())
}

fn preprocessing() -> ColumnTransformer {
    ColumnTransformer::empty()
        .with_branch("num", Chain::new(vec![Box::new(Imputer::new(ImputeStrategy::Median)), Box::new(StandardScaler::new())]), &["age", "fare"])
        .with_branch("cat", OneHotEncoder::new(), &["pclass", "sex", "embarked"])
}

/// Small models that between them use every type that can be saved.
fn fitted_pipelines(table: &Table) -> Vec<(&'static str, FittedPipeline, ModelSpec)> {
    let inputs = ["pclass", "sex", "age", "fare", "embarked"];
    let fit = |model: Box<dyn AnyEstimator>, targets: &[&str]| {
        let spec = ModelSpec::new(&inputs, targets);
        let pipe = Pipeline::new(vec![Box::new(preprocessing())], model);
        (pipe.fit(&ModelData::new(table, spec.clone()).unwrap()).unwrap(), spec)
    };
    let forest = RandomForestSpec::new().n_trees(5).max_depth(Some(4)).seed(1).build().unwrap();
    let forest_two_targets = RandomForestSpec::regressor().n_trees(5).max_depth(Some(4)).seed(1).build().unwrap();
    let boosting = GradientBoostingSpec::new().n_rounds(5).max_depth(Some(3)).build().unwrap();
    let boosting_regressor =
        GradientBoostingSpec::regressor().n_rounds(20).early_stopping_rounds(Some(3)).subsample(0.8).build().unwrap();

    let named = |name, (fitted, spec)| (name, fitted, spec);
    vec![
        named("forest_classifier", fit(Box::new(forest), &["survived"])),
        named("forest_two_targets", fit(Box::new(forest_two_targets), &["survived", "sibsp"])),
        named("boosting_classifier", fit(Box::new(boosting), &["survived"])),
        named("boosting_regressor", fit(Box::new(boosting_regressor), &["parch"])),
    ]
}

fn predictions_as_rows(fitted: &FittedPipeline, table: &Table) -> Vec<Vec<f64>> {
    fitted.predict(table).unwrap().rows().into_iter().map(|row| row.to_vec()).collect()
}

#[test]
#[ignore = "writes fixture files; run it once per release that changes what is saved"]
fn write_files_for_this_release() {
    let table = passengers();
    let version = env!("CARGO_PKG_VERSION");
    std::fs::create_dir_all(FOLDER).unwrap();
    for (name, fitted, spec) in fitted_pipelines(&table) {
        save(&fitted, &spec, &format!("{FOLDER}/{name}-{version}.json")).unwrap();
        let expected = serde_json::to_string(&predictions_as_rows(&fitted, &table)).unwrap();
        std::fs::write(format!("{FOLDER}/{name}-{version}.predictions"), expected).unwrap();
    }
}

#[test]
fn files_from_earlier_releases_load_and_predict_the_same() {
    let table = passengers();
    let mut checked = 0;
    for entry in std::fs::read_dir(FOLDER).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let saved = load(path.to_str().unwrap()).unwrap_or_else(|error| panic!("{path:?} no longer loads: {error}"));
        let expected: Vec<Vec<f64>> =
            serde_json::from_str(&std::fs::read_to_string(path.with_extension("predictions")).unwrap()).unwrap();

        let predicted = predictions_as_rows(&saved.fitted, &table);
        assert_eq!(predicted.len(), expected.len(), "{path:?}");
        for (now, then) in predicted.iter().flatten().zip(expected.iter().flatten()) {
            // Not exact equality: `exp` may round differently on another machine.
            assert!((now - then).abs() < 1e-9, "{path:?} predicts {now} where it used to predict {then}");
        }
        checked += 1;
    }
    assert!(checked >= 4, "expected saved-model files in {FOLDER}, found {checked}");
}
