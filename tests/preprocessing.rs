use mlib::prelude::*;

fn table() -> Table {
    Table::new(vec![
        ("age", Column::Float(vec![Some(10.0), None, Some(30.0), Some(20.0)])),
        ("rooms", Column::Int(vec![Some(1), Some(1), None, Some(3)])),
        ("city", Column::texts(&["Paris", "Oslo", "Paris", "Rome"])),
    ])
    .unwrap()
}

#[test]
fn imputer_fills_with_the_statistic_learned_at_fit() {
    let mut imputer = Imputer::new(ImputeStrategy::Median);
    let out = imputer.fit_transform(&table()).unwrap();
    assert_eq!(out.column("age").unwrap(), &Column::floats(&[10.0, 20.0, 30.0, 20.0]));
    assert_eq!(out.column("rooms").unwrap(), &Column::floats(&[1.0, 1.0, 1.0, 3.0]));
    assert_eq!(out.column("city"), table().column("city"));

    // New data is filled with the *training* median, not its own.
    let new = Table::new(vec![
        ("age", Column::Float(vec![None, Some(99.0)])),
        ("rooms", Column::ints(&[5, 5])),
        ("city", Column::texts(&["Oslo", "Oslo"])),
    ])
    .unwrap();
    let filled = imputer.transform(&new).unwrap();
    assert_eq!(filled.column("age").unwrap(), &Column::floats(&[20.0, 99.0]));
}

#[test]
fn imputer_strategies() {
    let column = |strategy| {
        let mut imputer = Imputer::new(strategy);
        imputer.fit(&table()).unwrap();
        imputer.fill_values()["rooms"]
    };
    assert_eq!(column(ImputeStrategy::Mean), 5.0 / 3.0);
    assert_eq!(column(ImputeStrategy::MostFrequent), 1.0);
    assert_eq!(column(ImputeStrategy::Constant(-1.0)), -1.0);
}

#[test]
fn unfitted_transformers_refuse_to_transform() {
    let table = table();
    assert!(matches!(Imputer::new(ImputeStrategy::Mean).transform(&table), Err(Error::NotFitted)));
    assert!(matches!(StandardScaler::new().transform(&table), Err(Error::NotFitted)));
    assert!(matches!(OneHotEncoder::new().transform(&table), Err(Error::NotFitted)));
    assert!(matches!(OneHotEncoder::new().output_columns(&[]), Err(Error::NotFitted)));
}

#[test]
fn all_null_columns_are_rejected_at_fit() {
    let table = Table::new(vec![("x", Column::Float(vec![None, None]))]).unwrap();
    assert!(Imputer::new(ImputeStrategy::Mean).fit(&table).is_err());
    assert!(StandardScaler::new().fit(&table).is_err());
}

#[test]
fn scaler_standardises_and_maps_constant_columns_to_zero() {
    let table =
        Table::new(vec![("x", Column::floats(&[1.0, 2.0, 3.0])), ("constant", Column::ints(&[7, 7, 7]))]).unwrap();
    let out = StandardScaler::new().fit_transform(&table).unwrap();
    let Column::Float(x) = out.column("x").unwrap() else { panic!() };
    let std = (2.0f64 / 3.0).sqrt();
    assert_eq!(x, &[Some(-1.0 / std), Some(0.0), Some(1.0 / std)]);
    assert_eq!(out.column("constant").unwrap(), &Column::floats(&[0.0, 0.0, 0.0]));
}

#[test]
fn one_hot_encodes_unseen_categories_as_all_zeros() {
    let train = table().select(&["city"]).unwrap();
    let mut encoder = OneHotEncoder::new();
    let out = encoder.fit_transform(&train).unwrap();
    assert_eq!(out.schema().names(), ["city=Oslo", "city=Paris", "city=Rome"]);
    assert_eq!(out.column("city=Paris").unwrap(), &Column::floats(&[1.0, 0.0, 1.0, 0.0]));

    let new = Table::new(vec![("city", Column::Text(vec![Some("Tokyo".into()), None, Some("Rome".into())]))]).unwrap();
    let encoded = encoder.transform(&new).unwrap();
    assert_eq!(encoded.schema().names(), out.schema().names());
    assert_eq!(encoded.column("city=Oslo").unwrap(), &Column::floats(&[0.0, 0.0, 0.0]));
    assert_eq!(encoded.column("city=Rome").unwrap(), &Column::floats(&[0.0, 0.0, 1.0]));

    let strict = {
        let mut strict = OneHotEncoder::new().with_handle_unknown(HandleUnknown::Error);
        strict.fit(&train).unwrap();
        strict
    };
    assert!(matches!(strict.transform(&new), Err(Error::InvalidData(_))));
}

#[test]
fn one_hot_treats_int_and_categorical_columns_as_categories() {
    let table = Table::new(vec![("class", Column::ints(&[3, 1, 10, 1]))]).unwrap();
    let out = OneHotEncoder::new().fit_transform(&table).unwrap();
    assert_eq!(out.schema().names(), ["class=1", "class=3", "class=10"]);

    // A dictionary-encoded column encodes the same as its text form.
    let text = table_city();
    let categorical = text.with_categorical(&["city"]).unwrap();
    let mut encoder = OneHotEncoder::new();
    let from_text = encoder.fit_transform(&text).unwrap();
    assert_eq!(encoder.transform(&categorical).unwrap(), from_text);
}

fn table_city() -> Table {
    table().select(&["city"]).unwrap()
}

#[test]
fn column_transformer_routes_columns_and_tracks_features() {
    let mut pre = ColumnTransformer::empty()
        .with_branch(
            "num",
            Chain::new(vec![Box::new(Imputer::new(ImputeStrategy::Mean)), Box::new(StandardScaler::new())]),
            &["age"],
        )
        .with_branch("cat", OneHotEncoder::new(), &["city"]);
    let out = pre.fit_transform(&table()).unwrap();
    assert_eq!(out.schema().names(), ["age", "city=Oslo", "city=Paris", "city=Rome"]);

    let input = [Feature::numeric("age"), Feature::numeric("rooms"), Feature::numeric("city")];
    let features = pre.output_columns(&input).unwrap();
    assert_eq!(features[0], Feature::numeric("age"));
    assert_eq!(features[2], Feature::one_hot("city", "Paris"));
    assert_eq!(features.len(), 4);

    // Passthrough appends the unrouted columns.
    let mut keep_rest = pre.clone().with_remainder(Remainder::Passthrough);
    let out = keep_rest.fit_transform(&table()).unwrap();
    assert_eq!(out.schema().names().last(), Some(&"rooms"));
    assert_eq!(keep_rest.output_columns(&input).unwrap().len(), 5);

    // A missing routed column is a schema mismatch.
    let without_city = table().select(&["age", "rooms"]).unwrap();
    assert!(matches!(pre.transform(&without_city), Err(Error::SchemaMismatch(_))));
}

fn table_with_missing_labels() -> Table {
    let text = |values: &[Option<&str>]| Column::Text(values.iter().map(|value| value.map(str::to_string)).collect());
    Table::new(vec![
        ("city", text(&[Some("Rome"), None, Some("Oslo"), Some("Rome"), Some("Oslo")])),
        ("colour", text(&[Some("red"), Some("blue"), None, Some("red"), None])),
        ("owner", Column::Bool(vec![Some(true), None, Some(false), Some(true), Some(true)])),
        ("age", Column::Float(vec![Some(10.0), None, Some(30.0), Some(20.0), Some(5.0)])),
    ])
    .unwrap()
    .with_categorical(&["colour"])
    .unwrap()
}

#[test]
fn categorical_imputer_fills_with_the_most_frequent_label() {
    let table = table_with_missing_labels();
    let mut imputer = CategoricalImputer::most_frequent();
    let out = imputer.fit_transform(&table).unwrap();

    // "Oslo" and "Rome" are equally frequent: the one that sorts first wins.
    assert_eq!(out.column("city").unwrap(), &Column::texts(&["Rome", "Oslo", "Oslo", "Rome", "Oslo"]));
    assert_eq!(
        out.column("owner").unwrap(),
        &Column::Bool(vec![Some(true), Some(true), Some(false), Some(true), Some(true)])
    );
    // A categorical column stays categorical, a numeric one is left alone.
    let colour = out.column("colour").unwrap();
    assert_eq!(colour.data_type(), DataType::Categorical);
    assert_eq!((0..5).map(|i| colour.get_string(i).unwrap()).collect::<Vec<_>>(), ["red", "blue", "red", "red", "red"]);
    assert_eq!(out.column("age"), table.column("age"));

    // New data is filled with the *training* label, not its own most frequent one.
    let new = table.take_rows(&[1, 1, 2]);
    assert_eq!(imputer.transform(&new).unwrap().column("city").unwrap(), &Column::texts(&["Oslo", "Oslo", "Oslo"]));
    assert_eq!(imputer.fill_values()["colour"], "red");
}

#[test]
fn categorical_imputer_fills_with_a_chosen_label() {
    let table = table_with_missing_labels().select(&["city", "colour"]).unwrap();
    let out = CategoricalImputer::constant("missing").fit_transform(&table).unwrap();
    assert_eq!(out.column("city").unwrap(), &Column::texts(&["Rome", "missing", "Oslo", "Rome", "Oslo"]));
    // The new label joins the categorical column's categories.
    assert_eq!(out.column("colour").unwrap().get_string(2).unwrap(), "missing");
    assert_eq!(out.column("colour").unwrap().null_count(), 0);

    // The encoder then gives "missing" a feature of its own.
    let mut encoder = OneHotEncoder::new();
    encoder.fit(&out).unwrap();
    assert!(encoder.categories()["city"].contains(&"missing".to_string()));
}

#[test]
fn categorical_imputer_reports_what_it_cannot_fill() {
    let table = table_with_missing_labels();
    // A true/false column cannot take an arbitrary label.
    assert!(CategoricalImputer::constant("missing").fit(&table).is_err());
    // Nothing to learn from.
    let empty = Table::new(vec![("city", Column::Text(vec![None, None]))]).unwrap();
    assert!(CategoricalImputer::most_frequent().fit(&empty).is_err());
    assert!(matches!(CategoricalImputer::most_frequent().transform(&table), Err(Error::NotFitted)));

    let mut imputer = CategoricalImputer::most_frequent();
    imputer.fit(&table).unwrap();
    assert!(imputer.transform(&table.select(&["city", "age"]).unwrap()).is_err());

    // Reachable by name, for grid search.
    imputer.set_param("fill_value", &ParamValue::from("unknown")).unwrap();
    imputer.set_param("strategy", &ParamValue::from("most_frequent")).unwrap();
    assert!(imputer.set_param("strategy", &ParamValue::from("mean")).is_err());
    assert!(imputer.set_param("no_such_param", &ParamValue::Int(1)).is_err());
}

#[test]
fn every_transformer_passes_the_conformance_battery() {
    use mlib::testing::check_transformer;
    check_transformer(&Imputer::new(ImputeStrategy::Median));
    check_transformer(&CategoricalImputer::most_frequent());
    check_transformer(&CategoricalImputer::constant("missing"));
    check_transformer(&StandardScaler::new());
    check_transformer(&OneHotEncoder::new());
    check_transformer(&Chain::new(vec![Box::new(Imputer::new(ImputeStrategy::Mean)), Box::new(StandardScaler::new())]));
    check_transformer(
        &ColumnTransformer::empty().with_branch("num", StandardScaler::new(), &["height", "rooms"]).with_branch(
            "cat",
            OneHotEncoder::new(),
            &["city"],
        ),
    );
}
