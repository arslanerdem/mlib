use mlib::prelude::*;

const CSV: &str = "\
id,age,fare,member,city
1,22,7.25,true,Paris
2,,71.5,false,Oslo
3,35,NaN,true,
4,41,8,false,Paris
";

fn table() -> Table {
    Table::from_csv_reader(CSV.as_bytes()).unwrap()
}

#[test]
fn csv_types_are_inferred_narrowest_first() {
    let table = table();
    let types: Vec<DataType> = table.schema().iter().map(|(_, data_type)| data_type).collect();
    assert_eq!(types, [DataType::Int, DataType::Int, DataType::Float, DataType::Bool, DataType::Text]);
    assert_eq!(table.num_rows(), 4);
    assert_eq!(table.num_columns(), 5);
}

#[test]
fn empty_cells_and_nan_are_null() {
    let table = table();
    assert_eq!(table.column("age").unwrap().null_count(), 1);
    assert_eq!(table.column("fare").unwrap().null_count(), 1);
    assert_eq!(table.column("city").unwrap().null_count(), 1);
    assert_eq!(table.column("id").unwrap().null_count(), 0);
}

#[test]
fn ragged_rows_fail_to_load() {
    let result = Table::from_csv_reader("a,b\n1,2\n3\n".as_bytes());
    assert!(matches!(result, Err(Error::Csv(_))));
}

#[test]
fn select_and_take_rows_keep_the_requested_order() {
    let table = table();
    let picked = table.select(&["city", "id"]).unwrap();
    assert_eq!(picked.schema().names(), ["city", "id"]);
    assert!(matches!(table.select(&["nope"]), Err(Error::ColumnNotFound(_))));

    let rows = table.take_rows(&[3, 0]);
    assert_eq!(rows.column("id").unwrap(), &Column::ints(&[4, 1]));
}

#[test]
fn drop_nulls_removes_only_rows_missing_the_named_columns() {
    let table = table();
    assert_eq!(table.drop_nulls(&["city"]).unwrap().num_rows(), 3);
    assert_eq!(table.drop_nulls(&["age", "fare"]).unwrap().num_rows(), 2);
}

#[test]
fn with_categorical_dictionary_encodes_with_sorted_categories() {
    let table = table().with_categorical(&["city"]).unwrap();
    assert_eq!(table.schema().get("city"), Some(DataType::Categorical));
    let Column::Categorical { codes, categories } = table.column("city").unwrap() else { panic!() };
    assert_eq!(categories, &["Oslo", "Paris"]);
    assert_eq!(codes, &[Some(1), Some(0), None, Some(1)]);
}

#[test]
fn columns_must_have_equal_lengths_and_distinct_names() {
    let uneven = Table::new(vec![("a", Column::ints(&[1, 2])), ("b", Column::ints(&[1]))]);
    assert!(matches!(uneven, Err(Error::ShapeMismatch { .. })));
    let duplicate = Table::new(vec![("a", Column::ints(&[1])), ("a", Column::ints(&[1]))]);
    assert!(matches!(duplicate, Err(Error::InvalidData(_))));
}

#[test]
fn model_data_checks_columns_and_target_types() {
    let table = table();
    assert!(ModelData::new(&table, ModelSpec::new(&["age"], &["member"])).is_ok());
    assert!(matches!(
        ModelData::new(&table, ModelSpec::new(&["missing"], &["member"])),
        Err(Error::ColumnNotFound(_))
    ));
    assert!(matches!(
        ModelData::new(&table, ModelSpec::new(&["age"], &["city"])),
        Err(Error::TypeMismatch { .. })
    ));
}
