//! `Pipeline`: preprocessing steps plus a final model, fitted and applied
//! as one unit — so what is trained is exactly what predicts.

use crate::encoded::{rank_importances, EncodedData, Feature};
use crate::error::{Error, Result};
use crate::preprocessing::Chain;
use crate::spec::ModelData;
use crate::table::{DataType, Schema, Table};
use crate::traits::{AnyEstimator, Label, ParamValue, Predictor, Transformer};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// An unfitted pipeline. `fit` leaves it untouched, so one pipeline can
/// be fitted many times (once per cross-validation fold, for instance).
///
/// ```
/// use mlib::prelude::*;
///
/// // 40 customers; the older half churned.
/// let ages: Vec<f64> = (0..40).map(|row| 20.0 + row as f64).collect();
/// let cities: Vec<&str> = (0..40).map(|row| if row % 2 == 0 { "Oslo" } else { "Rome" }).collect();
/// let churned: Vec<i64> = (0..40).map(|row| (row >= 20) as i64).collect();
/// let table = Table::new(vec![
///     ("age", Column::floats(&ages)),
///     ("city", Column::texts(&cities)),
///     ("churned", Column::ints(&churned)),
/// ])?;
/// let data = ModelData::new(&table, ModelSpec::new(&["age", "city"], &["churned"]))?;
///
/// let pre = ColumnTransformer::empty()
///     .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age"])
///     .with_branch("cat", OneHotEncoder::new(), &["city"]);
/// let forest = RandomForestSpec::default().n_trees(20).seed(1).build()?;
/// let mut pipe = Pipeline::new(vec![Box::new(pre)], Box::new(forest));
///
/// // Settings can be changed by name, as grid search does.
/// pipe.set_param("model__max_depth", &3.into())?;
/// pipe.set_param("num__strategy", &"mean".into())?;
///
/// let fitted = pipe.fit(&data)?;
/// assert!(fitted.score(&data)? > 0.9);
/// # Ok::<(), Error>(())
/// ```
pub struct Pipeline {
    steps: Vec<Box<dyn Transformer>>,
    model: Box<dyn AnyEstimator>,
}

impl Clone for Pipeline {
    fn clone(&self) -> Pipeline {
        Pipeline { steps: self.steps.iter().map(|step| step.clone_box()).collect(), model: self.model.clone_box() }
    }
}

impl Pipeline {
    /// The preprocessing steps, applied in order, and the model that
    /// trains on their result. With no steps, every x column must
    /// already be numeric and complete.
    pub fn new(steps: Vec<Box<dyn Transformer>>, model: Box<dyn AnyEstimator>) -> Pipeline {
        Pipeline { steps, model }
    }

    /// Fit every step on the training data, then fit the model on the
    /// encoded result.
    pub fn fit(&self, data: &ModelData) -> Result<FittedPipeline> {
        let input = data.x_table();
        let mut steps = Chain::new(self.steps.iter().map(|step| step.clone_box()).collect());
        let encoded_table = steps.fit_transform(&input)?;
        let features = steps.output_columns(&data.input_features())?;

        let encoded = EncodedData::from_table(&encoded_table, features.clone(), data.y_matrix()?, data.weights())?;
        let model = self.model.fit_boxed(&encoded)?;
        Ok(FittedPipeline { steps: steps.steps, model, schema: input.schema, features })
    }

    /// Set a hyper-parameter by name. `model__n_trees` goes to the model.
    /// Anything else is offered to the steps — e.g. `num__strategy`
    /// reaches the `num` branch of a `ColumnTransformer`.
    pub fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        match name.strip_prefix("model__") {
            Some(param) => self.model.set_param_boxed(param, value),
            None => crate::preprocessing::set_param_on_first(&mut self.steps, name, value),
        }
    }
}

/// A fitted pipeline: fitted steps, fitted model, and the input it expects.
/// This is the unit that gets saved and loaded.
///
/// ```
/// use mlib::prelude::*;
///
/// // 40 customers; the older half churned.
/// let ages: Vec<f64> = (0..40).map(|row| 20.0 + row as f64).collect();
/// let cities: Vec<&str> = (0..40).map(|row| if row % 2 == 0 { "Oslo" } else { "Rome" }).collect();
/// let churned: Vec<i64> = (0..40).map(|row| (row >= 20) as i64).collect();
/// let table = Table::new(vec![
///     ("age", Column::floats(&ages)),
///     ("city", Column::texts(&cities)),
///     ("churned", Column::ints(&churned)),
/// ])?;
/// let data = ModelData::new(&table, ModelSpec::new(&["age", "city"], &["churned"]))?;
/// let pre = ColumnTransformer::empty()
///     .with_branch("num", Imputer::new(ImputeStrategy::Median), &["age"])
///     .with_branch("cat", OneHotEncoder::new(), &["city"]);
/// let forest = RandomForestSpec::default().n_trees(20).seed(1).build()?;
/// let fitted = Pipeline::new(vec![Box::new(pre)], Box::new(forest)).fit(&data)?;
///
/// // New rows need only the x columns; the fitted steps are replayed on them.
/// let new = Table::new(vec![("age", Column::floats(&[25.0, 55.0])), ("city", Column::texts(&["Rome", "Oslo"]))])?;
/// assert_eq!(fitted.predict_single(&new)?, [0.0, 1.0]);
/// assert_eq!(fitted.predict(&new)?.dim(), (2, 1));       // rows × targets
/// assert_eq!(fitted.predict_proba(&new)?.dim(), (2, 2)); // rows × classes
/// assert_eq!(fitted.target_classes(), [[0.0, 1.0]]);
///
/// // What the model saw, and what it made of it.
/// let names: Vec<String> = fitted.features().iter().map(Feature::display_name).collect();
/// assert_eq!(names, ["age", "city=Oslo", "city=Rome"]);
/// let (most_important, _) = &fitted.feature_importances().unwrap()[0];
/// assert_eq!(most_important, "age");
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct FittedPipeline {
    steps: Vec<Box<dyn Transformer>>,
    model: Box<dyn Predictor>,
    schema: Schema,
    features: Vec<Feature>,
}

impl FittedPipeline {
    /// The fitted preprocessing steps, in the order they are applied.
    pub fn steps(&self) -> &[Box<dyn Transformer>] {
        &self.steps
    }

    /// The fitted model.
    pub fn model(&self) -> &dyn Predictor {
        self.model.as_ref()
    }

    /// The x columns the pipeline was fitted on: names and types.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The features the model sees, fixed at fit time.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Predict for every row of a table: one row per table row, one
    /// column per target. The table must contain the x columns seen at
    /// fit time (extra columns are ignored); otherwise this fails with
    /// `Error::SchemaMismatch`.
    pub fn predict(&self, table: &Table) -> Result<Array2<f64>> {
        self.model.predict(&self.encode(table)?)
    }

    /// Like `predict`, for a model with one target: a plain list with one
    /// value per table row.
    pub fn predict_single(&self, table: &Table) -> Result<Vec<Label>> {
        self.model.predict_single(&self.encode(table)?)
    }

    /// Class probabilities for every row, one column per class
    /// (classifiers only). With several targets, each target's classes
    /// sit side by side in target order; `target_classes` names the columns.
    pub fn predict_proba(&self, table: &Table) -> Result<Array2<f64>> {
        self.model.predict_proba(&self.encode(table)?)
    }

    /// The class labels of each target, sorted, one list per target.
    /// Empty for a regressor.
    pub fn target_classes(&self) -> Vec<Vec<Label>> {
        self.model.target_classes()
    }

    /// The model's default score on labelled data: accuracy for
    /// classifiers, R² for regressors.
    pub fn score(&self, data: &ModelData) -> Result<f64> {
        self.model.score(&self.encode_with_targets(data)?)
    }

    /// Features with their importance, most important first. `None` if
    /// the model does not report importances.
    pub fn feature_importances(&self) -> Option<Vec<(String, f64)>> {
        Some(rank_importances(&self.features, &self.model.feature_importances()?))
    }

    /// Run a table's x columns through the fitted steps.
    pub fn encode(&self, table: &Table) -> Result<EncodedData> {
        let mut encoded_table = self.select_input(table)?;
        for step in &self.steps {
            encoded_table = step.transform(&encoded_table)?;
        }
        let no_targets = Array2::zeros((table.num_rows(), 0));
        EncodedData::from_table(&encoded_table, self.features.clone(), no_targets, None)
    }

    /// Like `encode`, with the targets and weights the spec names.
    pub fn encode_with_targets(&self, data: &ModelData) -> Result<EncodedData> {
        let mut encoded = self.encode(data.table)?;
        encoded.y = data.y_matrix()?;
        encoded.weights = data.weights();
        Ok(encoded)
    }

    /// Pick the fit-time x columns out of a table, checking their types.
    fn select_input(&self, table: &Table) -> Result<Table> {
        for (name, expected) in &self.schema.fields {
            let column = table.column(name).ok_or_else(|| Error::SchemaMismatch(format!("missing column '{name}'")))?;
            let all_null = column.null_count() == column.len();
            if !all_null && !compatible(*expected, column.data_type()) {
                return Err(Error::SchemaMismatch(format!(
                    "column '{name}' was {expected:?} at fit time but is {:?} now",
                    column.data_type()
                )));
            }
        }
        table.select(&self.schema.names())
    }
}

/// Whether a column of type `found` can stand in for one fitted as `expected`.
/// Whole numbers are fine where decimals were seen, and text is text
/// whether or not it is dictionary-encoded.
fn compatible(expected: DataType, found: DataType) -> bool {
    use DataType::*;
    expected == found || matches!((expected, found), (Float, Int) | (Text, Categorical) | (Categorical, Text))
}
