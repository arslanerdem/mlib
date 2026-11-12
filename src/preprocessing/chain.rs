//! `Chain`: a sequence of transformers that is itself a transformer.

use crate::encoded::Feature;
use crate::error::Result;
use crate::table::Table;
use crate::traits::{ParamValue, Transformer};
use serde::{Deserialize, Serialize};

/// Runs its steps in order — e.g. impute then scale inside one
/// `ColumnTransformer` branch.
#[derive(Debug, Serialize, Deserialize)]
pub struct Chain {
    pub steps: Vec<Box<dyn Transformer>>,
}

impl Chain {
    pub fn new(steps: Vec<Box<dyn Transformer>>) -> Chain {
        Chain { steps }
    }
}

impl Clone for Chain {
    fn clone(&self) -> Chain {
        Chain { steps: self.steps.iter().map(|step| step.clone_box()).collect() }
    }
}

#[typetag::serde]
impl Transformer for Chain {
    fn fit(&mut self, table: &Table) -> Result<()> {
        self.fit_transform(table).map(|_| ())
    }

    fn transform(&self, table: &Table) -> Result<Table> {
        let mut table = table.clone();
        for step in &self.steps {
            table = step.transform(&table)?;
        }
        Ok(table)
    }

    fn fit_transform(&mut self, table: &Table) -> Result<Table> {
        let mut table = table.clone();
        for step in &mut self.steps {
            table = step.fit_transform(&table)?;
        }
        Ok(table)
    }

    fn output_columns(&self, input: &[Feature]) -> Result<Vec<Feature>> {
        let mut features = input.to_vec();
        for step in &self.steps {
            features = step.output_columns(&features)?;
        }
        Ok(features)
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }

    /// Offers the parameter to each step; the first step that knows it takes it.
    fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<()> {
        set_param_on_first(&mut self.steps, name, value)
    }
}

/// Offer a parameter to each step in turn; the first that accepts it wins.
/// If none does, the last step's error is returned.
pub(crate) fn set_param_on_first(
    steps: &mut [Box<dyn Transformer>],
    name: &str,
    value: &ParamValue,
) -> Result<()> {
    let mut outcome = Err(crate::Error::InvalidSpec(format!("unknown parameter: {name}")));
    for step in steps {
        outcome = step.set_param(name, value);
        if outcome.is_ok() {
            break;
        }
    }
    outcome
}
