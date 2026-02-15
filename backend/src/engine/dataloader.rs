use crate::engine::types::InputType;
use anyhow::{Context, Result};
use serde::Serialize;
use tera::Tera;

#[derive(Serialize)]
struct DataLoaderContext {
    file_path: String,
    features: Vec<String>,
    targets: Vec<String>,
    batch_size: usize,
    sequence_length: usize,
}

pub fn generate_data_loader(input: &InputType, batch_size: usize) -> Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template(
        "dataloader.py",
        include_str!("../../templates/dataloader.py.j2"),
    )
    .context("Failed to load dataloader template")?;

    let context = match input {
        InputType::Tabular {
            file_path,
            features,
            targets,
            sequence_length,
            ..
        } => DataLoaderContext {
            file_path: file_path.clone(),
            features: features.clone(),
            targets: targets.clone(),
            batch_size,
            sequence_length: *sequence_length,
        },

        // For future me
        _ => {
            return Err(anyhow::anyhow!(
                "Input type not supported for dataloader generation"
            ));
        }
    };

    let code = tera.render("dataloader.py", &tera::Context::from_serialize(&context)?)?;

    Ok(code)
}
