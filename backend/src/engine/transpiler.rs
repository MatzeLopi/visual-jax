use crate::engine::types::{ActivationType, InputType, LayerType, LossType, MetricType, NodeKind};
use crate::schemas::training::TrainParams;
use anyhow::Result;
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};
use tera::Tera;

// Define the data structure passed to the Jinja2 template
#[derive(Serialize)]
struct ModelContext {
    init_lines: Vec<String>,
    call_blocks: Vec<String>,
}

#[derive(Serialize)]
struct DataLoaderContext {
    file_path: String,
    features: Vec<String>,
    targets: Vec<String>,
    batch_size: usize,
    sequence_length: usize,
    separator: String,
}

pub fn transpile_model(
    sorted_nodes: Vec<(String, NodeKind)>,
    incoming_map: HashMap<String, Vec<String>>,
    tera: &Arc<Tera>,
) -> Result<String> {
    // 2. State Tracking
    let mut init_lines = Vec::new();
    let mut call_blocks = Vec::new();

    // Flags & tables
    let mut var_map: HashMap<String, String> = HashMap::new(); // Maps NodeID -> Python Var Name
    let mut last_output_var = "x".to_string(); // Default to input 'x' if graph is empty

    // 3. Iterate Nodes in Topological Order
    for (id, kind) in sorted_nodes {
        let clean_id = id.replace("-", "_");
        let current_out_var = format!("out_{}", clean_id);

        // --- A. Resolve Inputs ---
        let parents = incoming_map.get(&id).cloned().unwrap_or_default();

        // Collect all parent Python variable names
        let mut parent_vars = Vec::new();
        for p in &parents {
            let var = var_map.get(p).ok_or_else(|| {
                anyhow::anyhow!(
                    "Compilation Error: Parent node {} not processed before child {}.",
                    p,
                    id
                )
            })?;
            parent_vars.push(var.clone());
        }

        let input_var_name = if parent_vars.is_empty() {
            "x".to_string()
        } else {
            parent_vars[0].clone()
        };

        // --- B. Generate Node Logic ---
        match kind {
            NodeKind::Input(_) => {
                // Alias the node's output to the input 'x'
                call_blocks.push(format!("{} = {}", current_out_var, input_var_name));
            }
            NodeKind::Layer(layer_type) => match layer_type {
                LayerType::Dense { dim_in, dim_out } => {
                    init_lines.push(format!(
                        "self.layer_{} = nnx.Linear({}, {}, rngs=rngs)",
                        clean_id, dim_in, dim_out
                    ));
                    call_blocks.push(format!(
                        "{} = self.layer_{}({})",
                        current_out_var, clean_id, input_var_name
                    ));
                }
                LayerType::GruCell { dim_in, dim_out } => {
                    // 1. Initialize the nnx cell
                    init_lines.push(format!(
                        "self.gru_cell_{} = nnx.GRUCell({}, {}, rngs=rngs)",
                        clean_id, dim_in, dim_out
                    ));

                    // 2. Generate the self-contained scan block
                    call_blocks.push(format!(
                        r#"
        # --- GRU Cell {0} ---
        seq_in_{0} = jnp.transpose({1}, (1, 0, 2))
        carry_{0} = self.gru_cell_{0}.initialize_carry(seq_in_{0}.shape[1:], rngs=self.rngs)
        def step_{0}(carry, x):
            new_carry, y = self.gru_cell_{0}(carry, x)
            return new_carry, y
        final_carry_{0}, seq_out_{0} = jax.lax.scan(step_{0}, carry_{0}, seq_in_{0})
        {2} = jnp.transpose(seq_out_{0}, (1, 0, 2))
        # ---------------------"#,
                        clean_id, input_var_name, current_out_var
                    ));
                }

                LayerType::Concat { axis } => {
                    if parent_vars.is_empty() {
                        return Err(anyhow::anyhow!(
                            "Concat node {} requires at least 1 input",
                            id
                        ));
                    }
                    call_blocks.push(format!(
                        "{} = jnp.concatenate([{}], axis={}) # Explicit Concat",
                        current_out_var,
                        parent_vars.join(", "),
                        axis
                    ));
                }
                LayerType::Add => {
                    if parent_vars.len() < 2 {
                        return Err(anyhow::anyhow!(
                            "Add node {} requires at least 2 inputs",
                            id
                        ));
                    }
                    call_blocks.push(format!(
                        "{} = {} # Explicit Element-wise Addition",
                        current_out_var,
                        parent_vars.join(" + ")
                    ));
                }
                LayerType::Flatten => {
                    // Standard JAX flattening: Keeps batch dim, flattens the rest
                    call_blocks.push(format!(
                        "{} = {}.reshape(({}.shape[0], -1)) # Explicit Flatten",
                        current_out_var, input_var_name, input_var_name
                    ));
                }
                // TODO: Correctly implement custon layers by injecting the code and replacing input/output variable names
                LayerType::Custom { code } => {
                    call_blocks.push(format!(
                        "# Custom Layer {}\n{} = {}\n",
                        clean_id, current_out_var, code
                    ));
                }
            },
            NodeKind::Activation(act_type) => {
                let func = match act_type {
                    ActivationType::Relu => "nnx.relu",
                    ActivationType::Sigmoid => "nnx.sigmoid",
                    ActivationType::Tanh => "nnx.tanh",
                    ActivationType::LeakyRelu { .. } => "nnx.leaky_relu",
                    ActivationType::Custom { .. } => "custom_act",
                };
                call_blocks.push(format!(
                    "{} = {}({})",
                    current_out_var, func, input_var_name
                ));
            }
        }

        // --- C. Register Output ---
        var_map.insert(id.clone(), current_out_var.clone());
        last_output_var = current_out_var;
    }

    call_blocks.push(format!("return {}", last_output_var));

    // 6. Render Final Template
    let context = ModelContext {
        init_lines,
        call_blocks,
    };

    Ok(tera.render("model.py.j2", &tera::Context::from_serialize(&context)?)?)
}

pub fn transpile_dataloader(
    input: &InputType,
    batch_size: usize,
    tera: &Arc<Tera>,
) -> Result<String> {
    let context = match input {
        InputType::Tabular {
            file_path,
            features,
            targets,
            sequence_length,
            separator,
        } => DataLoaderContext {
            file_path: file_path.clone(),
            features: features.clone(),
            targets: targets.clone(),
            batch_size,
            sequence_length: *sequence_length,
            separator: separator.to_owned(),
        },

        // For future me
        _ => {
            return Err(anyhow::anyhow!(
                "Input type not supported for dataloader generation"
            ));
        }
    };

    let code = tera.render(
        "dataloader.py.j2",
        &tera::Context::from_serialize(&context)?,
    )?;

    Ok(code)
}

pub fn transpile_training(params: TrainParams, tera: &Arc<Tera>) -> Result<String> {
    let mut context = tera::Context::new();

    context.insert("epochs", &params.epochs);
    context.insert("batch_size", &params.batchsize);

    let loss_code = match &params.loss {
        LossType::MeanAbsoluteError => "loss = jnp.mean((preds - y) ** 2)".to_string(),
        LossType::CrossEntropy => {
            "loss = optax.softmax_cross_entropy(logits=preds, labels=y).mean()".to_string()
        }
        LossType::Custom { code } => code.to_owned(),
    };
    context.insert("loss_calculation_code", &loss_code);

    let mut metric_calcs = Vec::new();
    let mut metric_assigns = Vec::new();

    if let Some(metrics) = &params.metrics {
        for metric in metrics {
            match metric {
                MetricType::Accuracy => {
                    // The calculation
                    metric_calcs.push("acc = jnp.mean(jnp.argmax(preds, axis=-1) == jnp.argmax(batch_y, axis=-1))".to_string());
                    // The assignment
                    metric_assigns.push("metrics_dict['accuracy'] = float(acc)".to_string());
                }
                MetricType::MeanAbsoluteError => {
                    metric_calcs.push("mae = jnp.mean(jnp.abs(preds - batch_y))".to_string());
                    metric_assigns.push("metrics_dict['mae'] = float(mae)".to_string());
                }
            }
        }
    }

    context.insert("metric_calcs", &metric_calcs);
    context.insert("metric_assigns", &metric_assigns);

    let code = tera.render("training.py.j2", &context)?;
    Ok(code)
}
