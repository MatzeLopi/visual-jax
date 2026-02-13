use crate::engine::types::{ActivationType, LayerType, NodeKind};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use tera::Tera;

// Define the data structure passed to the Jinja2 template
#[derive(Serialize)]
struct MainContext {
    extra_classes: String,
    init_lines: Vec<String>,
    call_blocks: Vec<String>,
}

pub fn transpile(
    sorted_nodes: Vec<(String, NodeKind)>,
    incoming_map: HashMap<String, Vec<String>>,
) -> Result<String> {
    // 1. Initialize Template Engine
    let mut tera = Tera::default();
    tera.add_raw_template("model.py", include_str!("../../templates/model.py.j2"))?;

    // 2. State Tracking
    let mut init_lines = Vec::new();
    let mut call_blocks = Vec::new();
    let mut extra_classes = String::new();

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
        # 1. Transpose input to (Sequence, Batch, Features) for scanning
        seq_in_{0} = jnp.transpose({1}, (1, 0, 2))
        
        batch_size_{0} = {1}.shape[0]
        carry_{0} = jnp.zeros((batch_size_{0}, {2}), dtype=jnp.float32)
        
        def step_{0}(carry, x):
            new_carry, y = self.gru_cell_{0}(carry, x)
            return new_carry, y

        final_carry_{0}, seq_out_{0} = jax.lax.scan(step_{0}, carry_{0}, seq_in_{0})

        {3} = jnp.transpose(seq_out_{0}, (1, 0, 2))
        # ---------------------"#,
                        clean_id, input_var_name, dim_out, current_out_var
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
                LayerType::Custom { code } => {
                    call_blocks.push(format!(
                        "# Custom Layer {}\n{} = ... ",
                        clean_id, current_out_var
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
    let context = MainContext {
        extra_classes,
        init_lines,
        call_blocks,
    };

    Ok(tera.render("model.py", &tera::Context::from_serialize(&context)?)?)
}
