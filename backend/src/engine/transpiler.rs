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
    tera.add_raw_template("gru_module", include_str!("../../templates/gru.py.j2"))?;

    // 2. State Tracking
    let mut init_lines = Vec::new();
    let mut call_blocks = Vec::new();
    let mut extra_classes = String::new();

    // Flags & tables
    let mut has_gru = false;
    let mut var_map: HashMap<String, String> = HashMap::new(); // Maps NodeID -> Python Var Name
    let mut last_output_var = "x".to_string(); // Default to input 'x' if graph is empty
    let mut has_explicit_return = false;

    // 3. Iterate Nodes in Topological Order
    for (id, kind) in sorted_nodes {
        let clean_id = id.replace("-", "_");
        let current_out_var = format!("out_{}", clean_id);

        // --- A. Resolve Inputs ---
        // Who connects to me?
        let parents = incoming_map.get(&id).cloned().unwrap_or_default();

        let input_var_name = if parents.is_empty() {
            // No parents = Model Input
            "x".to_string()
        } else if parents.len() == 1 {
            // Single parent = Direct connection
            var_map
                .get(&parents[0])
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Compilation Error: Parent node {} not processed before child {}.",
                        parents[0],
                        id
                    )
                })?
                .clone()
        } else {
            // Multiple parents = Concatenate
            // Collect all parent variable names
            let mut parent_vars = Vec::new();
            for p in &parents {
                let var = var_map.get(p).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Compilation Error: Parent node {} missing for merge at {}.",
                        p,
                        id
                    )
                })?;
                parent_vars.push(var.clone());
            }

            // Generate concatenation code
            let concat_var = format!("concat_{}", clean_id);
            call_blocks.push(format!(
                "{} = jnp.concatenate([{}], axis=-1) # Merge branches",
                concat_var,
                parent_vars.join(", ")
            ));
            concat_var
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
                LayerType::GRU {
                    dim_in,
                    dim_hidden,
                    n_hidden,
                } => {
                    has_gru = true;
                    init_lines.push(format!(
                        "self.layer_{} = GRU(dim_in={}, n_hidden={}, dim_hidden={}, rngs=rngs)",
                        clean_id, dim_in, n_hidden, dim_hidden
                    ));
                    // GRU requires explicit carry handling
                    call_blocks.push(format!(
                        r#"
        # --- GRU Layer {0} ---
        carry_{0} = self.layer_{0}.initialize_carry({1}.shape[0], rngs)
        {2} = self.layer_{0}(carry_{0}, {1})
        # ---------------------"#,
                        clean_id, input_var_name, current_out_var
                    ));
                }
                LayerType::Custom { code } => {
                    // Placeholder for custom code injection
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
            NodeKind::Output { .. } => {
                // If we reach an Output node, return the input immediately
                call_blocks.push(format!("return {}", input_var_name));
                has_explicit_return = true;
                continue; // Skip registering this node since we returned
            }
        }

        // --- C. Register Output ---
        var_map.insert(id.clone(), current_out_var.clone());
        last_output_var = current_out_var;
    }

    // 4. Fallback Return
    // If the graph didn't contain an explicit 'Output' node, return the last calculated variable.
    if !has_explicit_return {
        call_blocks.push(format!("return {}", last_output_var));
    }

    // 5. Inject Helper Classes
    if has_gru {
        let gru_code = tera.render("gru_module", &tera::Context::new())?;
        extra_classes.push_str(&gru_code);
        extra_classes.push_str("\n\n");
    }

    // 6. Render Final Template
    let context = MainContext {
        extra_classes,
        init_lines,
        call_blocks,
    };

    Ok(tera.render("model.py", &tera::Context::from_serialize(&context)?)?)
}
