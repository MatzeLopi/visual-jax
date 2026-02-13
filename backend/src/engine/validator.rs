use crate::engine::types::{ActivationType, LayerType, NodeKind};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Fixed(isize),
    Batch,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<Dim>);

pub trait ShapeValidator {
    fn validate_and_propagate(&self, input_shapes: &[Shape]) -> Result<Shape, String>;
}

impl ShapeValidator for LayerType {
    fn validate_and_propagate(&self, input_shapes: &[Shape]) -> Result<Shape, String> {
        let input_shape = &input_shapes[0].0;
        let last_dim = input_shape.last().ok_or("Input shape cannot be empty")?;

        match self {
            LayerType::Dense { dim_in, dim_out } => {
                match last_dim {
                    Dim::Fixed(val) if val != dim_in => {
                        return Err(format!(
                            "Shape mismatch! Incoming connection has dimension {}, but Dense layer expects dim_in={}",
                            val, dim_in
                        ));
                    }
                    _ => {}
                }
                let mut out_shape = input_shape.clone();
                let last_idx = out_shape.len() - 1;
                out_shape[last_idx] = Dim::Fixed(*dim_out);

                Ok(Shape(out_shape))
            }

            LayerType::GRU {
                dim_in,
                dim_hidden,
                n_hidden: _,
            } => {
                match last_dim {
                    Dim::Fixed(val) if val != dim_in => {
                        return Err(format!(
                            "Shape mismatch! Incoming connection has dimension {}, but GRU layer expects dim_in={}",
                            val, dim_in
                        ));
                    }
                    _ => {}
                }

                let mut out_shape = input_shape.clone();
                let last_idx = out_shape.len() - 1;
                out_shape[last_idx] = Dim::Fixed(*dim_hidden);

                Ok(Shape(out_shape))
            }

            LayerType::Concat { axis } => {
                if input_shapes.is_empty() {
                    return Err("Concat layer requires at least one input".to_string());
                }

                let base_shape = &input_shapes[0].0;
                let rank = base_shape.len() as isize;

                let actual_axis = if *axis < 0 {
                    (rank + *axis) as usize
                } else {
                    *axis as usize
                };

                if actual_axis >= base_shape.len() {
                    return Err(format!("Invalid axis {} for shape of rank {}", axis, rank));
                }

                let mut sum_concat_dim = 0;
                let mut has_unknown = false;

                for (i, shape) in input_shapes.iter().enumerate() {
                    let current_shape = &shape.0;

                    // 1. Check rank
                    if current_shape.len() != base_shape.len() {
                        return Err(format!(
                            "Rank mismatch in Concat! Input 0 has {} dims, but Input {} has {} dims",
                            base_shape.len(),
                            i,
                            current_shape.len()
                        ));
                    }

                    for (dim_idx, dim) in current_shape.iter().enumerate() {
                        if dim_idx == actual_axis {
                            match dim {
                                Dim::Fixed(val) => sum_concat_dim += val,
                                Dim::Unknown => has_unknown = true,
                                Dim::Batch => {
                                    return Err(
                                        "Cannot concatenate along the Batch dimension!".to_string()
                                    );
                                }
                            }
                        } else {
                            if dim != &base_shape[dim_idx] {
                                if *dim != Dim::Unknown && base_shape[dim_idx] != Dim::Unknown {
                                    return Err(format!(
                                        "Shape mismatch in Concat at dimension {}! Input 0 has {:?} but Input {} has {:?}",
                                        dim_idx, base_shape[dim_idx], i, dim
                                    ));
                                }
                            }
                        }
                    }
                }

                let mut out_shape = base_shape.clone();
                if has_unknown {
                    out_shape[actual_axis] = Dim::Unknown;
                } else {
                    out_shape[actual_axis] = Dim::Fixed(sum_concat_dim);
                }

                Ok(Shape(out_shape))
            }

            LayerType::Add => {
                if input_shapes.len() < 2 {
                    return Err("Add layer requires at least two inputs".to_string());
                }

                let base_shape = &input_shapes[0].0;

                for (i, shape) in input_shapes.iter().enumerate().skip(1) {
                    if shape.0 != *base_shape {
                        return Err(format!(
                            "Shape mismatch in Add layer! Input 0 has shape {:?} but Input {} has shape {:?}",
                            base_shape, i, shape.0
                        ));
                    }
                }

                Ok(Shape(base_shape.clone()))
            }

            LayerType::Flatten => {
                let input_shape = &input_shapes[0].0;

                let mut flat_size = 1;
                let mut has_unknown = false;
                let mut output_shape = Vec::new();

                for dim in input_shape {
                    match dim {
                        Dim::Fixed(val) => flat_size *= val,
                        Dim::Batch => output_shape.push(Dim::Batch),
                        Dim::Unknown => has_unknown = true,
                    }
                }

                if has_unknown {
                    output_shape.push(Dim::Unknown);
                } else {
                    output_shape.push(Dim::Fixed(flat_size));
                }

                Ok(Shape(output_shape))
            }

            LayerType::Custom { code: _ } => {
                let mut out_shape = input_shape.clone();
                let last_idx = out_shape.len() - 1;
                out_shape[last_idx] = Dim::Unknown;

                Ok(Shape(out_shape))
            }
        }
    }
}

impl ShapeValidator for ActivationType {
    fn validate_and_propagate(&self, input_shapes: &[Shape]) -> Result<Shape, String> {
        Ok(input_shapes[0].clone())
    }
}

pub fn validate_graph(
    sorted_nodes: &[(String, NodeKind)],
    incoming_map: &HashMap<String, Vec<String>>,
) -> Result<(), String> {
    let mut shape_env: HashMap<String, Shape> = HashMap::new();

    for (id, kind) in sorted_nodes {
        let parent_ids = incoming_map.get(id).cloned().unwrap_or_default();
        let mut node_input_shapes = Vec::new();

        for p_id in &parent_ids {
            if let Some(shape) = shape_env.get(p_id) {
                node_input_shapes.push(shape.clone());
            } else {
                return Err(format!(
                    "Validation Error: Node {} missing input shape from {}",
                    id, p_id
                ));
            }
        }

        let output_shape = match kind {
            // FIX: Cast to usize AND prepend the Batch dimension!
            NodeKind::Input(input_data) => match input_data {
                crate::engine::types::InputType::Tabular {
                    features, dim_out, ..
                } => {
                    let feature_count = if !features.is_empty() {
                        features.len() as isize
                    } else {
                        *dim_out as isize
                    };

                    Shape(vec![Dim::Batch, Dim::Fixed(feature_count)])
                }
            },
            NodeKind::Layer(layer) => layer.validate_and_propagate(&node_input_shapes)?,
            NodeKind::Activation(_) => {
                if node_input_shapes.is_empty() {
                    return Err(format!("Activation node {} has no inputs", id));
                }
                node_input_shapes[0].clone()
            }
        };

        // 3. Store the result so the next node can use it
        shape_env.insert(id.clone(), output_shape);
    }

    Ok(())
}
