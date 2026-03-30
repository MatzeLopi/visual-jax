// Types used in the engine module
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn default_seq_len() -> usize {
    1
}

fn default_sep() -> String {
    ";".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum InputType {
    Tabular {
        file_path: String,
        features: Vec<String>,
        targets: Vec<String>,
        #[serde(default = "default_seq_len")]
        sequence_length: usize,
        #[serde(default = "default_sep")]
        separator: String,
    },
}

impl InputType {
    pub fn dim_out(&self) -> usize {
        match self {
            InputType::Tabular { targets, .. } => targets.len(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum LayerType {
    Dense { dim_in: isize, dim_out: isize },
    Concat { axis: isize },
    GruCell { dim_in: isize, dim_out: isize },
    Add,
    Flatten,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]

pub enum ActivationType {
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu { alpha: f32 },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum LossType {
    MeanAbsoluteError,
    CrossEntropy,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum MetricType {
    Accuracy,
    MeanAbsoluteError,
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
pub enum OptimizerType {
    Adam { learning_rate: f32 },
    SGD { learning_rate: f32, momentum: f32 },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NodeKind {
    Input(InputType),
    Layer(LayerType),
    Activation(ActivationType),
}
