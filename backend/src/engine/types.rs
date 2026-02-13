// Types used in the engine module
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum InputType {
    Tabular {
        file_path: String,
        features: Vec<String>,
        target: String,
        dim_out: isize,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum LayerType {
    Dense { dim_in: isize, dim_out: isize },
    Concat { axis: isize },
    GruCell { dim_in: isize, dim_out: isize },
    Add,
    Flatten,
    Custom { code: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]

pub enum ActivationType {
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu { alpha: f32 },
    Custom { code: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum LossType {
    MSE,
    CrossEntropy,
    Custom { code: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
#[serde(tag = "type", content = "config")]
pub enum MetricType {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Custom { code: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]
pub enum OptimizerType {
    Adam { learning_rate: f32 },
    SGD { learning_rate: f32, momentum: f32 },
    Custom { code: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NodeKind {
    Input(InputType),
    Layer(LayerType),
    Activation(ActivationType),
}
