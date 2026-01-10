// Graph definitions for the Frontend
use crate::engine::types::NodeKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(flatten)]
    pub kind: NodeKind,
    #[serde(default)]
    pub position: Value, // Position im Graph Editor
    #[serde(default)]
    pub data: Value, // Additional data from frontend
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,

    #[serde(default)]
    pub source_handle: Option<String>,
    #[serde(default)]
    pub target_handle: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
