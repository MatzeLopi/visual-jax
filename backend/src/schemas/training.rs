use crate::engine::types::{LossType, MetricType};
use crate::schemas::graph::NeuralGraph;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct TrainParams {
    pub loss: LossType,
    pub metrics: Option<Vec<MetricType>>,
    pub epochs: usize,
    pub batchsize: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrainRequestPayload {
    pub graph: NeuralGraph,
    pub params: TrainParams,
}
