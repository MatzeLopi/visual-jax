// Router for compiler related endpoints
use crate::engine::{
    graph::GraphProcessor,
    transpiler,
    types::{ActivationType, InputType, LayerType, LossType, MetricType, OptimizerType},
    validator,
};
use crate::http::{AppState, error::Error as HTTPError};
use crate::schemas::graph::NeuralGraph;
use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, get, post},
};
use schemars::schema_for;
use std::sync::Arc;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/compiler/layer-types", get(get_layer_types))
        .route("/compiler/input-types", get(get_input_types))
        .route("/compiler/activation-types", get(get_activation_types))
        .route("/compiler/optimizer-types", get(get_optimizer_types))
        .route("/compiler/loss-types", get(get_loss_types))
        .route("/compiler/metric-types", get(get_metric_types))
        .route("/compiler/compile", post(compile_graph))
        .with_state(state)
}

async fn get_layer_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching layer types");
    let layer_types = schema_for!(LayerType);
    Ok((StatusCode::OK, Json(layer_types)))
}

async fn get_input_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching input types");
    let input_types = schema_for!(InputType);
    Ok((StatusCode::OK, Json(input_types)))
}

async fn get_activation_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching activation types");
    let activation_types = schema_for!(ActivationType);
    Ok((StatusCode::OK, Json(activation_types)))
}

async fn get_optimizer_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching optimizer types");
    let optimizer_types = schema_for!(OptimizerType);
    Ok((StatusCode::OK, Json(optimizer_types)))
}

async fn get_loss_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching loss types");
    let loss_types = schema_for!(LossType);
    Ok((StatusCode::OK, Json(loss_types)))
}

async fn get_metric_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching metric types");
    let metric_types = schema_for!(MetricType);
    Ok((StatusCode::OK, Json(metric_types)))
}

async fn compile_graph(
    Json(graph): Json<NeuralGraph>,
) -> Result<impl axum::response::IntoResponse, HTTPError> {
    let processor = GraphProcessor::new(graph);

    let sorted_nodes = processor
        .validate_and_sort()
        .map_err(|e| HTTPError::BadRequest(e.to_string()))?;

    let incoming_map = processor.get_incoming_map();

    // Check code
    validator::validate_graph(&sorted_nodes, &incoming_map)
        .map_err(|e| HTTPError::InternalServerError(e.to_string()))?;

    let python_code = transpiler::transpile(sorted_nodes, incoming_map)
        .map_err(|e| HTTPError::InternalServerError(e.to_string()))?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({ "code": python_code })),
    ))
}
