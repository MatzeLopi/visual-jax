// Router for compiler related endpoints
use crate::engine::{
    graph::GraphProcessor,
    transpiler,
    types::{ActivationType, InputType, LayerType, NodeKind, OptimizerType},
    validator,
};
use crate::http::{AppState, error::Error as HTTPError};
use crate::schemas::training::TrainRequestPayload;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, get, post},
};
use log::debug;
use schemars::schema_for;
use std::{path::Path, sync::Arc};
use tokio::fs::create_dir_all;
use uuid::Uuid;

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
    let loss_types = vec!["MeanAbsoluteError", "CrossEntropy"];
    Ok((StatusCode::OK, Json(loss_types)))
}

async fn get_metric_types() -> Result<impl IntoResponse, HTTPError> {
    log::debug!("Fetching metric types");
    let metric_types = vec!["Accuracy", "MeanAbsoluteError"];
    Ok((StatusCode::OK, Json(metric_types)))
}

async fn compile_graph(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TrainRequestPayload>,
) -> Result<impl axum::response::IntoResponse, HTTPError> {
    let processor = GraphProcessor::new(payload.graph);

    let sorted_nodes = processor
        .validate_and_sort()
        .map_err(|e| HTTPError::BadRequest(e.to_string()))?;

    let incoming_map = processor.get_incoming_map();

    // Check code
    validator::validate_graph(&sorted_nodes, &incoming_map)
        .map_err(|e| HTTPError::InternalServerError(e.to_string()))?;

    let (_, node_kind) = sorted_nodes.first().unwrap();

    let dataloader_code = match node_kind {
        NodeKind::Input(input_node) => transpiler::transpile_dataloader(
            input_node,
            payload.params.batchsize,
            &state.tera_engine,
        ),

        _ => Err(anyhow::anyhow!("First node must be Input Node")),
    }?;

    let python_code = transpiler::transpile_model(sorted_nodes, incoming_map, &state.tera_engine)
        .map_err(|e| HTTPError::InternalServerError(e.to_string()))?;

    let training_code = transpiler::transpile_training(payload.params, &state.tera_engine)
        .map_err(|e| HTTPError::InternalServerError(e.to_string()))?;

    let code = format!("{}\n{}\n{}", dataloader_code, python_code, training_code);

    let uid = Uuid::new_v4();
    let file_name = format!("./files/models/{uid}.py");
    create_dir_all("./files/models/").await.map_err(|e| {
        debug!("Create Dir all in compiler router failed with: {:?}", e);
        HTTPError::InternalServerError(format!("An error ocured when saving the model file. {e} "))
    })?;
    let p = Path::new(&file_name);
    // Temp, remove clone in future, when return to frontend is not done anymore
    tokio::fs::write(p, code.clone()).await.map_err(|e| {
        debug!("File write in compiler router failed with {:?}", e);
        HTTPError::InternalServerError(format!("An error ocured when saving the model file. {e} "))
    })?;

    Ok((StatusCode::OK, Json(serde_json::json!({ "code": code }))))
}
