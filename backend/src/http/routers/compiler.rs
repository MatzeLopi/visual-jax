// Router for compiler related endpoints
use crate::http::{AppState, error::Error as HTTPError};
use crate::schemas::training::TrainRequestPayload;
use crate::{
    engine::{
        graph::GraphProcessor,
        runner::Runner,
        transpiler,
        types::{ActivationType, InputType, LayerType, NodeKind, OptimizerType},
        validator,
    },
    schemas::models::Model,
};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, get, post},
};
use log::{debug, error, info};

use schemars::schema_for;
use std::{path::Path, sync::Arc};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/compiler/layer-types", get(get_layer_types))
        .route("/compiler/input-types", get(get_input_types))
        .route("/compiler/activation-types", get(get_activation_types))
        .route("/compiler/optimizer-types", get(get_optimizer_types))
        .route("/compiler/loss-types", get(get_loss_types))
        .route("/compiler/metric-types", get(get_metric_types))
        .route("/compiler/compile", post(compile_graph))
        .route("/training/start", post(start_training))
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
    let model = Model::default();
    let processor = GraphProcessor::new(payload.graph);

    let sorted_nodes = processor.validate_and_sort().map_err(|e| {
        error!("Graph sort failed with: {:?}", e);
        HTTPError::BadRequest("Graph sort failed".to_string())
    })?;

    let incoming_map = processor.get_incoming_map();

    validator::validate_graph(&sorted_nodes, &incoming_map).map_err(|e| {
        error!("Graph sort failed with: {:?}", e);
        HTTPError::InternalServerError("Graph validation failed".to_string())
    })?;

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
        .map_err(|e| {
        error!("Model transpile failed {:?}", e);
        HTTPError::InternalServerError("Model transpile failed".to_string())
    })?;

    let training_code = transpiler::transpile_training(payload.params, &state.tera_engine)
        .map_err(|e| {
            error!("Trainer trainspile failed: {:?}", e);
            HTTPError::InternalServerError("Trainer transpile failed".to_string())
        })?;

    let code = format!("{}\n{}\n{}", dataloader_code, python_code, training_code);
    let p: &Path = Path::new(&model.model_path);
    tokio::fs::write(p, code).await.map_err(|e| {
        error!("File write in compiler router failed with {:?}", e);
        HTTPError::InternalServerError("An error ocured when saving the model file.".to_string())
    })?;
    Ok((StatusCode::OK, Json(model)))
}

async fn load_graph() {}

async fn update_model() {}

async fn start_training(
    State(state): State<Arc<AppState>>,
    Json(model): Json<Model>,
) -> Result<impl axum::response::IntoResponse, HTTPError> {
    let runner = Runner::new(model).await?;
    runner.run(&state.db).await?;
    Ok(StatusCode::OK)
}
