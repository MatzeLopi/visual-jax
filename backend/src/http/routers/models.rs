use crate::crud;
use crate::http::{AppState, error::Error as HTTPError};
use crate::schemas::models::ModelQueryOptions;
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, get, post},
};
use log::{debug, error, info};

use schemars::schema_for;
use std::{path::Path, sync::Arc};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/models", get(get_models))
        .with_state(state)
}

async fn get_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelQueryOptions>,
) -> Result<impl axum::response::IntoResponse, HTTPError> {
    let models = crud::models::get_models(query, &state.db).await?;
    Ok(Json(models))
}
