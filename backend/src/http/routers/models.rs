use crate::http::{AppState, dependencies::OptionalAuthUser};
use crate::schemas::models::{Model, NewModel};
use axum::{
    extract::{Path, State},
    Json, Router,
    response::IntoResponse,
    routing::{get, post},
    http::StatusCode,
};
use std::sync::Arc;
use uuid::Uuid;
use serde_json::json;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/models", post(save_model))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model))
        .with_state(state)
}

async fn save_model(
    State(state): State<Arc<AppState>>,
    auth_user: OptionalAuthUser,
    Json(payload): Json<NewModel>,
) -> impl IntoResponse {
    let user_id = auth_user.user_id();

    // Determine visibility
    let is_public = if user_id.is_none() {
        true
    } else {
        payload.is_public.unwrap_or(false)
    };

    // Calculate version
    let version: i32 = if let Some(uid) = user_id {
        sqlx::query!(
            "SELECT MAX(version) as max_ver FROM models WHERE name = $1 AND user_id = $2",
            payload.name,
            uid
        )
        .fetch_one(&state.db)
        .await
        .map(|r| r.max_ver.unwrap_or(0) + 1)
        .unwrap_or(1)
    } else {
         sqlx::query!(
            "SELECT MAX(version) as max_ver FROM models WHERE name = $1 AND user_id IS NULL",
            payload.name
        )
        .fetch_one(&state.db)
        .await
        .map(|r| r.max_ver.unwrap_or(0) + 1)
        .unwrap_or(1)
    };

    let model_id = Uuid::new_v4();

    let result = sqlx::query_as!(
        Model,
        r#"
        INSERT INTO models (model_id, user_id, name, version, is_public, graph_json)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING model_id, user_id, name, version, is_public, graph_json, created_at, updated_at
        "#,
        model_id,
        user_id,
        payload.name,
        version,
        is_public,
        payload.graph_json
    )
    .fetch_one(&state.db)
    .await;

    match result {
        Ok(model) => Json(json!(model)),
        Err(e) => {
            log::error!("Failed to save model: {}", e);
            // Return error with status code? Json usually implies 200 unless wrapped.
            // But here we return impl IntoResponse, so we can return (Status, Json).
            // For now, returning Json with error field is what other handlers seemed to do (based on datasets.rs)
            // Actually datasets.rs returned Json(serde_json::json!({ "error": ... })) which is 200 OK with error body.
            // I'll stick to that pattern or improve it.
            Json(json!({ "error": format!("Failed to save model: {}", e) }))
        }
    }
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    auth_user: OptionalAuthUser,
) -> impl IntoResponse {
    let user_id = auth_user.user_id();

    let models = sqlx::query_as!(
        Model,
        r#"
        SELECT model_id, user_id, name, version, is_public, graph_json, created_at, updated_at
        FROM models
        WHERE is_public = true OR (user_id = $1 AND $1 IS NOT NULL)
        ORDER BY created_at DESC
        "#,
        user_id
    )
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    Json(models)
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<Uuid>,
    auth_user: OptionalAuthUser,
) -> impl IntoResponse {
    let result = sqlx::query_as!(
        Model,
        r#"
        SELECT model_id, user_id, name, version, is_public, graph_json, created_at, updated_at
        FROM models
        WHERE model_id = $1
        "#,
        model_id
    )
    .fetch_optional(&state.db)
    .await;

    match result {
        Ok(Some(model)) => {
            let user_id = auth_user.user_id();
            if !model.is_public && model.user_id != user_id {
                return (StatusCode::FORBIDDEN, Json(json!({ "error": "Access denied" }))).into_response();
            }
            Json(json!(model)).into_response()
        }
        Ok(None) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Model not found" }))).into_response(),
        Err(e) => {
            log::error!("Failed to fetch model: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Database error" }))).into_response()
        }
    }
}
