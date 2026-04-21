use crate::http::{AppState, dependencies::OptionalAuthUser, error::Error as HTTPError};
use crate::schemas::datasets::Dataset;

use axum::{
    Json, Router,
    extract::{Multipart, Path as PathExtract, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use log::error;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use uuid::Uuid;

// Import OptionalAuthUser instead of AuthUser
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/datasets/upload", post(upload_dataset))
        .route("/datasets/list", get(list_datasets))
        .route("/datasets/{id}/file", get(download_dataset))
        .with_state(state)
}

async fn list_datasets(
    State(state): State<Arc<AppState>>,
    auth_user: OptionalAuthUser,
) -> impl IntoResponse {
    let user_id = auth_user.user_id(); // returns Option<Uuid>

    let datasets = sqlx::query_as!(
        Dataset,
        r#"
        SELECT dataset_id, name, file_path, created_at 
        FROM datasets 
        WHERE (user_id = $1::uuid) OR (user_id IS NULL)
        ORDER BY created_at DESC
        "#,
        user_id
    )
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    Json(datasets)
}

async fn upload_dataset(
    State(state): State<Arc<AppState>>,
    auth_user: OptionalAuthUser,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, HTTPError> {
    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("file") {
            let raw_file_name = field.file_name().unwrap_or("unknown.csv");
            let file_name = Path::new(raw_file_name)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown.csv")
                .to_string();
            let data = field.bytes().await.unwrap();

            let unique_name = format!("{}_{}", Uuid::new_v4(), file_name);
            let upload_dir = "./uploads";
            let _ = fs::create_dir_all(upload_dir).await;
            let file_path = Path::new(upload_dir).join(&unique_name);

            if let Err(e) = fs::write(&file_path, &data).await {
                error!("Error when writing upload file: {:?}", e);
                return Err(HTTPError::InternalServerError(
                    "File write failed".to_string(),
                ));
            }

            let dataset_id = Uuid::new_v4();
            let path_str = file_path.to_string_lossy().to_string();
            let user_id = auth_user.user_id(); // This might be None (Public)

            let result = sqlx::query!(
                r#"
                INSERT INTO datasets (dataset_id, user_id, name, file_path)
                VALUES ($1, $2::uuid, $3, $4)
                RETURNING dataset_id, name, file_path, created_at
                "#,
                dataset_id,
                user_id,
                file_name,
                path_str
            )
            .fetch_one(&state.db)
            .await;

            return match result {
                Ok(rec) => Ok(Json(serde_json::json!({
                    "dataset_id": rec.dataset_id,
                    "name": rec.name,
                    "file_path": rec.file_path,
                    "is_public": user_id.is_none()
                }))),
                Err(e) => {
                    error!("Error with database: {:?}", e);
                    Err(HTTPError::InternalServerError("Database error".to_string()))
                }
            };
        }
    }

    error!("Generic error on file upload");
    Err(HTTPError::BadRequest("Bad request".to_string()))
}

async fn download_dataset(
    State(state): State<Arc<AppState>>,
    PathExtract(dataset_id): PathExtract<Uuid>,
    auth: OptionalAuthUser,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let rec = sqlx::query!(
        "SELECT file_path, name, user_id FROM datasets WHERE dataset_id = $1::uuid",
        dataset_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::NOT_FOUND, "Dataset not found".to_string()))?;

    let requester_id = auth.user_id();
    if rec.user_id.is_some() && rec.user_id != requester_id {
        return Err((StatusCode::FORBIDDEN, "Access denied".to_string()));
    }

    let content = fs::read(&rec.file_path).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read file: {}", e),
        )
    })?;

    let response = Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", rec.name),
        )
        .body(axum::body::Body::from(content))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}
