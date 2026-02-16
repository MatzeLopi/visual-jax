use crate::schemas::datasets::Dataset;
use axum::{
    Json, Router,
    extract::{Multipart, Path as PathExtract, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use uuid::Uuid;
use crate::http::{AppState, dependencies::OptionalAuthUser};

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
    let user_id = auth_user.user_id();

    let datasets = sqlx::query_as!(
        Dataset,
        r#"
        SELECT dataset_id, name, file_path, created_at, version, is_public
        FROM datasets
        WHERE is_public = true OR (user_id = $1::uuid AND $1::uuid IS NOT NULL)
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
) -> impl IntoResponse {
    let mut file_path_saved: Option<String> = None;
    let mut file_name_final = String::new();
    let mut is_public_req = false;

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            let file_name = field.file_name().unwrap_or("unknown.csv").to_string();
            file_name_final = file_name.clone();
            let data = field.bytes().await.unwrap();

            let unique_name = format!("{}_{}", Uuid::new_v4(), file_name);
            let upload_dir = "./uploads";
            let _ = fs::create_dir_all(upload_dir).await;
            let file_path = Path::new(upload_dir).join(&unique_name);

            if let Err(e) = fs::write(&file_path, &data).await {
                return Json(serde_json::json!({ "error": format!("Failed to save file: {}", e) }));
            }
            file_path_saved = Some(file_path.to_string_lossy().to_string());
        } else if name == "is_public" {
            let val = field.text().await.unwrap();
            is_public_req = val == "true";
        }
    }

    if let Some(path_str) = file_path_saved {
        let dataset_id = Uuid::new_v4();
        let user_id = auth_user.user_id();

        // Determine visibility
        let is_public = if user_id.is_none() {
            true
        } else {
            is_public_req
        };

        // Determine version
        // Logic: Find max version for (user_id, name)
        let version: i32 = if let Some(uid) = user_id {
             sqlx::query!(
                "SELECT MAX(version) as max_ver FROM datasets WHERE name = $1 AND user_id = $2",
                file_name_final,
                uid
            )
            .fetch_one(&state.db)
            .await
            .map(|r| r.max_ver.unwrap_or(0) + 1)
            .unwrap_or(1)
        } else {
             sqlx::query!(
                "SELECT MAX(version) as max_ver FROM datasets WHERE name = $1 AND user_id IS NULL",
                file_name_final
            )
            .fetch_one(&state.db)
            .await
            .map(|r| r.max_ver.unwrap_or(0) + 1)
            .unwrap_or(1)
        };

        let result = sqlx::query!(
            r#"
            INSERT INTO datasets (dataset_id, user_id, name, file_path, version, is_public)
            VALUES ($1, $2::uuid, $3, $4, $5, $6)
            RETURNING dataset_id, name, file_path, created_at, version, is_public
            "#,
            dataset_id,
            user_id,
            file_name_final,
            path_str,
            version,
            is_public
        )
        .fetch_one(&state.db)
        .await;

        return match result {
            Ok(rec) => Json(serde_json::json!({
                "dataset_id": rec.dataset_id,
                "name": rec.name,
                "file_path": rec.file_path,
                "version": rec.version,
                "is_public": rec.is_public
            })),
            Err(e) => Json(serde_json::json!({ "error": format!("Database error: {}", e) })),
        };
    }

    Json(serde_json::json!({ "error": "No file found" }))
}

async fn download_dataset(
    State(state): State<Arc<AppState>>,
    PathExtract(dataset_id): PathExtract<Uuid>,
    auth: OptionalAuthUser,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let rec = sqlx::query!(
        "SELECT file_path, name, user_id, is_public FROM datasets WHERE dataset_id = $1::uuid",
        dataset_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::NOT_FOUND, "Dataset not found".to_string()))?;

    // Visibility check
    let requester_id = auth.user_id();
    if !rec.is_public && rec.user_id != requester_id {
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
