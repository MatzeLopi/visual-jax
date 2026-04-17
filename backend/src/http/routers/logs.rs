use std::str::FromStr;
use std::sync::Arc;

use crate::crud::logs::get_log;
use crate::http::{AppState, error::Error as HTTPError};

use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::{Router, get, post},
};
use log::error;

use uuid::Uuid;
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/logs/:uid", get(get_logs))
        .with_state(state)
}

// TODO: In future make this protected so that only user with correct auth can get the model
async fn get_logs(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(uid): axum::extract::Path<String>,
) -> Result<impl IntoResponse, HTTPError> {
    let uid = Uuid::from_str(&uid).map_err(|e| {
        error!("Could not convert String to Uuid: {:?}", e);
        HTTPError::BadRequest(e.to_string())
    })?;
    let logs = get_log(uid, 100, &state.db).await?;

    Ok(Json(logs))
}
