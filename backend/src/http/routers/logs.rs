use std::str::FromStr;
use std::sync::Arc;

use crate::crud::{logs::get_log, models::get_model_owner};
use crate::http::{dependencies::OptionalAuthUser, AppState, error::Error as HTTPError};

use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::{Router, get},
};
use log::error;

use uuid::Uuid;
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/logs/{uid}", get(get_logs))
        .with_state(state)
}

async fn get_logs(
    State(state): State<Arc<AppState>>,
    auth_user: OptionalAuthUser,
    axum::extract::Path(uid): axum::extract::Path<String>,
) -> Result<impl IntoResponse, HTTPError> {
    let uid = Uuid::from_str(&uid).map_err(|e| {
        error!("Could not convert String to Uuid: {:?}", e);
        HTTPError::BadRequest(e.to_string())
    })?;

    let owner_id = get_model_owner(uid, &state.db).await?;
    let requester_id = auth_user.user_id();

    if owner_id.is_some() && owner_id != requester_id {
        return Err(HTTPError::Forbidden);
    }

    let logs = get_log(uid, 100, &state.db).await?;

    Ok(Json(logs))
}
