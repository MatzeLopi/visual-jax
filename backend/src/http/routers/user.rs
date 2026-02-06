use crate::{
    crud,
    http::{AppState, dependencies, error::Error as HTTPError},
    schemas::users::{NewUser, UpdatePassword, User},
};
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, delete, get, post},
};
use std::sync::Arc;
use uuid::Uuid;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/users/create-user", post(create_user))
        .route("/users/delete-user", delete(delete_user))
        .route("/users/me", get(me))
        .route("/users/me/update-password", post(update_password))
        .route("/users/verify/{username}/{token}", post(verify_user))
        .with_state(state)
}

async fn me(
    State(state): State<Arc<AppState>>,
    auth_user: dependencies::OptionalAuthUser,
) -> Result<impl IntoResponse, HTTPError> {
    let user_id = match auth_user.0 {
        Some(user) => user.user_id,
        _ => {
            let user = User {
                id: Uuid::nil(),
                username: String::from(""),
                email: String::from(""),
                verified: true,
            };
            return Ok((StatusCode::NO_CONTENT, Json(user)));
        }
    };

    let user = crud::user::get_user_by_id(&user_id, &state.db).await?;
    Ok((StatusCode::OK, Json(user)))
}

async fn create_user(
    State(state): State<Arc<AppState>>,
    Json(user): Json<NewUser>,
) -> Result<impl IntoResponse, HTTPError> {
    log::debug!("New user creation started");
    let NewUser {
        username,
        email,
        password,
    } = user;

    let password_hash = dependencies::hash_password(password)?;

    _ = crud::user::create_user(&username, &email, &password_hash, state).await?;
    log::debug!("Successfully created new user");
    Ok((StatusCode::CREATED, "User created successfully"))
}

async fn update_password(
    State(state): State<Arc<AppState>>,
    auth_user: dependencies::AuthUser,
    Json(update_struct): Json<UpdatePassword>,
) -> Result<impl IntoResponse, HTTPError> {
    let user = crud::user::get_user_by_id(&auth_user.user_id, &state.db).await?;
    let old_hash = crud::user::get_hash(&user.username, &state.db).await?.1;
    let pw_hash = dependencies::hash_password(update_struct.new_password)?;

    dependencies::validate_password(update_struct.old_password, &old_hash)?;
    if crud::user::update_password(&auth_user.user_id, &pw_hash, &state.db).await? == true {
        Ok(StatusCode::OK)
    } else {
        log::error!("Failed to update password");
        Err(HTTPError::InternalServerError(
            "Failed to update password".to_string(),
        ))
    }
}

async fn verify_user(
    State(state): State<Arc<AppState>>,
    Path((username, token)): Path<(String, String)>,
) -> Result<impl IntoResponse, HTTPError> {
    if crud::user::get_verification_token(&username, &state.db).await? == token {
        _ = crud::user::verify_user(&username, &state.db).await?;
        Ok((StatusCode::OK, "User successfully verified"))
    } else {
        Err(HTTPError::Forbidden)
    }
}

async fn delete_user(
    State(state): State<Arc<AppState>>,
    _: dependencies::CsrfValidator,
    auth_user: dependencies::AuthUser,
) -> Result<impl IntoResponse, HTTPError> {
    _ = crud::user::delete_user(&auth_user.user_id, &state.db).await?;
    Ok((StatusCode::OK, "Successfully deleted user"))
}
