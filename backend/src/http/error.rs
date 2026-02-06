use axum::{Json, http::StatusCode, response::IntoResponse, response::Response};
use serde_json::json;
use sqlx::error::DatabaseError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("authentication required")]
    Unauthorized,

    #[error("user may not perform that action")]
    Forbidden,

    #[error("request path not found")]
    NotFound,

    #[error("Internal Server Error")]
    InternalServerError(String),

    #[error("an error occurred with the database")]
    Sqlx(#[from] sqlx::Error),

    #[error("an internal server error occurred")]
    Anyhow(#[from] anyhow::Error),

    #[error("conflict, resource already exists")]
    Conflict,

    #[error("bad request")]
    BadRequest(String),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            // Use the custom message passed to BadRequest
            Error::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            Error::Conflict => (StatusCode::CONFLICT, "Conflict".to_string()),
            Error::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".to_string()),
            Error::Forbidden => (StatusCode::FORBIDDEN, "Forbidden".to_string()),
            Error::NotFound => (StatusCode::NOT_FOUND, "Not Found".to_string()),
            Error::InternalServerError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Error::Anyhow(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            Error::Sqlx(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
        }));

        (status, body).into_response()
    }
}

pub trait ResultExt<T> {
    fn on_constraint(
        self,
        name: &str,
        f: impl FnOnce(Box<dyn DatabaseError>) -> Error,
    ) -> Result<T, Error>;
}

impl<T, E> ResultExt<T> for Result<T, E>
where
    E: Into<Error>,
{
    fn on_constraint(
        self,
        name: &str,
        map_err: impl FnOnce(Box<dyn DatabaseError>) -> Error,
    ) -> Result<T, Error> {
        self.map_err(|e| match e.into() {
            Error::Sqlx(sqlx::Error::Database(dbe)) if dbe.constraint() == Some(name) => {
                map_err(dbe)
            }
            e => e,
        })
    }
}
