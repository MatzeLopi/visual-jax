// Router for auth and csrf token generation
use crate::{
    http::{
        AppState,
        dependencies::{self, OptionalAuthUser},
        error::Error as HTTPError,
        utils,
    },
    schemas::users::UserLogin,
};

use axum_extra::extract::cookie::{Cookie, CookieJar, Expiration};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{Router, get, post},
};
use serde_json::json;
use std::sync::Arc;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(ok))
        .route("/token/get", post(token))
        .route("/token/renew", post(update_token))
        .route("/logout", get(logout))
        .with_state(state)
}

async fn ok() -> impl IntoResponse {
    (StatusCode::OK, axum::Json(json!({ "status": "ok" })))
}

async fn get_csfr(mut jar: CookieJar) -> CookieJar {
    // Generate a random token
    let csft: String = utils::random_string(32);
    let expiration = Expiration::from(time::OffsetDateTime::now_utc() + time::Duration::hours(24));
    let server_cookie = Cookie::build(("s_csft", csft.clone()))
        .path("/")
        .secure(true)
        .http_only(true)
        .expires(expiration)
        .build();
    let client_cookie = Cookie::build(("x_csft", csft))
        .path("/")
        .secure(true)
        .http_only(false)
        .expires(expiration)
        .build();
    jar = jar
        .remove(Cookie::from("s_csft"))
        .remove(Cookie::from("x_csft"))
        .add(server_cookie)
        .add(client_cookie);

    jar
}

async fn token(
    State(state): State<Arc<AppState>>,
    mut jar: CookieJar,
    maybe_user: OptionalAuthUser,
    Json(user): Json<UserLogin>,
) -> Result<impl IntoResponse, HTTPError> {
    if maybe_user.0.is_some() {
        return Ok((StatusCode::FOUND, jar));
    }

    let db = &state.db;
    let UserLogin { username, password } = user;
    let auth_user = dependencies::auth_user(&username, password, db).await;
    match auth_user {
        Ok(auth_user) => {
            let token = auth_user.to_jwt(&state)?;
            jar = get_csfr(jar).await;
            let token_cookie = Cookie::build((dependencies::DEFAULT_AUTH, token.clone()))
                .secure(true)
                .http_only(true)
                .expires(Expiration::from(
                    time::OffsetDateTime::now_utc() + time::Duration::hours(24),
                ))
                .path("/")
                .build();

            jar = jar.add(token_cookie);

            Ok((StatusCode::OK, jar))
        }
        Err(e) => Err(e),
    }
}

async fn update_token(
    State(state): State<Arc<AppState>>,
    _: dependencies::CsrfValidator,
    mut jar: CookieJar,
    user: dependencies::AuthUser,
) -> Result<impl IntoResponse, HTTPError> {
    let token = user.to_jwt(&state)?;
    let token_cookie = Cookie::build((dependencies::DEFAULT_AUTH, token.clone()))
        .secure(true)
        .http_only(true)
        .expires(Expiration::from(
            time::OffsetDateTime::now_utc() + time::Duration::hours(24),
        ))
        .path("/")
        .build();
    jar = get_csfr(jar).await;
    jar = jar.add(token_cookie);

    Ok((StatusCode::OK, jar))
}

async fn logout(mut jar: CookieJar) -> impl IntoResponse {
    jar = jar
        .remove(Cookie::from("x_csft"))
        .remove(Cookie::from("s_csft"))
        .remove(Cookie::from(dependencies::DEFAULT_AUTH));
    (StatusCode::OK, jar)
}
