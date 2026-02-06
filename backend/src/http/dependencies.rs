// External Crates
use crate::crud;
use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};
use axum::{
    extract::FromRequestParts,
    http::{header::COOKIE, request::Parts},
};
use axum_extra::extract::cookie::CookieJar;
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use sqlx::PgPool;
use time::OffsetDateTime;
use uuid::Uuid;

// Internal Modules
use crate::http::{AppState, error::Error as HTTPError};

const DEFAULT_SESSION_DURATION: time::Duration = time::Duration::weeks(1);

pub const DEFAULT_AUTH: &str = "jwt";

pub struct AuthUser {
    pub user_id: Uuid,
}

// Use in handler if auth is optional
pub struct OptionalAuthUser(pub Option<AuthUser>);

pub struct CsrfValidator;

#[derive(serde::Serialize, serde::Deserialize)]
struct AuthClaims {
    sub: Uuid,
    exp: i64,
}

pub fn hash_password(password: String) -> Result<String, HTTPError> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();

    match argon2.hash_password(password.as_bytes(), &salt) {
        Ok(password_hash) => Ok(password_hash.to_string()),
        Err(e) => {
            log::debug!("Failed to hash password: {:?}", e);
            Err(HTTPError::InternalServerError(
                "Internal Server Error".to_string(),
            ))
        }
    }
}

pub fn validate_password(password: String, password_hash: &str) -> Result<bool, HTTPError> {
    let parsed_hash = PasswordHash::new(password_hash).map_err(|e| {
        log::debug!("Invalid password hash format: {:?}", e);
        HTTPError::Unauthorized
    })?;
    let result = Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok();

    match result {
        true => Ok(true),
        false => Err(HTTPError::Unauthorized),
    }
}

pub async fn auth_user(
    username: &str,
    password: String,
    db: &PgPool,
) -> Result<AuthUser, HTTPError> {
    // Fetch password hash from the database
    let (id, password_hash) = crud::user::get_hash(username, db).await?;

    // Validate the password
    validate_password(password, &password_hash)?;

    Ok(AuthUser { user_id: id })
}

impl AuthUser {
    pub(in crate::http) fn to_jwt(&self, context: &AppState) -> Result<String, HTTPError> {
        let secret = &context.config.hmac_key;
        let token = encode(
            &Header::default(),
            &AuthClaims {
                sub: self.user_id,
                exp: (OffsetDateTime::now_utc() + DEFAULT_SESSION_DURATION).unix_timestamp(),
            },
            &EncodingKey::from_secret(secret.as_ref()),
        );

        match token {
            Ok(token) => {
                log::debug!("Token generated successfully");
                Ok(token)
            }
            Err(e) => {
                log::debug!("Failed to encode token: {:?}", e);
                Err(HTTPError::InternalServerError(
                    "Internal Server Error".to_string(),
                ))
            }
        }
    }
    fn from_authorization(ctx: &AppState, jwt_token: &str) -> Result<Self, HTTPError> {
        let secret = &ctx.config.hmac_key;
        // `token` is a struct with 2 fields: `header` and `claims` where `claims` is your own struct.
        let token = decode::<AuthClaims>(
            jwt_token,
            &DecodingKey::from_secret(secret.as_ref()),
            &Validation::default(),
        )
        .map_err(|e| {
            log::debug!("Failed to decode token: {:?}", e);
            HTTPError::Unauthorized
        });

        match token {
            Ok(token) => Ok(AuthUser {
                user_id: token.claims.sub,
            }),
            Err(e) => Err(e),
        }
    }
}

#[allow(dead_code)]
impl OptionalAuthUser {
    pub fn user_id(&self) -> Option<Uuid> {
        self.0.as_ref().map(|auth_user| auth_user.user_id)
    }
}

impl<S> FromRequestParts<S> for CsrfValidator
where
    S: Send + Sync,
{
    type Rejection = HTTPError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let client_token = parts.headers.get("x_csft").and_then(|v| v.to_str().ok());
        let server_token = parts
            .headers
            .get(COOKIE)
            .and_then(|v| v.to_str().ok())
            .and_then(|cookie_header| {
                cookie_header
                    .split(';')
                    .map(str::trim)
                    .find_map(|cookie| cookie.strip_prefix("s_csft="))
            });

        // Validate that both tokens exist and are equal
        match (client_token, server_token) {
            (Some(client_token), Some(server_token)) if client_token == server_token => Ok(Self),
            _ => {
                log::debug!("CSFT verification failed.");
                Err(HTTPError::Unauthorized)
            }
        }
    }
}

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync + AsRef<AppState>,
{
    type Rejection = HTTPError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // Extract the `ApiContext` extension
        let ctx = state.as_ref();

        // Parse cookies using CookieJar
        let jar = CookieJar::from_request_parts(parts, state).await;

        match jar {
            Ok(jar) => {
                let cookie = jar.get(DEFAULT_AUTH).ok_or_else(|| {
                    log::debug!("JWT cookie is missing");
                    HTTPError::Unauthorized
                })?;
                AuthUser::from_authorization(ctx, cookie.value())
            }
            Err(_) => Err(HTTPError::Unauthorized),
        }
    }
}

impl<S> FromRequestParts<S> for OptionalAuthUser
where
    S: Send + Sync + AsRef<AppState>,
{
    type Rejection = HTTPError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // Extract the `ApiContext` extension
        let ctx = state.as_ref();

        let jar = CookieJar::from_request_parts(parts, state).await;

        match jar {
            Ok(jar) => {
                // Try to get the cookie for JWT authorization
                if let Some(cookie) = jar.get(DEFAULT_AUTH) {
                    if let Ok(auth_user) = AuthUser::from_authorization(ctx, cookie.value()) {
                        return Ok(Self(Some(auth_user)));
                    }
                }
            }
            Err(_) => {}
        }

        Ok(Self(None))
    }
}
