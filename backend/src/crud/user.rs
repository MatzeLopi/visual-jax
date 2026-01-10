use std::sync::Arc;

use crate::{
    http::{AppState, error::Error as HTTPError, utils::random_string, utils::send_verification},
    schemas::users::User,
};
use sqlx::PgPool;
use uuid::Uuid;

pub async fn get_verification_token(username: &str, db: &PgPool) -> Result<String, HTTPError> {
    /// Get the verification token of a user
    ///
    /// # Arguments
    ///  username: &str - The username of the user
    ///  db: PgPool - The database connection pool
    ///
    /// # Returns
    ///  Result<String, HTTPError> - The verification token if found, an error otherwise
    let result = sqlx::query!(
        "SELECT verification_token FROM users WHERE username = $1",
        username
    )
    .fetch_one(db)
    .await
    .map_err(HTTPError::from)?;

    match result.verification_token {
        Some(token) => Ok(token),
        None => Err(HTTPError::NotFound),
    }
}

pub async fn verify_user(username: &str, db: &PgPool) -> Result<(), HTTPError> {
    /// Verify a user
    ///
    /// # Arguments
    ///  username: &str - The username of the user
    ///  db: PgPool - The database connection pool
    ///
    /// # Returns
    ///  Result<(), HTTPError> - The result of the operation
    let result = sqlx::query!(
        "UPDATE users SET is_verified = true WHERE username = $1",
        username
    )
    .execute(db)
    .await;

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            log::error!("Error verifying user: {:?}", e);
            Err(HTTPError::from(e))
        }
    }
}

pub async fn update_password(
    id: &Uuid,
    password_hash: &str,
    db: &PgPool,
) -> Result<bool, HTTPError> {
    let result = sqlx::query!(
        "update users set password_hash = $1 where user_id = $2;",
        password_hash,
        id
    )
    .execute(db)
    .await;

    match result {
        Ok(_) => Ok(true),
        Err(e) => {
            log::error!("Error updating password: {:?}", e);
            Err(HTTPError::from(e))
        }
    }
}

pub async fn get_user_by_id(id: &Uuid, db: &PgPool) -> Result<User, HTTPError> {
    /// Get a user by their id
    ///
    /// # Arguments
    ///  id: &Uuid - The user id
    ///  db: PgPool - The database connection pool
    ///
    /// # Returns
    ///  Result<User, HTTPError> - The user if found, an error otherwise
    let result = sqlx::query!("SELECT * FROM users WHERE user_id = $1", id)
        .fetch_one(db)
        .await;

    match result {
        Ok(row) => Ok(User {
            id: row.user_id,
            username: row.username,
            email: row.email,
            verified: row.is_verified,
        }),
        Err(e) => Err(HTTPError::from(e)),
    }
}

pub async fn check_username(username: &str, db: &PgPool) -> bool {
    /// Check if the username exists in the DB
    ///
    /// # Arguments
    ///  username: &str - The username to check
    ///  db: PgPool - The database connection pool
    ///
    /// # Returns
    ///  bool - True if the username exists, false otherwise
    let result = sqlx::query!("SELECT username FROM users WHERE username = $1", username)
        .fetch_one(db)
        .await;

    match result {
        Ok(_) => true,
        Err(_) => false,
    }
}

pub async fn check_email(email: &str, db: &PgPool) -> bool {
    /// Check if the email exists in the DB
    ///
    /// # Arguments
    ///  email: &str - The email to check
    ///  db: PgPool - The database connection pool
    ///
    /// # Returns
    ///  bool - True if the email exists, false otherwise
    let result = sqlx::query!("SELECT email FROM users WHERE email = $1", email)
        .fetch_one(db)
        .await;

    match result {
        Ok(_) => true,
        Err(_) => false,
    }
}

pub async fn create_user(
    username: &str,
    email: &str,
    password_hash: &str,
    state: Arc<AppState>,
) -> Result<(), HTTPError> {
    /// Create a new user in DB
    ///
    /// # Arguments
    ///  username: &str - The username of the user
    ///  email: &str - The email of the user
    ///  password_hash: &str - The password hash of the user
    ///
    /// # Returns
    ///  Result<(), HTTPError> - The result of the operation
    let db = &state.db;

    let uid = Uuid::new_v4();
    let verification_token = random_string(8);

    if check_username(username, db).await || check_email(email, db).await {
        return Err(HTTPError::Unauthorized);
    }

    let result = sqlx::query!(
        "INSERT INTO users (user_id, username, email, password_hash, verification_token, is_verified) VALUES ($1, $2, $3, $4, $5 , true)",
        uid,
        username,
        email,
        password_hash,
        verification_token
    ).execute(db).await;

    tokio::spawn(send_verification(
        email.to_string(),
        username.to_string(),
        verification_token,
        state,
    ));

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            log::error!("Error creating user: {}", e);
            Err(HTTPError::Unauthorized)
        }
    }
}

pub async fn delete_user(uid: &Uuid, db: &PgPool) -> Result<(), HTTPError> {
    /// Delete a user from DB
    ///
    /// # Arguments
    ///  uid: &Uuid - The user id of the user
    ///  db: &PgPool - The database connection pool
    ///
    /// # Returns
    ///  Result<(), sqlx::Error> - The result of the operation
    let result = sqlx::query!("DELETE FROM users WHERE user_id = $1", uid)
        .bind(uid)
        .execute(db)
        .await;
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(HTTPError::from(e)),
    }
}
pub async fn get_hash(username: &str, db: &PgPool) -> Result<(Uuid, String), sqlx::Error> {
    /// Get the user's id and password hash from the DB
    ///
    /// # Arguments
    ///   username: &str - The username of the user
    ///   db: &PgPool - The database connection pool
    ///
    /// # Returns
    ///  (Uuid, String) - The user's id and password hash
    let row = sqlx::query!(
        "SELECT user_id, password_hash FROM users WHERE username = $1",
        username
    )
    .fetch_one(db)
    .await?;

    Ok((row.user_id, row.password_hash))
}
