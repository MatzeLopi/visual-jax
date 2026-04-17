use crate::{http::error::Error as HTTPError, schemas::logs::Logs};
use log::error;
use sqlx::PgPool;
use uuid::Uuid;

pub async fn create_log(log: Logs, db: &PgPool) -> Result<(), HTTPError> {
    let _ = sqlx::query!(
        r#"INSERT INTO logs (origin_uid, logs) VALUES ($1, $2)"#,
        log.origin,
        log.text,
    )
    .execute(db)
    .await
    .map_err(|e| {
        error!(
            "Error when logging to database for uuid {:?}: {:?}",
            log.origin, e
        );
        HTTPError::InternalServerError("Error while logging data".to_string())
    })?;

    Ok(())
}

pub async fn get_log(uid: Uuid, limit: i64, db: &PgPool) -> Result<Vec<Logs>, HTTPError> {
    let result: Vec<Logs> = sqlx::query_as!(
        Logs,
        r#"
        SELECT 
            origin_uid as "origin!:_",
            logs as "text!:_",
            created_at as "created_at!:_"
        FROM logs 
        WHERE origin_uid = $1 
        ORDER BY created_at DESC 
        LIMIT $2
        "#,
        uid,
        limit
    )
    .fetch_all(db)
    .await
    .map_err(|e| {
        error!("Error when reading logs from database: {:?}", e);
        HTTPError::InternalServerError("Error reading from db.".to_string())
    })?;

    Ok(result)
}
