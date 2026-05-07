use crate::{
    http::error::Error as HTTPError,
    schemas::logs::{LogSeverity, Logs},
};
use log::error;
use sqlx::PgPool;
use uuid::Uuid;

pub async fn create_log(log: Logs, db: &PgPool) -> Result<(), HTTPError> {
    let _ = sqlx::query!(
        r#"INSERT INTO logs (origin_uid, run_id, logs, severity) VALUES ($1, $2, $3, $4::log_severity)"#,
        log.origin,
        log.run_id,
        log.text,
        log.severity as LogSeverity,
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
            run_id as "run_id",
            logs as "text!:_",
            severity as "severity:_",
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
