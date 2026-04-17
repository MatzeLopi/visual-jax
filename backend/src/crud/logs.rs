use crate::{http::error::Error as HTTPError, schemas::logs::Logs};
use log::error;
use sqlx::PgPool;

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
