use crate::http::error::Error as HTTPError;
use log::error;
use sqlx::PgPool;
use uuid::Uuid;

pub async fn get_path(uid: Uuid, version: Option<i32>, db: &PgPool) -> Result<String, HTTPError> {
    let model_path = match version {
        Some(v) => sqlx::query_scalar!(
            r#"
                SELECT model_path 
                FROM models 
                WHERE model_id = $1 AND version_ = $2
                "#,
            uid,
            v
        )
        .fetch_one(db)
        .await
        .map_err(|e| {
            error!("Error when retriving the model path {:?}", e);
            HTTPError::InternalServerError("Error in database.".to_string())
        })?,
        None => sqlx::query_scalar!(
            r#"
                SELECT model_path 
                FROM models 
                WHERE model_id = $1 
                ORDER BY version_ DESC
                "#,
            uid
        )
        .fetch_one(db)
        .await
        .map_err(|e| {
            error!("Error when retriving the model path {:?}", e);
            HTTPError::InternalServerError("Error in database.".to_string())
        })?,
    };

    Ok(model_path)
}
