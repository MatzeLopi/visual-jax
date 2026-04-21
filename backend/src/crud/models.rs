use crate::{
    http::error::Error as HTTPError,
    schemas::models::{Model, ModelQueryOptions},
};
use log::error;
use sqlx::{PgPool, QueryBuilder};
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

pub async fn get_models(query: ModelQueryOptions, db: &PgPool) -> Result<Vec<Model>, HTTPError> {
    // 1. Start with the base query
    // We use WHERE 1=1 as a trick so we can safely append " AND ..." for all filters
    let mut builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT model_id, user_id, version_, model_name, model_description, model_path 
         FROM models 
         WHERE 1=1",
    );

    // 2. Dynamically add filters
    if let Some(uid) = query.user_id {
        builder.push(" AND user_id = ");
        builder.push_bind(uid);
    }

    if let Some(mid) = query.model_id {
        builder.push(" AND model_id = ");
        builder.push_bind(mid);
    }

    if let Some(search_term) = query.search {
        builder.push(" AND model_name ILIKE ");
        builder.push_bind(format!("%{}%", search_term)); // ILIKE is case-insensitive search
    }

    // 3. Add sorting
    builder.push(" ORDER BY version_ DESC");

    // 4. Add pagination (Limit & Offset)
    if let Some(limit) = query.limit {
        builder.push(" LIMIT ");
        builder.push_bind(limit);
    }

    if let Some(offset) = query.offset {
        builder.push(" OFFSET ");
        builder.push_bind(offset);
    }

    // 5. Build and execute the query
    let result: Vec<Model> = builder
        .build_query_as::<Model>()
        .fetch_all(db)
        .await
        .map_err(|e| {
            error!("Error when reading models from database: {:?}", e);
            HTTPError::InternalServerError("Error reading from db.".to_string())
        })?;

    Ok(result)
}

pub async fn get_model_owner(uid: Uuid, db: &PgPool) -> Result<Option<Uuid>, HTTPError> {
    let user_id: Option<Option<Uuid>> = sqlx::query_scalar(
        "SELECT user_id FROM models WHERE model_id = $1 LIMIT 1"
    )
    .bind(uid)
    .fetch_optional(db)
    .await
    .map_err(|e| {
        error!("Error when retrieving model owner: {:?}", e);
        HTTPError::InternalServerError("Error in database.".to_string())
    })?;

    user_id.ok_or(HTTPError::NotFound)
}
