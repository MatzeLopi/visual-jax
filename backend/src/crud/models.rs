use crate::{
    http::error::Error as HTTPError,
    schemas::models::{Model, ModelQueryOptions},
};
use log::error;
use sqlx::{PgPool, QueryBuilder};
use uuid::Uuid;

pub async fn update_name(
    user_id: &Uuid,
    model_id: &Uuid,
    name: String,
    db: &PgPool,
) -> Result<(), HTTPError> {
    Ok(())
}

pub async fn insert_model(model: &Model, db: &PgPool) -> Result<i32, HTTPError> {
    let mut tx = db.begin().await.map_err(|e| {
        error!("Transaktionsstart fehlgeschlagen: {:?}", e);
        HTTPError::InternalServerError("Database error.".to_string())
    })?;

    sqlx::query!(
        r#"
        INSERT INTO models (model_id, user_id, model_name, model_description)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (model_id) DO UPDATE
        SET model_name = EXCLUDED.model_name,
            model_description = EXCLUDED.model_description,
            updated_at = NOW()
        "#,
        model.model_id,
        model.user_id,
        model.model_name,
        model.model_description
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        error!("Fehler beim Insert in models: {:?}", e);
        HTTPError::InternalServerError("Failed to save model metadata.".to_string())
    })?;

    let version = sqlx::query_scalar!(
        r#"
        INSERT INTO model_versions (model_id, model_path)
        VALUES ($1, $2)
        RETURNING version_
        "#,
        model.model_id,
        model.model_path
    )
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| {
        error!("Fehler beim Insert in model_versions: {:?}", e);
        HTTPError::InternalServerError("Failed to save model version.".to_string())
    })?;

    tx.commit().await.map_err(|e| {
        error!("Transaktions-Commit fehlgeschlagen: {:?}", e);
        HTTPError::InternalServerError("Transaction failed.".to_string())
    })?;

    Ok(version)
}
pub async fn get_path(uid: Uuid, version: Option<i32>, db: &PgPool) -> Result<String, HTTPError> {
    let model_path = match version {
        Some(v) => {
            sqlx::query_scalar!(
                r#"
            SELECT model_path
            FROM model_versions
            WHERE model_id = $1 AND version_ = $2
            "#,
                uid,
                v
            )
            .fetch_one(db)
            .await
        }
        None => {
            sqlx::query_scalar!(
                r#"
            SELECT model_path
            FROM model_versions
            WHERE model_id = $1
            ORDER BY version_ DESC
            LIMIT 1
            "#,
                uid
            )
            .fetch_one(db)
            .await
        }
    }
    .map_err(|e| {
        error!("Fehler beim Abrufen des model_path: {:?}", e);
        HTTPError::InternalServerError("Error reading from database.".to_string())
    })?;

    Ok(model_path)
}

pub async fn get_models(query: ModelQueryOptions, db: &PgPool) -> Result<Vec<Model>, HTTPError> {
    let mut builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        "SELECT m.model_id, m.user_id, mv.version_, m.model_name, m.model_description, mv.model_path
         FROM models m
         INNER JOIN model_versions mv ON m.model_id = mv.model_id
         WHERE 1=1",
    );

    if let Some(uid) = query.user_id {
        builder.push(" AND m.user_id = ");
        builder.push_bind(uid);
    }

    if let Some(mid) = query.model_id {
        builder.push(" AND m.model_id = ");
        builder.push_bind(mid);
    }

    if let Some(search_term) = query.search {
        builder.push(" AND m.model_name ILIKE ");
        builder.push_bind(format!("%{}%", search_term));
    }

    builder.push(" ORDER BY m.updated_at DESC, mv.version_ DESC");

    if let Some(limit) = query.limit {
        builder.push(" LIMIT ");
        builder.push_bind(limit);
    }

    if let Some(offset) = query.offset {
        builder.push(" OFFSET ");
        builder.push_bind(offset);
    }

    let result: Vec<Model> = builder
        .build_query_as::<Model>()
        .fetch_all(db)
        .await
        .map_err(|e| {
            error!("Fehler beim Lesen der model_versions: {:?}", e);
            HTTPError::InternalServerError("Error reading from db.".to_string())
        })?;

    Ok(result)
}
