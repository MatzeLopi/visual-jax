use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub model_id: Uuid,
    pub user_id: Option<Uuid>,
    pub name: String,
    pub version: i32,
    pub is_public: bool,
    pub graph_json: Value,
    pub created_at: time::OffsetDateTime,
    pub updated_at: Option<time::OffsetDateTime>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NewModel {
    pub name: String,
    pub graph_json: Value,
    pub is_public: Option<bool>,
}
