use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
pub struct Dataset {
    pub dataset_id: Uuid,
    pub name: String,
    pub file_path: String,
    pub created_at: time::OffsetDateTime,
    pub version: i32,
    pub is_public: bool,
}
