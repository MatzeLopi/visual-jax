use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct Logs {
    pub origin: Uuid,
    pub text: String,
    pub created_at: Option<OffsetDateTime>,
}
