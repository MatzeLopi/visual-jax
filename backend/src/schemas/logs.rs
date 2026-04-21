use serde::{Deserialize, Serialize};
use sqlx;
use time::OffsetDateTime;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, sqlx::Type, Clone)]
#[sqlx(type_name = "log_severity", rename_all = "UPPERCASE")]
pub enum LogSeverity {
    Info,
    Error,
    Warn,
    Debug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct Logs {
    pub origin: Uuid,
    pub text: String,
    pub created_at: Option<OffsetDateTime>,
    pub severity: LogSeverity,
}
