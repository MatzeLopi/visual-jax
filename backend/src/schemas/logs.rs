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
    pub run_id: Option<Uuid>,
    pub text: String,
    #[serde(with = "time::serde::rfc3339::option")]
    pub created_at: Option<OffsetDateTime>,
    pub severity: LogSeverity,
}
