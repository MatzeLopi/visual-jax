use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub model_id: Uuid,
    pub user_id: Uuid,
    pub version_: i32,
    pub model_name: Option<String>,
    pub model_description: Option<String>,
    pub model_path: String,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            model_id: Uuid::new_v4(),
            user_id: Uuid::nil(),
            version_: 1,
            model_name: None,
            model_description: None,
            model_path: format!("/app/files/models/{}.py", Uuid::now_v7()),
        }
    }
}
