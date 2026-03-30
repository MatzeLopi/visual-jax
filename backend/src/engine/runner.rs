// Runs python subbrocess to execute jax code for training and inference
use crate::http::error::Error as HTTPError;
use log::debug;
use std::process::Stdio;
use tokio::process;
pub struct Runner {
    pub code: String,
}

impl Runner {
    pub fn new(code: String) -> Self {
        Runner { code }
    }

    pub async fn run(self) -> Result<(), HTTPError> {
        let mut child = process::Command::new("python")
            .arg(file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                debug!("Error spawning python process {:?}", e);
                HTTPError::InternalServerError(format!("Error spawning python process {e}"))
            })?;
        Ok(())
    }
}
