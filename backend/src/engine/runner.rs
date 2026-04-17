use crate::crud::logs::create_log;
use crate::http::error::Error as HTTPError;
use crate::schemas::logs::Logs;
use crate::schemas::models::Model;
use bollard::Docker;
use bollard::models::HostConfig;
use bollard::plugin::ContainerCreateBody;
use bollard::query_parameters::{CreateContainerOptionsBuilder, LogsOptionsBuilder};
use futures::{StreamExt, TryStreamExt};
use log::{debug, error, info};
use sqlx::PgPool;
use uuid::Uuid;

pub struct Runner {
    pub uid: Uuid,
    pub name: String,
    pub model: Model,
}

impl Runner {
    pub async fn new(model: Model) -> Result<Self, HTTPError> {
        let uid = Uuid::new_v4();
        Ok(Runner {
            uid: uid,
            name: format!("jax-runner-{}", uid).to_string(),
            model: model,
        })
    }

    pub async fn run(self, db: &PgPool) -> Result<(), HTTPError> {
        let docker = Docker::connect_with_socket_defaults().map_err(|e| {
            error!("Fehler bei der Verbindung zum Docker-Daemon: {:?}", e);
            HTTPError::InternalServerError(format!("Docker connection failed"))
        })?;

        let host_config = HostConfig {
            binds: Some(vec!["models:/app/files/models".to_string()]),
            auto_remove: Some(true),
            ..Default::default()
        };

        let config = ContainerCreateBody {
            image: Some("jax-runner".to_string()),
            cmd: Some(vec!["python".to_string(), self.model.model_path]),
            host_config: Some(host_config),
            ..Default::default()
        };

        let options = CreateContainerOptionsBuilder::new()
            .name(&self.name)
            .build();

        let create_response = docker
            .create_container(Some(options), config)
            .await
            .map_err(|e| {
                error!("Error creating container {:?}", e);
                HTTPError::InternalServerError("Unable to create container".to_string())
            })?;

        debug!("Container Create response: {:?}", create_response);
        docker
            .start_container(&self.name, None)
            .await
            .map_err(|e| {
                error!("Error starting container {:?}", e);
                HTTPError::InternalServerError("Unable to start container".to_string())
            })?;

        create_log(
            Logs {
                origin: self.model.model_id,
                text: "Container started.".to_string(),
            },
            db,
        )
        .await?;

        let log_options = LogsOptionsBuilder::new().stderr(true).stdout(true).build();
        let mut log_stream = docker.logs(&self.name, Some(log_options));
        let origin_uid = self.model.model_id.clone();
        let bg_db = db.clone();

        tokio::spawn(async move {
            while let Some(log_result) = log_stream.next().await {
                match log_result {
                    Ok(log_output) => {
                        let log_text = log_output.to_string();
                        let _ = create_log(
                            Logs {
                                origin: origin_uid,
                                text: log_text,
                            },
                            &bg_db,
                        )
                        .await;
                    }
                    Err(e) => {
                        error!("Error reading log stream for model {}: {:?}", origin_uid, e);
                    }
                }
            }
            let _ = create_log(
                Logs {
                    origin: origin_uid,
                    text: "Container stopped.".to_string(),
                },
                &bg_db,
            )
            .await;
            debug!("Log stream finished for model {}", origin_uid);
        });

        Ok(())
    }
}
