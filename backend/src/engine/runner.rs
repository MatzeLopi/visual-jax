use crate::crud::models;
use crate::http::error::Error as HTTPError;
use bollard::Docker;
use bollard::models::HostConfig;
use bollard::plugin::ContainerCreateBody;
use bollard::query_parameters::CreateContainerOptionsBuilder;
use log::{debug, error, info};
use sqlx::PgPool;
use uuid::Uuid;
pub struct Runner {
    pub uid: Uuid,
    pub path: String,
    pub name: String,
}

impl Runner {
    pub async fn new(uid: Uuid, db: &PgPool) -> Result<Self, HTTPError> {
        Ok(Runner {
            uid: uid,
            path: models::get_path(uid, None, db).await?,
            name: format!("jax_run_{}", uid),
        })
    }

    pub async fn run(self) -> Result<String, HTTPError> {
        let docker = Docker::connect_with_socket_defaults().map_err(|e| {
            error!("Fehler bei der Verbindung zum Docker-Daemon: {:?}", e);
            HTTPError::InternalServerError(format!("Docker connection failed"))
        })?;

        let script = format!("/app/files/models/{}.py", self.uid);

        let host_config = HostConfig {
            binds: Some(vec!["models:/app/files/models".to_string()]),
            ..Default::default()
        };

        let config = ContainerCreateBody {
            image: Some("jax-runner".to_string()),
            cmd: Some(vec!["python".to_string(), script]),
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
        docker
            .start_container(&self.name, None)
            .await
            .map_err(|e| {
                error!("Error starting container {:?}", e);
                HTTPError::InternalServerError("Unable to start container".to_string())
            })?;

        Ok(self.name)
    }
}
