pub mod config;
pub mod crud;
pub mod engine;
pub mod http;
pub mod schemas;

use deadpool::managed::{Manager, RecycleResult};
use mail_send::{Error, SmtpClient, SmtpClientBuilder};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;

#[derive(Clone, Debug)]
pub struct SmtpManager {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
}

impl Manager for SmtpManager {
    type Type = SmtpClient<TlsStream<TcpStream>>;
    type Error = Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        SmtpClientBuilder::new(self.host.clone(), self.port)
            .credentials((self.username.clone(), self.password.clone()))
            .implicit_tls(true)
            .connect()
            .await
    }

    async fn recycle(
        &self,
        _: &mut Self::Type,
        _: &deadpool::managed::Metrics,
    ) -> RecycleResult<Self::Error> {
        Ok(())
    }
}
