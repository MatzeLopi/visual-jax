use crate::http::error::Error as HTTPError;
use anyhow::Context; // Needed for context to work
use clap::Parser; // Needed for parse to work
use deadpool::managed::Pool;
use log::debug;
use rust_backend::crud::dataset;
use rust_backend::http;
use rust_backend::{SmtpManager, config::Config};
use sqlx::postgres::PgPoolOptions;
use tera::Tera;
use tokio::{fs::create_dir_all, time::Duration, time::sleep};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check if .env file exists, init logger, loda config
    dotenv::dotenv().ok();
    env_logger::init();

    let provider = tokio_rustls::rustls::crypto::aws_lc_rs::default_provider();

    provider
        .install_default()
        .expect("Failed to install rustls crypto provider");

    // Load config
    let config = Config::parse();

    // Create SMTP pool
    let smtp_manager = SmtpManager {
        host: config.mail_host.clone(),
        port: config.mail_port,
        username: config.mail_username.clone(),
        password: config.mail_password.clone(),
    };

    let smtp_pool: Pool<SmtpManager> = Pool::builder(smtp_manager).max_size(10).build().unwrap();

    // Create DB pool
    let db = PgPoolOptions::new()
        .max_connections(50)
        .connect(&config.database_url)
        .await
        .context("could not connect to database_url")?;

    // Migrate the DB
    sqlx::migrate!()
        .run(&db)
        .await
        .context("could not run migrations")?;

    // Create Tera engine
    let tera_context = Tera::new("templates/**/*").unwrap();

    // Create some dirs needed
    create_dir_all("./files/models/").await.map_err(|e| {
        debug!("Create Dir all in compiler router failed with: {:?}", e);
        HTTPError::InternalServerError(format!("An error ocured when saving the model file. {e} "))
    })?;

    let bg_db = db.clone();
    tokio::spawn(async move {
        loop {
            dataset::cleanup_stale_datasets(&bg_db).await;
            sleep(Duration::from_secs(24 * 60 * 60)).await;
        }
    });
    // Start Server
    http::serve(config, db, smtp_pool, tera_context)
        .await
        .unwrap();

    Ok(())
}
