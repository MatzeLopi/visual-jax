use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use deadpool::managed::Pool;
use rust_backend::{
    config::Config,
    http::{create_router, AppState, SmtpManager},
};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
#[ignore]
async fn health_check() {
    // This test requires a running database.
    // Ensure you have the database running and the connection string is correct.

    let config = Config {
        database_url: "postgres://postgres:password@localhost:5432/visual_jax".to_string(),
        hmac_key: "secret".to_string(),
        mail_sender: "noreply@example.com".to_string(),
        mail_from: "No Reply".to_string(),
        mail_host: "localhost".to_string(),
        mail_port: 1025,
        mail_username: "user".to_string(),
        mail_password: "password".to_string(),
    };

    // We expect this to fail if no DB is running, hence the expect.
    // If you run `cargo test -- --ignored`, ensure DB is up.
    let db = PgPoolOptions::new()
        .max_connections(5)
        .connect(&config.database_url)
        .await
        .expect("Failed to connect to database");

    let smtp_manager = SmtpManager {
        host: config.mail_host.clone(),
        port: config.mail_port,
        username: config.mail_username.clone(),
        password: config.mail_password.clone(),
    };
    let smtp_pool = Pool::builder(smtp_manager).max_size(1).build().unwrap();

    let shared_state = Arc::new(AppState {
        config: Arc::new(config),
        db,
        smtp_pool: Arc::new(smtp_pool),
    });

    let app = create_router(&shared_state);

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
