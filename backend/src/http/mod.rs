use crate::SmtpManager;
use crate::config::Config;
use anyhow::Context;
use axum::Router;
use axum::http::header::HeaderValue;
use deadpool::managed::Pool;
use http::{Method, header};
use sqlx::PgPool;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

mod dependencies;
pub mod error;
mod routers;
pub mod utils;

async fn clean_db(db: PgPool) {
    // Clean the database every 12 hours
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(3600 * 12)).await;

        let result = sqlx::query!(
            "DELETE FROM users WHERE is_verified = false AND created_at < NOW() - INTERVAL '1 day'"
        )
        .execute(&db)
        .await;

        if let Err(e) = result {
            log::error!("Error cleaning the database: {:?}", e);
        }
    }
}
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: PgPool,
    pub smtp_pool: Arc<Pool<SmtpManager>>,
}

pub async fn serve(config: Config, db: PgPool, smtp_pool: Pool<SmtpManager>) -> anyhow::Result<()> {
    // Create shared state
    let shared_state = Arc::new(AppState {
        config: Arc::new(config),
        db,
        smtp_pool: Arc::new(smtp_pool),
    });

    let origin = "http://localhost:3000".parse::<HeaderValue>().unwrap();

    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::PUT])
        .allow_origin(origin)
        .allow_credentials(true)
        .allow_headers([
            header::CONTENT_TYPE,
            header::RANGE,
            header::ACCEPT,
            header::AUTHORIZATION,
            header::ORIGIN,
            header::HOST,
            header::COOKIE,
            header::SET_COOKIE,
            header::HeaderName::from_static("x_csft"),
            header::HeaderName::from_static("s_csft"),
            header::HeaderName::from_static("jwt"),
        ]);

    // Build the app router
    let app = create_router(&shared_state).layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();

    // Start the database cleaner
    tokio::spawn(clean_db(shared_state.db.clone()));

    // Start the server using the listener
    axum::serve(listener, app)
        .await
        .context("Error running the server")
}

// Create Router
fn create_router(shared_state: &Arc<AppState>) -> Router {
    Router::new()
        .merge(routers::auth::router(shared_state.clone())) // Add auth router
        .merge(routers::user::router(shared_state.clone())) // Add user router
        .merge(routers::compiler::router(shared_state.clone()))
}
