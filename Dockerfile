# Frontend Builder 
FROM node:25-bookworm AS frontend-builder
WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Rust Base
FROM rust:1.93-bookworm AS rust-base
WORKDIR /app
RUN cargo install cargo-chef

# Chef Planner
FROM rust-base AS planner
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src
RUN cargo chef prepare --recipe-path recipe.json

# Backend Builder
FROM rust-base AS backend-builder
COPY --from=planner /app/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json

COPY backend/src ./src
COPY backend/.sqlx ./.sqlx 
COPY backend/migrations ./migrations
COPY backend/templates ./templates

ENV SQLX_OFFLINE=true

RUN cargo build --release


# Runtime
FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p ./uploads ./files/models && chown -R appuser:appuser /app

COPY --from=backend-builder /app/target/release/rust_backend ./backend_app

COPY --from=backend-builder /app/migrations ./migrations
COPY --from=backend-builder /app/templates ./templates

COPY --from=frontend-builder /app/out ./static

USER appuser

EXPOSE 8080

CMD ["./backend_app"]