# Frontend
FROM node:25-bookworm AS frontend-builder
WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build 

# Cargo Chef Planner
FROM rust:1.93-bookworm AS planner
WORKDIR /app
RUN cargo install cargo-chef
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/src ./src
RUN cargo chef prepare --recipe-path recipe.json


# Cargo Chef Cook & Build Backend
FROM rust:1.93-bookworm AS backend-builder
WORKDIR /app
RUN cargo install cargo-chef

COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY backend/src ./src
COPY backend/.sqlx ./.sqlx 
COPY backend/migrations ./migrations
COPY backend/templates ./templates
ENV SQLX_OFFLINE=true

COPY --from=frontend-builder /app/out ./static

RUN cargo build --release

# Production Runtime

FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=backend-builder /app/target/release/rust_backend ./backend_app

COPY --from=backend-builder /app/migrations ./migrations
COPY --from=backend-builder /app/templates ./templates
COPY --from=backend-builder /app/static ./static

EXPOSE 8080

# Run the binary
CMD ["./backend_app"]