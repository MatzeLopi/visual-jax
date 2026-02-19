FROM rust:1.93-bookworm

WORKDIR /app

COPY Cargo.toml Cargo.lock ./

COPY src ./src
COPY migrations ./migrations
COPY templates ./templates

EXPOSE 8080

CMD ["cargo", "run"]