# AGENTS.md

This document outlines the strict constraints, architecture boundaries, and workflow rules for all AI agents (including Jules) operating within this repository. 

## Setup & Execution

- **Start the Development Stack:** `docker compose -f docker-compose_dev.yaml up --build`
- **Service Mapping:** Rust Axum API (`:8080`), PostgreSQL (`:5432`).
- **Constraint:** Do NOT modify `docker-compose.yaml` or `docker-compose_dev.yaml` unless explicitly instructed by the user.

## Architecture Boundaries & Conventions

- **Single Source of Truth:** Rust (`backend/src/engine/types.rs` and `backend/src/schemas/`) is the absolute source of truth for all graph node types and API schemas. DO NOT manually create or duplicate TypeScript interfaces for graph nodes in the frontend. The Next.js UI builds itself dynamically via JSON schemas (`schemars`).
- **Graph Validation:** Validation logic belongs exclusively in the Rust backend (`petgraph`). DO NOT implement validation checks in the Next.js frontend or Python execution layers.
- **Database Rules:** Use `SQLx` for all database interactions. Schema changes MUST be executed via new SQL migration files in `backend/migrations/`. DO NOT write raw SQL outside of the `backend/src/crud/` directory.
- **Python Generation:** DO NOT write static `.py` files for the core training loop. All Python code is transpiled dynamically. To change execution logic, modify the Tera templates located exactly at `backend/templates/*.py.j2`.

## Workflows & Development Cycle

- **Committing Code (Jules' Rule):** The build MUST be completely successful before Jules (or any agent) is allowed to commit. Do not commit broken code, failing tests, or code that causes the Docker build to crash.
- **Adding a New Layer / Modifying Types:**
  1. Define the new `struct`/`enum` in `backend/src/engine/types.rs` (for internal engine logic) or the appropriate file in `backend/src/schemas/` (for API inputs/outputs).
  2. Map the new type to JAX/Flax code inside `backend/src/engine/transpiler.rs`.
  3. Verify the Rust compilation (`cargo check`). The frontend will automatically update from the schemas endpoint.
- **Database & SQL Changes:** If any SQL code (`sqlx::query!`) or migrations are added or modified, you MUST run `cargo sqlx prepare` inside the `backend/` directory before building or committing. This updates the offline `.sqlx` query data required for successful builds.
- **Testing & Verification:** Before concluding a task, verify the integrity of the codebase. Check Rust code using `cargo clippy` or `cargo test` if applicable within the `backend/` directory.

## Security Constraints

- **Execution Safety:** NEVER execute the dynamically generated Python code directly on the host system. This prevents arbitrary code execution vulnerabilities.
- **Production Builds:** DO NOT run `npm run build` or `pnpm build` inside the frontend agent session. Doing so overrides the `.next` development assets and breaks hot module replacement.
- **Environment Variables:** Do not log or print environment variables or secrets to the console during debugging.