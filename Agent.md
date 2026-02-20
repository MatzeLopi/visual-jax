# AGENTS.md

## Setup Commands

- Start the full development stack: `docker compose up`
- Services map to: Next.js (`:3000`), Rust Axum API (`:8080`), PostgreSQL (`:5432`).
- Do NOT modify the `docker-compose.yaml`.

## Architecture Boundaries & Conventions

- **Single Source of Truth:** Rust (`backend/src/engine/types.rs`) is the absolute source of truth for all graph node types. Do NOT manually create or duplicate TypeScript interfaces for graph nodes in the frontend. The Next.js UI builds itself dynamically via JSON schemas (`schemars`).
- **Graph Validation:** Validation logic belongs exclusively in the Rust backend (`petgraph`). Do not implement validation checks in the Next.js frontend or Python execution layers.
- **Database Rules:** Use `SQLx` for all database interactions. Schema changes MUST be executed via new SQL migration files in `backend/migrations/`. Do not write raw SQL outside of the `backend/src/crud/` directory.
- **Python Generation:** Do not write static `.py` files for the core training loop. All Python code is transpiled dynamically. To change execution logic, modify the Tera templates located exactly at `backend/templates/*.py.j2`.

## Workflows

- **Adding a New Layer:**
  1. Define the new struct/enum in `backend/src/engine/types.rs`.
  2. Map the new type to JAX/Flax code inside `backend/src/engine/transpiler.rs`.
  3. The frontend should automatically update from the schemas endpoint

## Security Constraints

- **Execution Safety:** Never execute the dynamically generated Python code directly on the host system to prevent arbitrary code execution vulnerabilities.
- **Production Builds:** Do not run `npm run build` or `pnpm build` inside the frontend agent session, as it overrides the `.next` development assets and breaks hot module replacement.
