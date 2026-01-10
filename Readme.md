# JAX/Flax Visual Graph Editor

A high-performance, node-based graphical interface for building, compiling, and training Neural Networks using **JAX** and **Flax**.

This tool bridges the gap between visual prototyping and high-performance research code. It allows users to design network architectures visually using **React Flow**, which are then compiled by a type-safe **Rust** backend into optimized **Python/JAX** code.

## 🏗 Architecture

The system is built on a strict separation of concerns:

1.  **Frontend (UI/UX):**
    * **Framework:** Next.js (React) + TypeScript.
    * **Editor:** React Flow for the node-graph canvas.
    * **Role:** Handles user interaction, graph manipulation, and validates node connections visually based on JSON Schemas provided by the backend.

2.  **Backend (The Compiler & API):**
    * **Framework:** Rust (Axum).
    * **Database:** PostgreSQL (via SQLx) for user management and project storage.
    * **Engine:**
        * Uses `petgraph` for topological sorting and cycle detection.
        * Uses `tera` templates to transpile the graph into valid Python code.
        * Uses `schemars` to expose Rust types as JSON Schemas to the frontend.

3.  **Execution Layer (The Runner):**
    * **Language:** Python 3.10+.
    * **Libraries:** JAX, Flax (Neural Net), Optax (Optimizers), Polars (Data Loading).
    * **Role:** Receives generated scripts from Rust and executes training loops on the CPU/GPU.

## 🚀 Features

* **Visual Node Editor:** Drag-and-drop interface for Layers (Dense, Conv2D), Activations, and Data Inputs.
* **Type-Safe Compilation:** Rust guarantees that the graph structure is valid (DAG) before any Python code is generated.
* **Dynamic Schema Loading:** The Frontend UI is generated dynamically from Rust Structs. Adding a new Layer type in Rust automatically updates the UI.
* **Smart Data Loading:**
    * **Tabular:** Uses Polars for high-speed CSV/Parquet loading.
    * **Images:** Uses NumPy/TensorFlow datasets.
* **JAX Optimization:** Automatically handles `jit` compilation and `vmap` logic in the generated code.

## 🛠 Tech Stack

| Component       | Technology     | Description                 |
| :-------------- | :------------- | :-------------------------- |
| **Backend**     | **Rust**       | Axum, Tokio, SQLx, Serde    |
| **Graph Logic** | **Rust**       | Petgraph, Tera, Schemars    |
| **Frontend**    | **TypeScript** | Next.js 13+, Tailwind CSS   |
| **Visuals**     | **React**      | React Flow / XYFlow         |
| **ML Core**     | **Python**     | JAX, Flax, Optax, Polars    |
| **Database**    | **PostgreSQL** | User auth & Project storage |

## 📦 Project Structure

```text
.
├── backend/                # Rust / Axum Application
│   ├── migrations/         # SQLx Database migrations
│   ├── templates/          # Tera templates (.py.j2) for code generation
│   └── src/
│       ├── config.rs       # Env configuration
│       ├── crud/           # Database interactions
│       ├── engine/         # CORE LOGIC: Types, Graph, Compiler
│       │   ├── types.rs    # Node definitions (LayerType, InputType)
│       │   └── mod.rs
│       ├── http/           # API Routers & Handlers
│       └── main.rs         # Entry point
│
└── frontend/               # Next.js Application
    ├── app/
    │   ├── editor/         # The main Graph Editor Page
    │   ├── components/     # UI Components (Sidebar, Inspector)
    │   └── lib/            # API Clients & Utilities
    └── public/             # Static assets