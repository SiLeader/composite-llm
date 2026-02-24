# Copilot Instructions

## Build, Test, and Lint

### Build
- **Default (OpenAI backend only):** `cargo build`
- **All backends:** `cargo build --all-features`
- **Specific backend:** `cargo build --features backend-bedrock`
- **Type check:** `cargo check --all-features`

### Test
- **Run all tests:** `cargo test --all-features`
- **Run specific backend tests:** `cargo test --no-default-features --features backend-azure`
- **Run single test:** `cargo test --all-features -- test_name_substring`

### Lint
- **Format:** `cargo fmt --all --check`
- **Clippy:** `cargo clippy --all-features -- -D warnings`

## High-Level Architecture

This crate (`composite-llm`) provides a **unified OpenAI-compatible chat completion interface** across multiple providers.

1.  **Core Trait (`src/backend/mod.rs`):** `ChatCompletionBackend` defines `chat_completion` and `chat_completion_stream`. All backends must implement this.
2.  **Unified Client (`src/lib.rs`):** `CompositeClient` is an enum wrapping backends, gated by feature flags. It uses a `dispatch!` macro to delegate calls to the active variant.
3.  **Common Types:** All backends accept `async_openai` request types and return `async_openai` response types.
4.  **Conversion Layer (`src/convert/`):** Translates between OpenAI types and provider-native SDK types (e.g., Bedrock, Vertex).

## Key Conventions

- **Rust Edition:** Uses **Rust 2024** features (e.g., `let` chains).
- **Feature Flags:** Development on a specific backend *must* enable the corresponding feature flag (e.g., `backend-bedrock`, `backend-vertex`).
- **New Backends:**
    1.  Implement `ChatCompletionBackend` in `src/backend/<name>.rs`.
    2.  Add conversion logic in `src/convert/<name>.rs` if needed.
    3.  Add feature flag in `Cargo.toml`.
    4.  Register in `CompositeClient` and `CompositeLlmError`.
- **Error Handling:** Use `CompositeLlmError` (`thiserror`) with feature-gated variants for each backend.
