# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build                                    # Build with default feature (backend-openai)
cargo build --all-features                     # Build with all backends
cargo build --features backend-bedrock         # Build a specific backend
cargo check --all-features                     # Quick type-check all backends
```

No tests or lints are configured yet.

## Rust Edition

This project uses **Rust edition 2024** — `let` chains and other 2024 features are used in the codebase (e.g., `if let Some(x) = ... && condition`).

## Architecture

This is a Rust library crate (`composite-llm`) that provides a **unified OpenAI-compatible chat completion interface** over multiple LLM provider backends. All backends accept `async_openai::types::chat::CreateChatCompletionRequest` and return the corresponding OpenAI response types, so consumers write against one API regardless of the underlying provider.

### Core Trait

`ChatCompletionBackend` (in `src/backend/mod.rs`) defines two async methods: `chat_completion` (single response) and `chat_completion_stream` (returns a `Pin<Box<dyn Stream>>`). Every backend implements this trait.

### CompositeClient Dispatch

`CompositeClient` (in `src/lib.rs`) is an enum with one variant per backend, gated by feature flags. A `dispatch!` macro delegates method calls to the active variant, avoiding manual match boilerplate.

### Backends (feature-gated)

| Feature | Module | Underlying SDK | Notes |
|---|---|---|---|
| `backend-openai` (default) | `backend::openai` | `async-openai` with `OpenAIConfig` | Thin wrapper, direct passthrough |
| `backend-azure` | `backend::azure` | `async-openai` with `AzureConfig` | Same passthrough pattern as OpenAI |
| `backend-bedrock` | `backend::bedrock` | `aws-sdk-bedrockruntime` | Uses Converse API; streaming via mpsc channel bridge |
| `backend-vertex` | `backend::vertex` | `reqwest` + `gcp_auth` | Raw HTTP to Vertex AI REST API; custom `SseStream` for streaming |

### Conversion Layer

`src/convert/` handles translating between OpenAI request/response types and provider-native formats:
- `convert::bedrock` — Converts OpenAI messages to Bedrock Converse messages/system blocks, builds inference and tool configs, converts responses back
- `convert::vertex` — Converts to/from Vertex AI's `generateContent` JSON format, includes SSE parsing for streaming
- `convert::mod.rs` — Shared utilities: `generate_chat_cmpl_id()` and `unix_timestamp()`

### Error Handling

`CompositeLlmError` (in `src/error.rs`) is a `thiserror` enum with feature-gated variants per backend plus shared `Serde` and `Unsupported` variants.

### Key Pattern

When adding a new backend: create `src/backend/<name>.rs` implementing `ChatCompletionBackend`, add a conversion module under `src/convert/<name>.rs` if the provider doesn't speak OpenAI natively, add a feature flag in `Cargo.toml`, and wire it into `CompositeClient` and `CompositeLlmError` with the appropriate `#[cfg(feature = ...)]` gates.
