# Composite LLM

A unified, OpenAI-compatible chat completion interface for multiple LLM backends in Rust.

`composite-llm` allows you to write code against a single API (based on `async-openai`) and switch between different LLM providers like OpenAI, Azure OpenAI, Amazon Bedrock, and Google Vertex AI using feature flags or runtime configuration.

## Features

- **Unified Interface**: Use `async_openai::types::chat::CreateChatCompletionRequest` and get `CreateChatCompletionResponse` regardless of the backend.
- **Multiple Backends**:
  - **OpenAI**: Direct support via `async-openai`.
  - **Azure OpenAI**: Support for Azure-hosted OpenAI models.
  - **Amazon Bedrock**: Support for models like Claude 3 via the Bedrock Converse API.
  - **Google Vertex AI**: Support for Gemini models via the Vertex AI API.
- **Streaming Support**: Unified streaming interface (`ChatCompletionStream`) across all backends.
- **Extensible**: easy to add new backends by implementing the `ChatCompletionBackend` trait.

## Installation

Add `composite-llm` to your `Cargo.toml`. You must enable at least one backend feature.

```toml
[dependencies]
composite-llm = { version = "0.1.0", features = ["backend-openai", "backend-bedrock"] }
```

## Usage

### 1. Initialize the Client

The `CompositeClient` enum wraps the specific backend implementation. You can initialize it with the desired backend.

```rust
use composite_llm::{CompositeClient, OpenAIBackend, BedrockBackend};
use async_openai::config::OpenAIConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // OpenAI Example
    let openai_config = OpenAIConfig::new().with_api_key("sk-...");
    let client = CompositeClient::OpenAI(OpenAIBackend::new(openai_config));

    // Bedrock Example
    // let client = CompositeClient::Bedrock(BedrockBackend::from_env("anthropic.claude-3-sonnet-20240229-v1:0").await);

    Ok(())
}
```

### 2. Make a Chat Completion Request

Use the `chat_completion` method with standard OpenAI request types.

```rust
use async_openai::types::chat::{CreateChatCompletionRequestArgs, ChatCompletionRequestUserMessageArgs};

let request = CreateChatCompletionRequestArgs::default()
    .model("gpt-4o") // Model name is backend-specific but passed through
    .messages([
        ChatCompletionRequestUserMessageArgs::default()
            .content("Hello, world!")
            .build()?
            .into()
    ])
    .build()?;

let response = client.chat_completion(request).await?;

for choice in response.choices {
    println!("Response: {}", choice.message.content.unwrap_or_default());
}
```

### 3. Streaming Responses

Use `chat_completion_stream` for streaming responses.

```rust
use tokio_stream::StreamExt;

let mut stream = client.chat_completion_stream(request).await?;

while let Some(result) = stream.next().await {
    match result {
        Ok(response) => {
            for choice in response.choices {
                if let Some(content) = choice.delta.content {
                    print!("{}", content);
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Feature Flags

- `backend-openai` (default): Enables the OpenAI backend.
- `backend-azure`: Enables the Azure OpenAI backend.
- `backend-bedrock`: Enables the Amazon Bedrock backend (requires AWS credentials).
- `backend-vertex`: Enables the Google Vertex AI backend (requires GCP authentication).

## License

This project is licensed under the Apache-2.0 License.
