use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use tokio_stream::StreamExt;

use super::{ChatCompletionBackend, ChatCompletionStream};
use crate::error::CompositeLlmError;
use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse};

/// A backend implementation for OpenAI.
///
/// This backend uses the `async-openai` crate to communicate with the OpenAI API.
pub struct OpenAIBackend {
    client: Client<OpenAIConfig>,
}

impl OpenAIBackend {
    /// Creates a new `OpenAIBackend` with the given configuration.
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            client: Client::with_config(config),
        }
    }

    /// Creates a new `OpenAIBackend` from environment variables.
    ///
    /// This uses `async_openai::Client::new()` which reads `OPENAI_API_KEY` and other
    /// environment variables.
    pub fn from_env() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[async_trait]
impl ChatCompletionBackend for OpenAIBackend {
    async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
        self.client
            .chat()
            .create(req)
            .await
            .map_err(CompositeLlmError::from)
    }

    async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError> {
        let stream = self
            .client
            .chat()
            .create_stream(req)
            .await
            .map_err(CompositeLlmError::from)?;

        Ok(Box::pin(stream.map(|r| r.map_err(CompositeLlmError::from))))
    }
}
