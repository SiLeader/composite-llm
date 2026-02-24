use async_openai::{Client, config::AzureConfig};
use async_trait::async_trait;
use tokio_stream::StreamExt;

use super::{ChatCompletionBackend, ChatCompletionStream};
use crate::error::CompositeLlmError;
use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse};

/// A backend implementation for Azure OpenAI.
///
/// This backend uses the `async-openai` crate with `AzureConfig`.
pub struct AzureBackend {
    client: Client<AzureConfig>,
}

impl AzureBackend {
    /// Creates a new `AzureBackend` with the given configuration.
    pub fn new(config: AzureConfig) -> Self {
        Self {
            client: Client::with_config(config),
        }
    }
}

#[async_trait]
impl ChatCompletionBackend for AzureBackend {
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
