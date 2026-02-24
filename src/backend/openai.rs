use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use tokio_stream::StreamExt;

use super::{ChatCompletionBackend, ChatCompletionStream};
use crate::error::CompositeLlmError;
use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse};

pub struct OpenAIBackend {
    client: Client<OpenAIConfig>,
}

impl OpenAIBackend {
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            client: Client::with_config(config),
        }
    }

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
