use std::pin::Pin;

use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use async_trait::async_trait;
use futures_core::Stream;

use crate::error::CompositeLlmError;

#[cfg(feature = "backend-openai")]
pub mod openai;

#[cfg(feature = "backend-azure")]
pub mod azure;

#[cfg(feature = "backend-bedrock")]
pub mod bedrock;

#[cfg(feature = "backend-vertex")]
pub mod vertex;

pub type ChatCompletionStream = Pin<
    Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse, CompositeLlmError>> + Send>,
>;

#[async_trait]
pub trait ChatCompletionBackend: Send + Sync {
    async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError>;

    async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError>;
}
