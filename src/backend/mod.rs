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

/// A pinned, boxed stream of chat completion stream responses.
///
/// This type aliases a `Stream` that yields `Result<CreateChatCompletionStreamResponse, CompositeLlmError>`.
pub type ChatCompletionStream = Pin<
    Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse, CompositeLlmError>> + Send>,
>;

/// A trait for LLM backends that support chat completion.
///
/// All backends (OpenAI, Azure, Bedrock, Vertex) must implement this trait
/// to be usable with `CompositeClient`.
#[async_trait]
pub trait ChatCompletionBackend: Send + Sync {
    /// Sends a chat completion request to the backend.
    async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError>;

    /// Sends a streaming chat completion request to the backend.
    async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError>;
}
