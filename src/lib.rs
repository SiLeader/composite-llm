pub mod backend;
pub mod convert;
pub mod error;

pub use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
pub use backend::ChatCompletionBackend;
pub use backend::ChatCompletionStream;
pub use error::CompositeLlmError;

#[cfg(feature = "backend-azure")]
pub use backend::azure::AzureBackend;
#[cfg(feature = "backend-bedrock")]
pub use backend::bedrock::BedrockBackend;
#[cfg(feature = "backend-openai")]
pub use backend::openai::OpenAIBackend;
#[cfg(feature = "backend-vertex")]
pub use backend::vertex::VertexBackend;

/// A unified client for multiple LLM backends.
///
/// This enum wraps the specific backend implementation (OpenAI, Azure, Bedrock, Vertex)
/// and delegates method calls to the active backend.
///
/// Use feature flags to enable specific backends.
pub enum CompositeClient {
    #[cfg(feature = "backend-openai")]
    /// The OpenAI backend.
    OpenAI(OpenAIBackend),
    #[cfg(feature = "backend-azure")]
    /// The Azure OpenAI backend.
    Azure(AzureBackend),
    #[cfg(feature = "backend-bedrock")]
    /// The Amazon Bedrock backend.
    Bedrock(BedrockBackend),
    #[cfg(feature = "backend-vertex")]
    /// The Google Vertex AI backend.
    Vertex(VertexBackend),
}

macro_rules! dispatch {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            #[cfg(feature = "backend-openai")]
            CompositeClient::OpenAI(b) => b.$method($($arg),*).await,
            #[cfg(feature = "backend-azure")]
            CompositeClient::Azure(b) => b.$method($($arg),*).await,
            #[cfg(feature = "backend-bedrock")]
            CompositeClient::Bedrock(b) => b.$method($($arg),*).await,
            #[cfg(feature = "backend-vertex")]
            CompositeClient::Vertex(b) => b.$method($($arg),*).await,
            #[cfg(not(any(
                feature = "backend-openai",
                feature = "backend-azure",
                feature = "backend-bedrock",
                feature = "backend-vertex",
            )))]
            _ => unreachable!("no backend feature enabled"),
        }
    };
}

impl CompositeClient {
    /// Sends a chat completion request to the configured backend.
    ///
    /// # Arguments
    ///
    /// * `req` - A `CreateChatCompletionRequest` containing the model, messages, and other parameters.
    ///
    /// # Returns
    ///
    /// * `Result<CreateChatCompletionResponse, CompositeLlmError>` - The response from the backend or an error.
    pub async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
        dispatch!(self, chat_completion, req)
    }

    /// Sends a streaming chat completion request to the configured backend.
    ///
    /// # Arguments
    ///
    /// * `req` - A `CreateChatCompletionRequest` containing the model, messages, and other parameters.
    ///
    /// # Returns
    ///
    /// * `Result<ChatCompletionStream, CompositeLlmError>` - A stream of response chunks or an error.
    pub async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError> {
        dispatch!(self, chat_completion_stream, req)
    }
}
