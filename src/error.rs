use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompositeLlmError {
    #[error("OpenAI error: {0}")]
    #[cfg(any(feature = "backend-openai", feature = "backend-azure"))]
    OpenAI(#[from] async_openai::error::OpenAIError),

    #[error("Bedrock error: {0}")]
    #[cfg(feature = "backend-bedrock")]
    Bedrock(String),

    #[error("Vertex AI error: {0}")]
    #[cfg(feature = "backend-vertex")]
    Vertex(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Unsupported: {0}")]
    Unsupported(String),
}
