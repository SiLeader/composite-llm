use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
};
use composite_llm::{ChatCompletionBackend, VertexBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_id = std::env::var("GCP_PROJECT_ID")?;
    let location = std::env::var("GCP_LOCATION").unwrap_or_else(|_| "us-central1".to_string());

    // Uses Application Default Credentials (ADC) for authentication.
    let backend = VertexBackend::new(project_id, location, "gemini-2.0-flash").await?;

    let req = CreateChatCompletionRequest {
        model: "gemini-2.0-flash".to_string(),
        messages: vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("What is the capital of France?")
                .build()?,
        )],
        ..Default::default()
    };

    let response = backend.chat_completion(req).await?;

    if let Some(choice) = response.choices.first() {
        println!("{}", choice.message.content.as_deref().unwrap_or(""));
    }

    Ok(())
}
