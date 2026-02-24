use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
};
use composite_llm::{ChatCompletionBackend, OpenAIBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reads OPENAI_API_KEY from the environment automatically.
    let backend = OpenAIBackend::from_env();

    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
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
