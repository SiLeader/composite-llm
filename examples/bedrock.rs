use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest,
};
use composite_llm::{BedrockBackend, ChatCompletionBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Uses the default AWS credential chain (env vars, ~/.aws/credentials, IAM role, etc.)
    let backend =
        BedrockBackend::from_env("anthropic.claude-3-5-sonnet-20241022-v2:0").await;

    let req = CreateChatCompletionRequest {
        model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
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
