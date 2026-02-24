use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
};
use composite_llm::{ChatCompletionBackend, OpenAIBackend};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = OpenAIBackend::from_env();

    let req = CreateChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Write a short poem about Rust programming.")
                .build()?,
        )],
        ..Default::default()
    };

    let mut stream = backend.chat_completion_stream(req).await?;

    while let Some(result) = stream.next().await {
        let chunk = result?;
        if let Some(choice) = chunk.choices.first() {
            if let Some(ref content) = choice.delta.content {
                print!("{content}");
            }
        }
    }
    println!();

    Ok(())
}
