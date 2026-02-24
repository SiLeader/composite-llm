use async_openai::config::AzureConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest,
};
use composite_llm::{AzureBackend, ChatCompletionBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AzureConfig::new()
        .with_api_base(std::env::var("AZURE_OPENAI_ENDPOINT")?)
        .with_api_key(std::env::var("AZURE_OPENAI_API_KEY")?)
        .with_deployment_id(std::env::var("AZURE_OPENAI_DEPLOYMENT_ID")?)
        .with_api_version(
            std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2024-02-01".to_string()),
        );

    let backend = AzureBackend::new(config);

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
