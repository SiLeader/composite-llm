use async_trait::async_trait;
use aws_sdk_bedrockruntime::Client as BedrockClient;

use super::{ChatCompletionBackend, ChatCompletionStream};
use crate::convert::bedrock::{
    build_inference_config, build_tool_config, convert_converse_response,
    extract_system_and_messages, stream_event_to_response,
};
use crate::convert::generate_chat_cmpl_id;
use crate::error::CompositeLlmError;
use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};

pub struct BedrockBackend {
    client: BedrockClient,
    model_id: String,
}

impl BedrockBackend {
    pub fn new(client: BedrockClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }

    pub async fn from_env(model_id: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Self {
            client: BedrockClient::new(&config),
            model_id: model_id.into(),
        }
    }
}

#[async_trait]
impl ChatCompletionBackend for BedrockBackend {
    async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
        let model = req.model.clone();
        let (system_blocks, messages) = extract_system_and_messages(req.messages.clone())?;
        let inference_config = build_inference_config(&req);
        let tool_config = build_tool_config(&req)?;

        let mut builder = self
            .client
            .converse()
            .model_id(&self.model_id)
            .set_messages(Some(messages));

        if !system_blocks.is_empty() {
            builder = builder.set_system(Some(system_blocks));
        }
        if let Some(config) = inference_config {
            builder = builder.inference_config(config);
        }
        if let Some(tc) = tool_config {
            builder = builder.tool_config(tc);
        }

        let output = builder
            .send()
            .await
            .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?;

        convert_converse_response(&output, &model)
    }

    async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError> {
        let model = req.model.clone();
        let (system_blocks, messages) = extract_system_and_messages(req.messages.clone())?;
        let inference_config = build_inference_config(&req);
        let tool_config = build_tool_config(&req)?;

        let mut builder = self
            .client
            .converse_stream()
            .model_id(&self.model_id)
            .set_messages(Some(messages));

        if !system_blocks.is_empty() {
            builder = builder.set_system(Some(system_blocks));
        }
        if let Some(config) = inference_config {
            builder = builder.inference_config(config);
        }
        if let Some(tc) = tool_config {
            builder = builder.tool_config(tc);
        }

        let mut output = builder
            .send()
            .await
            .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?;

        let id = generate_chat_cmpl_id();

        // Use a channel to bridge the async recv() loop into a Stream
        let (tx, rx) = tokio::sync::mpsc::channel::<
            Result<CreateChatCompletionStreamResponse, CompositeLlmError>,
        >(32);

        tokio::spawn(async move {
            loop {
                match output.stream.recv().await {
                    Ok(Some(event)) => {
                        if let Some(resp) = stream_event_to_response(&event, &model, &id)
                            && tx.send(Ok(resp)).await.is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx
                            .send(Err(CompositeLlmError::Bedrock(e.to_string())))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}
