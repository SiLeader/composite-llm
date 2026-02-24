use async_openai::types::chat::{
    ChatChoice, ChatChoiceStream, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCalls, ChatCompletionRequestMessage,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
    CompletionUsage, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse, FinishReason, Role, StopConfiguration,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestDeveloperMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionTools,
};
use async_openai::types::chat::FunctionCall;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ContentBlockDelta, ConversationRole, ConverseStreamOutput,
    InferenceConfiguration, Message, StopReason, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock,
    ToolSpecification, ToolUseBlock,
};

use crate::error::CompositeLlmError;

use super::{generate_chat_cmpl_id, unix_timestamp};

/// Convert `serde_json::Value` to `aws_smithy_types::Document`.
fn json_to_document(value: serde_json::Value) -> aws_smithy_types::Document {
    match value {
        serde_json::Value::Null => aws_smithy_types::Document::Null,
        serde_json::Value::Bool(b) => aws_smithy_types::Document::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(i as u64))
            } else if let Some(f) = n.as_f64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                aws_smithy_types::Document::Null
            }
        }
        serde_json::Value::String(s) => aws_smithy_types::Document::String(s),
        serde_json::Value::Array(arr) => {
            aws_smithy_types::Document::Array(arr.into_iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => {
            aws_smithy_types::Document::Object(
                map.into_iter().map(|(k, v)| (k, json_to_document(v))).collect(),
            )
        }
    }
}

/// Convert `aws_smithy_types::Document` to `serde_json::Value`.
fn document_to_json(doc: &aws_smithy_types::Document) -> serde_json::Value {
    match doc {
        aws_smithy_types::Document::Null => serde_json::Value::Null,
        aws_smithy_types::Document::Bool(b) => serde_json::Value::Bool(*b),
        aws_smithy_types::Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(i) => serde_json::json!(*i),
            aws_smithy_types::Number::NegInt(i) => serde_json::json!(*i),
            aws_smithy_types::Number::Float(f) => serde_json::json!(*f),
        },
        aws_smithy_types::Document::String(s) => serde_json::Value::String(s.clone()),
        aws_smithy_types::Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json).collect())
        }
        aws_smithy_types::Document::Object(map) => {
            serde_json::Value::Object(
                map.iter().map(|(k, v)| (k.clone(), document_to_json(v))).collect(),
            )
        }
    }
}

pub fn extract_system_and_messages(
    messages: Vec<ChatCompletionRequestMessage>,
) -> Result<(Vec<SystemContentBlock>, Vec<Message>), CompositeLlmError> {
    let mut system_blocks = Vec::new();
    let mut bedrock_messages = Vec::new();

    for msg in messages {
        match msg {
            ChatCompletionRequestMessage::System(s) => {
                let text = match s.content {
                    ChatCompletionRequestSystemMessageContent::Text(t) => t,
                    ChatCompletionRequestSystemMessageContent::Array(parts) => {
                        parts
                            .into_iter()
                            .map(|ChatCompletionRequestSystemMessageContentPart::Text(t)| t.text)
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };
                system_blocks.push(SystemContentBlock::Text(text));
            }
            ChatCompletionRequestMessage::Developer(d) => {
                let text = match d.content {
                    ChatCompletionRequestDeveloperMessageContent::Text(t) => t,
                    ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
                        parts
                            .into_iter()
                            .map(|ChatCompletionRequestDeveloperMessageContentPart::Text(t)| t.text)
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };
                system_blocks.push(SystemContentBlock::Text(text));
            }
            ChatCompletionRequestMessage::User(u) => {
                let text = match u.content {
                    ChatCompletionRequestUserMessageContent::Text(t) => t,
                    ChatCompletionRequestUserMessageContent::Array(parts) => {
                        parts
                            .into_iter()
                            .filter_map(|p| match p {
                                ChatCompletionRequestUserMessageContentPart::Text(t) => Some(t.text),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };
                bedrock_messages.push(
                    Message::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(text))
                        .build()
                        .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
                );
            }
            ChatCompletionRequestMessage::Assistant(a) => {
                let mut contents = Vec::new();
                if let Some(content) = a.content {
                    let text = match content {
                        ChatCompletionRequestAssistantMessageContent::Text(t) => t,
                        ChatCompletionRequestAssistantMessageContent::Array(parts) => {
                            parts
                                .into_iter()
                                .filter_map(|p| match p {
                                    ChatCompletionRequestAssistantMessageContentPart::Text(t) => Some(t.text),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };
                    if !text.is_empty() {
                        contents.push(ContentBlock::Text(text));
                    }
                }
                if let Some(tool_calls) = a.tool_calls {
                    for tc in tool_calls {
                        if let ChatCompletionMessageToolCalls::Function(func_call) = tc {
                            let input: serde_json::Value =
                                serde_json::from_str(&func_call.function.arguments)
                                    .unwrap_or_default();
                            contents.push(ContentBlock::ToolUse(
                                ToolUseBlock::builder()
                                    .tool_use_id(&func_call.id)
                                    .name(&func_call.function.name)
                                    .input(json_to_document(input))
                                    .build()
                                    .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
                            ));
                        }
                    }
                }
                if !contents.is_empty() {
                    let mut builder = Message::builder().role(ConversationRole::Assistant);
                    for c in contents {
                        builder = builder.content(c);
                    }
                    bedrock_messages.push(
                        builder
                            .build()
                            .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
                    );
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                let text = match t.content {
                    ChatCompletionRequestToolMessageContent::Text(text) => text,
                    ChatCompletionRequestToolMessageContent::Array(parts) => {
                        parts
                            .into_iter()
                            .map(|ChatCompletionRequestToolMessageContentPart::Text(t)| t.text)
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };
                let result = ToolResultBlock::builder()
                    .tool_use_id(&t.tool_call_id)
                    .content(ToolResultContentBlock::Text(text))
                    .build()
                    .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?;
                bedrock_messages.push(
                    Message::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::ToolResult(result))
                        .build()
                        .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
                );
            }
            _ => {}
        }
    }

    Ok((system_blocks, bedrock_messages))
}

pub fn build_inference_config(
    req: &CreateChatCompletionRequest,
) -> Option<InferenceConfiguration> {
    let has_params = req.temperature.is_some()
        || req.top_p.is_some()
        || req.max_completion_tokens.is_some()
        || req.stop.is_some();

    if !has_params {
        return None;
    }

    let mut builder = InferenceConfiguration::builder();

    if let Some(temp) = req.temperature {
        builder = builder.temperature(temp);
    }
    if let Some(top_p) = req.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(max_tokens) = req.max_completion_tokens {
        builder = builder.max_tokens(max_tokens as i32);
    }
    if let Some(ref stop) = req.stop {
        match stop {
            StopConfiguration::String(s) => {
                builder = builder.stop_sequences(s.clone());
            }
            StopConfiguration::StringArray(arr) => {
                for s in arr {
                    builder = builder.stop_sequences(s.clone());
                }
            }
        }
    }

    Some(builder.build())
}

pub fn build_tool_config(
    req: &CreateChatCompletionRequest,
) -> Result<Option<ToolConfiguration>, CompositeLlmError> {
    let tools = match &req.tools {
        Some(t) if !t.is_empty() => t,
        _ => return Ok(None),
    };

    let mut tool_list = Vec::new();
    for tool in tools {
        let func = match tool {
            ChatCompletionTools::Function(t) => &t.function,
            _ => continue,
        };

        let input_schema = if let Some(ref params) = func.parameters {
            ToolInputSchema::Json(json_to_document(params.clone()))
        } else {
            ToolInputSchema::Json(json_to_document(
                serde_json::json!({"type": "object", "properties": {}}),
            ))
        };

        let mut spec_builder = ToolSpecification::builder()
            .name(&func.name)
            .input_schema(input_schema);

        if let Some(ref desc) = func.description {
            spec_builder = spec_builder.description(desc);
        }

        tool_list.push(Tool::ToolSpec(
            spec_builder
                .build()
                .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
        ));
    }

    let mut config_builder = ToolConfiguration::builder();
    for tool in tool_list {
        config_builder = config_builder.tools(tool);
    }

    Ok(Some(
        config_builder
            .build()
            .map_err(|e| CompositeLlmError::Bedrock(e.to_string()))?,
    ))
}

pub fn convert_stop_reason(reason: &StopReason) -> FinishReason {
    match reason {
        StopReason::EndTurn | StopReason::StopSequence => FinishReason::Stop,
        StopReason::MaxTokens => FinishReason::Length,
        StopReason::ToolUse => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

#[allow(deprecated)]
pub fn convert_converse_response(
    output: &aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    model: &str,
) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
    let mut text_content = String::new();
    let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();

    if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(ref msg)) = output.output {
        for block in msg.content() {
            match block {
                ContentBlock::Text(t) => {
                    text_content.push_str(t);
                }
                ContentBlock::ToolUse(tu) => {
                    let args = serde_json::to_string(&document_to_json(tu.input()))
                        .unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(ChatCompletionMessageToolCalls::Function(
                        ChatCompletionMessageToolCall {
                            id: tu.tool_use_id().to_string(),
                            function: FunctionCall {
                                name: tu.name().to_string(),
                                arguments: args,
                            },
                        },
                    ));
                }
                _ => {}
            }
        }
    }

    let finish_reason = convert_stop_reason(output.stop_reason());

    let message = ChatCompletionResponseMessage {
        content: if text_content.is_empty() {
            None
        } else {
            Some(text_content)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        role: Role::Assistant,
        function_call: None,
        refusal: None,
        audio: None,
        annotations: None,
    };

    let usage = output.usage().map(|u| CompletionUsage {
        prompt_tokens: u.input_tokens() as u32,
        completion_tokens: u.output_tokens() as u32,
        total_tokens: (u.input_tokens() + u.output_tokens()) as u32,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    Ok(CreateChatCompletionResponse {
        id: generate_chat_cmpl_id(),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage,
        system_fingerprint: None,
        service_tier: None,
    })
}

#[allow(deprecated)]
pub fn stream_event_to_response(
    event: &ConverseStreamOutput,
    model: &str,
    id: &str,
) -> Option<CreateChatCompletionStreamResponse> {
    match event {
        ConverseStreamOutput::ContentBlockDelta(delta) => {
            let content = delta.delta().and_then(|d| match d {
                ContentBlockDelta::Text(t) => Some(t.to_string()),
                _ => None,
            });

            content.map(|text| CreateChatCompletionStreamResponse {
                id: id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created: unix_timestamp(),
                model: model.to_string(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: Some(text),
                        tool_calls: None,
                        role: None,
                        function_call: None,
                        refusal: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
                service_tier: None,
            })
        }
        ConverseStreamOutput::MessageStop(stop) => {
            let finish_reason = convert_stop_reason(stop.stop_reason());

            Some(CreateChatCompletionStreamResponse {
                id: id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created: unix_timestamp(),
                model: model.to_string(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: None,
                        tool_calls: None,
                        role: None,
                        function_call: None,
                        refusal: None,
                    },
                    finish_reason: Some(finish_reason),
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
                service_tier: None,
            })
        }
        ConverseStreamOutput::Metadata(meta) => {
            let usage = meta.usage().map(|u| CompletionUsage {
                prompt_tokens: u.input_tokens() as u32,
                completion_tokens: u.output_tokens() as u32,
                total_tokens: (u.input_tokens() + u.output_tokens()) as u32,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            });

            usage.map(|u| CreateChatCompletionStreamResponse {
                id: id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created: unix_timestamp(),
                model: model.to_string(),
                choices: vec![],
                usage: Some(u),
                system_fingerprint: None,
                service_tier: None,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    };

    #[test]
    fn test_extract_system_and_messages() {
        let messages = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are helpful.")
                    .build()
                    .unwrap(),
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Hello")
                    .build()
                    .unwrap(),
            ),
        ];

        let (system, msgs) = extract_system_and_messages(messages).unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_build_inference_config_none() {
        let req = CreateChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            ..Default::default()
        };
        assert!(build_inference_config(&req).is_none());
    }

    #[test]
    fn test_build_inference_config_some() {
        let req = CreateChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: Some(0.7),
            top_p: Some(0.9),
            ..Default::default()
        };
        let config = build_inference_config(&req);
        assert!(config.is_some());
    }
}
