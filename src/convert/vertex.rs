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
    ChatCompletionTools, ChatCompletionToolChoiceOption, ResponseFormat,
    ToolChoiceOptions,
};
use async_openai::types::chat::FunctionCall;
use serde::{Deserialize, Serialize};

use crate::error::CompositeLlmError;

use super::{generate_chat_cmpl_id, unix_timestamp};

// ── Vertex AI REST API types ──

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexRequest {
    pub contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<VertexTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<VertexToolConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VertexContent {
    pub role: String,
    pub parts: Vec<VertexPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<VertexFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<VertexFunctionResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VertexFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VertexFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexTool {
    pub function_declarations: Vec<VertexFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub struct VertexFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
pub struct FunctionCallingConfig {
    pub mode: String,
}

// ── Vertex AI Response types ──

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexResponse {
    pub candidates: Option<Vec<VertexCandidate>>,
    pub usage_metadata: Option<VertexUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexCandidate {
    pub content: Option<VertexContent>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexUsageMetadata {
    pub prompt_token_count: Option<u32>,
    pub candidates_token_count: Option<u32>,
    pub total_token_count: Option<u32>,
}

// ── Conversion functions ──

pub fn convert_request(
    req: &CreateChatCompletionRequest,
) -> Result<VertexRequest, CompositeLlmError> {
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();

    for msg in &req.messages {
        match msg {
            ChatCompletionRequestMessage::System(s) => {
                let text = match &s.content {
                    ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
                    ChatCompletionRequestSystemMessageContent::Array(parts) => parts
                        .iter()
                        .map(|ChatCompletionRequestSystemMessageContentPart::Text(t)| t.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                system_parts.push(VertexPart {
                    text: Some(text),
                    function_call: None,
                    function_response: None,
                });
            }
            ChatCompletionRequestMessage::Developer(d) => {
                let text = match &d.content {
                    ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
                    ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
                        .iter()
                        .map(|ChatCompletionRequestDeveloperMessageContentPart::Text(t)| t.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                system_parts.push(VertexPart {
                    text: Some(text),
                    function_call: None,
                    function_response: None,
                });
            }
            ChatCompletionRequestMessage::User(u) => {
                let text = match &u.content {
                    ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                    ChatCompletionRequestUserMessageContent::Array(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            ChatCompletionRequestUserMessageContentPart::Text(t) => {
                                Some(t.text.clone())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                contents.push(VertexContent {
                    role: "user".to_string(),
                    parts: vec![VertexPart {
                        text: Some(text),
                        function_call: None,
                        function_response: None,
                    }],
                });
            }
            ChatCompletionRequestMessage::Assistant(a) => {
                let mut parts = Vec::new();
                if let Some(ref content) = a.content {
                    let text = match content {
                        ChatCompletionRequestAssistantMessageContent::Text(t) => t.clone(),
                        ChatCompletionRequestAssistantMessageContent::Array(arr) => arr
                            .iter()
                            .filter_map(|p| match p {
                                ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                                    Some(t.text.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    };
                    if !text.is_empty() {
                        parts.push(VertexPart {
                            text: Some(text),
                            function_call: None,
                            function_response: None,
                        });
                    }
                }
                if let Some(ref tool_calls) = a.tool_calls {
                    for tc in tool_calls {
                        if let ChatCompletionMessageToolCalls::Function(func_call) = tc {
                            let args: serde_json::Value =
                                serde_json::from_str(&func_call.function.arguments)
                                    .unwrap_or_default();
                            parts.push(VertexPart {
                                text: None,
                                function_call: Some(VertexFunctionCall {
                                    name: func_call.function.name.clone(),
                                    args,
                                }),
                                function_response: None,
                            });
                        }
                    }
                }
                if !parts.is_empty() {
                    contents.push(VertexContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                let response_text = match &t.content {
                    ChatCompletionRequestToolMessageContent::Text(text) => text.clone(),
                    ChatCompletionRequestToolMessageContent::Array(parts) => parts
                        .iter()
                        .map(|ChatCompletionRequestToolMessageContentPart::Text(t)| t.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                let response_value = serde_json::from_str(&response_text)
                    .unwrap_or_else(|_| serde_json::json!({"result": response_text}));
                contents.push(VertexContent {
                    role: "user".to_string(),
                    parts: vec![VertexPart {
                        text: None,
                        function_call: None,
                        function_response: Some(VertexFunctionResponse {
                            name: t.tool_call_id.clone(),
                            response: response_value,
                        }),
                    }],
                });
            }
            _ => {}
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(VertexContent {
            role: "user".to_string(),
            parts: system_parts,
        })
    };

    let generation_config = build_generation_config(req);
    let tools = build_vertex_tools(req);
    let tool_config = build_vertex_tool_config(req);

    Ok(VertexRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
        tool_config,
    })
}

fn build_generation_config(req: &CreateChatCompletionRequest) -> Option<GenerationConfig> {
    let has_params = req.temperature.is_some()
        || req.top_p.is_some()
        || req.max_completion_tokens.is_some()
        || req.stop.is_some()
        || req.response_format.is_some();

    if !has_params {
        return None;
    }

    let stop_sequences = req.stop.as_ref().map(|s| match s {
        StopConfiguration::String(s) => vec![s.clone()],
        StopConfiguration::StringArray(arr) => arr.clone(),
    });

    let response_mime_type = req.response_format.as_ref().and_then(|rf| match rf {
        ResponseFormat::JsonObject => Some("application/json".to_string()),
        ResponseFormat::JsonSchema { .. } => Some("application/json".to_string()),
        _ => None,
    });

    Some(GenerationConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_completion_tokens,
        stop_sequences,
        response_mime_type,
    })
}

fn build_vertex_tools(req: &CreateChatCompletionRequest) -> Option<Vec<VertexTool>> {
    let tools = match &req.tools {
        Some(t) if !t.is_empty() => t,
        _ => return None,
    };

    let declarations: Vec<VertexFunctionDeclaration> = tools
        .iter()
        .filter_map(|t| match t {
            ChatCompletionTools::Function(f) => Some(VertexFunctionDeclaration {
                name: f.function.name.clone(),
                description: f.function.description.clone(),
                parameters: f.function.parameters.clone(),
            }),
            _ => None,
        })
        .collect();

    if declarations.is_empty() {
        None
    } else {
        Some(vec![VertexTool {
            function_declarations: declarations,
        }])
    }
}

fn build_vertex_tool_config(req: &CreateChatCompletionRequest) -> Option<VertexToolConfig> {
    req.tool_choice.as_ref().map(|tc| {
        let mode = match tc {
            ChatCompletionToolChoiceOption::Mode(m) => match m {
                ToolChoiceOptions::None => "NONE",
                ToolChoiceOptions::Auto => "AUTO",
                ToolChoiceOptions::Required => "ANY",
            },
            ChatCompletionToolChoiceOption::Function(_) => "ANY",
            _ => "AUTO",
        };
        VertexToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: mode.to_string(),
            },
        }
    })
}

fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[allow(deprecated)]
pub fn convert_vertex_response(
    resp: &VertexResponse,
    model: &str,
) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
    let mut choices = Vec::new();

    if let Some(ref candidates) = resp.candidates {
        for (i, candidate) in candidates.iter().enumerate() {
            let (text, tool_calls) = extract_parts(candidate);

            let finish_reason = candidate
                .finish_reason
                .as_deref()
                .map(convert_finish_reason)
                .unwrap_or(FinishReason::Stop);

            choices.push(ChatChoice {
                index: i as u32,
                message: ChatCompletionResponseMessage {
                    content: if text.is_empty() { None } else { Some(text) },
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
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            });
        }
    }

    let usage = resp.usage_metadata.as_ref().map(|u| CompletionUsage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    Ok(CreateChatCompletionResponse {
        id: generate_chat_cmpl_id(),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: model.to_string(),
        choices,
        usage,
        system_fingerprint: None,
        service_tier: None,
    })
}

fn extract_parts(
    candidate: &VertexCandidate,
) -> (String, Vec<ChatCompletionMessageToolCalls>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(ref content) = candidate.content {
        for part in &content.parts {
            if let Some(ref t) = part.text {
                text.push_str(t);
            }
            if let Some(ref fc) = part.function_call {
                tool_calls.push(ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4().as_simple()),
                        function: FunctionCall {
                            name: fc.name.clone(),
                            arguments: serde_json::to_string(&fc.args)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    },
                ));
            }
        }
    }

    (text, tool_calls)
}

#[allow(deprecated)]
pub fn convert_vertex_stream_chunk(
    resp: &VertexResponse,
    model: &str,
    id: &str,
) -> Option<CreateChatCompletionStreamResponse> {
    let candidates = resp.candidates.as_ref()?;
    let candidate = candidates.first()?;

    let (text, _tool_calls) = extract_parts(candidate);

    let finish_reason = candidate
        .finish_reason
        .as_deref()
        .map(convert_finish_reason);

    let usage = resp.usage_metadata.as_ref().map(|u| CompletionUsage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    Some(CreateChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: unix_timestamp(),
        model: model.to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: if text.is_empty() { None } else { Some(text) },
                tool_calls: None,
                role: Some(Role::Assistant),
                function_call: None,
                refusal: None,
            },
            finish_reason,
            logprobs: None,
        }],
        usage,
        system_fingerprint: None,
        service_tier: None,
    })
}

/// Parse SSE data lines from a byte buffer, returning parsed responses and remaining bytes.
pub fn parse_sse_events(buffer: &[u8]) -> (Vec<VertexResponse>, Vec<u8>) {
    let text = String::from_utf8_lossy(buffer);
    let mut responses = Vec::new();

    // Find the last complete event boundary (double newline)
    let last_boundary = text.rfind("\n\n");

    let (complete_text, remaining_text) = match last_boundary {
        Some(pos) => {
            let complete = &text[..pos + 2];
            let remaining = &text[pos + 2..];
            (complete, remaining)
        }
        None => {
            // No complete event found
            return (vec![], buffer.to_vec());
        }
    };

    for line in complete_text.lines() {
        let trimmed = line.trim();
        if let Some(json_str) = trimmed.strip_prefix("data: ")
            && let Ok(resp) = serde_json::from_str::<VertexResponse>(json_str) {
                responses.push(resp);
            }
    }

    let remaining = if remaining_text.trim().is_empty() {
        Vec::new()
    } else {
        remaining_text.as_bytes().to_vec()
    };

    (responses, remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    };

    #[test]
    fn test_convert_request_basic() {
        let req = CreateChatCompletionRequest {
            model: "gemini-pro".to_string(),
            messages: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content("Be helpful.")
                        .build()
                        .unwrap(),
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("Hello")
                        .build()
                        .unwrap(),
                ),
            ],
            ..Default::default()
        };

        let vertex_req = convert_request(&req).unwrap();
        assert!(vertex_req.system_instruction.is_some());
        assert_eq!(vertex_req.contents.len(), 1);
        assert_eq!(vertex_req.contents[0].role, "user");
    }

    #[test]
    fn test_convert_vertex_response() {
        let resp = VertexResponse {
            candidates: Some(vec![VertexCandidate {
                content: Some(VertexContent {
                    role: "model".to_string(),
                    parts: vec![VertexPart {
                        text: Some("Hello!".to_string()),
                        function_call: None,
                        function_response: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }]),
            usage_metadata: Some(VertexUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
        };

        let result = convert_vertex_response(&resp, "gemini-pro").unwrap();
        assert_eq!(result.choices.len(), 1);
        assert_eq!(
            result.choices[0].message.content.as_deref(),
            Some("Hello!")
        );
        assert_eq!(result.choices[0].finish_reason, Some(FinishReason::Stop));
        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_parse_sse_events() {
        let data = b"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n";
        let (responses, remaining) = parse_sse_events(data);
        assert_eq!(responses.len(), 1);
        assert!(remaining.is_empty());
    }
}
