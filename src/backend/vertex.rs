use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use gcp_auth::TokenProvider;
use reqwest::Client;

use super::{ChatCompletionBackend, ChatCompletionStream};
use crate::convert::generate_chat_cmpl_id;
use crate::convert::vertex::{
    VertexResponse, convert_request, convert_vertex_response, convert_vertex_stream_chunk,
    parse_sse_events,
};
use crate::error::CompositeLlmError;
use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};

pub struct VertexBackend {
    client: Client,
    auth: Arc<dyn TokenProvider>,
    project_id: String,
    location: String,
    model_id: String,
}

impl VertexBackend {
    pub async fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<Self, CompositeLlmError> {
        let auth = gcp_auth::provider()
            .await
            .map_err(|e| CompositeLlmError::Vertex(e.to_string()))?;

        Ok(Self {
            client: Client::new(),
            auth,
            project_id: project_id.into(),
            location: location.into(),
            model_id: model_id.into(),
        })
    }

    fn base_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}",
            self.location, self.project_id, self.location, self.model_id
        )
    }

    async fn get_token(&self) -> Result<String, CompositeLlmError> {
        let scopes = &["https://www.googleapis.com/auth/cloud-platform"];
        let token = self
            .auth
            .token(scopes)
            .await
            .map_err(|e| CompositeLlmError::Vertex(e.to_string()))?;
        Ok(token.as_str().to_string())
    }
}

#[async_trait]
impl ChatCompletionBackend for VertexBackend {
    async fn chat_completion(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, CompositeLlmError> {
        let model = req.model.clone();
        let vertex_req = convert_request(&req)?;
        let token = self.get_token().await?;

        let url = format!("{}:generateContent", self.base_url());
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .json(&vertex_req)
            .send()
            .await
            .map_err(|e| CompositeLlmError::Vertex(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(CompositeLlmError::Vertex(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let vertex_resp: VertexResponse = resp
            .json()
            .await
            .map_err(|e| CompositeLlmError::Vertex(e.to_string()))?;

        convert_vertex_response(&vertex_resp, &model)
    }

    async fn chat_completion_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, CompositeLlmError> {
        let model = req.model.clone();
        let vertex_req = convert_request(&req)?;
        let token = self.get_token().await?;

        let url = format!("{}:streamGenerateContent?alt=sse", self.base_url());
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .json(&vertex_req)
            .send()
            .await
            .map_err(|e| CompositeLlmError::Vertex(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(CompositeLlmError::Vertex(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let id = generate_chat_cmpl_id();
        let byte_stream = resp.bytes_stream();

        let stream = SseStream {
            inner: Box::pin(byte_stream),
            buffer: Vec::new(),
            model,
            id,
            done: false,
            pending: Vec::new(),
        };

        Ok(Box::pin(stream))
    }
}

struct SseStream {
    inner: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    buffer: Vec<u8>,
    model: String,
    id: String,
    done: bool,
    pending: Vec<CreateChatCompletionStreamResponse>,
}

impl Stream for SseStream {
    type Item = Result<CreateChatCompletionStreamResponse, CompositeLlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Return pending items first
        if !this.pending.is_empty() {
            return Poll::Ready(Some(Ok(this.pending.remove(0))));
        }

        if this.done {
            return Poll::Ready(None);
        }

        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                this.buffer.extend_from_slice(&bytes);
                let (responses, remaining) = parse_sse_events(&this.buffer);
                this.buffer = remaining;

                for resp in responses {
                    if let Some(chunk) = convert_vertex_stream_chunk(&resp, &this.model, &this.id) {
                        this.pending.push(chunk);
                    }
                }

                if this.pending.is_empty() {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(Some(Ok(this.pending.remove(0))))
                }
            }
            Poll::Ready(Some(Err(e))) => {
                this.done = true;
                Poll::Ready(Some(Err(CompositeLlmError::Vertex(e.to_string()))))
            }
            Poll::Ready(None) => {
                this.done = true;
                // Process any remaining buffer
                if !this.buffer.is_empty() {
                    let (responses, _) = parse_sse_events(&this.buffer);
                    this.buffer.clear();
                    for resp in responses {
                        if let Some(chunk) =
                            convert_vertex_stream_chunk(&resp, &this.model, &this.id)
                        {
                            this.pending.push(chunk);
                        }
                    }
                    if !this.pending.is_empty() {
                        return Poll::Ready(Some(Ok(this.pending.remove(0))));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
