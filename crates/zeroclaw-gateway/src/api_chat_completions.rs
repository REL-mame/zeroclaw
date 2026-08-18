//! OpenAI-compatible `POST /v1/chat/completions` adapter.
//!
//! This module owns the chat-completions wire contract: request/response
//! types, the OpenAI-compatible error envelope, request validation, and
//! (added incrementally) agent routing, handler orchestration, SSE/JSON
//! dispatch, and the tool whitelist.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

// ── Request wire types ──────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String, // missing -> "" (default-agent shorthand)
    pub messages: Vec<ChatCompletionMessage>, // required
    #[serde(default)]
    pub stream: bool, // missing -> false
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub stop: Option<serde_json::Value>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub tools: Option<Vec<ChatCompletionTool>>,
    pub tool_choice: Option<serde_json::Value>,
    pub stream_options: Option<StreamOptions>,
    pub n: Option<u32>,
    pub response_format: Option<serde_json::Value>,
    pub seed: Option<i64>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub user: Option<String>,
    pub logit_bias: Option<serde_json::Value>,
    pub max_completion_tokens: Option<u32>,
    // Behavior-changing fields: modeled so `validate_unsupported_params` can
    // reject them explicitly rather than silently dropping them.
    pub parallel_tool_calls: Option<bool>,
    pub service_tier: Option<String>,
    pub functions: Option<serde_json::Value>,
    pub function_call: Option<serde_json::Value>,
    pub reasoning_effort: Option<String>,
    pub modalities: Option<serde_json::Value>,
    pub audio: Option<serde_json::Value>,
    pub prediction: Option<serde_json::Value>,
    pub web_search_options: Option<serde_json::Value>,
    // Benign annotation fields: modeled + tolerated, never rejected.
    pub metadata: Option<serde_json::Value>,
    pub store: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ChatCompletionTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value, // missing -> Null; OpenAI allows omitting parameters
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ── Response wire types ─────────────────────────────────────────────────────

// Consumed by the handler; defined here so the wire contract is fixed before
// orchestration lands.

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct ChatCompletionResponse {
    id: String, // "chatcmpl-{uuid}"
    object: &'static str, // "chat.completion"
    created: u64, // Unix seconds
    model: String, // echoes the request model
    choices: Vec<NonStreamChoice>,
    usage: CompletionUsage,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct NonStreamChoice {
    index: u32,
    message: AssistantMessage,
    finish_reason: String, // "stop"
    pub logprobs: Option<()>, // always null; placeholder to keep the field
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct AssistantMessage {
    role: &'static str, // "assistant"
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct ResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ResponseFunctionCall,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct ResponseFunctionCall {
    name: String,
    arguments: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct CompletionUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// ── Error envelope ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct ErrorResponse {
    pub(crate) error: ErrorDetail,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub(crate) struct ErrorDetail {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) error_type: String,
    pub(crate) code: Option<String>,
    pub(crate) param: Option<String>, // the rejected field name; message-level rejections use "messages"
    pub(crate) status: u16, // HTTP status redundantly carried in the body (OpenAI-compatible)
}

#[allow(dead_code)]
pub(crate) fn error_response(
    status: StatusCode,
    error_type: &str,
    message: &str,
    code: Option<&str>,
    param: Option<&str>,
) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: code.map(String::from),
                param: param.map(String::from),
                status: status.as_u16(),
            },
        }),
    )
        .into_response()
}

// ── Request validation ──────────────────────────────────────────────────────

/// Reject the 23 request-level fields ZeroClaw does not support, each with a
/// precise `param` + `message`. 14 explicit `if`s for generation-settings
/// fields + 9 array-loop over behavior-control fields.
///
/// A field is rejected only when it is present (`Some`), whatever its value —
/// "explicit 400 instead of silent ignore". `metadata`/`store` are benign
/// annotations and intentionally skipped here.
#[allow(dead_code, clippy::result_large_err)]
fn validate_unsupported_params(req: &ChatCompletionRequest) -> Result<(), Response> {
    // 5.1 — generation-settings fields, each with a distinct message.
    if req.max_tokens.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens is not supported per-request; configure in provider settings",
            None,
            Some("max_tokens"),
        ));
    }
    if req.top_p.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_p is not supported per-request",
            None,
            Some("top_p"),
        ));
    }
    if req.stop.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "stop is not supported per-request",
            None,
            Some("stop"),
        ));
    }
    if req.presence_penalty.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "presence_penalty is not supported per-request",
            None,
            Some("presence_penalty"),
        ));
    }
    if req.frequency_penalty.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "frequency_penalty is not supported per-request",
            None,
            Some("frequency_penalty"),
        ));
    }
    if req.n.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "n is not supported; single completion per request",
            None,
            Some("n"),
        ));
    }
    if req.response_format.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "response_format is not supported; configure output format in provider settings",
            None,
            Some("response_format"),
        ));
    }
    if req.seed.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "seed is not supported; configure in provider settings",
            None,
            Some("seed"),
        ));
    }
    if req.logprobs.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logprobs is not supported",
            None,
            Some("logprobs"),
        ));
    }
    if req.top_logprobs.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_logprobs is not supported",
            None,
            Some("top_logprobs"),
        ));
    }
    if req.user.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "user is not supported",
            None,
            Some("user"),
        ));
    }
    if req.logit_bias.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logit_bias is not supported",
            None,
            Some("logit_bias"),
        ));
    }
    if req.max_completion_tokens.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_completion_tokens is not supported; use provider settings",
            None,
            Some("max_completion_tokens"),
        ));
    }
    // Omission keeps the routed agent's configured temperature; explicit
    // per-request temperature is rejected.
    if req.temperature.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "temperature is not supported per-request; set `temperature` on the routed agent's provider model",
            None,
            Some("temperature"),
        ));
    }

    // 5.2 — behavior-control fields, short messages, array-able.
    for (present, param, message) in [
        (
            req.parallel_tool_calls.is_some(),
            "parallel_tool_calls",
            "parallel_tool_calls is not supported; tools executed transparently",
        ),
        (
            req.service_tier.is_some(),
            "service_tier",
            "service_tier is not applicable; routing is ZeroClaw config",
        ),
        (
            req.functions.is_some(),
            "functions",
            "legacy function-calling is not supported; use `tools`",
        ),
        (
            req.function_call.is_some(),
            "function_call",
            "legacy function_call is not supported; use `tool_choice`",
        ),
        (
            req.reasoning_effort.is_some(),
            "reasoning_effort",
            "reasoning_effort is not supported per-request; configure model in provider settings",
        ),
        (req.modalities.is_some(), "modalities", "only text output supported"),
        (req.audio.is_some(), "audio", "audio output not supported"),
        (req.prediction.is_some(), "prediction", "predicted outputs not supported"),
        (
            req.web_search_options.is_some(),
            "web_search_options",
            "web search not supported per-request",
        ),
    ] {
        if present {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
                None,
                Some(param),
            ));
        }
    }
    Ok(())
}

/// Validate the message list: `messages` must be non-empty, and each message
/// is checked against 4 message-level rejections — `name`, `tool_call_id`,
/// role allow-list (`system`/`developer`/`user`/`assistant`, with
/// `tool`/`function` explicitly 400), and `tool_calls`. Message-level errors
/// all use `param = "messages"` with the fine-grained index in the message
/// text.
///
/// `tools`/`tool_choice` deep validation is handled in a later step and is
/// not done here.
#[allow(dead_code, clippy::result_large_err)]
fn validate_request(req: &ChatCompletionRequest) -> Result<(), Response> {
    if req.messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
            None,
            Some("messages"),
        ));
    }
    for (i, msg) in req.messages.iter().enumerate() {
        // ① name: not propagated under transparent execution.
        if msg.name.is_some() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!(
                    "messages[{i}].name is not supported; tool results are transparently executed"
                ),
                None,
                Some("messages"),
            ));
        }
        // ② tool_call_id: same rationale.
        if msg.tool_call_id.is_some() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("messages[{i}].tool_call_id is not supported; tool execution is transparent"),
                None,
                Some("messages"),
            ));
        }
        // ③ role allow-list (4); tool/function roles are explicitly rejected
        // (RFC line 36), not silently folded into prompt text.
        if !matches!(msg.role.as_str(), "system" | "developer" | "user" | "assistant") {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!(
                    "messages[{i}].role '{}' is not supported; allowed: system, developer, user, \
                     assistant (tool/function roles are transparently executed and rejected)",
                    msg.role
                ),
                None,
                Some("messages"),
            ));
        }
        // ④ tool_calls: meaningless under transparent execution.
        if msg.tool_calls.is_some() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("messages[{i}].tool_calls is not supported; tool execution is transparent"),
                None,
                Some("messages"),
            ));
        }
    }
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use serde_json::json;

    fn parse_request(v: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(v).expect("request must deserialize")
    }

    /// Run both validators; the first rejection wins (unsupported params
    /// checked before message-level validation).
    #[allow(clippy::result_large_err)]
    fn run_validators(req: &ChatCompletionRequest) -> Result<(), Response> {
        validate_unsupported_params(req).and_then(|_| validate_request(req))
    }

    async fn response_json(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    async fn assert_rejected(req: serde_json::Value, expected_param: &str) {
        let r = parse_request(req);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], expected_param);
    }

    fn base_request() -> serde_json::Value {
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        })
    }

    #[tokio::test]
    async fn rejects_max_tokens() {
        let mut v = base_request();
        v["max_tokens"] = json!(1);
        assert_rejected(v, "max_tokens").await;
    }

    #[tokio::test]
    async fn rejects_top_p() {
        let mut v = base_request();
        v["top_p"] = json!(0.5);
        assert_rejected(v, "top_p").await;
    }

    #[tokio::test]
    async fn rejects_stop() {
        let mut v = base_request();
        v["stop"] = json!("END");
        assert_rejected(v, "stop").await;
    }

    #[tokio::test]
    async fn rejects_seed() {
        let mut v = base_request();
        v["seed"] = json!(42);
        assert_rejected(v, "seed").await;
    }

    #[tokio::test]
    async fn rejects_logprobs() {
        let mut v = base_request();
        v["logprobs"] = json!(true);
        assert_rejected(v, "logprobs").await;
    }

    #[tokio::test]
    async fn rejects_all_23_unsupported() {
        let cases: &[(&str, serde_json::Value)] = &[
            ("max_tokens", json!(1)),
            ("top_p", json!(0.5)),
            ("stop", json!("END")),
            ("presence_penalty", json!(0.0)),
            ("frequency_penalty", json!(0.0)),
            ("n", json!(1)),
            ("response_format", json!({"type": "text"})),
            ("seed", json!(42)),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(5)),
            ("user", json!("u-1")),
            ("logit_bias", json!({})),
            ("max_completion_tokens", json!(100)),
            ("temperature", json!(0.7)),
            ("parallel_tool_calls", json!(true)),
            ("service_tier", json!("auto")),
            ("functions", json!([{"name": "f"}])),
            ("function_call", json!("auto")),
            ("reasoning_effort", json!("medium")),
            ("modalities", json!(["text"])),
            ("audio", json!({})),
            ("prediction", json!({})),
            ("web_search_options", json!({})),
        ];
        assert_eq!(cases.len(), 23);
        for (param, value) in cases {
            let mut v = base_request();
            v[param] = value.clone();
            assert_rejected(v, param).await;
        }
    }

    #[tokio::test]
    async fn accepts_none_unsupported() {
        let r = parse_request(base_request());
        assert!(run_validators(&r).is_ok());
    }

    #[tokio::test]
    async fn rejects_explicit_temperature() {
        let mut v = base_request();
        v["temperature"] = json!(0.7);
        assert_rejected(v, "temperature").await;
    }

    #[tokio::test]
    async fn omits_temperature_uses_agent_config() {
        // No temperature in the request -> passes; the routed agent's
        // configured temperature is used (actual value verified by the handler).
        let r = parse_request(base_request());
        assert!(run_validators(&r).is_ok());
    }

    #[tokio::test]
    async fn rejects_n_eq_1() {
        // Strict: any `n`, including n=1, is rejected.
        let mut v = base_request();
        v["n"] = json!(1);
        assert_rejected(v, "n").await;
    }

    #[tokio::test]
    async fn tolerates_metadata_and_store() {
        let mut v = base_request();
        v["metadata"] = json!({"trace_id": "abc", "user_tags": ["x"]});
        v["store"] = json!(true);
        let r = parse_request(v);
        assert!(run_validators(&r).is_ok());
    }

    #[tokio::test]
    async fn rejects_unknown_role() {
        let v = json!({
            "model": "gpt-4o",
            "messages": [{"role": "admin", "content": "hi"}],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "messages");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].role"));
        assert!(msg.contains("admin"));
    }

    #[tokio::test]
    async fn rejects_tool_role() {
        // Rejected at the role check itself, not indirectly via tool_call_id.
        let v = json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": "result"}],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "messages");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].role"));
        assert!(msg.contains("tool"));
    }

    #[tokio::test]
    async fn rejects_function_role() {
        let v = json!({
            "model": "gpt-4o",
            "messages": [{"role": "function", "content": "legacy"}],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "messages");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].role"));
        assert!(msg.contains("function"));
    }

    #[tokio::test]
    async fn rejects_tool_calls_in_history() {
        let v = json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "assistant",
                "content": "ok",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            }],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "messages");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].tool_calls"));
    }

    #[tokio::test]
    async fn rejects_name_and_tool_call_id() {
        // name alone.
        let v = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (_, body) = response_json(err).await;
        assert_eq!(body["error"]["param"], "messages");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].name"));

        // tool_call_id alone.
        let v2 = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi", "tool_call_id": "call_1"}],
        });
        let r2 = parse_request(v2);
        let err2 = run_validators(&r2).unwrap_err();
        let (_, body2) = response_json(err2).await;
        assert_eq!(body2["error"]["param"], "messages");
        let msg2 = body2["error"]["message"].as_str().unwrap();
        assert!(msg2.contains("messages[0].tool_call_id"));
    }

    #[tokio::test]
    async fn rejects_empty_messages() {
        let v = json!({"model": "gpt-4o", "messages": []});
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "messages");
        assert_eq!(body["error"]["message"], "messages must not be empty");
    }

    #[tokio::test]
    async fn error_envelope_shape() {
        let mut v = base_request();
        v["max_tokens"] = json!(1);
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (status, body) = response_json(err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].is_string());
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["code"].is_null());
        assert_eq!(body["error"]["param"], "max_tokens");
        assert_eq!(body["error"]["status"], 400);
    }

    #[tokio::test]
    async fn null_unsupported_fields_treated_as_unset() {
        // `null` folds into `None` for Option fields (indistinguishable from
        // omission), matching OpenAI's "null means unset" convention — so none
        // of the 23 rejected fields trips validation when passed as null.
        let fields = [
            "max_tokens", "top_p", "stop", "presence_penalty",
            "frequency_penalty", "n", "response_format", "seed", "logprobs",
            "top_logprobs", "user", "logit_bias", "max_completion_tokens",
            "temperature", "parallel_tool_calls", "service_tier", "functions",
            "function_call", "reasoning_effort", "modalities", "audio",
            "prediction", "web_search_options",
        ];
        assert_eq!(fields.len(), 23);
        for param in fields {
            let mut v = base_request();
            v[param] = serde_json::Value::Null;
            let r = parse_request(v);
            assert!(
                run_validators(&r).is_ok(),
                "null for `{param}` should be treated as unset"
            );
        }
    }

    #[tokio::test]
    async fn request_level_checked_before_message_level() {
        // Request-level rejections run first; a message-level violation is not
        // reached when a request-level field is also present.
        let mut v = base_request();
        v["max_tokens"] = json!(1);
        v["messages"][0]["role"] = json!("admin");
        assert_rejected(v, "max_tokens").await;
    }

    #[tokio::test]
    async fn max_tokens_beats_temperature() {
        // Both present: the 14 explicit checks run in field order, with
        // `temperature` last — so max_tokens wins.
        let mut v = base_request();
        v["max_tokens"] = json!(1);
        v["temperature"] = json!(0.7);
        assert_rejected(v, "max_tokens").await;
    }

    #[tokio::test]
    async fn second_message_index_reported() {
        // The index in the message text is the 0-based position of the
        // offending message, not always 0.
        let v = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "admin", "content": "second"},
            ],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (_, body) = response_json(err).await;
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[1].role"));
    }

    #[tokio::test]
    async fn first_message_violation_wins_within_message() {
        // Within one message the checks run name → tool_call_id → role →
        // tool_calls, so `name` wins when both name and role are invalid.
        let v = json!({
            "model": "gpt-4o",
            "messages": [{"role": "admin", "content": "hi", "name": "alice"}],
        });
        let r = parse_request(v);
        let err = run_validators(&r).unwrap_err();
        let (_, body) = response_json(err).await;
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("messages[0].name"));
        assert!(!msg.contains("messages[0].role"));
    }

    #[test]
    fn deserialization_defaults_and_tolerance() {
        // Missing model -> "", missing stream -> false, missing Options -> None,
        // unknown fields (e.g. a typo'd known field) silently ignored — no
        // deny_unknown_fields.
        let r: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "unknown_field": "ignored",
            "max_tokenss": 5,
        }))
        .unwrap();
        assert_eq!(r.model, "");
        assert!(!r.stream);
        assert_eq!(r.messages.len(), 1);
        assert_eq!(r.messages[0].role, "user");
        assert_eq!(r.messages[0].content.as_deref(), Some("hi"));
        assert!(r.messages[0].name.is_none());
        assert!(r.max_tokens.is_none());
        assert!(r.temperature.is_none());
        assert!(r.metadata.is_none());
        assert!(r.store.is_none());
    }

    #[test]
    fn deserialization_full_openai_payload() {
        let r: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "stream": true,
            "stream_options": {"include_usage": true},
            "metadata": {"k": "v"},
            "store": false,
        }))
        .unwrap();
        assert_eq!(r.model, "gpt-4o");
        assert!(r.stream);
        let so = r.stream_options.expect("stream_options present");
        assert!(so.include_usage);
        assert_eq!(r.metadata.as_ref().unwrap()["k"], "v");
        // store is Option<bool>: explicit false is distinct from unset (None).
        assert_eq!(r.store, Some(false));
    }
}
