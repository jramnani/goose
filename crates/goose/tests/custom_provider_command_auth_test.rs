//! End-to-end coverage for command-based auth on custom providers (issue #11329):
//! a `DeclarativeProviderConfig.auth` command is run to fetch the credential
//! instead of a static `api_key_env`, and the credential is refreshed
//! reactively when the upstream API returns a 401.

use goose::conversation::message::Message;
use goose::providers::base::Provider;
use goose::providers::{anthropic_def, openai_def};
use goose_providers::declarative::{AuthConfig, DeclarativeProviderConfig, ProviderEngine};
use goose_providers::model::ModelConfig;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn custom_config_with_auth(base_url: &str, auth: AuthConfig) -> DeclarativeProviderConfig {
    DeclarativeProviderConfig {
        name: "custom_command_auth".to_string(),
        engine: ProviderEngine::OpenAI,
        display_name: "Custom Command Auth".to_string(),
        description: None,
        api_key_env: String::new(),
        base_url: base_url.to_string(),
        models: vec![goose_providers::base::ModelInfo::new("test-model")],
        headers: None,
        session_id_header_override: None,
        timeout_seconds: None,
        supports_streaming: Some(true),
        requires_auth: true,
        catalog_provider_id: None,
        base_path: None,
        env_vars: None,
        auth: Some(auth),
        dynamic_models: Some(false),
        skip_canonical_filtering: false,
        model_doc_link: None,
        setup_steps: vec![],
        toolshim: false,
        preserves_thinking: false,
        emit_clear_thinking: false,
        setup: None,
    }
}

/// A command whose stdout differs on every invocation (this process's own
/// PID / a fresh pseudo-random draw), so two captured values back-to-back
/// prove whether the underlying command was actually re-run.
#[cfg(unix)]
fn distinct_value_command() -> (&'static str, Vec<&'static str>) {
    ("sh", vec!["-c", "echo $$"])
}
#[cfg(windows)]
fn distinct_value_command() -> (&'static str, Vec<&'static str>) {
    ("cmd", vec!["/C", "echo %RANDOM%%RANDOM%%RANDOM%"])
}

fn chat_completions_sse() -> String {
    format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        json!({
            "choices": [{
                "delta": {"content": "hi", "role": "assistant"},
                "index": 0
            }],
            "created": 1755133833,
            "id": "chatcmpl-test",
            "model": "test-model"
        }),
        json!({
            "choices": [],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}
        })
    )
}

async fn complete(provider: &dyn Provider) -> anyhow::Result<()> {
    let message = Message::user().with_text("hello");
    let model_config = ModelConfig::new("test-model");
    provider
        .complete(&model_config, "system", &[message], &[])
        .await?;
    Ok(())
}

#[tokio::test]
async fn command_auth_credential_is_sent_as_bearer_token() {
    let server = MockServer::start().await;
    let captured_auth: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = captured_auth.clone();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let auth = req
                .headers
                .get("authorization")
                .map(|v| v.to_str().unwrap().to_string())
                .unwrap_or_default();
            capture.lock().unwrap().push(auth);
            ResponseTemplate::new(200)
                .set_body_string(chat_completions_sse())
                .insert_header("content-type", "text/event-stream")
        })
        .mount(&server)
        .await;

    let config = custom_config_with_auth(
        &server.uri(),
        AuthConfig {
            command: "echo".to_string(),
            args: vec!["test-token-123".to_string()],
            refresh_interval: 3600,
            timeout_seconds: None,
            cwd: None,
            header_name: None,
            header_value_prefix: None,
        },
    );
    let provider = openai_def::from_custom_config(config, None).unwrap();

    complete(&provider).await.unwrap();

    assert_eq!(
        captured_auth.lock().unwrap().as_slice(),
        ["Bearer test-token-123".to_string()]
    );
}

#[tokio::test]
async fn command_auth_refreshes_credential_on_401_and_retries() {
    let server = MockServer::start().await;
    let captured_auth: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = captured_auth.clone();
    let call_count = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let auth = req
                .headers
                .get("authorization")
                .map(|v| v.to_str().unwrap().to_string())
                .unwrap_or_default();
            capture.lock().unwrap().push(auth);

            if call_count.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(401)
                    .set_body_json(json!({"error": {"message": "token expired"}}))
            } else {
                ResponseTemplate::new(200)
                    .set_body_string(chat_completions_sse())
                    .insert_header("content-type", "text/event-stream")
            }
        })
        .mount(&server)
        .await;

    // A value that differs on every invocation (this process's own PID / a
    // fresh pseudo-random draw), so two distinct captured Authorization
    // headers prove the auth command actually ran twice (once up front, once
    // after `refresh_credentials` invalidated the cache on the 401), rather
    // than the same token being reused.
    let (command, args) = distinct_value_command();
    let config = custom_config_with_auth(
        &server.uri(),
        AuthConfig {
            command: command.to_string(),
            args: args.into_iter().map(str::to_string).collect(),
            refresh_interval: 3600,
            timeout_seconds: None,
            cwd: None,
            header_name: None,
            header_value_prefix: None,
        },
    );
    let provider = openai_def::from_custom_config(config, None).unwrap();

    complete(&provider)
        .await
        .expect("request should succeed after refreshing credentials and retrying");

    let captured = captured_auth.lock().unwrap();
    assert_eq!(captured.len(), 2, "expected an initial request and a retry");
    assert_ne!(
        captured[0], captured[1],
        "the retried request should carry a freshly-fetched credential, not the stale one"
    );
}

fn anthropic_messages_sse() -> String {
    [
        r#"data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"test-model","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        r#"data: {"type":"content_block_stop","index":0}"#,
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
        r#"data: {"type":"message_stop"}"#,
    ]
    .join("\n\n")
        + "\n\n"
}

#[derive(Debug, PartialEq)]
struct CapturedCredentialHeaders {
    x_api_key: Option<String>,
    authorization: Option<String>,
}

/// Mounts a `/v1/messages` endpoint and returns the credential headers of
/// each request it receives.
async fn mount_anthropic_messages_endpoint(
    server: &MockServer,
) -> Arc<Mutex<Vec<CapturedCredentialHeaders>>> {
    let captured_headers: Arc<Mutex<Vec<CapturedCredentialHeaders>>> =
        Arc::new(Mutex::new(Vec::new()));
    let capture = captured_headers.clone();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &Request| {
            let header_value = |name: &str| {
                req.headers
                    .get(name)
                    .map(|value| value.to_str().unwrap().to_string())
            };
            capture.lock().unwrap().push(CapturedCredentialHeaders {
                x_api_key: header_value("x-api-key"),
                authorization: header_value("authorization"),
            });
            ResponseTemplate::new(200)
                .set_body_string(anthropic_messages_sse())
                .insert_header("content-type", "text/event-stream")
        })
        .mount(server)
        .await;

    captured_headers
}

fn anthropic_config_with_auth(base_url: &str, auth: AuthConfig) -> DeclarativeProviderConfig {
    DeclarativeProviderConfig {
        engine: ProviderEngine::Anthropic,
        ..custom_config_with_auth(base_url, auth)
    }
}

fn echo_auth_config(header_name: Option<&str>, header_value_prefix: Option<&str>) -> AuthConfig {
    AuthConfig {
        command: "echo".to_string(),
        args: vec!["test-token-123".to_string()],
        refresh_interval: 3600,
        timeout_seconds: None,
        cwd: None,
        header_name: header_name.map(str::to_string),
        header_value_prefix: header_value_prefix.map(str::to_string),
    }
}

#[tokio::test]
async fn anthropic_command_auth_sends_x_api_key_by_default() {
    let server = MockServer::start().await;
    let captured_headers = mount_anthropic_messages_endpoint(&server).await;
    let config = anthropic_config_with_auth(&server.uri(), echo_auth_config(None, None));
    let provider = anthropic_def::from_custom_config(config, None).unwrap();

    complete(&provider).await.unwrap();

    assert_eq!(
        captured_headers.lock().unwrap().as_slice(),
        [CapturedCredentialHeaders {
            x_api_key: Some("test-token-123".to_string()),
            authorization: None,
        }]
    );
}

#[tokio::test]
async fn anthropic_command_auth_honors_configured_bearer_header() {
    let server = MockServer::start().await;
    let captured_headers = mount_anthropic_messages_endpoint(&server).await;
    let config = anthropic_config_with_auth(
        &server.uri(),
        echo_auth_config(Some("Authorization"), Some("Bearer ")),
    );
    let provider = anthropic_def::from_custom_config(config, None).unwrap();

    complete(&provider).await.unwrap();

    assert_eq!(
        captured_headers.lock().unwrap().as_slice(),
        [CapturedCredentialHeaders {
            x_api_key: None,
            authorization: Some("Bearer test-token-123".to_string()),
        }]
    );
}
