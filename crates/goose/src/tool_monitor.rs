use crate::config::GooseMode;
use crate::conversation::message::{Message, MessageContent, ToolRequest};
use crate::tool_inspection::{InspectionAction, InspectionResult, ToolInspector};
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
struct InternalToolCall {
    name: String,
    parameters: Value,
}

impl InternalToolCall {
    fn matches(&self, other: &InternalToolCall) -> bool {
        self.name == other.name && self.parameters == other.parameters
    }

    fn from_tool_call(tool_call: &CallToolRequestParams) -> Self {
        let name = tool_call.name.to_string();
        let parameters = tool_call
            .arguments
            .as_ref()
            .map(|obj| Value::Object(obj.clone()))
            .unwrap_or(Value::Null);
        Self { name, parameters }
    }
}

/// Consecutive malformed (unparseable) tool calls tolerated before denying.
/// Malformed tool_use blocks are a model-layer failure that few-shot-poisons
/// the context, so the model repeats the malformation in an unrecoverable loop
/// (anthropics/claude-code#63604). A small cap allows transient parse errors
/// while breaking a sustained run so the model can respond in text.
const MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS: u32 = 3;

#[derive(Debug)]
pub struct RepetitionInspector {
    max_repetitions: Option<u32>,
    last_call: Option<InternalToolCall>,
    repeat_count: u32,
    call_counts: HashMap<String, u32>,
}

impl RepetitionInspector {
    pub fn new(max_repetitions: Option<u32>) -> Self {
        Self {
            max_repetitions,
            last_call: None,
            repeat_count: 0,
            call_counts: HashMap::new(),
        }
    }

    pub fn check_tool_call(&mut self, tool_call: CallToolRequestParams) -> bool {
        let internal_call = InternalToolCall::from_tool_call(&tool_call);
        let total_calls = self
            .call_counts
            .entry(internal_call.name.clone())
            .or_insert(0);
        *total_calls += 1;

        if self.max_repetitions.is_none() {
            self.last_call = Some(internal_call);
            self.repeat_count = 1;
            return true;
        }

        if let Some(last) = &self.last_call {
            if last.matches(&internal_call) {
                self.repeat_count += 1;
                if self.repeat_count > self.max_repetitions.unwrap() {
                    return false;
                }
            } else {
                self.repeat_count = 1;
            }
        } else {
            self.repeat_count = 1;
        }

        self.last_call = Some(internal_call);
        true
    }

    pub fn reset(&mut self) {
        self.last_call = None;
        self.repeat_count = 0;
        self.call_counts.clear();
    }

    /// Counts how many times `target` appears within the trailing `window`
    /// tool calls of history. A windowed count catches *alternating* loops
    /// (A → B → A → B …, aaif-goose/goose#9640) that the consecutive-only
    /// `check_tool_call` misses, since its counter resets on any differing call.
    fn count_in_recent_window(
        &self,
        target: &InternalToolCall,
        history_calls: &[InternalToolCall],
        window: usize,
    ) -> u32 {
        let start = history_calls.len().saturating_sub(window);
        history_calls[start..]
            .iter()
            .filter(|call| call.matches(target))
            .count() as u32
    }

    /// Pulls the parsed tool calls out of the conversation history, oldest to
    /// newest, so `inspect` can scan a recent window for repetition patterns.
    fn collect_history_tool_calls(messages: &[Message]) -> Vec<InternalToolCall> {
        messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|content| match content {
                MessageContent::ToolRequest(tool_request) => tool_request
                    .tool_call
                    .as_ref()
                    .ok()
                    .map(InternalToolCall::from_tool_call),
                _ => None,
            })
            .collect()
    }

    /// Counts the trailing run of malformed (`Err`) tool calls in history. Only
    /// the trailing run counts: a since-recovered parse error must not keep
    /// tripping the guard — we care about an ongoing malformed loop.
    fn count_trailing_malformed_tool_calls(messages: &[Message]) -> u32 {
        let mut trailing_malformed = 0u32;
        'outer: for message in messages.iter().rev() {
            for content in message.content.iter().rev() {
                match content {
                    MessageContent::ToolRequest(tool_request) => {
                        if tool_request.tool_call.is_err() {
                            trailing_malformed += 1;
                        } else {
                            // A well-formed call ends the trailing run.
                            break 'outer;
                        }
                    }
                    // Non-tool content (text/thinking) is ignored; it neither
                    // extends nor breaks the malformed run.
                    _ => {}
                }
            }
        }
        trailing_malformed
    }
}

#[async_trait]
impl ToolInspector for RepetitionInspector {
    fn name(&self) -> &'static str {
        "repetition"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn inspect(
        &self,
        _session_id: &str,
        tool_requests: &[ToolRequest],
        messages: &[Message],
        _goose_mode: GooseMode,
    ) -> Result<Vec<InspectionResult>> {
        let mut results = Vec::new();

        // Disabled when no limit is configured: behave as a no-op.
        let max_repetitions = match self.max_repetitions {
            Some(limit) => limit,
            None => return Ok(results),
        };

        // Guard 1 — malformed-tool-call loop (anthropics/claude-code#63604).
        let trailing_malformed = Self::count_trailing_malformed_tool_calls(messages);
        if trailing_malformed >= MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS {
            for tool_request in tool_requests {
                if tool_request.tool_call.is_err() {
                    results.push(InspectionResult {
                        tool_request_id: tool_request.id.clone(),
                        action: InspectionAction::Deny,
                        reason: format!(
                            "Model emitted {} malformed tool calls in a row; stopping the retry \
                             loop so it can respond in text.",
                            trailing_malformed
                        ),
                        confidence: 1.0,
                        inspector_name: "repetition".to_string(),
                        finding_id: Some("REP-002".to_string()),
                    });
                }
            }
            if !results.is_empty() {
                return Ok(results);
            }
        }

        // Guard 2 — windowed repetition, catching consecutive and alternating
        // loops. Window of 2 * max_repetitions keeps the scan bounded.
        let window = (max_repetitions as usize).saturating_mul(2);
        let history_calls = Self::collect_history_tool_calls(messages);

        for tool_request in tool_requests {
            if let Ok(tool_call) = &tool_request.tool_call {
                let candidate = InternalToolCall::from_tool_call(tool_call);
                // +1 for the pending call itself.
                let occurrences =
                    self.count_in_recent_window(&candidate, &history_calls, window) + 1;

                if occurrences > max_repetitions {
                    results.push(InspectionResult {
                        tool_request_id: tool_request.id.clone(),
                        action: InspectionAction::Deny,
                        reason: format!(
                            "Tool '{}' called with identical arguments {} times in the last {} \
                             tool calls; denying to break the repetition loop.",
                            tool_call.name, occurrences, window
                        ),
                        confidence: 1.0,
                        inspector_name: "repetition".to_string(),
                        finding_id: Some("REP-001".to_string()),
                    });
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::{Message, MessageContent, ToolRequest};
    use rmcp::model::{CallToolRequestParams, ErrorCode, ErrorData};
    use rmcp::object;

    /// Builds a well-formed tool call against the `developer__text_editor`
    /// tool. The `marker` is embedded in the `path` argument so two calls with
    /// different markers have genuinely different (name + arguments) signatures
    /// while sharing the same tool name.
    fn named_call(tool: &str, marker: &str) -> CallToolRequestParams {
        CallToolRequestParams::new(tool.to_string()).with_arguments(object!({
            "command": "str_replace",
            "path": marker,
        }))
    }

    /// Builds a `ToolRequest` carrying a well-formed `developer__text_editor`
    /// edit against `target_path` — the exact shape of the runaway edit/write
    /// loop this inspector exists to stop.
    fn make_edit_request(request_id: &str, target_path: &str) -> ToolRequest {
        ToolRequest {
            id: request_id.to_string(),
            tool_call: Ok(named_call("developer__text_editor", target_path)),
            metadata: None,
            tool_meta: None,
        }
    }

    /// Builds a *malformed* pending `ToolRequest`: one whose `tool_call` is the
    /// `Err` arm because the provider could not parse it into a
    /// `CallToolRequestParams`.
    fn make_malformed_request(request_id: &str) -> ToolRequest {
        ToolRequest {
            id: request_id.to_string(),
            tool_call: Err(ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: std::borrow::Cow::from("malformed"),
                data: None,
            }),
            metadata: None,
            tool_meta: None,
        }
    }

    /// Wraps a sequence of well-formed tool calls into one assistant `Message`
    /// per call, oldest-to-newest, matching how `inspect` scans history.
    fn history_from_calls(calls: &[CallToolRequestParams]) -> Vec<Message> {
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                Message::assistant().with_tool_request(format!("hist-{index}"), Ok(call.clone()))
            })
            .collect()
    }

    /// Builds a single assistant `Message` carrying one *malformed* tool call,
    /// used to construct a trailing run of unparseable calls in history.
    fn malformed_history_message(request_id: &str) -> Message {
        Message::assistant().with_content(MessageContent::ToolRequest(ToolRequest {
            id: request_id.to_string(),
            tool_call: Err(ErrorData {
                code: ErrorCode::INVALID_PARAMS,
                message: std::borrow::Cow::from("malformed"),
                data: None,
            }),
            metadata: None,
            tool_meta: None,
        }))
    }

    /// When no limit is configured the inspector is a complete no-op: even an
    /// enormous history of identical calls plus an identical pending request
    /// must produce zero findings.
    #[tokio::test]
    async fn disabled_inspector_allows_unlimited_identical_calls() {
        let inspector = RepetitionInspector::new(None);
        let identical_call = named_call("developer__text_editor", "/src/Foo.kt");
        let history = history_from_calls(&vec![identical_call.clone(); 50]);
        let pending_request = make_edit_request("req-pending", "/src/Foo.kt");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert!(
            results.is_empty(),
            "a disabled inspector must produce no findings, even for 50 identical calls"
        );
    }

    /// A straight run of identical calls (A, A, A in history) plus one more
    /// identical pending call exceeds the limit of 3 and must be denied with a
    /// REP-001 finding.
    #[tokio::test]
    async fn consecutive_identical_calls_are_denied_at_the_limit() {
        let inspector = RepetitionInspector::new(Some(3));
        let edit_call = named_call("developer__text_editor", "/src/Foo.kt");
        let history = history_from_calls(&[
            edit_call.clone(),
            edit_call.clone(),
            edit_call.clone(),
        ]);
        let pending_request = make_edit_request("req-pending", "/src/Foo.kt");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert_eq!(results.len(), 1, "exactly one finding expected for the looping call");
        assert_eq!(results[0].action, InspectionAction::Deny);
        assert_eq!(results[0].inspector_name, "repetition");
        assert_eq!(results[0].finding_id.as_deref(), Some("REP-001"));
        assert_eq!(results[0].tool_request_id, "req-pending");
    }

    /// An alternating two-tool loop (A, B, A, B, A, B in history) is a loop the
    /// consecutive-only counter would miss, because no two adjacent calls are
    /// identical. The windowed guard must still deny the next A with REP-001.
    /// See aaif-goose/goose#9640.
    #[tokio::test]
    async fn alternating_two_tool_loop_is_denied() {
        let inspector = RepetitionInspector::new(Some(3));
        // Same tool name, different arguments, so signatures differ.
        let call_a = named_call("developer__text_editor", "schema.sql");
        let call_b = named_call("developer__text_editor", ".env");
        let history = history_from_calls(&[
            call_a.clone(),
            call_b.clone(),
            call_a.clone(),
            call_b.clone(),
            call_a.clone(),
            call_b.clone(),
        ]);
        let pending_request = make_edit_request("req-pending", "schema.sql");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert_eq!(results.len(), 1, "the alternating A/B loop must produce one finding");
        assert_eq!(results[0].action, InspectionAction::Deny);
        assert_eq!(results[0].inspector_name, "repetition");
        assert_eq!(results[0].finding_id.as_deref(), Some("REP-001"));
        assert_eq!(results[0].tool_request_id, "req-pending");
    }

    /// Calls to the same tool with *different* arguments are legitimate work,
    /// not a loop. Four distinct markers in history plus a fifth, new marker
    /// pending must produce no findings.
    #[tokio::test]
    async fn varied_arguments_on_the_same_tool_are_allowed() {
        let inspector = RepetitionInspector::new(Some(3));
        let history = history_from_calls(&[
            named_call("developer__text_editor", "/src/A.kt"),
            named_call("developer__text_editor", "/src/B.kt"),
            named_call("developer__text_editor", "/src/C.kt"),
            named_call("developer__text_editor", "/src/D.kt"),
        ]);
        let pending_request = make_edit_request("req-pending", "/src/E.kt");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert!(
            results.is_empty(),
            "varied arguments on the same tool must not be treated as a repetition loop"
        );
    }

    /// A sustained trailing run of malformed (unparseable) tool calls is a
    /// model-layer failure loop. Three malformed history messages plus a
    /// malformed pending request must be denied with REP-002.
    /// See anthropics/claude-code#63604.
    #[tokio::test]
    async fn trailing_malformed_tool_calls_are_denied() {
        let inspector = RepetitionInspector::new(Some(3));
        let history = vec![
            malformed_history_message("hist-0"),
            malformed_history_message("hist-1"),
            malformed_history_message("hist-2"),
        ];
        let pending_request = make_malformed_request("req-malformed");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert_eq!(results.len(), 1, "a sustained malformed run must produce one finding");
        assert_eq!(results[0].action, InspectionAction::Deny);
        assert_eq!(results[0].inspector_name, "repetition");
        assert_eq!(results[0].finding_id.as_deref(), Some("REP-002"));
        assert_eq!(results[0].tool_request_id, "req-malformed");
    }

    /// A single, isolated malformed call is a transient parse error, not a
    /// loop. One malformed history message plus a malformed pending request is
    /// below the threshold and must be tolerated (no findings).
    #[tokio::test]
    async fn an_isolated_malformed_call_is_tolerated() {
        let inspector = RepetitionInspector::new(Some(3));
        let history = vec![malformed_history_message("hist-0")];
        let pending_request = make_malformed_request("req-malformed");

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&pending_request),
                &history,
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert!(
            results.is_empty(),
            "a single transient malformed call must not be denied"
        );
    }

    /// The stateful `check_tool_call` tracks *consecutive* identical calls and
    /// returns `false` once the repeat count exceeds the limit: attempts 1..=3
    /// are allowed, the 4th is denied.
    #[test]
    fn check_tool_call_denies_consecutive_identical_calls() {
        let mut inspector = RepetitionInspector::new(Some(3));
        let repeated_call = named_call("developer__text_editor", "/src/Foo.kt");

        for attempt in 1..=3 {
            assert!(
                inspector.check_tool_call(repeated_call.clone()),
                "attempt {attempt} should be allowed (<= limit of 3)"
            );
        }

        assert!(
            !inspector.check_tool_call(repeated_call.clone()),
            "the 4th consecutive identical call must be denied (> limit of 3)"
        );
    }
}
