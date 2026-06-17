use crate::config::GooseMode;
use crate::conversation::message::{Message, ToolRequest};
use crate::tool_inspection::{InspectionAction, InspectionResult, ToolInspector};
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use serde_json::Value;
use std::collections::HashMap;

// Helper struct for internal tracking
#[derive(Debug, Clone)]
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
        _messages: &[Message],
        _goose_mode: GooseMode,
    ) -> Result<Vec<InspectionResult>> {
        let mut results = Vec::new();

        // Check repetition limits for each tool request
        for tool_request in tool_requests {
            if let Ok(tool_call) = &tool_request.tool_call {
                // Create a temporary clone to check without modifying state
                let mut temp_inspector = RepetitionInspector::new(self.max_repetitions);
                temp_inspector.last_call = self.last_call.clone();
                temp_inspector.repeat_count = self.repeat_count;
                temp_inspector.call_counts = self.call_counts.clone();

                if !temp_inspector.check_tool_call(tool_call.clone()) {
                    results.push(InspectionResult {
                        tool_request_id: tool_request.id.clone(),
                        action: InspectionAction::Deny,
                        reason: format!(
                            "Tool '{}' has exceeded maximum repetitions",
                            tool_call.name
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
    use crate::conversation::message::ToolRequest;
    use rmcp::model::CallToolRequestParams;
    use rmcp::object;

    /// Builds a `ToolRequest` whose tool_call is an identical-looking `edit`
    /// against the same file — the exact shape of the runaway edit/write loop
    /// this inspector exists to stop.
    fn make_edit_request(request_id: &str, target_path: &str) -> ToolRequest {
        ToolRequest {
            id: request_id.to_string(),
            tool_call: Ok(CallToolRequestParams::new("developer__text_editor")
                .with_arguments(object!({
                    "command": "str_replace",
                    "path": target_path,
                    "old_str": "fn foo() {}",
                    "new_str": "fn foo() { bar(); }"
                }))),
            metadata: None,
            tool_meta: None,
        }
    }

    /// When no limit is configured, the inspector must never deny — it should
    /// behave exactly as the pre-existing `RepetitionInspector::new(None)` did.
    #[tokio::test]
    async fn disabled_inspector_allows_unlimited_identical_calls() {
        let inspector = RepetitionInspector::new(None);
        let identical_request = make_edit_request("req-1", "/src/Foo.kt");

        for _attempt in 0..50 {
            let results = inspector
                .inspect(
                    "test-session",
                    std::slice::from_ref(&identical_request),
                    &[],
                    GooseMode::Auto,
                )
                .await
                .expect("inspection should not error");

            assert!(
                results.is_empty(),
                "a disabled inspector must produce no findings, even for repeated identical calls"
            );
        }
    }

    /// With a limit of 3, repeating the *same* edit must be denied once the
    /// consecutive count exceeds the limit. We drive the inspector's real
    /// counting state via `check_tool_call` (the same method `inspect` uses).
    #[test]
    fn identical_calls_are_denied_once_the_limit_is_exceeded() {
        let mut inspector = RepetitionInspector::new(Some(3));
        let repeated_call = CallToolRequestParams::new("developer__text_editor")
            .with_arguments(object!({
                "command": "str_replace",
                "path": "/src/Foo.kt",
                "old_str": "a",
                "new_str": "b"
            }));

        // Attempts 1..=3 are within the limit and must be allowed.
        for attempt in 1..=3 {
            assert!(
                inspector.check_tool_call(repeated_call.clone()),
                "attempt {attempt} should be allowed (<= limit of 3)"
            );
        }

        // The 4th identical attempt exceeds the limit and must be denied.
        assert!(
            !inspector.check_tool_call(repeated_call.clone()),
            "the 4th consecutive identical call must be denied (> limit of 3)"
        );
    }

    /// A different tool call between repeats resets the consecutive counter, so
    /// legitimate interleaved work is never falsely blocked.
    #[test]
    fn a_different_call_resets_the_consecutive_counter() {
        let mut inspector = RepetitionInspector::new(Some(2));
        let edit_call = CallToolRequestParams::new("developer__text_editor")
            .with_arguments(object!({"command": "str_replace", "path": "/src/Foo.kt"}));
        let read_call = CallToolRequestParams::new("developer__text_editor")
            .with_arguments(object!({"command": "view", "path": "/src/Foo.kt"}));

        // Two identical edits — still within the limit of 2.
        assert!(inspector.check_tool_call(edit_call.clone()));
        assert!(inspector.check_tool_call(edit_call.clone()));

        // A genuinely different call (a read) breaks the streak and resets.
        assert!(inspector.check_tool_call(read_call.clone()));

        // Now two more identical edits are again allowed because the counter
        // was reset — proving interleaved work is not penalized.
        assert!(inspector.check_tool_call(edit_call.clone()));
        assert!(inspector.check_tool_call(edit_call.clone()));
    }

    /// End-to-end through the `ToolInspector::inspect` trait method: once the
    /// limit is exceeded, the inspector emits a Deny finding tagged "repetition".
    #[tokio::test]
    async fn inspect_emits_a_deny_finding_when_limit_exceeded() {
        let mut inspector = RepetitionInspector::new(Some(2));
        let request = make_edit_request("req-loop", "/src/DashboardRepository.kt");

        // Prime the inspector's internal counter past the limit using the same
        // call signature the request carries, so the next `inspect` denies it.
        let primed_call = request
            .tool_call
            .clone()
            .expect("test request must hold a valid tool_call");
        assert!(inspector.check_tool_call(primed_call.clone()));
        assert!(inspector.check_tool_call(primed_call.clone()));

        let results = inspector
            .inspect(
                "test-session",
                std::slice::from_ref(&request),
                &[],
                GooseMode::Auto,
            )
            .await
            .expect("inspection should not error");

        assert_eq!(results.len(), 1, "exactly one finding expected for the looping call");
        assert_eq!(results[0].action, InspectionAction::Deny);
        assert_eq!(results[0].inspector_name, "repetition");
        assert_eq!(results[0].tool_request_id, "req-loop");
    }
}
