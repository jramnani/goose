use crate::session::last_message_snippet::{
    message_from_recent_row, normalized_user_text, RecentMessageRow,
};
use crate::session::session_manager::{
    message_text_match_clause, user_visible_message_sql, Session,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::{AssertSqlSafe, Pool, Sqlite};
use std::collections::HashMap;

const MATCHED_SNIPPET_MAX_CHARS: usize = 128;
const MATCHED_SNIPPET_LEAD_CHARS: usize = 32;
const MATCHED_MESSAGE_SCAN_LIMIT: usize = 3;

/// Which surface of a session matched an active keyword filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchedIn {
    #[serde(rename = "message")]
    Message,
    #[serde(rename = "title")]
    Title,
    #[serde(rename = "title+message")]
    TitleAndMessage,
}

/// Populate each listed session with "why it matched" data for an active
/// keyword filter: which surface matched (`matched_in`) and a snippet of the
/// most recent matching, user-visible message.
pub(super) async fn hydrate_matched_message_snippets(
    pool: &Pool<Sqlite>,
    sessions: &mut [Session],
    terms: &[String],
) -> Result<()> {
    if sessions.is_empty() || terms.is_empty() {
        return Ok(());
    }

    let session_ids = sessions
        .iter()
        .map(|session| session.id.clone())
        .collect::<Vec<_>>();
    let rows = matched_message_rows(pool, &session_ids, terms).await?;

    let mut rows_by_session: HashMap<String, Vec<RecentMessageRow>> =
        HashMap::with_capacity(session_ids.len());
    for row in rows {
        rows_by_session
            .entry(row.session_id.clone())
            .or_default()
            .push(row);
    }

    for session in sessions.iter_mut() {
        let title_matched = terms
            .iter()
            .any(|term| session.name.to_lowercase().contains(term));
        let message_matched = rows_by_session
            .get(&session.id)
            .is_some_and(|rows| !rows.is_empty());
        session.matched_in = match (title_matched, message_matched) {
            (true, true) => Some(MatchedIn::TitleAndMessage),
            (true, false) => Some(MatchedIn::Title),
            (false, true) => Some(MatchedIn::Message),
            (false, false) => None,
        };
        session.matched_message_snippet = None;
        if let Some(rows) = rows_by_session.get_mut(&session.id) {
            for row in rows.drain(..) {
                let Some(message) = message_from_recent_row(row)? else {
                    continue;
                };
                let Some(normalized) = normalized_user_text(&message) else {
                    continue;
                };
                session.matched_message_snippet = Some(term_centered_snippet(&normalized, terms));
                break;
            }
        }
    }

    Ok(())
}

async fn matched_message_rows(
    pool: &Pool<Sqlite>,
    session_ids: &[String],
    terms: &[String],
) -> Result<Vec<RecentMessageRow>> {
    if session_ids.is_empty() || terms.is_empty() {
        return Ok(Vec::new());
    }

    let visible = user_visible_message_sql("metadata_json");
    let text_match = message_text_match_clause(terms.len());
    let branch = format!(
        r#"
        SELECT row_id, session_id, role, content_json, created_timestamp, metadata_json, message_id
        FROM (
            SELECT id AS row_id, session_id, role, content_json, created_timestamp, metadata_json, message_id
            FROM messages
            WHERE session_id = ?
              AND {visible}
              AND EXISTS (
                  SELECT 1
                  FROM json_each(messages.content_json)
                  WHERE json_extract(value, '$.type') = 'text'
                    AND ({text_match})
              )
            ORDER BY created_timestamp DESC, id DESC
            LIMIT {MATCHED_MESSAGE_SCAN_LIMIT}
        )
        "#
    );
    let sql = std::iter::repeat_n(branch, session_ids.len())
        .collect::<Vec<_>>()
        .join(" UNION ALL ");

    let mut query = sqlx::query_as::<_, RecentMessageRow>(AssertSqlSafe(sql));
    for session_id in session_ids {
        query = query.bind(session_id);
        for term in terms {
            query = query.bind(term);
        }
    }

    let mut rows = query.fetch_all(pool).await?;
    rows.sort_by(|left, right| {
        left.session_id
            .cmp(&right.session_id)
            .then_with(|| right.created_timestamp.cmp(&left.created_timestamp))
            .then_with(|| right.row_id.cmp(&left.row_id))
    });
    Ok(rows)
}

/// A snippet of `normalized` that keeps the first occurrence of any term
/// visible: a window of at most `MATCHED_SNIPPET_MAX_CHARS` characters
/// starting up to `MATCHED_SNIPPET_LEAD_CHARS` before the occurrence, with `…`
/// marking either truncated end. Falls back to a plain head slice when no term
/// occurs (possible when SQL matched an assistant-audience-only block).
fn term_centered_snippet(normalized: &str, terms: &[String]) -> String {
    let Some(occurrence) = earliest_term_occurrence(normalized, terms) else {
        return head_snippet(normalized);
    };
    let chars: Vec<char> = normalized.chars().collect();
    if occurrence >= chars.len() {
        return head_snippet(normalized);
    }

    let start = occurrence.saturating_sub(MATCHED_SNIPPET_LEAD_CHARS);
    let prefix = start > 0;
    let suffix = start + MATCHED_SNIPPET_MAX_CHARS - usize::from(prefix) < chars.len();
    let budget = MATCHED_SNIPPET_MAX_CHARS - usize::from(prefix) - usize::from(suffix);
    let end = (start + budget).min(chars.len());

    let mut snippet = String::with_capacity(MATCHED_SNIPPET_MAX_CHARS + 2);
    if prefix {
        snippet.push('…');
    }
    snippet.extend(chars[start..end].iter());
    if end < chars.len() {
        snippet.push('…');
    }
    snippet
}

/// The character index in `normalized` of the earliest case-insensitive
/// occurrence of any term.
fn earliest_term_occurrence(normalized: &str, terms: &[String]) -> Option<usize> {
    let lowered = normalized.to_lowercase();
    terms
        .iter()
        .filter_map(|term| lowered.find(term.as_str()))
        .map(|byte| lowered.get(..byte).unwrap_or_default().chars().count())
        .min()
}

fn head_snippet(normalized: &str) -> String {
    let mut chars = normalized.chars();
    let mut result: String = chars.by_ref().take(MATCHED_SNIPPET_MAX_CHARS - 1).collect();
    if chars.next().is_some() {
        let end = result.trim_end().len();
        result.truncate(end);
        result.push('…');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::conversation::message::{Message, MessageMetadata};
    use crate::session::session_manager::{
        SessionListCursor, SessionListFilters, SessionListPage, SessionListPageQuery,
        SessionManager, SessionType,
    };
    use std::path::PathBuf;
    use tempfile::TempDir;

    async fn create_session(sm: &SessionManager, name: &str) -> String {
        sm.create_session(
            PathBuf::from("/tmp/matched-snippet"),
            name.to_string(),
            SessionType::User,
            GooseMode::default(),
        )
        .await
        .unwrap()
        .id
    }

    async fn add_message(sm: &SessionManager, id: &str, text: &str) {
        sm.add_message(id, &Message::user().with_text(text))
            .await
            .unwrap();
    }

    async fn list_page(
        sm: &SessionManager,
        keyword: Option<&str>,
        page_size: usize,
        cursor: Option<&SessionListCursor>,
    ) -> SessionListPage {
        let types = [SessionType::User];
        sm.list_sessions_paged(SessionListPageQuery {
            filters: SessionListFilters {
                types: Some(&types),
                keyword,
                only_sessions_with_messages: true,
                ..Default::default()
            },
            cursor,
            page_size,
            include_last_message_snippet: false,
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_hydrate_returns_most_recent_matching_message() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "Matched snippet session").await;

        add_message(&sm, &id, "first PROJ-1234 report").await;
        add_message(&sm, &id, "PROJ-1234 failed in the nightly run").await;

        let page = list_page(&sm, Some("proj-1234"), 10, None).await;
        let session = page
            .sessions
            .iter()
            .find(|session| session.id == id)
            .unwrap();

        assert_eq!(session.matched_in, Some(MatchedIn::Message));
        let snippet = session.matched_message_snippet.as_deref().unwrap();
        assert!(snippet.contains("PROJ-1234"));
        assert!(snippet.contains("nightly"));
    }

    #[tokio::test]
    async fn test_snippet_is_term_centered_and_truncated() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "Long matched session").await;

        let text = format!(
            "{}PROJ-1234 {}",
            "lorem ipsum dolor ".repeat(15),
            "sit amet consectetur ".repeat(15)
        );
        add_message(&sm, &id, &text).await;

        let page = list_page(&sm, Some("proj-1234"), 10, None).await;
        let session = page
            .sessions
            .iter()
            .find(|session| session.id == id)
            .unwrap();

        let snippet = session.matched_message_snippet.as_deref().unwrap();
        assert!(snippet.chars().count() <= 128);
        assert!(snippet.starts_with('…'));
        assert!(snippet.to_lowercase().contains("proj-1234"));
    }

    #[tokio::test]
    async fn test_hydrate_matches_or_terms() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let postgres_id = create_session(&sm, "Alpha session").await;
        let sqlite_id = create_session(&sm, "Beta session").await;

        add_message(&sm, &postgres_id, "tuned the Postgres replica today").await;
        add_message(&sm, &sqlite_id, "migrated the SQLite file").await;

        let page = list_page(&sm, Some("postgres sqlite"), 10, None).await;
        let by_id = |id: &str| {
            page.sessions
                .iter()
                .find(|session| session.id == id)
                .unwrap()
        };

        let postgres = by_id(&postgres_id);
        assert_eq!(postgres.matched_in, Some(MatchedIn::Message));
        assert!(postgres
            .matched_message_snippet
            .as_deref()
            .unwrap()
            .contains("Postgres"));

        let sqlite = by_id(&sqlite_id);
        assert_eq!(sqlite.matched_in, Some(MatchedIn::Message));
        assert!(sqlite
            .matched_message_snippet
            .as_deref()
            .unwrap()
            .contains("SQLite"));
    }

    #[tokio::test]
    async fn test_empty_keyword_leaves_match_fields_none() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "Plain session").await;
        add_message(&sm, &id, "an ordinary message").await;

        for keyword in [Some("   "), None] {
            let page = list_page(&sm, keyword, 10, None).await;
            let session = page
                .sessions
                .iter()
                .find(|session| session.id == id)
                .unwrap();
            assert_eq!(session.matched_in, None);
            assert_eq!(session.matched_message_snippet, None);
        }
    }

    #[tokio::test]
    async fn test_hydration_applies_to_each_page() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let first_id = create_session(&sm, "First paged session").await;
        let second_id = create_session(&sm, "Second paged session").await;
        add_message(&sm, &first_id, "PROJ-77 opening notes").await;
        add_message(&sm, &second_id, "PROJ-77 closing notes").await;

        let first_page = list_page(&sm, Some("proj-77"), 1, None).await;
        assert_eq!(first_page.sessions.len(), 1);
        assert!(first_page
            .sessions
            .iter()
            .any(|session| session.matched_message_snippet.is_some()));

        let second_page = list_page(&sm, Some("proj-77"), 1, first_page.next_cursor.as_ref()).await;
        assert_eq!(second_page.sessions.len(), 1);
        assert_ne!(
            second_page.sessions[0].id, first_page.sessions[0].id,
            "cursor must advance past the first page's session"
        );
        assert!(second_page
            .sessions
            .iter()
            .any(|session| session.matched_message_snippet.is_some()));
    }

    #[tokio::test]
    async fn test_hydrate_skips_agent_only_matches() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "Agent-only match session").await;

        add_message(&sm, &id, "older PROJ-55 visible note").await;
        sm.add_message(
            &id,
            &Message::assistant()
                .with_text("PROJ-55 hidden agent summary")
                .with_metadata(MessageMetadata::agent_only()),
        )
        .await
        .unwrap();

        let page = list_page(&sm, Some("proj-55"), 10, None).await;
        let session = page
            .sessions
            .iter()
            .find(|session| session.id == id)
            .unwrap();

        assert_eq!(session.matched_in, Some(MatchedIn::Message));
        let snippet = session.matched_message_snippet.as_deref().unwrap();
        assert!(snippet.contains("older"));
        assert!(!snippet.contains("hidden"));
    }

    #[tokio::test]
    async fn test_title_only_match_gets_matched_in_title() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "PROJ-99 nightly fix").await;
        add_message(&sm, &id, "unrelated work happened here").await;

        let page = list_page(&sm, Some("proj-99"), 10, None).await;
        let session = page
            .sessions
            .iter()
            .find(|session| session.id == id)
            .unwrap();

        assert_eq!(session.matched_in, Some(MatchedIn::Title));
        assert_eq!(session.matched_message_snippet, None);
    }

    #[tokio::test]
    async fn test_title_and_message_match_reports_both() {
        let temp_dir = TempDir::new().unwrap();
        let sm = SessionManager::new(temp_dir.path().to_path_buf());
        let id = create_session(&sm, "PROJ-88 sync").await;
        add_message(&sm, &id, "PROJ-88 progress update").await;

        let page = list_page(&sm, Some("proj-88"), 10, None).await;
        let session = page
            .sessions
            .iter()
            .find(|session| session.id == id)
            .unwrap();

        assert_eq!(session.matched_in, Some(MatchedIn::TitleAndMessage));
        assert!(session
            .matched_message_snippet
            .as_deref()
            .unwrap()
            .contains("PROJ-88"));
    }
}
