---
name: goose-test-isolation
description: Run cargo tests in the goose repo with an isolated config root (GOOSE_PATH_ROOT) so the developer's active Goose installation cannot pollute test results. Use whenever running or debugging cargo test in this repository. Non-hermetic tests tracked in upstream issue #13188.
---

# Goose test isolation

Tests in this repo resolve paths through `Config::global()` and `Paths`, so
without `GOOSE_PATH_ROOT` they read the developer's active Goose installation
(`~/.config/goose/...`). A customized `config.yaml` or a user prompt override
makes tests fail or flake. Tracked upstream in #13188; until it is fixed,
always run the suite against an empty config root.

## Run the suite isolated

```console
GOOSE_PATH_ROOT=$(mktemp -d) GOOSE_DISABLE_KEYRING=1 RUST_MIN_STACK=8388608 \
  cargo test -p goose --no-default-features \
  --features rustls-tls,code-mode,tree-sitter,live-voice,scheduler,platform-apps,chat-recall,acp-http
```

- Set `GOOSE_PATH_ROOT` on the whole cargo invocation, never per-test.
  `Config::global()` is a process-global `OnceLock`, so a parallel test that
  initializes it first from the real config can poison the run even when the
  target test locks its own environment.
- A plain workspace `cargo test` also exercises the goose lib through feature
  unification via goose-cli (the `scheduler` tests appear there); the command
  above includes `scheduler` explicitly so it matches that surface.
- Bedrock provider tests need the additional `--features aws-providers`
  flag; they live behind that feature gate.

## Failure signatures of a polluted run

- `scheduler::full::tests::test_scheduled_run_uses_recipe_model_settings`
  times out after 10 seconds waiting for a session model config:
  `execute_job` loads every config-enabled extension first, so one slow or
  unreachable MCP server (autovisualiser, datagripmcp, tutorial) stalls
  session creation before the model config is persisted.
- `agents::prompt_manager::{test_typical_setup, test_basic, test_one_extension}`
  and `prompt_template::tests::test_get_template` fail snapshot assertions
  when `~/.config/goose/prompts/system.md` exists, because the user override
  wins over the bundled `src/prompts/system.md`.

## Not this bug

`tests/providers.rs::{test_openrouter_provider, test_claude_code_provider}`
call live networks; failures there are environmental, not regressions.
