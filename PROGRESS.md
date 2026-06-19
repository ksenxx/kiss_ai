# Progress — run all tests and report causes of failures

- Read `SORCAR.md` (empty) then `PROGRESS.md` (had a previous-session log).
- Checked CPU cores: 10 → split tests into 8 parallel splits (cores − 2).
- Collected all pytest node IDs with `uv run pytest --collect-only -q --no-cov -m ''` → 3749 tests (all markers enabled).
- Split node IDs into 8 files `tmp/split_1..8.txt` via a small Python script.
- Ran all 8 splits in parallel via `run_parallel`. Each agent ran:
  `xargs -a tmp/split_N.txt uv run pytest --no-cov -m '' --tb=short -q -p no:randomly`
  and wrote a concise `tmp/results_N.md` report. NO code was modified.
- Read every `tmp/results_N.md` and synthesized the final failure report below.

## Aggregate result
- Total: 3749 tests
- Passed: 3692
- Failed: 17
- Skipped: 37 (29 in split 1 + 8 in split 8; all environment-/marker-gated)
- Errors: 0

## Failures grouped by root cause

### A. `openSubagentTab` events not emitted/captured (6 failures)
Tests asserting that running `run_parallel` emits `openSubagentTab` stream
events see an empty event list. The producer side appears to emit
`new_tab` / `subagentDone` instead (event-name / wiring drift between
producer and consumer of the subagent-tab event stream).
- `src/kiss/tests/agents/sorcar/test_run_parallel_integration.py::TestNestedParallelReal::test_nested_parallel_subagent_tab_events` — `assert 0 == 2` (expected 2 outer `openSubagentTab` events as direct children of root).
- `src/kiss/tests/agents/sorcar/test_run_parallel_integration.py::TestSubagentTabEventsE2E::test_subagent_tab_events_broadcast` — `assert 0 == 2`.
- `src/kiss/tests/agents/sorcar/test_run_parallel_integration.py::TestSubagentTabEventsE2E::test_subagent_streaming_events_have_tab_ids` — `assert 0 > 0` (no streaming events tagged with sub-agent `tab_id`).
- `src/kiss/tests/agents/sorcar/test_run_parallel_integration.py::TestSubagentTabEventsE2E::test_description_field_in_open_event` — `assert 0 == 1` (no `openSubagentTab` event to inspect `description` on).
- `src/kiss/tests/agents/sorcar/test_subagent_tabs.py::TestSubagentTabEvents::test_parallel_creates_subagent_tab_events` — `assert len(open_events) >= 2`, got 0.
- `src/kiss/tests/agents/sorcar/test_subagent_tabs.py::TestSubagentTabEvents::test_parallel_subagent_events_have_correct_types` — `assert len(open_events) >= 1`, got 0.

### B. VSCode server stop test never observes status `running=true` (1 failure)
- `src/kiss/tests/agents/sorcar/test_vscode_stop.py::TestVSCodeServerStop::test_stop_command_interrupts_running_task` — `AssertionError: Never saw status running=true. Events: []`. The status-streaming channel the test subscribes to is not being populated (server didn't start/emit status within the wait window or status feed is not wired in this environment).

### C. Stderr-drain timing-too-tight (1 failure, looks like flake / host-speed-dependent)
- `src/kiss/tests/agents/vscode/test_web_server_security.py::TestH6StderrReaderCleanup::test_reader_keeps_draining_stderr_after_url_found` — `subprocess.TimeoutExpired` after 10 s. The helper subprocess writes 2000 stderr lines with `time.sleep(0.001)` between each line and does not finish within `proc.wait(timeout=10)` on this host. The test's timing budget is too tight rather than an actual stderr-drain bug.

### D. Missing external CLI binary `/usr/bin/claude` (2 failures)
Real-CLI integration tests that spawn `/usr/bin/claude`, which is not installed.
- `src/kiss/tests/core/models/test_cc_opus_live_thinking.py::TestCCOpusLiveThinking::test_no_empty_thinking_bar` — `FileNotFoundError: '/usr/bin/claude'` → `KISSError: Failed to start Claude Code CLI`.
- `src/kiss/tests/core/models/test_claude_code_model.py::TestGenerateIntegration::test_generate_token_counts` — same `FileNotFoundError: '/usr/bin/claude'`.
- `src/kiss/tests/core/models/test_claude_code_model.py::TestGenerateIntegration::test_generate_streaming` — same `FileNotFoundError: '/usr/bin/claude'`.

### E. Missing external CLI binary `/usr/bin/codex` (6 failures)
Real-CLI integration tests that spawn `/usr/bin/codex`, which is not installed.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_token_counts` — `FileNotFoundError: '/usr/bin/codex'`.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_streaming` — same.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_failure_raises` — expected regex `'Codex CLI failed'` but actual error is the same startup `FileNotFoundError: '/usr/bin/codex'`.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_can_modify_files` — same `FileNotFoundError`.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_streams_command_progress` — same.
- `src/kiss/tests/core/models/test_codex_model.py::TestGenerateIntegration::test_generate_and_process_with_tools_runs` — same.

(Recount: A = 6, B = 1, C = 1, D = 3, E = 6 → 17 failures.)

## No code was modified during this task.
