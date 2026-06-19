# Task: Apply limit=16*1024*1024 to all asyncio.open_unix_connection calls

## Status

Applying `limit=16*1024*1024` parameter to every `asyncio.open_unix_connection` call in the codebase to prevent buffer overflow vulnerabilities similar to the CLI "Daemon connection lost" bug.

## Files already DONE (production fix from prior task)

- src/kiss/agents/sorcar/cli_client.py (line ~378) — DONE in prior session.

## Files already MODIFIED in this session

1. src/kiss/tests/agents/sorcar/test_cli_daemon_live_stream.py (line 150)
1. src/kiss/tests/agents/sorcar/test_cli_history_click_resumes_live_stream.py (lines 140, 356)
1. src/kiss/tests/agents/vscode/test_bughunt3_autocomplete_echo_strip.py (line 105)
1. src/kiss/tests/agents/vscode/test_update_available_check.py (line 154)
1. src/kiss/tests/agents/vscode/test_web_extension_parity.py (line 98)

## Files REMAINING

6. src/kiss/tests/agents/vscode/test_web_server_uds.py — lines 118, 140, 169, 209, 272, 321 (6 sites)
1. src/kiss/tests/agents/vscode/test_replay_event_coalescing.py — line 262
1. src/kiss/tests/agents/vscode/test_bughunt5_stale_state_after_all_done.py — line 143
1. src/kiss/tests/agents/vscode/test_welcome_suggestions_not_broadcast.py — lines 118, 145
1. src/kiss/tests/agents/vscode/test_settings_work_dir.py — line 327
1. src/kiss/tests/agents/vscode/test_server_reset.py — line 111
1. src/kiss/tests/agents/vscode/test_simplification_lockdown_web_server.py — line 220
1. src/kiss/tests/agents/vscode/test_per_window_reply_isolation.py — line 146
1. src/kiss/tests/agents/vscode/test_per_window_work_dir.py — line 116
1. src/kiss/tests/agents/vscode/test_per_window_autocomplete.py — line 98

## Edit pattern

Replace:

```
reader, writer = await asyncio.open_unix_connection(
    str(self.uds_path),
)
```

with:

```
reader, writer = await asyncio.open_unix_connection(
    str(self.uds_path),
    limit=16 * 1024 * 1024,
)
```

Single-line variant: add `, limit=16 * 1024 * 1024` as 2nd kwarg.

## After all edits

- Run `uv run check --full`
- Run impacted tests (the modified test files)
- Verify production fix in cli_client.py still has the limit
- finish(success=True)

## Verification command

After edits run:

```
grep -rn "open_unix_connection" --include="*.py" src/ | grep -v "limit="
```

Only doc/comment mentions should remain. All call-site lines must include limit=.

## NEVER kill PID 34767 — that's the live kiss-web daemon.
