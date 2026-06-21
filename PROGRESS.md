# Progress

## Task: Restrict non-interactive sorcar CLI to SorcarAgent only and remove options -c/--chat-id, -l/--list-chat-id, --use-chat, --cleanup, --use-worktree.

### Status: ✅ COMPLETE

### Final shape of the change

- `src/kiss/agents/sorcar/cli_helpers.py` — removed `-c/--chat-id`,
  `-l/--list-chat-id`, `--cleanup`, `--use-chat`, `--use-worktree`
  from `_build_arg_parser`. `_apply_chat_args` now reads `args.new`
  / `args.chat_id` via `getattr` so third-party agents that still
  bolt these flags onto their own parsers keep working.
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` — removed the
  `_resolve_cli_modes` and `_build_cli_agent` helpers and rewrote
  `main()`. Non-interactive (`-t TASK` / `-f FILE`) constructs a
  bare `SorcarAgent("Sorcar Agent")` and dispatches it through
  `run_with_steering` — no `ChatSorcarAgent` / `WorktreeSorcarAgent`
  is built, and the legacy `[c]ommit / [d]iscard?` worktree prompt
  is gone. Interactive mode still hands the daemon `run_client(...)`
  with the worktree / parallel / auto-commit flags from argparse.
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` —
  re-adds `-c/--chat-id` and `-l/--list-chat-id` directly onto the
  channel-agent parser so background channel agents (Slack, Discord,
  …) preserve their existing chat-resume CLI surface.
- `src/kiss/tests/agents/sorcar/test_cli_only_sorcar_agent.py` —
  new integration test pinning the contract:
  - the five removed flags all trigger `SystemExit` in the parser;
  - `main()` with `-t TASK` invokes `run_with_steering` with a
    `type(agent) is SorcarAgent` instance (parametrized on
    `--no-worktree`);
  - `_resolve_cli_modes` / `_build_cli_agent` are no longer
    importable from `worktree_sorcar_agent`;
  - the parser still exposes `--worktree` / `--no-worktree` /
    `-n/--new`;
  - `_apply_chat_args` is still importable for third-party agents.
- `src/kiss/tests/agents/sorcar/test_cli_default_modes.py` — removed
  `TestLegacyAliases`, `TestBuildCliAgent`, and `_build_cli_agent` /
  `_resolve_cli_modes` imports. `TestArgParserDefaults`,
  `TestArgParserOptOuts`, `TestRunKwargsPropagation`, and
  `TestAutoCommitEnabledGate` remain.

### Verification

- `uv run check --full` — all checks pass (ruff, mypy, pyright,
  mdformat).
- `uv run pytest src/kiss/tests/agents/sorcar/` — **1696 passed**,
  28 deselected.
- `uv run pytest src/kiss/tests/agents/channels/` — **425 passed**,
  29 skipped.
- The new test reproduces the issue before the fix and locks the
  contract afterwards.
