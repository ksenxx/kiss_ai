# Progress

## Task

Make all third-party agents use `SorcarAgent` instead of
`ChatSorcarAgent`, and completely get rid of the chat-session CLI
options (`-n/--new`, `-c/--chat-id`, `-l/--list-chat-id`) from the
project. Reproduce the issue first with an integration test, then
fix it. Use gpt-5.5 (non-codex) for thorough review and the
original model (claude-opus-4-7) for coding, bug fixing, and tests.

## Round 1 (previous context)

- Wrote the integration test
  `src/kiss/tests/agents/channels/test_channel_agents_no_chat_session.py`
  that locks the post-fix contract (channel agents inherit
  `SorcarAgent` but NOT `ChatSorcarAgent`; `channel_main` rejects
  every chat-session flag; `_apply_chat_args` no longer exists).
- Flipped all 23 channel agent classes from `ChatSorcarAgent` to
  `SorcarAgent` (`bluebubbles_agent.py`, `discord_agent.py`,
  `feishu_agent.py`, `gmail_agent.py`, `googlechat_agent.py`,
  `imessage_agent.py`, `irc_agent.py`, `line_agent.py`,
  `matrix_agent.py`, `mattermost_agent.py`, `msteams_agent.py`,
  `nextcloud_talk_agent.py`, `nostr_agent.py`,
  `phone_control_agent.py`, `signal_agent.py`, `slack_agent.py`,
  `sms_agent.py`, `synology_chat_agent.py`, `telegram_agent.py`,
  `tlon_agent.py`, `twitch_agent.py`, `whatsapp_agent.py`,
  `zalo_agent.py`).

## Round 2 (this context)

### Production code

- **`src/kiss/agents/third_party_agents/_channel_agent_utils.py`**:

  - `BaseChannelAgent` class docstring + MRO example flipped to
    `SorcarAgent`.
  - `ChannelRunner` class docstring + `run_once()` docstring
    updated to say "SorcarAgent" (was "ChatSorcarAgent").
  - `ChannelRunner._handle_message()` now imports
    `SorcarAgent` from `sorcar_agent` and constructs a bare
    `SorcarAgent(self._agent_name)`; the `agent.new_chat()`
    call was dropped (no chat-session state on a plain
    `SorcarAgent`).
  - `channel_main()`:
    - Dropped `-c/--chat-id`, `-l/--list-chat-id`, `-n/--new`
      local parser extensions and the explanatory comment.
    - Dropped `[-n] [--chat-id ID] [-l]` from the usage line.
    - Dropped the `if args.list_chat_id:` block.
    - Dropped the `_apply_chat_args(agent, args, ...)` call.
    - Dropped `_apply_chat_args` and `_print_recent_chats` from
      the import list.

- **`src/kiss/agents/sorcar/cli_helpers.py`**:

  - Removed the `_apply_chat_args` helper entirely.
  - `_print_run_stats` parameter type changed from
    `ChatSorcarAgent` to `SorcarAgent`, and the
    `"\nChat ID: {agent.chat_id}"` line was dropped.
  - `print_outcome()`: dropped the
    `isinstance(agent, ChatSorcarAgent)` branch and the runtime
    `ChatSorcarAgent` import; both code paths now just call
    `_print_run_stats(agent, elapsed)`.
  - Dropped `ChatSorcarAgent` from the `TYPE_CHECKING` block.

- **Pollers** (`slack_sorcar_poller.py` /
  `slack_channel_sorcar_poller.py`): intentionally left alone —
  they use `ChatSorcarAgent` (and `WorktreeSorcarAgent`) directly
  through the Python API to map Slack threads to chat sessions
  via `chat_id` / `resume_chat_by_id`. This is a programmatic
  use of chat-session persistence, not a CLI option, and the
  task is about removing CLI surface.

### Test updates

- **`src/kiss/tests/agents/channels/test_base_channel_agent.py`**:
  deleted `test_channel_main_list_chats_exits` and the now-unused
  `sys` and `channel_main` imports.

- **`src/kiss/tests/agents/channels/test_slack_agent.py`**: deleted
  the entire `TestSlackAgentChatPersistence` class plus the
  `_redirect_db`, `_restore_db`, `_intercept_run` helpers it owned.
  Removed the now-unused `Any`, `cast`, `kiss.agents.sorcar.persistence as th`,
  `SorcarAgent` imports. Updated module docstring.

- **`src/kiss/tests/agents/sorcar/test_100pct_branch_coverage.py`**:
  removed `test_apply_chat_args_chat_id` and
  `test_apply_chat_args_no_options`, the
  `_apply_chat_args` import, the now-unused `ChatSorcarAgent`
  import, and the now-unused `argparse` import.

- **`src/kiss/tests/agents/sorcar/test_cli_only_sorcar_agent.py`**:
  replaced `TestApplyChatArgsStillExported` (which asserted the
  helper IS importable) with `TestApplyChatArgsRemoved` (which
  asserts it no longer exists).

- **`src/kiss/tests/agents/sorcar/test_simplification_lockdown_cli_tools.py`**:
  `test_print_run_stats_exact_lines` updated to use a plain
  `SorcarAgent` (no `_chat_id`) and assert the new
  three-line output `Time / Cost / Total tokens`. Added a
  `SorcarAgent` import.

## Verification

- `uv run check --full` → ruff, mypy, pyright, mdformat all PASS.
- `uv run pytest src/kiss/tests/agents/channels/test_channel_agents_no_chat_session.py -v`
  → **53 passed** (the new integration test).
- `uv run pytest src/kiss/tests/agents/channels/test_base_channel_agent.py src/kiss/tests/agents/channels/test_slack_agent.py src/kiss/tests/agents/sorcar/test_cli_only_sorcar_agent.py src/kiss/tests/agents/sorcar/test_cli_non_interactive_flag_validation.py src/kiss/tests/agents/sorcar/test_100pct_branch_coverage.py`
  → **207 passed**.
- `uv run pytest src/kiss/tests/agents/channels/ -k 'not slack_agent'`
  → **442 passed, 29 skipped**.
- `uv run pytest src/kiss/tests/agents/channels/test_slack_agent.py`
  → **30 passed**.
- `uv run pytest src/kiss/tests/agents/sorcar/test_cli_launch_work_dir.py`
  in isolation → **6 passed** (full sorcar suite hits the known
  `socket.accept() out of system resource` on this machine when too
  many sockets open concurrently; same condition was documented in
  the prior task and is unrelated to this change).
- `uv run pytest src/kiss/tests/agents/sorcar/test_cli_repl.py`
  in isolation → **27 passed**.
- `uv run pytest src/kiss/tests/agents/sorcar/test_simplification_lockdown_cli_tools.py`
  → **30 passed**.

The contract is locked end-to-end: every channel agent inherits
`SorcarAgent` (not `ChatSorcarAgent`), every chat-session CLI flag
is rejected by `channel_main`, the `_apply_chat_args` helper is
deleted, and the only `print_outcome` path that ever printed a
`Chat ID:` line is gone.
