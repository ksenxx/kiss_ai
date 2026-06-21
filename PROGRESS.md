# Progress

## Task: Remove unused options from the Sorcar CLI

Use gpt-5.5 (non-codex) for thorough review and the original model
(claude-opus-4-7) for coding, bug fixing, and test creation.

### gpt-5.5 review — identifying unused options

Reviewed every option declared in
`kiss.agents.sorcar.cli_helpers._build_arg_parser` against every
consumer in the sorcar CLI:

- `main()` in `worktree_sorcar_agent.py` (interactive +
  non-interactive paths)
- `_build_run_kwargs` in `cli_helpers.py`
- `run_client(...)` (interactive client) — parameters:
  `use_worktree`, `use_parallel`, `auto_commit`
- `run_with_steering(...)` (non-interactive driver) — reads
  `run_kwargs`

Findings:

| Option | Non-interactive use | Interactive use |
|--------|---------------------|-----------------|
| `-m/--model_name` | run_kwargs | `model_name=` |
| `-e/--endpoint` | model_config | dropped (daemon owns it) |
| `--header` | model_config | dropped (daemon owns it) |
| `-b/--max_budget` | run_kwargs | dropped (daemon owns it) |
| `-w/--work_dir` | run_kwargs + main | `work_dir=` |
| `-v/--verbose` | run_kwargs + print_outcome | dropped |
| `--no-web` | run_kwargs | dropped |
| `-p/--parallel` | run_kwargs | `use_parallel=` |
| `--worktree` | rejected (interactive-only) | `use_worktree=` |
| `--auto-commit` | rejected (interactive-only) | `auto_commit=` |
| `-t/--task` | task selection | n/a (mode selector) |
| `-f/--file` | task selection | n/a (mode selector) |
| **`-n/--new`** | **rejected (interactive-only)** | **NEVER READ** |

`run_client()` has no `new` parameter, so the interactive path
silently ignores `args.new`. The non-interactive path rejects it
via `_reject_interactive_only_flags`. Net effect: `-n/--new` is
parsed by the sorcar CLI but never has any effect in any code path
of the sorcar CLI.

Third-party channel agents (Slack, Discord, …) DO need a `--new`
flag — they're long-running `ChatSorcarAgent` subclasses that
use `_apply_chat_args` (which reads `args.new` → `agent.new_chat()`).
But those agents already extend the parser locally with `--chat-id`
/ `-l/--list-chat-id` in `channel_main`; they can add `-n/--new`
the same way without keeping the dead flag in the shared parser.

### Plan (claude-opus-4-7)

1. Remove `-n/--new` from `_build_arg_parser` in `cli_helpers.py`.
1. Remove `"-n"`, `"--new"` from `_INTERACTIVE_ONLY_FLAGS` in
   `worktree_sorcar_agent.py` and update the docstrings of
   `_INTERACTIVE_ONLY_FLAGS`, `_reject_interactive_only_flags`,
   and `main()`.
1. Update the `_apply_chat_args` docstring in `cli_helpers.py` —
   the sorcar CLI no longer exposes `--new`; the helper is used
   exclusively by `channel_main`.
1. Add `-n/--new` to the local parser in
   `_channel_agent_utils.channel_main` so the channel agent CLI
   surface stays unchanged.
1. Update tests:
   - `test_cli_non_interactive_flag_validation.py` — drop the
     `-n`/`--new`/`--ne` rows from the rejection parametrize and
     the interactive-acceptance parametrize.
   - `test_cli_only_sorcar_agent.py` — replace
     `test_new_flag_still_present` with
     `test_new_flag_is_rejected` (locks the new contract).
1. Update PROGRESS.md.

### Verification

After implementation, run:

- `uv run check --full`
- `uv run pytest src/kiss/tests/agents/sorcar/` and the channel
  agent tests touching `channel_main`.
