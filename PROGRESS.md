# Progress

## Task: Multi-round review/fix cycle for the previous "non-interactive sorcar = SorcarAgent only" change

Use gpt-5.5 (non-codex) for reviews and claude-opus-4-7 for code/tests,
repeating until the gpt-5.5 review finds nothing actionable.

### Status: ✅ COMPLETE — 4 review rounds done, all bugs fixed.

### Round 1 — gpt-5.5 review of the previous task (commit c4444f4e)

Bugs identified:

- **BUG-1 (high)**: Non-interactive mode silently accepted and ignored
  `--worktree` / `--no-worktree` / `--auto-commit` / `--no-auto-commit`
  / `-n` / `--new`. The parser still exposed them (and `--worktree`
  defaulted to `True`), so users running `sorcar -t TASK` silently lost
  worktree isolation — edits now landed directly in the working tree.
- **BUG-2 (minor)**: `main()` docstring claimed "no chat persistence"
  but `_build_run_kwargs` installs a `RecordingConsolePrinter` and
  `run_with_steering` mints a `chat_id` via `_allocate_chat_id`.

### Round 1 — claude-opus-4-7 fix (commit 48651884)

- Added `_INTERACTIVE_ONLY_FLAGS` and `_reject_interactive_only_flags()`
  in `worktree_sorcar_agent.py`. Called from `main()` for non-interactive
  runs AFTER argparse but BEFORE building the agent / spending budget.
- The helper exits with `sys.exit(2)` (argparse convention) and an error
  naming every offending flag.
- Amended the `main()` docstring to drop the false "no chat persistence"
  claim and document the new validation.
- Wrote `test_cli_non_interactive_flag_validation.py` reproducing
  BUG-1 (15 tests; failed before the fix) and locking the contract:
  - Each flag in the interactive-only set is rejected with a non-zero
    exit code, an error message naming the flag, and no call into
    `run_with_steering`.
  - Non-interactive still accepts `--no-parallel` and `--no-web`.
  - Interactive (no `-t/-f`) still accepts every flag (forwarded to the
    daemon).
- Removed obsolete `test_no_worktree_still_uses_plain_sorcar_agent`
  from `test_cli_only_sorcar_agent.py` (the flag is now rejected).

### Round 2 — gpt-5.5 review of the round 1 fix

Bugs identified:

- **BUG-3 (high)**: `argparse.ArgumentParser` defaults to
  `allow_abbrev=True`, so the parser silently expanded abbreviations
  like `--auto` → `--auto-commit`, `--no-auto` → `--no-auto-commit`,
  `--worktr` → `--worktree`, `--ne` → `--new`. The literal-token
  guard didn't match the abbreviated tokens, reintroducing the silent
  no-op for every flag in the interactive-only set.
- **BUG-4 (cleanup)**: `_INTERACTIVE_ONLY_FLAGS` carried a `reason`
  string per entry that was collected but never displayed.

### Round 2 — claude-opus-4-7 fix (commit 2731f4fe)

- Set `allow_abbrev=False` on `_build_arg_parser` (docstring explains
  why — to keep the non-interactive guard reliable).
- Refactored `_INTERACTIVE_ONLY_FLAGS` from `tuple[tuple[str, str]]`
  to `frozenset[str]` and simplified `_reject_interactive_only_flags`
  to a set-membership check.
- Added a parametrized `test_argparse_abbreviations_are_also_rejected`
  test that reproduces BUG-3 (failed before the fix) and pins the
  contract.

### Round 3 — gpt-5.5 review of the round 2 fix

No new actionable bugs found. Reviewed:

- `allow_abbrev=False` cleanly closes the abbreviation hole.
- `frozenset` lookup matches every literal flag token.
- Channel-agent `channel_main` re-uses `_build_arg_parser()` so its
  parser also disables abbreviations (intentional, uniform behaviour;
  no in-tree caller relies on abbreviations — `grep -rn "allow_abbrev" src/` is empty).
- Pre-existing notes (out of scope): `-n/--new` is silently ignored
  in interactive mode because `run_client` does not take a `new`
  parameter (pre-dates this task); `RecordingConsolePrinter`
  registers an `atexit` handler per instance (pre-dates this task).

### Round 4 — gpt-5.5 final pass

Read the diff end-to-end one more time after running the test
suite. No actionable findings.

### Verification

- `uv run check --full` — ruff, mypy, pyright, mdformat all pass.
- `uv run pytest src/kiss/tests/agents/sorcar/` — **1714 passed**, 28
  deselected (one pre-existing flaky timing test passed on retry).
- `uv run pytest src/kiss/tests/agents/channels/ -k "not slack"` —
  **370 passed**, 29 skipped. (Two slack-API live integration tests
  intermittently fail on SSL handshake; pass in isolation. Unrelated
  to this change.)
- The new test reproduces BUG-1 and BUG-3 before the fixes; locks the
  contract afterwards.

### Files changed (cumulative)

- `src/kiss/agents/sorcar/cli_helpers.py` — `_build_arg_parser` sets
  `allow_abbrev=False`.
- `src/kiss/agents/sorcar/worktree_sorcar_agent.py` — added
  `_INTERACTIVE_ONLY_FLAGS` and `_reject_interactive_only_flags`;
  `main()` validates non-interactive runs before building the agent.
- `src/kiss/tests/agents/sorcar/test_cli_non_interactive_flag_validation.py`
  — NEW: 20 tests pinning the fail-fast contract for the
  interactive-only flag set, including the abbreviation guard.
- `src/kiss/tests/agents/sorcar/test_cli_only_sorcar_agent.py` —
  removed the now-obsolete `--no-worktree` parametrization.
