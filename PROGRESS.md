# Task: sorcar CLI — pressing `@` must show a live file/folder picker (Up/Down navigate, Tab/Enter select)

## Progress log

1. Diagnosis: the REPL (`src/kiss/agents/sorcar/cli_repl.py`) used `readline`, which
   only completes on Tab and cannot render an arrow-navigable dropdown. Solution:
   use `prompt_toolkit` on interactive TTYs, keep readline as off-TTY fallback.
1. `pyproject.toml`: added `prompt_toolkit>=3.0.0` to dependencies; `uv sync` OK
   (prompt-toolkit 3.0.52 installed).
1. NEW file `src/kiss/agents/sorcar/cli_prompt.py`:
   - `_AT_RE` / `_MODEL_CMD_RE` moved here (single source; cli_repl imports them).
   - `PtkCompleter(Completer)` adapts `CliCompleter`: `@query` → `rank_file_suggestions`
     entries displayed as bare path with meta folder/recent/file, inserted as
     `PWD/<path> ` with `start_position=-(len(query)+1)`; `/model <partial>` →
     `rank_model_suggestions`; `/cmd` → `cli._slash_matches` with SLASH_COMMANDS help
     meta (lazy import of cli_repl to avoid circularity); whole-line predictive
     matches only when `complete_event.completion_requested` (Tab).
   - `_KEY_BINDINGS`: Enter and Tab with `completion_is_selected` set
     `buffer.complete_state = None` → confirm highlighted item without submitting.
   - `_migrate_readline_history`: one-time seed of `<hist>.ptk` (FileHistory format
     `+line`) from old readline history.
   - `PtkLineReader`: `PromptSession(completer=PtkCompleter, complete_while_typing=True, key_bindings=_KEY_BINDINGS, history=FileHistory(<hist>.ptk), reserve_space_for_menu=8)`;
     `read(prompt)` wraps prompt in `ANSI(...)`.
1. `cli_repl.py` changes:
   - module docstring updated (prompt_toolkit primary, readline fallback).
   - imports `_AT_RE`, `_MODEL_CMD_RE`, `PtkLineReader` from cli_prompt; local regex
     definitions removed.
   - NEW `_make_ptk_reader(completer, history_path)` → PtkLineReader only when
     stdin AND stdout are TTYs, else None.
   - NEW `_read_line_ptk(reader, prompt)` → prints panel top, reads via
     reader.read(framed prompt), backslash continuation preserved, prints bottom
     after (also on EOF/Ctrl+C; EOF→None, Ctrl+C re-raised).
   - `_read_line(prompt, reader=None)` delegates to `_read_line_ptk` when reader set.
   - `run_repl`: `reader = _make_ptk_reader(...)`; `_setup_readline` only when
     reader is None; `_save_history` guarded by `using_readline` (else empty
     in-process readline history would clobber the history file).
1. Smoke test: `from kiss.agents.sorcar import cli_repl, cli_prompt` imports OK.

## Verification (all done)

1. NEW tests `src/kiss/tests/agents/sorcar/test_at_mention_picker.py` (9 tests, all
   pass): @ pops files+folders immediately; @query filters and replaces the whole
   token; folder meta; slash menu with help; /model partial; predictive gated to
   Tab; end-to-end pipe-input tests (real PromptSession via `create_pipe_input` +
   `create_app_session`): `@` + Down + Enter inserts the mention without
   submitting, and `@alp` + Down + Tab confirms (final submitted lines
   `look at PWD/<file>  please` and `PWD/alpha.py now`); readline→ptk history
   migration is idempotent.
1. Existing REPL tests: 99 passed; `test_bughunt4_prompt_markers.py` fails
   identically on pre-change code (verified via `git stash`) — pre-existing
   environment flake, not a regression.
1. `uv run check --full` → ✅ All checks passed (after `mdformat PROGRESS.md`).
1. PTY sanity script (real terminal): typing `@` rendered the dropdown with
   `alpha.py`/`subdir/`, Down+Enter inserted `PWD/alpha.py` without submitting,
   final submitted line was `see PWD/alpha.py now`. Temp script deleted.
1. New files `cli_prompt.py` and `test_at_mention_picker.py` added to the git
   index.
