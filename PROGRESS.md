# Progress — Add `start_line` parameter to Read tools

## Plan

1. Write failing integration tests that exercise reading a window starting at
   `start_line` on:
   - `kiss.agents.sorcar.useful_tools.UsefulTools.Read`
   - `kiss.docker.docker_tools.DockerTools.Read`
1. Add `start_line: int = 1` parameter (1-indexed) to both `Read` implementations.
1. Confirm tests pass with `uv run pytest` and the project is clean under
   `uv run check --full`.
1. Hand the diff to `gpt-5.5` for a thorough review.

## Decisions

- `start_line` is 1-indexed (matches `head`/`tail`/IDE jump-to-line UX).
- `start_line=1` is the default → fully backward compatible.
- Out-of-range (`start_line > total_lines`) returns a clear sentinel string
  rather than silently returning empty content.
- `start_line < 1` is rejected with an error string (avoid surprising the model
  with 0/negative indexing semantics).
- The truncation footer continues to count the lines that were skipped at the
  *end* of the window so the model knows how many remain.
- Empty-file behaviour (`(file is empty)`) is preserved regardless of
  `start_line`.

## Status

- [x] Reproduce bug with failing integration tests (claude-opus-4-7).
- [x] Implement `start_line` on `UsefulTools.Read` (claude-opus-4-7).
- [x] Implement `start_line` on `DockerTools.Read` shell snippet (claude-opus-4-7).
- [x] `uv run pytest` for new + existing Read tests: 39+16 passing.
- [x] `uv run check --full`: passing.
- [x] Thorough review by gpt-5.5 — passes.

## gpt-5.5 review notes

- **Backward compatibility (Python)**: default `start_line=1` reproduces prior
  behavior. The new path uses `text.splitlines(keepends=True)` + `"".join(window)`
  which is a lossless round-trip for the universal-newline-translated string
  returned by `Path.read_text()`.
- **Empty / single-newline files**: `(file is empty)` sentinel preserved;
  `"\n"`-only files still return `"\n"` (covered by existing
  `test_text_file_with_only_newlines_is_not_empty`).
- **Truncation footer**: now `total - (start_line - 1) - len(window)` — counts
  only the tail beyond the returned window, never double-counting the prefix.
- **Docker shell snippet**: `tail -n +"$START" | head -n "$MAX"` is POSIX-portable;
  `wc -l` whitespace is tolerated by `[ -gt ]` and `$(())`. Out-of-range guard
  prints the error and `exit 0` so the bash wrapper surfaces it as stdout
  (same convention as "File not found" already used).
- **Tool schema**: `model.py` derives JSON Schema from `inspect.signature` +
  docstring; no hardcoded schema mentions `Read` or `max_lines` anywhere — the
  new `start_line` parameter is automatically exposed to the LLM.
- **Validation**: `start_line < 1` is rejected with an explicit message; past-EOF
  returns a sentinel rather than silently leaking either nothing or earlier
  content. Both contracts are pinned by the new tests.
- **Regressions**: ran 1,636 tests across sorcar+docker suites. Five sporadic
  failures live in `test_cli_client*` (CLI/REPL contention, pre-existing); each
  passes in isolation and none touch the Read code path.
- **Lint/typecheck**: `uv run check --full` passes (ruff + mypy + pyright +
  mdformat).

## Files to change

- `src/kiss/agents/sorcar/useful_tools.py` — `UsefulTools.Read`
- `src/kiss/docker/docker_tools.py` — `DockerTools.Read`
- `src/kiss/tests/agents/sorcar/test_read_start_line.py` — new test file
- `src/kiss/tests/docker/test_docker_tools_start_line.py` — new test file
