---
title: 'uv run check --full: stages, flags and pitfalls'
uuid: 2745a81e-0a58-438e-b062-f65ee6036cc2
summary: What `uv run check` / `check --full` runs (uv sync, generate-api-docs, compileall,
  ruff, mypy, pyright, VS Code extension typecheck and lint), its flags, and common
  failures.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# uv run check --full: stages, flags and pitfalls

`check` is a console script declared in `pyproject.toml` (`check = "kiss.scripts.check:main"`),
so `uv run check --full` runs `src/kiss/scripts/check.py`.

## Order of work in `main()`
1. Unless `--no-clean`: `clean_build_artifacts()` deletes build dirs, `__pycache__` dirs and
   `*.pyc` files. `__pycache__`/`*.pyc` paths inside `.git` or reported as ignored by
   `git check-ignore` are skipped (`_should_skip_path`); non-ignored ones are removed.
   `--clean-only` stops after this.
2. `_sync_extension_version()` copies `__version__` from `src/kiss/core/_version.py` into the
   `version` field of `src/kiss/agents/vscode/package.json` when they differ.
3. The stages, in this order:
   | Stage | Command |
   |---|---|
   | Install dependencies | `uv sync` |
   | Generate API docs | `uv run generate-api-docs` (rewrites `API.md`, see `dev-api-docs-generation`) |
   | Syntax check | `uv run python -m compileall -q src/` |
   | Lint | `uv run ruff check src/` |
   | Type check | `uv run mypy src/` |
   | `--full` only: pyright | `uv run pyright src/` |
   | `--full` only: extension typecheck | `npm --prefix src/kiss/agents/vscode run typecheck` (`tsc --noEmit -p ./`) |
   | `--full` only: extension lint | `npm --prefix src/kiss/agents/vscode run lint` (eslint, stylelint, htmlhint) |

   The two npm stages are added only when `src/kiss/agents/vscode/node_modules` exists and
   `npm` is on PATH. In a worktree `node_modules` is a symlink to the main checkout's copy.
   `check --full` never runs the JS tests.

## Every stage runs (except after a failed `uv sync`)
`run_checks` keeps going after a failed stage and returns all failures. The one exception is
`uv sync`: without dependencies every later stage would fail spuriously, so a failed sync stops
the run. The digest at the end
repeats each failed stage's error lines (`error_lines`: mypy/pyright `error:` lines, ruff
`-->` locations, eslint/stylelint `file:line:col`), capped per stage. The reason is recorded in
`src/kiss/tests/scripts/test_check_runs_all_stages.py`: an audit found agents re-running the
check once per failing stage, which cost a model step and about a minute each time. So fix
everything the report lists, then re-run once.

Exit code is 0 with "All checks passed!", otherwise 1.

## Pitfalls
- There is no `ruff format` stage, and `mdformat` was removed from the check (commit
  "remove mdformat checks/formatting from check and API doc generation"). Do not reformat whole
  files to satisfy a formatter the check does not run.
- `mypy src/` also type-checks `benchmarkings/harnesstax/*` because tests import it.
- htmlhint is blocking. `src/kiss/agents/vscode/.htmlhintrc` lists the template placeholders
  `{{BODY_CLASS_ATTR}}`, `{{ENTERKEYHINT}}` and `{{NONCE_ATTR}}` under `attr-lowercase`, so
  `media/chat.html` lints cleanly. (Older notes that call htmlhint advisory, `|| true`, are
  out of date: `lint:html` in `package.json` no longer has `|| true`.)
- Ruff: `line-length = 100`, `target-version = "py313"`, rules `E, F, W, I, N, UP`.
  `src = [".", "src", "projects"]` makes isort treat `swedefend` (in `projects/`) as first-party.
- pyright follows `pyrightconfig.json`. mypy config is in `[tool.mypy]` (Python 3.13, with
  per-module overrides for `kiss.tests.*` and `kiss.viz_trajectory.*`).
- A new version literal only has to be edited in `src/kiss/core/_version.py`. The check
  syncs `package.json` from it, and hatch reads the version from the same file
  (`[tool.hatch.version]`).

## Sources
- `src/kiss/scripts/check.py` (`main`, `run_checks`, `clean_build_artifacts`, `_sync_extension_version`, `error_lines`)
- `src/kiss/agents/vscode/package.json` (`typecheck`, `lint`, `lint:html` scripts), `src/kiss/agents/vscode/.htmlhintrc`
- `pyproject.toml` (`[project.scripts]`, `[tool.ruff]`, `[tool.mypy]`), `pyrightconfig.json`
- `src/kiss/tests/scripts/test_check_runs_all_stages.py`
