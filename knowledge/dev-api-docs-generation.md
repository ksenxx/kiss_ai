---
title: 'API.md generation: generate-api-docs allowlist and behavior'
uuid: bfe2316f-90cb-4ee2-b8a1-1c52fe138088
summary: How generate_api_docs.py builds API.md by AST from an allowlist (INCLUDE_FILES/INCLUDE_DIRS),
  runs in every uv run check, and why third_party_agents never appear.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# API.md generation (`generate-api-docs`)

`uv run generate-api-docs` (console script -> `kiss.scripts.generate_api_docs:main`) rewrites
the repo-root `API.md` (`OUTPUT = PROJECT_ROOT / "API.md"`). It is the second stage of every
`uv run check` run, so running the check can leave a modified `API.md` in the working tree.
Commit it together with the API change that caused it.

## How it works
- Pure AST parsing, no imports. `discover_modules` walks `src/kiss` (`KISS_SRC`) and keeps
  only files that pass `_is_included`:
  - `INCLUDE_FILES` (relative to `src/kiss`): `core/kiss_agent.py`,
    `agents/sorcar/relentless_agent.py`, `agents/sorcar/sorcar_agent.py`,
    `agents/sorcar/chat_sorcar_agent.py`, `agents/sorcar/worktree_sorcar_agent.py`,
    `agents/sorcar/daemon_client.py` (the synchronous client API `run` / `TaskResult`) and
    `server/sorcar.py`, which re-exports it;
  - `INCLUDE_DIRS`: `{Path("third_party_agents")}`.
- For a package `__init__.py` it honours `__all__` and follows re-exports to their defining file
  (`_parse_imports`, `_find_def_in_file`). Packages whose `__init__.py` mentions "deprecated"
  in its first 500 characters are skipped. `main` functions are skipped (`SKIP_FUNCTIONS`).
- Google-style docstrings are parsed into summary, Args and Returns
  (`_parse_google_docstring`), then rendered as Markdown with a heading per module, class and
  method (`generate_markdown`, `_render_class`, `_render_function`).

## Gotcha: `INCLUDE_DIRS` matches nothing
Paths are relative to `src/kiss`, and the channel agents live in
`src/kiss/agents/third_party_agents/`, so `Path("third_party_agents")` matches no file. There
is no `src/kiss/third_party_agents/`. As a result, `API.md` currently contains no
third-party (channel) agent entries. If they should be documented, the entry must be
`Path("agents/third_party_agents")`, and the resulting `API.md` diff needs review.

## Making something appear in API.md
Add its file to `INCLUDE_FILES`, give public functions and classes Google-style docstrings (the
project requires full docstrings on public methods), then run `uv run generate-api-docs`.
Tests: `src/kiss/tests/scripts/test_generate_api_docs.py` and the `generate_api_docs` case in
`src/kiss/tests/scripts/test_partial_branch_coverage.py`.

## Sources
- `src/kiss/scripts/generate_api_docs.py` (`INCLUDE_FILES`, `INCLUDE_DIRS`, `_is_included`, `discover_modules`, `generate_markdown`, `main`)
- `src/kiss/scripts/check.py` (`main`: "Generate API docs" stage)
- `API.md`
