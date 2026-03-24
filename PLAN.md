# Plan: Remove Web-Based Sorcar, Retain VSCode Extension Code

## Overview

The web-based Sorcar UI (`sorcar.py` + `chatbot_ui.py`) runs a Starlette/uvicorn server with an embedded code-server iframe and a full browser-based chat interface. We want to remove this web UI entirely while keeping everything the VS Code extension (`src/kiss/agents/vscode/`) and other agents depend on.

## Current File Analysis (src/kiss/agents/sorcar/)

| File | Lines | Purpose | Used by VSCode? | Used by Others? | Action |
|---|---|---|---|---|---|
| `sorcar.py` | 1425 | Web chatbot UI: `run_chatbot()`, `main()`, Starlette routes, SSE, code-server management, `_auto_update()` | **No** | **No** | **DELETE** |
| `chatbot_ui.py` | 2111 | HTML/CSS/JS templates for web chatbot (`_build_html()`, `CHATBOT_CSS`, `CHATBOT_JS`, `_THEME_PRESETS`, etc.) | **No** | **No** (only `sorcar.py` imports from it) | **DELETE** |
| `browser_ui.py` | 747 | `BaseBrowserPrinter`, `find_free_port()`, CSS/JS constants (`BASE_CSS`, `OUTPUT_CSS`, `EVENT_HANDLER_JS`, `HTML_HEAD`) | `BaseBrowserPrinter` (vscode `server.py` subclasses it) | tests | **KEEP** (but remove web-only CSS/JS/HTML constants and `find_free_port()`) |
| `code_server.py` | 1033 | `_setup_code_server()`, `_scan_files()`, `_git()`, `_parse_diff_hunks()`, `_prepare_merge_view()`, `_snapshot_files()`, etc. | **Yes** (many functions imported) | tests | **KEEP** (remove `_setup_code_server()`, `_CS_SETTINGS`, `_CS_STATE_ENTRIES`, `_CS_EXTENSION_JS`; keep `_log_exc()` — used by many remaining functions) |
| `sorcar_agent.py` | 357 | `SorcarAgent` class | **Yes** | `coding_agents`, `channels`, `create_and_optimize_agent` | **KEEP** |
| `task_history.py` | 574 | SQLite task/model/file history | **Yes** | tests | **KEEP** |
| `shared_utils.py` | 139 | `model_vendor()`, `generate_followup_text()`, `rank_file_suggestions()`, `clip_autocomplete_suggestion()`, `clean_llm_output()` | **Yes** | **No** | **KEEP** |
| `useful_tools.py` | 408 | `UsefulTools` (Bash, Read, Edit, Write) | Indirectly via `SorcarAgent` | `relentless_agent` | **KEEP** |
| `web_use_tool.py` | 476 | `WebUseTool` (Playwright browser automation) | Indirectly via `SorcarAgent` | **No** | **KEEP** |
| `config.py` | 42 | Pydantic config for SorcarAgent | Indirectly via `SorcarAgent._reset()` | **No** | **KEEP** |
| `__init__.py` | 3 | Imports `config` | **Yes** | **Yes** | **KEEP** |
| `SORCAR.md` | 329 | Docs for web UI | - | - | **DELETE** |

## Step-by-Step Plan

### Step 1: Delete web-only files

Delete these files entirely:

- `src/kiss/agents/sorcar/sorcar.py` — the entire web chatbot UI server
- `src/kiss/agents/sorcar/chatbot_ui.py` — HTML/CSS/JS templates for web UI (nothing outside sorcar.py imports from it)

### Step 2: Clean up `browser_ui.py`

`browser_ui.py` contains:

1. `BaseBrowserPrinter` — **KEEP** (used by vscode `VSCodePrinter`, and tests)
1. `_DISPLAY_EVENT_TYPES`, `_coalesce_events()` — **KEEP** (used by `BaseBrowserPrinter.stop_recording()`)
1. `find_free_port()` — **DELETE** (only used by deleted `sorcar.py`; no non-test consumers remain)
1. `BASE_CSS`, `OUTPUT_CSS`, `EVENT_HANDLER_JS`, `HTML_HEAD` — **DELETE** (only imported by deleted `chatbot_ui.py` and `sorcar.py`)

### Step 3: Clean up `code_server.py`

Functions used by vscode `server.py` (all **KEEP**):

- `_capture_untracked`, `_cleanup_merge_data`, `_git`, `_merge_data_dir`, `_parse_diff_hunks`, `_prepare_merge_view`, `_save_untracked_base`, `_snapshot_files`
- `_scan_files` (imported inside a method)

Functions/constants only used by deleted `sorcar.py` or by `_setup_code_server` itself:

- `_setup_code_server()` — **DELETE** (only called from deleted `sorcar.py`)
- `_CS_SETTINGS` — **DELETE** (only used by `_setup_code_server`)
- `_CS_STATE_ENTRIES` — **DELETE** (only used by `_setup_code_server`)
- `_CS_EXTENSION_JS` — **DELETE** (only used by `_setup_code_server` at lines 600, 602)
- `_log_exc()` — **KEEP** (used by 7+ remaining functions in code_server.py)

Functions only used by deleted `sorcar.py` and tests:

- `_restore_merge_files()` — **DELETE** (only production consumer was deleted `sorcar.py`; remaining uses are only in tests)

### Step 4: Update `pyproject.toml`

Remove:

- The `[project.scripts]` entry: `sorcar = "kiss.agents.sorcar.sorcar:main"`
- The `[project.optional-dependencies]` `sorcar` group: BUT `playwright>=1.40.0` is still needed by `web_use_tool.py` (which is kept). Either:
  - Move `playwright>=1.40.0` to another deps group (e.g. main deps or a new group), **or**
  - Rename/restructure the optional deps to keep `playwright` while removing `uvicorn` and `starlette`
  - `uvicorn>=0.30.0` — **SAFE TO REMOVE** (only used in deleted `sorcar.py`)
  - `starlette>=0.38.0` — **SAFE TO REMOVE** (only used in deleted `sorcar.py`)

### Step 5: Update `__init__.py`

No changes needed — it only imports `config` which is retained.

### Step 6: Delete or update test files

#### Tests under `src/kiss/tests/agents/sorcar/` that import from deleted files:

These test files import from `sorcar.py` (the deleted web UI) and should be **DELETED**:

- `test_sorcar_integration.py` — imports `run_chatbot` from `sorcar.py`
- `test_sorcar_cs_integ.py` — imports `run_chatbot` from `sorcar.py`
- `test_sorcar_race_conditions.py` — imports from `sorcar.py`
- `test_stop_agent_thread.py` — imports `_StopRequested` from `sorcar.py`

These test files import from `chatbot_ui.py` (the deleted templates) and should be **DELETED**:

- `test_chatbot_ui.py` — tests chatbot UI rendering
- `test_vscode_panel.py` — imports `chatbot_ui` module directly
- `test_chat_history_events.py` — imports `CHATBOT_JS`
- `test_code_review_fixes.py` — imports `CHATBOT_JS`
- `test_run_prompt_button.py` — imports `CHATBOT_CSS`, `CHATBOT_JS`, `CHATBOT_THEME_CSS`
- `test_sorcar_run_selection.py` — imports `CHATBOT_JS`
- `test_sorcar_coverage.py` — imports `_THEME_PRESETS` and `CHATBOT_JS` (via other imports)
- `test_scm.py` — imports `CHATBOT_JS`
- `test_code_server.py` — imports `CHATBOT_JS`

These test helper files import from deleted `sorcar.py` and should be **DELETED**:

- `_sorcar_test_server.py` — imports `sorcar` module
- `_sorcar_test_server_with_cov.py` — wraps `_sorcar_test_server.py`
- `_sorcar_merge_test_server.py` — likely imports sorcar module

This test file imports `sorcar` module directly and should be **DELETED**:

- `test_sse_reconnection.py` — imports `from kiss.agents.sorcar import sorcar`

These test files do NOT import from deleted files and should be **KEPT**:

- `test_sorcar_agent.py` — tests `SorcarAgent` (kept)
- `test_task_history.py` — tests `task_history` (kept)
- `test_useful_tools.py` — tests `UsefulTools` (kept)
- `test_web_use_tool.py` — tests `WebUseTool` (kept)
- `test_merge_view.py` — **EDIT**: remove `_restore_merge_files` import and 4 tests that use it (`test_restore_handles_subdirectory_files`, `test_pending_merge_json_detected_after_restore_returns_zero`, `test_no_false_positive_without_pending_merge`, `test_empty_pending_merge_no_hunks`); keep 3 tests that only use kept functions (`test_deleted_file_excluded_from_merge`, `test_deleted_file_excluded_but_modified_file_kept`, `test_pre_existing_diff_excluded_on_second_task`)
- `test_print_to_browser.py` — tests `BaseBrowserPrinter` (kept)
- `test_ask_user.py` — tests ask user functionality (kept)
- `test_current_editor_file.py` — tests SorcarAgent parameter (kept)
- `test_file_usage.py` — tests file usage tracking (kept)
- `test_vscode_server.py` — tests vscode server (kept)
- `test_vscode_stop.py` — tests vscode stop (kept)
- `integration_test_*.py` — all import from kept files (`SorcarAgent`, `WebUseTool`) → **KEEP**

#### Tests under `src/kiss/tests/core/` that need updating:

- `test_coverage_integration.py` — remove test `test_atomic_write_text` (imports `_atomic_write_text` from deleted `sorcar.py`); remove test `test_find_free_port` (imports deleted `find_free_port`); remove test `test_restore_merge_files` (imports deleted `_restore_merge_files`); keep tests for `task_history`, `UsefulTools`, `BaseBrowserPrinter`, `_snapshot_files`
- `test_race_conditions.py` — remove test that imports `_atomic_write_text` from deleted `sorcar.py`; keep `BaseBrowserPrinter` tests
- `test_uncovered_branches.py` — **EDIT**: remove `test_restore_merge_files_no_data` (imports deleted `_restore_merge_files`); keep all other tests
- `test_printer_parity.py` — keep (only imports `BaseBrowserPrinter` which is kept)

### Step 7: Verify no remaining dead code

After all deletions, grep for any remaining imports of deleted symbols:

- `from kiss.agents.sorcar.sorcar import ...`
- `from kiss.agents.sorcar.chatbot_ui import ...`
- `from kiss.agents.sorcar.browser_ui import find_free_port`
- `from kiss.agents.sorcar.browser_ui import BASE_CSS, OUTPUT_CSS, EVENT_HANDLER_JS, HTML_HEAD`
- `from kiss.agents.sorcar.code_server import _setup_code_server, _CS_SETTINGS, _CS_STATE_ENTRIES, _CS_EXTENSION_JS, _restore_merge_files`

### Step 8: Run tests

Run `uv run check --full` and fix any lint/type errors. Run the full test suite to ensure nothing is broken.

## Summary of Changes

| Action | Target |
|---|---|
| **DELETE file** | `sorcar.py` |
| **DELETE file** | `chatbot_ui.py` |
| **DELETE file** | `SORCAR.md` |
| **EDIT** `browser_ui.py` | Remove `BASE_CSS`, `OUTPUT_CSS`, `EVENT_HANDLER_JS`, `HTML_HEAD`, `find_free_port()` |
| **EDIT** `code_server.py` | Remove `_setup_code_server()`, `_CS_SETTINGS`, `_CS_STATE_ENTRIES`, `_CS_EXTENSION_JS`, `_restore_merge_files()` (keep `_log_exc()`) |
| **EDIT** `pyproject.toml` | Remove `sorcar` script entry; remove `uvicorn`/`starlette` from optional deps; keep `playwright` |
| **DELETE tests** | ~14 test files under `tests/agents/sorcar/` that test web UI (see Step 6) |
| **DELETE test helpers** | `_sorcar_test_server.py`, `_sorcar_test_server_with_cov.py`, `_sorcar_merge_test_server.py` |
| **EDIT tests** | `test_coverage_integration.py` — remove `test_atomic_write_text`, `test_find_free_port`, `test_restore_merge_files` |
| **EDIT tests** | `test_race_conditions.py` — remove test importing `_atomic_write_text` |
| **EDIT tests** | `test_merge_view.py` — remove 4 tests using deleted `_restore_merge_files` |
| **EDIT tests** | `test_uncovered_branches.py` — remove `test_restore_merge_files_no_data` |
| **NO CHANGE** | `sorcar_agent.py`, `task_history.py`, `shared_utils.py`, `useful_tools.py`, `web_use_tool.py`, `config.py`, `__init__.py` |
| **NO CHANGE** | `src/kiss/agents/vscode/` (all files untouched) |
| **NO CHANGE** | External consumers: `coding_agents/`, `channels/`, `create_and_optimize_agent/`, `relentless_agent.py` |

## Risk Assessment

- **Low risk:** Deleting `sorcar.py` and `chatbot_ui.py` — these are the web UI entry point and templates, not used by vscode or other agents.
- **Low risk:** Removing CSS/JS/HTML constants from `browser_ui.py` — only used by deleted files.
- **Low risk:** Removing `find_free_port()` — only used by deleted `sorcar.py`.
- **Low risk:** Removing `_setup_code_server()` and related constants from `code_server.py` — only called from deleted `sorcar.py`.
- **Medium risk:** Removing `uvicorn`/`starlette` deps — verify no other module needs them (confirmed: no other module imports them).
- **Must verify:** `playwright` dependency must be preserved (used by kept `web_use_tool.py`).
- **Must verify:** Each test file listed for deletion — confirm it doesn't test any kept functionality.
