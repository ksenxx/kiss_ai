---
title: Printers - Printer ABC, event types, ConsolePrinter, html_to_rich
uuid: 67da501d-3e6d-45d9-9e09-6a61a0bdcca6
summary: Printer ABC (print, token_callback, thinking_callback, reset), event types
  KISSAgent emits, thread-safe ConsolePrinter with usage offsets, html_to_rich result
  rendering.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Printers

## Interface (`src/kiss/core/printer.py`)
`Printer(ABC)` has these methods:
- `print(content, type="text", **kwargs) -> str` (abstract),
- `token_callback(token)` (abstract) for streamed text,
- `thinking_callback(is_start)` (no-op by default) to switch between the thinking and text panels,
- `reset()` (abstract) to clear streaming state between messages.

Implementations: `ConsolePrinter` (`core/print_to_console.py`), `JsonPrinter` (`server/json_printer.py`,
which broadcasts events to the VS Code / web clients and owns the stop check) and `WebPrinter(JsonPrinter)`
(`server/web_server.py`).

The module also has shared helpers: `parse_result_yaml` (returns a dict only when the YAML has `summary`),
`truncate_result` (removes base64 attachment payloads and keeps the first and last 1500 of
`MAX_RESULT_LEN = 3000` chars), `lang_for_path` / `LANG_MAP`, `extract_path_and_lang`, and `extract_extras`
(tool args other than `KNOWN_KEYS`).

## Printer selection (`Base.set_printer`)
An explicit printer always wins. Otherwise a `ConsolePrinter` is used unless `verbose is False`, in which
case there is none. The model's streaming callbacks are the printer's `token_callback` / `thinking_callback`,
bound in `_reset` and rebound after a fallback swap.

## Event types KISSAgent emits
| type | when | notable kwargs |
|---|---|---|
| `system_prompt` | `run()` with a system prompt; fallback-swap notice | |
| `prompt` | after the template is filled (if `print_prompts`) | |
| `usage_info` | after each agentic-step model response (`_execute_step`, not `_generate_once`) | `total_tokens`, `cost`, `total_steps`, `cache_read`, `model` |
| `tool_call` | before each tool | `tool_input`, `call_id` (used for per-panel Stop) |
| `tool_result` | after each tool or block | `tool_name`, `tool_input`, `is_error`, `interrupted` |
| `result` | run end | `step_count`, `total_tokens`, `cost` |

`ConsolePrinter` also handles `text`, `message`, `bash_stream` and `notification`.

## ConsolePrinter
- It is shared across threads: `run_tasks_parallel` passes one printer to every sub-agent, and the live
  usage monitor prints from another thread. Every public entry point holds an `RLock` for its whole body,
  so a panel is printed as one unit.
- The state that belongs to the printing agent (`bash_streamed`, `block_type`, `tokens_offset`,
  `budget_offset`, `steps_offset`) is **thread-local** (`_PrinterThreadState`). The terminal cursor
  (`_mid_line`) is shared.
- Offsets: `RelentlessAgent` and Sorcar set `printer.tokens_offset` / `budget_offset` so the Result panel
  includes spend from earlier sessions and sub-agents. ConsolePrinter applies the offsets to `result`
  panels only; the `usage_info` line is printed as received.
- The `result` event renders a green "Result" panel. When the content is finish YAML it shows "Status:
  Continue" or "Status: FAILED", then the HTML summary through `html_to_rich`, then "Suggested next: ...". Other content is rendered as Markdown.
- Read tool results can get syntax highlighting (`_should_syntax_highlight_read`).

## `html_to_rich` (`src/kiss/core/html_render.py`)
`ensure_html` guarantees that finish summaries are HTML, so the console needs an HTML renderer. It is a
stdlib `HTMLParser` producing `rich.text.Text`: headings and bold render bold, emphasis italic, code is
highlighted, list items get bullets, links show their targets, and `script`/`style`/`head` are skipped.
It falls back to plain `Text(html)` on a parser exception.

## Sources
- `src/kiss/core/printer.py`, `src/kiss/core/print_to_console.py` (`ConsolePrinter`, `_PrinterThreadState`), `src/kiss/core/html_render.py` (`html_to_rich`)
- `src/kiss/core/base.py` (`Base.set_printer`)
- `src/kiss/tests/core/test_console_printer_thread_safety.py`, `test_html_render_coverage.py`
