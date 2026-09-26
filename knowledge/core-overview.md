---
title: Core agent loop area overview (KISSAgent, Base, compaction, interrupts, printers)
uuid: 0d6717bc-80ac-497f-abf0-5e3fd498631f
summary: Map of the core agent loop files in src/kiss/core (kiss_agent.py, base.py,
  context_compaction.py, prompt_cache_keepalive.py, tool_interrupt.py, stop_signal.py,
  printers) and links to detail pages.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Core agent loop area overview

`KISSAgent` (in `src/kiss/core/kiss_agent.py`) is the ReAct loop that every higher-level agent
builds on. `RelentlessAgent`, `SorcarAgent` and the SEAs are covered in other areas. Here
the flow is: prompt, model call, tool calls, tool results, and so on until `finish`.

## File map

| File | Role | Detail page |
|---|---|---|
| `src/kiss/core/kiss_agent.py` | `KISSAgent.run`, agentic loop, tool execution, limits, usage accounting, model fallback | `core-agent-run-loop`, `core-tool-execution-and-hooks`, `core-budget-and-token-accounting`, `core-limits-and-context-handoff`, `core-errors-retry-and-model-fallback`, `core-finish-contract-and-implicit-finish`, `models-cli-backed-cc-codex` |
| `src/kiss/core/base.py` | `Base` class (id counter, printer setup, `messages` history, trajectory YAML save); loads `SYSTEM_PROMPT` / `SYSTEM_PROMPT_LITE` | `core-trajectory-persistence`, `sorcar-system-prompt-assembly` |
| `src/kiss/core/models/model.py` (`Model._function_to_openai_tool`) | Tool schema generation from Python functions (the only models file this area covers) | `core-tool-schema-from-python-functions` |
| `src/kiss/core/context_compaction.py` | Stubs old large tool outputs, gated by cache cost | `core-context-compaction` |
| `src/kiss/core/prompt_cache_keepalive.py` | Pings Anthropic's prompt cache during long tool calls | `models-prompt-caching` |
| `src/kiss/core/tool_interrupt.py` | Per-tool-call Stop (cooperative, then async exception injection) | `core-tool-call-interrupt` |
| `src/kiss/core/stop_signal.py` | Per-thread task stop event visible below the agent | `core-stop-signal` |
| `src/kiss/core/kiss_error.py` | `KISSError` hierarchy | `core-error-types` |
| `src/kiss/core/printer.py`, `print_to_console.py`, `html_render.py` | Printer ABC, Rich console printer, HTML to Rich text | `core-printers` |
| `src/kiss/agents/kiss.py` | Small ready-made agents (prompt refiner, docker bash agent, simple coding agent) | `core-kiss-helper-agents` |
| `src/kiss/SYSTEM.md`, `SYSTEM_LITE.md` | Default system prompts | `sorcar-system-prompt-assembly` |
| `src/kiss/INJECTIONS.md` | Bundled "Inject instruction" tricks for the UI | `core-injections-tricks` |

## Tests
These live in `src/kiss/tests/core/`: `test_kiss_agent.py`, `test_context_compaction.py`,
`test_prompt_cache_keepalive.py`, `test_tool_interrupt.py`,
`test_conc2026_kiss_agent_usage_atomicity.py`, `test_console_printer_thread_safety.py` and
`test_html_render_coverage.py`.

## Config knobs used by this area (`src/kiss/core/config.py`)
- `KISS_TOOL_OUTPUT_COMPACTION` (default on), `KISS_COMPACTION_START_TOKENS` (100000),
  `KISS_COMPACTION_STEP_TOKENS` (100000)
- `KISS_CONTEXT_LIMIT_FRACTION` (0.7). Values outside (0, 1] fall back to 0.7 in `kiss_agent.py`.

## Sources
- `src/kiss/core/kiss_agent.py` (`KISSAgent`)
- `src/kiss/core/base.py` (`Base`, `SYSTEM_PROMPT`)
- `src/kiss/core/config.py` (`Config`)
