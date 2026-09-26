---
title: Browser automation area overview (WebUseTool, web_stealth)
uuid: e0bb1f18-eadd-435b-b800-9b543d77f38f
summary: Map of Sorcar's browser tool files (web_use_tool.py WebUseTool, web_stealth.py
  Patchright/Xvfb/challenge helpers), the nine web tools, and links to detail pages.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Browser automation area overview

Sorcar agents browse the web through one class, `WebUseTool`, which drives a Chromium through
the sync Playwright API (Patchright when installed). Pages are read as an accessibility (ARIA)
snapshot in which interactive elements get numeric `[N]` ids; the model acts on those ids.

## File map

| File | Role |
|---|---|
| `src/kiss/agents/sorcar/web_use_tool.py` | `WebUseTool`: launch, profile locking, AX-tree numbering, input with human-like pointer/typing, challenge waiting, show/hide window, crash and hang recovery |
| `src/kiss/agents/sorcar/web_stealth.py` | Helpers consumed by `WebUseTool`: `playwright_api`, `chrome_channel`, `virtual_display` (Xvfb), `mouse_path`, `typing_chunks`, `challenge_vendor`, `search_fallback_url` |
| `src/kiss/agents/sorcar/sorcar_agent.py` | Builds the tool in `SorcarAgent._get_tools` and closes it at the end of `run` |

## Tools exposed to the model

`WebUseTool.get_tools()` returns nine callables: `go_to_url`, `click`, `type_text`, `press_key`,
`scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser`. `close()` is not
exposed (it also stops Playwright and deletes an ephemeral profile). Every public tool returns a
string; failures are returned as `"Error <doing X>: <message>"` rather than raised
(`_try_ensure_browser`).

`go_to_url` also accepts `"tab:list"` (list tabs) and `"tab:N"` (switch to tab N, 0-based).
`click`, `type_text`, `press_key` and `go_to_url` return the fresh accessibility tree.

## Detail pages

- `web-accessibility-tree-ids`: how `[N]` ids are assigned and resolved back to elements.
- `web-browser-profile-and-launch`: persistent profile, `_N` escalation, locks, crash and hang recovery.
- `web-stealth-bot-protection`: Patchright, headed Chromium on Xvfb, challenge detection, Bing fallback.
- `web-show-browser-human-handoff`: `show_browser`, cookie carry-over, when to hand control to a person.
- `web-sorcar-integration`: when the agent gets web tools, sub-agent profiles, Docker interaction.

## Sources
- `src/kiss/agents/sorcar/web_use_tool.py` (`WebUseTool`, `WebUseTool.get_tools`, `WebUseTool.go_to_url`)
- `src/kiss/agents/sorcar/web_stealth.py`
