---
title: 'Sorcar bundled user docs: TIPS.md, SAMPLE_TASKS.md and SORCAR.md'
uuid: 9ae2fafa-5cb7-4f67-872b-0644bdb6d9ef
summary: What src/kiss/TIPS.md (webview tips), src/kiss/SAMPLE_TASKS.md (welcome task
  chips) and SORCAR.md (user prefs inlined from ~/.kiss/SORCAR.md) are and who reads
  them.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Bundled user-facing docs

## `src/kiss/TIPS.md`
Fresh-install tips shown in the chat webview. The file is split into one tip per `# Tip` section, and
`{{PRODUCT_NAME}}` placeholders are substituted with the brand name. It is read by `getTips` in
`src/kiss/agents/vscode/src/SorcarTab.ts` and, for the remote webapp, by `kiss.server.tips`, which
`web_server._build_html` injects as `window.__TIPS__`. `KISS_TIPS_PATH` overrides the path (the tests use it).
The topics double as a feature list: `/ask` and slash commands (SEAs via `~/.kiss/SEAS.md`), the update
button, set_model and steering-on-the-fly, voice chat (talk), the remote web/mobile app, running tasks from Python
(`kiss.server.sorcar.run(..., chat_id=...)`), Docker, ssh servers, and merge conflicts.

## `src/kiss/SAMPLE_TASKS.md`
Sample task templates shown as suggestion chips on the welcome page, one per `## Task` section (e.g.
"Authenticate slack workspace <<workspace name>>", "Every 2 minutes, run a gateway tick on the Slack channel
sorcar, with pairing"). The VS Code extension combines them with the user's `~/.kiss/MY_TASK_TEMPLATES.md`
(`SorcarTab.ts`). The server deliberately does **not** broadcast suggestions, because an empty broadcast once
cleared the extension's chips (see the `web_server.py` comment and
`test_welcome_suggestions_not_broadcast.py`).

## `SORCAR.md`
The agent reads **`~/.kiss/SORCAR.md`** (`kiss_home() / "SORCAR.md"`) and inlines it at the end of every task's
system prompt (`RelentlessAgent.perform_task`). Use it for durable user preferences and pointers (the
repo-root `SORCAR.md` holds one such line: use `third_party_agents/govee.py` for home lights). The repo copy is
not loaded automatically, and SYSTEM.md no longer tells the agent to `Read("./SORCAR.md")` first (see
`sorcar-system-prompt-assembly`).

## Sources
- `src/kiss/TIPS.md`, `src/kiss/SAMPLE_TASKS.md`, `SORCAR.md`
- `src/kiss/server/tips.py`, `src/kiss/agents/vscode/src/SorcarTab.ts` (`getTips`)
- `src/kiss/agents/sorcar/relentless_agent.py` (`perform_task`)
