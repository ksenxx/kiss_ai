# KISS Sorcar: Complete Feature Inventory

Version 2026.9.18 · repository `/home/ksen/kiss` (4,496 commits) · task database `~/.kiss/sorcar.db` · compiled 2026-09-20

This report enumerates everything KISS Sorcar can do. Every item was checked against the source tree, not only the README: the tool functions registered in `src/kiss/agents/sorcar/sorcar_agent.py`, the daemon command handlers in `src/kiss/server/`, the browser client in `src/kiss/agents/vscode/media/`, the 43 channel-agent modules, the CLI entry points in `pyproject.toml`, and the system prompt in `src/kiss/SYSTEM.md`. The last section turns to the recorded trajectories in `sorcar.db` to show which of these capabilities are actually used and how hard they have been pushed.

**Contents**

1. [At a glance](#1-at-a-glance)
2. [Architecture](#2-architecture)
3. [Interfaces and prompt surfaces](#3-interfaces-and-prompt-surfaces)
4. [Agent runtime](#4-agent-runtime)
5. [Built-in tools](#5-built-in-tools)
6. [Models and multi-model routing](#6-models-and-multi-model-routing)
7. [Chat client features](#7-chat-client-features)
8. [Voice](#8-voice)
9. [Persistent memory](#9-persistent-memory-memoryfield)
10. [Task classifier](#10-task-classifier)
11. [Git integration and worktrees](#11-git-integration-and-worktrees)
12. [Scheduled automations](#12-scheduled-automations)
13. [Third-party agents and credential isolation](#13-third-party-agents-and-credential-isolation)
14. [Extension points: MCP, skills, SEAs](#14-extension-points-mcp-skills-seas)
15. [Persistence, accounting, observability](#15-persistence-accounting-observability)
16. [Installation, deployment, operations](#16-installation-deployment-operations)
17. [Long-horizon research drivers and benchmarks](#17-long-horizon-research-drivers-and-benchmarks)
18. [Developer tooling and tests](#18-developer-tooling-and-tests)
19. [What the trajectories show](#19-what-the-trajectories-show)
20. [Caveats and gaps](#20-caveats-and-gaps)

## 1. At a glance

- **665** models in the bundled catalog, 9 provider categories
- **43** third-party agents (32 messaging, 9 service, 2 infrastructure)
- **50** CLI entry points in `pyproject.toml`
- **~40** built-in agent tools (plus per-channel, MCP and skill tools)
- **6** prompt surfaces: VS Code, web/mobile, Python API, terminal, chat channels, OpenAI/A2A endpoints
- **1,228** Python test files, 8,669 test functions, plus a JS test suite for the extension
- **24,063** task rows in `sorcar.db` (5,648 top-level, 18,415 sub-agent)
- **$52.8K** recorded model spend across 35.3 billion tokens and 321K steps

## 2. Architecture

One local daemon, `kiss-web`, hosts every agent and serves every client. The agent is a five-layer class stack; each layer adds one concern. Lines of code below are from `wc -l` on 2026-09-20.

```
  VS Code extension   Web / mobile app   Python API             Chat channels        OpenAI-compat / A2A   Voice
  (TypeScript,        (cloudflared       (kiss.server.          (Telegram, Slack,    (kiss-oai, kiss-a2a)  (wake word
   webview)            tunnel)            sorcar.run)            email, ...)                                "Sorcar")
        |                  |                  |                      |                      |                 |
        +------------------+------------------+----------+-----------+----------------------+-----------------+
                                                         v
  +----------------------------------------------------------------------------------------------------------+
  | kiss-web daemon (src/kiss/server, 30,138 LoC)                                                            |
  |   Unix-socket JSON protocol . WebSocket web server . tab registry (mirrored tabs) . task runner & stop    |
  |   machinery . autocomplete (ghost text, @paths) . merge flow . Explorer/SCM providers . voice wake        |
  |   control . talk player . cron scheduler thread . SEA loader . tools-file loader . stall watchdog .       |
  |   update checker . remote password + cloudflared tunnel                                                   |
  +----------------------------------------------------------------------------------------------------------+
                                                         v
  +-------------------------------------------------------------------+  +-----------------------------------+
  | WorktreeSorcarAgent (1,916): git worktree per task, auto-commit,   |  | Model adapters                    |
  |                             merge / discard / leave-as-is          |  | (src/kiss/core/models)            |
  | ChatSorcarAgent     (727):  chat history & digest, sorcar.db       |  |   Anthropic . OpenAI-compatible   |
  |                             persistence, stop/steer                |  |   (2 generations) . Gemini .      |
  | SorcarAgent         (3,333): tools, browser, run_parallel,         |  |   Claude Code CLI . Codex CLI .   |
  |                             run_agent, memory, classifier, talk    |  |   Decisions (Jev) .               |
  | RelentlessAgent     (1,605): auto-continuation across context      |  |   MODEL_INFO.json (665)           |
  |                             exhaustion, usage ledger               |  +-----------------------------------+
  | KISSAgent           (1,201): native function calling,              |  | Side systems                      |
  |                             budget/step/context limits, retries,   |  |   memoryfield (vector index of    |
  |                             fallback                               |  |   Markdown pages) . Playwright    |
  +-------------------------------------------------------------------+  |   browser . Docker manager . MCP  |
                                    v                                    |   client . skills . Muse-auth     |
  +-----------------------------------------------------------------+    |   vault daemon                    |
  | 43 channel agents (SEAs) . cron agent . MCP servers . skills .  |    +-----------------------------------+
  | user tools files . custom SEAs                                  |
  +-----------------------------------------------------------------+
```

*Figure 1. Every surface talks to one daemon; the daemon builds an agent from the five-layer stack and gives it tools. Channel agents and the cron agent are themselves Sorcar Extension Agents dispatched by the `run_agent` tool.*

## 3. Interfaces and prompt surfaces

| Surface | What it does | Where |
|---|---|---|
| **VS Code extension** | Sidebar chat or editor-tab chats (`kissSorcar.editorTabsMode`); commands: open panel, new conversation, open settings, stop task, generate commit message, git commit, toggle focus, run selection, insert selection to chat, show history. Auto-starts the daemon; installs its own dependencies (`DependencyInstaller.ts`); update checker with snooze; macOS launchd integration. | `src/kiss/agents/vscode/` |
| **Web / mobile app** | The same chat UI served by the daemon over WebSocket; optional cloudflared tunnel plus password (`remote_password`); desktop mode adds a Task Info sidebar (tokens, cost, steps, elapsed, machine, work dir, budget, live `tmp/PROGRESS.md`); activity bar with Explorer, Source Control and Tasks views. | `src/kiss/server/web_server.py` (9,259 LoC) |
| **Python client API** | `kiss.server.sorcar.run(prompt, …)` blocks until a daemon task finishes and returns `TaskResult` (text, success, cost, tokens, steps, chat_id). Options: model, work_dir, scope_work_dir, chat_id, use_worktree, auto_commit, max_budget, model_config, use_web_tools, classify_tasks, use_memory, is_parallel, timeout, stop_on_timeout, sock_path, parent_task_id/parent_tab_id, tools file, system_prompt, append_to_system_prompt, append_to_prompt, append_basic_tools, extension_agent_path. | `src/kiss/server/sorcar.py`, `daemon_client.py` |
| **`sorcar` terminal command** | Runs a SorcarAgent in the current directory without the daemon: `sorcar -t "task"` or `sorcar -f task.txt`; flags for model, budget, work dir; answers `ask_user_question` in the terminal. | `sorcar_agent.py:main` |
| **`kiss-web` daemon CLI** | `kiss-web`, `--workdir DIR`, `--url` (print the tunnel URL). | `web_server.py:main` |
| **Chat channels as inbound surfaces** | Gateway mode: a poll tick drains new messages from Telegram, Slack, Discord, email, WhatsApp, … and runs each as a task; thread continuity, delivery ledger, per-channel model/budget, `--allow-users`, pairing handshake (`--pairing`, `--approve`, `--list-pending`). | `third_party_agents/_channel_cli.py` |
| **OpenAI-compatible server** | `kiss-oai` exposes the daemon behind an OpenAI chat API so Open WebUI, LibreChat or an `openai` SDK script become chat surfaces. | `openai_compat_agent.py` |
| **A2A agent** | `kiss-a2a` publishes an agent card at `/.well-known/agent-card.json`, accepts JSON-RPC `message/send` and `tasks/get`, and can call peer agents outbound. | `a2a_agent.py` |
| **Voice** | Wake word "Sorcar" via the mic button, speech-to-text, spoken replies through `talk`; also steers a running task. | `server/voice_wake*.py`, `media/voice.js` |
| **Per-channel CLIs** | `kiss-slack`, `kiss-gmail`, `kiss-whatsapp`, `kiss-telegram`, … 41 channel commands plus `kiss-cron`, `kiss-web`, `sorcar`, `check`, `generate-api-docs`, `swedefend-eval`. | `pyproject.toml [project.scripts]` |

## 4. Agent runtime

### KISSAgent (`src/kiss/core/kiss_agent.py`)

- Native function calling against every adapter; tool schemas are built once per tool set and rebuilt on model switch.
- Hard limits per run: `max_budget` (USD), `max_steps`, and context length (`CONTEXT_LIMIT_FRACTION` of the model's window); raises `BudgetExceededError` / `ContextWindowExceededError`.
- Retries with backoff; non-retryable errors (bad key, permission denied, low credit, model gated) trigger a one-shot **fallback model** registered in `MODEL_INFO.json`, copying the conversation so no context is lost.
- **Tool-output compaction** (`context_compaction.py`): once the context passes 100k tokens, old tool results longer than 2,000 chars are replaced by 200-char stubs, keeping the 20 newest; repeated every 50k of growth.
- Read dedupe and outlines: re-reading an unchanged range returns a one-line note; files over 2,000 lines return an outline (config flags `read_dedupe`, `read_outline_lines`).
- **Implicit finish**: a model that stops calling tools is wrapped into a finish result instead of erroring.
- Per-tool-call interrupt (`tool_interrupt.py`): the Stop button on one panel aborts that tool call while the task continues; the model sees `USER_INTERRUPTED_MESSAGE`.
- Streamed responses that go quiet are aborted and retried (`stream_abort.py`, `stream_stall_timeout`).
- Image and PDF attachments travel with the prompt (`Attachment`; HEIC/HEIF converted, oversized images re-encoded).
- Prompt caching on Anthropic (`cache_control: ephemeral`); usage/cost tracked per response from catalog prices.
- Non-agentic single-shot mode (used by the classifier, commit-message generation and TTS).

### RelentlessAgent (`relentless_agent.py`)

- Auto-continuation: when a session exhausts its context the agent calls `finish(is_continue=True)` with a structured progress summary and a fresh sub-session resumes from it (up to `max_sub_sessions`, default 10,000). Prior-session summaries are capped and re-injected.
- Append-only **usage ledger** with epochs so spend from sub-agents, abandoned children, classifier calls and TTS is counted exactly once even across stop-interrupts.
- Optional `docker_image`: Bash/Read/Write/Edit execute inside a container (`docker_manager.py`, `docker_tools.py`) with streaming output, per-exec kill tokens and output caps.
- Host settings (IP, machine, OS, PID, user id, chat/task ids) are placed in the system prompt.

### SorcarAgent (`sorcar_agent.py`)

- Builds the tool set (section 5), the Playwright browser, the memory tools, the classifier, `run_parallel` fan-out and `run_agent` dispatch.
- **Tool profiles**: `full`, `review` (Bash, Read, run_commands_parallel, memory reads, decide, summary) and `shell` (Bash, Read, run_commands_parallel); reviewer children get `review` automatically.
- **Review quota** (`fanout_guard.py`): at most 3 review rounds per task tree; "at most N% of the budget for reviewing" in the prompt clips reviewer budgets; sub-agent budgets are split from the parent's remainder with a $0.50 floor.
- Sub-agents open their own tabs, are persisted with `parent_task_id`, and their spend folds into the parent.
- Auto-commit with an LLM-generated commit message (`commit_message.py`) after non-worktree tasks.

### ChatSorcarAgent and WorktreeSorcarAgent

- Chat continuation: earlier tasks of the same `chat_id` are replayed as context, with a digest for older entries (`chat_history_digest`).
- Every event (system prompt, prompt, thinking, text, tool call, tool result, usage, result) is persisted to `sorcar.db` for live mirroring, resume and replay.
- Worktree isolation per task (section 11).

## 5. Built-in tools

Tools the model can call in a standard run, grouped by concern. Names are the exact function names registered in `SorcarAgent._get_tools`.

| Group | Tools | Notes |
|---|---|---|
| Files and shell | `Bash`, `Read`, `Write`, `Edit`, `run_commands_parallel` | Bash has per-call timeouts and output caps; parent-repo path guard rewrites paths into the active worktree; `run_commands_parallel` runs shell commands in threads with no LLM. |
| Browser | `go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser` | Playwright Chromium, headless by default, numbered accessibility tree, persistent profile, watchdog kill of stuck processes; `show_browser` makes the window visible for user login. |
| Delegation | `run_parallel(tasks, max_workers, model_name, tool_profile)`, `number_of_cores`, `run_agent(agent, task, model_name, max_budget, timeout, workspace)` | `run_parallel` spawns LLM sub-agents (nesting capped at 2); `run_agent` dispatches a channel agent, `cron`, or any agent-script `.py` file. |
| Decisions | `decide(state, questions)` | Typed `noul`/`choice`/`score` questions answered by OpenRouter's decisions model (Jev) with calibrated probabilities; present only when an OpenRouter key exists. |
| Memory | `memory_search`, `memory_pull`, `memory_read`, `memory_write`, `memory_list`, `memory_refresh`, `memory_delete` | Section 9. |
| User interaction | `ask_user_question`, `talk(language, text, emotion)` | Question modal in every client; spoken answer synthesized server-side. |
| Control | `set_model`, `summary`, `finish(success, is_continue, summary_in_html, suggested_next_task)` | `set_model` swaps the model mid-run and carries the conversation over; `summary` collapses the last ten steps in the UI. |
| Skills and MCP | `skill(name)`, `<server>_<tool>` | `skill` appears only when user or project skills exist; MCP tools are generated from each server's schema. |
| Channel tools (dispatched runs) | e.g. `check_slack_auth`, `authenticate_slack`, `post_message`, `read_messages`, `list_messages`, `send_whatsapp_message`, `gdrive_upload_file`, `github_search_repositories`, `cron_job` | Added when the run is a channel or cron agent. |

## 6. Models and multi-model routing

- **Catalog**: 665 models in `MODEL_INFO.json` with prices, context lengths and capability flags (`fc`, `gen`, `emb`, `dec`): OpenAI 102, Anthropic 14, Gemini 24, Together AI 103, Z.AI 8, Moonshot AI 10, OpenRouter 382, Claude Code CLI `cc/*` 14, Codex CLI `codex/*` 8. Rolling aliases such as `openrouter/~anthropic/claude-opus-latest`.
- **Adapters**: Anthropic native, OpenAI-compatible (Chat Completions and Responses API), Gemini, Claude Code CLI, Codex CLI, Decisions. Reasoning-effort variants (`-low/-medium/-high/-xhigh`) are catalog entries.
- **Mix vendors inside one task**: `set_model` mid-run, `run_parallel(model_name=…)` for sub-agents, per-channel and per-cron model overrides, model picker per tab, live model switch from the UI while a task runs.
- **Routing by prompt**: bundled "tricks" pair a builder model with a read-only reviewer (e.g. `claude-fable-5-1` + `gpt-5.6-sol`); a trick reads `./ROUTING.md` and asks the agent to update the routing strategy after each task.
- **Custom models**: `~/.kiss/MY_MODELS.json` entries (local vLLM/Ollama or any OpenAI-compatible endpoint) appear in the picker; custom endpoint, API key and HTTP headers per run via `model_config` or the Settings panel.
- **Catalog maintenance**: `update_models.py` fetches vendor pricing, tests generation/embedding/function-calling/decisions, and rewrites the catalog; `update_responses_api_support.py` live-verifies Responses API support; an "Update models" button in Settings.
- API keys live in `$KISS_HOME/api_keys.env`, settable from the Settings panel or environment.

## 7. Chat client features

Verified against element ids in `media/chat.html` and message types in `media/main.js`.

- **Input**: `@` file/folder mentions with ranked completion; ghost-text autocomplete from input history and frequent tasks; attachments by picker, paste or drag-and-drop (images, PDFs); input history; clear; tricks panel (promptlets inserted into the input); tips panel; sample-task chips on the welcome screen; "Suggested next" follow-up after each task.
- **Run controls**: model picker with search, per-task budget cap, Stop, per-panel tool interrupt, live steering (send a message into a running task), `<task>…</task>` to queue a follow-up task, model switch during a run.
- **Rendering**: streamed text and thinking, collapsible tool panels, syntax-highlighted code (highlight.js, light/dark theme toggle), inline images produced by tools, clickable file paths that open in the editor (existence verified via `checkPaths`), copy buttons, nested sub-agent panels, collapsed groups after every `summary`.
- **Tabs**: server-canonical tab registry mirrored across every VS Code window and web client on the same workspace; sub-agents open their own tabs; adjacent-task navigation within a chat.
- **History**: search, filters (workspace, running/completed/errors, date range, favorites), resume a chat, open a task from history, frequent-tasks panel, delete frequent task, share a chat or selected tasks as a standalone HTML page (`share.js`, e.g. `reports/chat-*.html`).
- **Workspace views** (web app): Explorer tree with New File/Folder, Rename, Delete, Cut/Copy/Paste, Find in Folder, Compare; Source Control view with branch, changes, commit graph, `git show`; file open/save from the browser.
- **Git actions**: autocommit button and progress, worktree merge/discard/leave prompts, main-tree actions, generate commit message.
- **Settings panel**: work dir, default max budget, worktree on/off, auto-commit, classify tasks, classify with Jev, web tools on/off, memory on/off and memory directory, editor-tabs mode, voice sensitivity and auto-submit, remote password, remote URL, API keys, custom endpoint/headers/model, custom models list, Update button, Update models, Server reset.
- **Status**: bottom bar with steps, tokens, budget, machine; Task Info drawer with task id, chat id, parent, model, date, time, work dir, worktree and parallel flags; daemon status and update-available notices.

## 8. Voice

- Wake-word listener (`voice_wake.py`, Vosk model auto-downloaded) runs as a daemon child process; fuzzy match on "Sorcar" (Levenshtein), sensitivity levels, silence trimming, speaker identification, language detection.
- Speech-to-text of the captured utterance through a catalog model; transcripts arrive as `Speaker #N says in the language xx that: …` (114 such top-level tasks recorded).
- Spoken replies: the `talk` tool synthesizes MP3 server-side with `gpt-audio-1.5` (emotion hint supported), broadcast to every client tab of the task, with a system-TTS fallback on the daemon (`talk_player.py`).
- Acknowledgement sound (`working-on-it.mp3`) and a listening overlay in the UI; voice auto-submit setting.

## 9. Persistent memory (memoryfield)

- Markdown pages with generated frontmatter under `~/.kiss/memories` (or a custom directory), indexed in SQLite with vector embeddings: OpenAI embeddings when a key exists, otherwise an offline hashed embedder.
- Seven tools (search, pull, read, write, list, refresh, delete); `memory_refresh` re-indexes external edits and reports near-duplicates and stale pages.
- A memory protocol in the system prompt: recall before work, record durable knowledge, never store secrets, keep per-session notes in `./tmp/PROGRESS.md`.
- Disabled for Docker runs, `cc/*`/`codex/*` models, runs without the built-in toolset, or runs with a caller-supplied system prompt; toggle with `KISS_USE_MEMORY` or the settings panel.
- `memoryfield/evaluate.py` measures Recall@k and MRR on real past tasks from `sorcar.db`.

## 10. Task classifier

- Runs once before each task to decide (a) whether the task edits files (worktree needed) and (b) whether it is simple (use `SYSTEM_LITE.md`, 4.1 KB, instead of `SYSTEM.md`, 20.7 KB).
- With an OpenRouter key it asks `~typesafe/jev-latest` one typed question (about 0.2 s, $0.00003); otherwise one non-agentic structured-output call on the run's model with a plain-text retry; skipped for CLI models.
- Verdicts are cached on disk; classifier spend folds into the task's totals; benchmark of 415 prompts in `benchmarkings/task_classifier/`.

## 11. Git integration and worktrees

- Each development task runs in a fresh git worktree on a unique branch (`.kiss-worktrees/`); a spare worktree is pre-created in the background (`worktree_pool.py`) so tasks start in well under a second on large repos.
- On success: auto-commit and merge back, or an interactive merge/discard/leave-as-is prompt; conflicts are detected and reported; stash-pop and merge-conflict warnings are surfaced; ignored files can be rescued on discard.
- Failed or stopped tasks leave their work committed on the branch; retired worktrees are cleaned up when no abandoned sub-agent still writes to them.
- Bash and file tools rewrite parent-repo paths into the worktree and block edits that would escape it.
- Non-worktree tasks get post-task auto-commit with a generated message; the extension exposes Generate Commit Message and Git Commit commands.
- 22,023 of 24,063 recorded task rows ran with worktree isolation.

## 12. Scheduled automations

- Natural-language requests are dispatched to the cron agent (`run_agent("cron", …)`; also `kiss-cron` from the shell). Its `cron_job` tool creates, lists, pauses, resumes, removes or immediately runs jobs stored in `~/.kiss/cron/jobs.json`.
- Schedules: 5-field cron expressions and one-shot timestamps; a scheduler thread in `kiss-web` ticks about once a minute.
- Two job kinds: **prompt jobs** run an unattended Sorcar task (never asks questions, replies `[SILENT]` to suppress delivery) and **command jobs** run a shell command without a model (whole process tree killed on timeout).
- Delivery of results to authenticated channels (`telegram:123456`, `email:user@example.com`, …; 25 of 32 channels), with `until_delivered` retry semantics; outputs kept under `~/.kiss/cron/output`.
- Recorded: 17 `cron_job` calls and 42 unattended cron-launched tasks; one job is currently configured on this machine.

## 13. Third-party agents and credential isolation

43 modules in `src/kiss/agents/third_party_agents/`, each a Sorcar Extension Agent whose `tools()` returns auth tools plus the backend's public methods once authenticated. Configuration lives under `~/.kiss/third_party_agents/<service>/`; authentication happens by chatting.

| Kind | Agents |
|---|---|
| Messaging and device channels (32) | BlueBubbles, DingTalk, Discord, Email (IMAP/SMTP), Feishu, Gmail, Google Chat, Home Assistant, iMessage, IRC, LINE, Matrix, Mattermost, Microsoft Teams, Nextcloud Talk, Nostr, ntfy, Phone Control, QQ, Signal, SimpleX, Slack, SMS, Synology Chat, Telegram, Tlon, Twitch, Webhook, WeCom, WeiXin, WhatsApp, Zalo |
| Service APIs (9) | Brave Search, Firecrawl, GitHub, Google Calendar, Google Docs, Google Drive, Google Sheets, Notion, PostgreSQL |
| Infrastructure (2) | A2A protocol agent, OpenAI-compatible server |
| Extra | `govee.py` smart-light CLI (on/off, brightness, color, color temperature) |

- **Dispatch**: the top-level session names the service and calls `run_agent`; channel names match forgivingly; multi-account workspaces via `KISS_CHANNEL_WORKSPACE`; dispatched sessions cannot dispatch further.
- **Gateway mode** for channels with an inbound stream (section 3) and **scheduled delivery** (section 12).
- **Muse auth** (Linux): credentials for 24 connectors live in a vault owned by a local auth daemon; the agent holds only surrogate tokens swapped at the network edge; every request is host-allowlisted, classified read/write and checked against an allow/deny/ask policy with an audit log; OAuth device grants (GitHub, Twitch, Microsoft Teams), Nextcloud Login Flow v2, Matrix OAuth device grant, Signal QR linking; CLI `python -m kiss.agents.third_party_agents.muse_auth` with `status`, `enroll`, `import`, `grant`, `revoke`, `audit`, `clear`, `daemon`, `stop`, `export`; opt out with `KISS_MUSE_AUTH=0`.
- The agent never asks for a password or 2FA code; login walls are handed to the user's own browser.

## 14. Extension points: MCP, skills, SEAs

- **MCP servers**: discovered from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, `<project>/.mcp.json`; stdio, HTTP and SSE transports; OAuth 2.1 with dynamic client registration and PKCE, tokens under `~/.kiss/mcp_auth/`; tools exposed as `<server>_<tool>` and filtered by `mcp_permissions` wildcard rules; save/remove servers programmatically.
- **Connector catalog** (`connectors/`): 16 curated privacy-first MCP servers (fetch, time, memory, sequential-thinking, deepwiki, context7, github, google, slack, twilio-sms, whatsapp, brave-search, notion, postgres, firecrawl, playwright) with `enable.py`/`verify.py` CLIs; credentials read from the shell, never stored.
- **Agent Skills**: loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills` and bundled skills; a single `skill(name)` tool loads `SKILL.md` and lists bundled resources; permission rules per skill.
- **Sorcar Extension Agents**: a Python file whose top-level getters (`prompt()`, `model()`, `max_budget()`, `system_prompt()`, `tools()`, `use_worktree()`, …) compute run parameters on the daemon, type-checked and applied atomically; `llm_call_hook` rewrites outgoing messages and `tool_call_hook` can veto tool calls. `run_agent` also runs any SEA file by path.
- **Tools file**: `tools="/path/my_tools.py"` whose `get_tools()` returns extra callables; `append_basic_tools=False` restricts the agent to `finish` plus those tools.
- **Prompt customization without code**: `~/.kiss/MY_TASK_TEMPLATES.md` (sample-task chips), `~/.kiss/MY_INJECTION.md` (tricks), `~/.kiss/MY_MODELS.json`, per-repo `SORCAR.md`, optional `ROUTING.md`; 12 bundled sample tasks and 8 bundled tricks.

## 15. Persistence, accounting, observability

- **`~/.kiss/sorcar.db`** (WAL SQLite): `task_history` (task, result, chat_id, model, work_dir, version, tokens, cost, steps, flags, start/end, favorite, parent_task_id, owner, max_budget), `events` (ordered JSON per task), `model_usage`, `file_usage`, `frequent_tasks`, `replayed_journals`. Journals protect against crashes; `lost_and_found` shows a past recovery.
- **Live accounting**: cost, tokens, steps and budget appear in every tool result, the status bar and the Task Info sidebar; sub-agent, classifier and TTS spend fold into the parent task.
- **Cost KPIs**: `scripts/cost_report.py` computes token-cost KPIs (peak context, hand-offs, compaction) over a database; `cost_levers_experiment.py` runs A/B experiments of the cost levers (`read_dedupe`, `tool_output_compaction`, `tool_profiles`, `review_budget_fraction`, `chat_history_digest`, `dispatch_path_rewrite`, `context_limit_fraction` in `config.py`).
- **Trajectory viewer**: `kiss.viz_trajectory.server` is a Flask app that renders saved trajectory YAML files under `.kiss.artifacts`; `/api/jobs` endpoints in the web server expose benchmark job trajectories.
- **Shareable chats**: any chat or task selection exports to a self-contained HTML page.
- **Daemon health**: GIL-independent stall watchdog, daemon health checks and restart verification in the extension, logs in `~/.kiss/kiss-web-*.log`, active-task queries (`scripts/check-kiss-web-active-tasks.py`).
- **Database tooling**: `sync_db.py` (one-way merge of task and event rows), `carry_over_tables.py`, `db_fingerprint.py`, `relocate_work_dir.py`, `running_tasks.py`, `scripts/sync-task-db.sh`.

## 16. Installation, deployment, operations

- **One-line install** (`install.sh`, 63 KB): bootstraps Homebrew/Xcode CLT, git, node, uv, Python 3.13, Playwright, builds and installs the VSIX, launches VS Code; interactive or `--non-interactive`; immune to terminal signals; logs to `~/.kiss/install.log`. Also `pipx install kiss-agent-framework` for the daemon and CLIs only.
- **Remote deployment** (`rsorcar user@host`): checks SSH, installs prerequisites, verifies no task is running and disk space suffices (with a bind-mount helper to move `$HOME`), copies SSH identity, syncs the checkout through `origin`, ships the task database, sets the remote password and work dir (`remote_config.py`), starts `kiss-web` and waits for the public cloudflared URL.
- **Docker**: `sorcar-docker [PORT] [--rebuild]` builds the image from `Dockerfile`, runs code-server with the extension in a container and opens the browser; per-task `docker_image` isolation for tools.
- **Updates**: Settings "Update" button re-runs the installer; the extension checks for new versions with a snooze shared between daemon and clients; `scripts/release.sh` packages releases.
- **Auth helpers**: `install-api-keys.sh`, `install-github-auth.sh`, `collect-github-auth.sh`, `install-ssh-identity.sh`.
- **Platforms**: macOS and Linux (x86_64, aarch64, arm64); Windows bash discovery exists in the tools; Muse auth is Linux-only.

## 17. Long-horizon research drivers and benchmarks

The system prompt contains procedures that turn one chat message into a multi-hour autonomous loop:

- **AI discovery / auto research / optimization** (`SYSTEM.md` §workflow): profile baseline, web-search state of the art, write `./tmp/ideas.md`, judge ideas pairwise, implement and evaluate end to end, log to `./tmp/explored-ideas.md`, compose winners, stop only when the metric goal holds on a held-out check.
- **Adversarial testing**: one sub-task tries to break the system with adversarial tests and workloads while another fixes it. **Adversarial training**: generate adversarial datasets until a model stops overfitting.
- **GEPA prompt optimization** and **repository optimization** as bundled sample prompts; recorded runs include xxHash (5,583 steps, $393), SQLite, LZ4 and the HydraKV and Bespoke OLAP case studies described in `papers/kisssorcar/kiss_sorcar.tex`.
- **Paper writing and review**: `templates/write_paper_prompt.md` and `RECIPES.md` give prompts for venue-conformant LaTeX papers with citation verification and AI-slop gates, and for deep conference reviews; the repo holds five paper directories (KISS Sorcar, SE KISS Sorcar, SorcarCCL, SweDefend, HydraKV) written this way.
- **Deep work, planning, file-browsing rules**: read-before-modify, concrete per-file plans for 3+ file changes, end-to-end tests with 100% branch coverage and no mocks, parallel test splitting by core count, single lint/typecheck at the end, three-item pre-finish check.
- **Web research protocol**: a confidence gate that mandates search whenever facts may be stale, a ten-site research procedure, and reports for answers over ~800 words.
- **Benchmarks** (`benchmarkings/`): Harbor agent for data-eng-bench (103 tasks, 309 trials, 69.58% accuracy, pass@3 0.728, $1,143 total); `harnesstax` runners for Terminal-Bench 2.0 (30 sampled tasks) and SWE-bench Lite with official graders; task-classifier benchmark; `projects/swedefend` (a prompt-injection defense pipeline with `swedefend-eval`).

## 18. Developer tooling and tests

- `uv run check --full`: syntax, ruff lint, pyright/mypy type check; JS side `npm run check` (tsc, eslint, stylelint, htmlhint, node tests).
- 1,228 Python test files with 8,669 test functions under `src/kiss/tests/`, mirroring every package (core, models, memoryfield, sorcar, server, third-party agents, vscode, viz, scripts, swedefend); JS tests for the extension in `agents/vscode/test/`.
- `generate-api-docs` builds `API.md` (82 KB) from the source by AST introspection; `redundancy_analyzer.py` finds redundant tests via branch coverage contexts.
- Type hints throughout (`py.typed`), pyproject-driven tooling, GitHub Actions workflow under `.github/`.

## 19. What the trajectories show

The task database records every task run on this machine and on machines whose databases were synced into it (71 distinct work directories; the main checkout accounts for 5,332 of the 5,648 top-level tasks). Rows before April 2026 are legacy imports; the six months from April to September 2026 hold the working history.

- **5,648** top-level tasks · 18,415 sub-agent rows · 2,845 chats
- **8.30 M** persisted events (5.0 M thinking deltas, 488K tool calls, 467K tool results)
- **235** top-level tasks longer than one hour; 23 with 1,000+ steps; 39 costing $100+
- **8,056** steps in the longest task ($762, a repo-wide race-condition audit); costliest task $845 with 687 M tokens

**Tool calls recorded in sorcar.db**

| Tool | Calls | Related tools |
|---|---:|---|
| `Bash` | 229,056 | |
| `Read` | 146,700 | |
| `Edit` | 39,306 | |
| `summary` | 22,116 | |
| `finish` | 17,329 | |
| `go_to_url` | 8,673 | |
| `Write` | 8,051 | |
| `set_model` | 4,174 | |
| `run_parallel` | 3,969 | |
| `memory_search` | 1,456 | `memory_pull` 755, `memory_write` 499, `memory_read` 182 |
| `screenshot` | 592 | |
| `click` | 468 | `type_text` 140, `press_key` 132, `scroll` 114, `get_page_content` 177 |
| `ask_user_question` | 336 | |
| `talk` | 253 | |
| `run_agent` | 209 | channel tools such as `list_messages` 108, `check_gmail_auth` 12, `cron_job` 17 |
| `run_commands_parallel` | 78 | added 2026-09 |
| `decide` | 59 | added 2026-09 |

Groups: files/shell (`Bash`, `Read`, `Edit`, `Write`, `run_commands_parallel`) · browser (`go_to_url`, `screenshot`, `click`, ...) · delegation and model switching (`set_model`, `run_parallel`, `run_agent`) · memory and decisions (`memory_*`, `decide`) · control and interaction (`summary`, `finish`, `ask_user_question`, `talk`).

*Figure 2. 487,656 tool calls. Shell and file tools dominate, as expected for a coding agent; `set_model` (4,174) and `run_parallel` (3,969) show that multi-model, multi-agent work is routine rather than exceptional. The 1,524 `code_graph` calls belong to a tool that no longer exists.*

**Top-level tasks per month, 2026**

| Month | Apr | May | Jun | Jul | Aug | Sep (to 20th) |
|---|---:|---:|---:|---:|---:|---:|
| Top-level tasks | 513 | 1,807 | 888 | 883 | 843 | 712 |

*Figure 3. Sustained daily use since April 2026: about 30 top-level tasks per day, each often fanning out into several sub-agents (3.3 sub-agent rows per top-level task on average).*

### Models actually used

| Model | Task rows | Model | Task rows |
|---|---|---|---|
| gpt-5.6-sol | 7,871 | claude-opus-5 | 505 |
| claude-fable-5 | 7,317 | claude-opus-4-8 | 388 |
| gpt-5.6-sol-xhigh | 2,972 | gpt-5.5 | 159 |
| claude-opus-4-7 | 2,615 | claude-sonnet-4-5 | 52 |
| claude-opus-4-6 | 1,098 | codex/gpt-5.5 · cc/opus | 43 · 37 |
| claude-fable-5-1 | 751 | openrouter/moonshotai/kimi-k3 · gpt-6-astra(-high) | 22 · 44 |

The split is the builder/reviewer pattern from the tricks: Anthropic models build, OpenAI `gpt-5.6-sol` reviews in read-only sub-agents. Thirteen further models appear in small numbers, including Gemini, Kimi, Sonnet, Haiku and the CLI-backed `cc/*` and `codex/*` namespaces.

### Kinds of work recorded

Keyword counts over the 5,648 top-level task texts (a task can fall in several rows):

| Kind of task | Tasks | Examples from the history |
|---|---|---|
| Software changes (fix, bug, implement, refactor, add) | 1,924 | "Extend Muse-auth to the remaining credentialed connectors…" (2,289 steps); "major refactoring of the project to significantly simplify the implementation" (2,281 steps) |
| Testing | 1,303 | "run all tests… split by the number of test methods into cores − 2 and run all splits in parallel" (a frequent task, up to 3,519 steps) |
| Code review and audits | 924 | "Find all race conditions, deadlocks, obvious bugs, missing wiring, redundancies…" (8,056 steps, $762); 8 automatic "Verify the listed changes" reviewer sub-tasks |
| Web research | 631 | trip planning to Yosemite with hotel availability, product comparisons, market reads, STT-for-Indian-languages survey |
| Git operations | 632 | "git push origin", "get rid of the last commit to main", "checkout main", merge-conflict help |
| Reports | 404 | 72 HTML reports in `reports/`: audits, optimization reports, architecture notes, comparisons |
| Papers and LaTeX | 400 | KISS Sorcar, SE KISS Sorcar, SorcarCCL, SweDefend, HydraKV papers; ICSE/ACL/OpenReview reviews |
| Deployment and remote machines | 320 | `rsorcar` deploys, VS Code server checks, disk-space remediation |
| Messaging (Slack, Gmail, WhatsApp, Telegram, iMessage, SMS, email, Discord) | 171 | Slack workspace authentication, Gmail checks, WhatsApp pairing, SMS sends, Slack gateway pollers |
| Shopping, travel, price comparison | 123 | non-stick cookware comparison, Yosemite trip, due-diligence research |
| Benchmarking and optimization | 120 | xxHash, SQLite, LZ4, DuckDB and TPC-H optimization runs; Terminal-Bench and SWE-bench jobs |
| Voice tasks ("Speaker #N says…") | 114 | spoken questions and steering through the wake word |
| Blog posts, LinkedIn posts, slides | 94 | SQLite and LZ4 optimization blogs, lecture PPTX and one-slide decks |
| Unattended cron jobs | 42 | VS Code release watcher with spoken notification; Slack gateway ticks |
| Docker | 40 | code-server container launches, Docker-isolated tool runs |
| Home devices (lights, Govee, Home Assistant) | 25 | Govee light control |

## 20. Caveats and gaps

- `SYSTEM.md`'s "Desktop Apps" rule refers to screenshot, keyboard and mouse tools, but the only `screenshot`/`click`/`press_key` tools registered operate on the Playwright browser page, not the desktop. Desktop control is available only through channel agents such as Phone Control or through shell commands.
- Cron is not a built-in tool of a normal run: it works through `run_agent("cron", …)`. The `skill` tool is registered only when a user or project skill directory exists; bundled skills alone do not add it.
- Memory, the classifier and prompt caching are unavailable for `cc/*` and `codex/*` runs, which hand the whole task to the external CLI.
- The `summary`-every-ten-steps rule is enforced by the prompt only; there is no mechanical check.
- Muse-auth isolation is a process boundary, not an OS boundary: an agent allowed to run arbitrary shell commands could read the vault unless tool permissions deny it (stated in `muse_auth/__init__.py`).
- The database contains artifacts: 64 task rows with an empty model, a minimum timestamp of 2009 (a clock artifact on an imported row), 1,524 calls to a retired `code_graph` tool, and a `lost_and_found` table from a past recovery.
- README counts (665 models, 43 agents, 24 Muse-covered connectors, 25 delivery-capable channels) were re-verified for the first two; the last two are taken from the README and third-party README.

Sources: `README.md`, `src/kiss/SYSTEM.md`, `src/kiss/SYSTEM_LITE.md`, `src/kiss/TIPS.md`, `src/kiss/INJECTIONS.md`, `src/kiss/SAMPLE_TASKS.md`, `RECIPES.md`, `pyproject.toml`, `src/kiss/core/*`, `src/kiss/agents/sorcar/*`, `src/kiss/server/*`, `src/kiss/agents/third_party_agents/README.md`, `connectors/README.md`, `src/kiss/agents/vscode/{package.json,media/chat.html,media/main.js}`, `benchmarkings/`, `papers/kisssorcar/kiss_sorcar.tex`, `install.sh`, `rsorcar`, `sorcar-docker`, and SQL queries against `~/.kiss/sorcar.db` on 2026-09-20.
