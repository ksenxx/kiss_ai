# KISS Sorcar: Complete Feature Inventory

Version 2026.9.20 · HEAD `85d9f3cda` (4,566 commits) · task database `~/.kiss/sorcar.db` · compiled 2026-09-21 · supersedes the 2026-09-20 inventory (version 2026.9.18)

This report enumerates everything KISS Sorcar can do, as of the source tree at version 2026.9.20. Every item was checked against the code rather than the README: the tool functions registered in `src/kiss/agents/sorcar/sorcar_agent.py`, the daemon command handlers in `src/kiss/server/`, the browser client in `src/kiss/agents/vscode/media/`, the 44 channel-agent modules and 3 bundled Sorcar-extending agents, the CLI entry points in `pyproject.toml`, and the system prompt in `src/kiss/SYSTEM.md`. Items marked **NEW** arrived in the 69 commits landed between the previous inventory (2026-09-20, commit `d684b3193`) and this one. The last sections turn to the recorded trajectories in `sorcar.db` to show which capabilities are actually used and how hard they have been pushed.

**Contents**

1. [At a glance](#1-at-a-glance)
2. [What changed since 2026-09-20](#2-what-changed-since-2026-09-20)
3. [Architecture](#3-architecture)
4. [Interfaces and prompt surfaces](#4-interfaces-and-prompt-surfaces)
5. [Agent runtime](#5-agent-runtime)
6. [Built-in tools](#6-built-in-tools)
7. [Models and multi-model routing](#7-models-and-multi-model-routing)
8. [Chat client features](#8-chat-client-features)
9. [Voice](#9-voice)
10. [Persistent memory](#10-persistent-memory-memoryfield)
11. [Task classifier](#11-task-classifier)
12. [Git integration and worktrees](#12-git-integration-and-worktrees)
13. [Scheduled automations](#13-scheduled-automations)
14. [Third-party agents and credential isolation](#14-third-party-agents-and-credential-isolation)
15. [Extension points: slash commands, SEAs, MCP, skills](#15-extension-points-slash-commands-seas-mcp-skills)
16. [Persistence, accounting, observability](#16-persistence-accounting-observability)
17. [Installation, deployment, operations](#17-installation-deployment-operations)
18. [Long-horizon research drivers and benchmarks](#18-long-horizon-research-drivers-and-benchmarks)
19. [Developer tooling and tests](#19-developer-tooling-and-tests)
20. [What the trajectories show](#20-what-the-trajectories-show)
21. [Caveats and gaps](#21-caveats-and-gaps)

## 1. At a glance

- **664** models in the bundled catalog, 9 provider categories (642 generation, 485 function-calling)
- **44 + 3** channel agents (32 messaging, 9 service, 2 infrastructure, 1 `/ask`) plus bundled `/merge`, `/sh`, `dummy` SEAs
- **49** CLI entry points in `pyproject.toml`
- **~41** built-in agent tools (plus per-channel, MCP and skill tools); 4 tool profiles
- **7** prompt surfaces: VS Code, web/mobile, slash commands, Python API, terminal, chat channels, OpenAI/A2A endpoints
- **1,260** Python test files, 9,576 test functions, plus 324 JS test files for the extension
- **24,308** task rows in `sorcar.db` (5,783 top-level, 18,525 sub-agent)
- **$53.9K** recorded model spend across 36.3 billion tokens and 333K steps

## 2. What changed since 2026-09-20

Sixty-nine commits, two version bumps (2026.9.19, 2026.9.20), 10,911 files touched. Grouped by what a user notices:

| Area | Change | Where |
|---|---|---|
| **Slash commands** | Every `xxx_sea.py` the daemon can see is a chat command `/xxx`; a "Commands" popup lists them as soon as the prompt starts with `/`. Bundled: `/slack`, `/gmail`, `/github`, … (44 channel agents), `/ask`, `/merge`, `/sh`. Own commands: list folders in `~/.kiss/SEAS.md`; rescanned every 2 s, no restart. | `sorcar/sea_commands.py`, `website/…/docs/sea-commands.md` |
| **`/ask` side channel** | Ask a question about the task that is currently running (`/ask why did the tests fail?`); an offline sub-agent reads the task's events from `sorcar.db` and answers in the transcript; the answer panel never auto-collapses. Chat replies are also delivered as answers to a pending `ask_user_question`. | `third_party_agents/ask_sea.py` |
| **`/sh` and `/merge`** | `/sh git status --short` runs one command with the `bash` tool profile directly in the tab's checkout and returns raw output. `/merge` resolves a conflicted merge; the same agent runs automatically, in-process, when an auto-commit worktree merge conflicts. | `agents/seas/sh_sea.py`, `merge_sea.py`, `server/merge_conflict_resolver.py` |
| **Background shell jobs** | `Bash(command, background=true)` returns a job id and log path at once; `bash_job(job_id, action="wait"\|"tail"\|"kill")` follows it. Long builds, servers and test suites no longer block a model step. | `sorcar/useful_tools.py` |
| **Browser vs. bot protection** | Patchright driver (no `Runtime.enable` fingerprint), installed Google Chrome preferred, headed Chromium on an Xvfb virtual display on servers, Bezier mouse paths, uneven typing cadence, challenge-vendor detection with wait-and-report. Measured 4/14 to 11/14 blocked hosts passed. | `sorcar/web_stealth.py`, `web_use_tool.py` |
| **Sign-in hand-off** | OAuth consent, device-code and bot-token pages are opened in the user's own default browser (`$BROWSER`, `open`, `xdg-open`) and the URL/code echoed in chat; the agent never drives them. | `third_party_agents/_browser_handoff.py` |
| **Cron** | Jobs run concurrently, each in its own scratch directory; per-job `work_dir`, `use_worktree`, `auto_commit`, `timeout`; prompt jobs are launched as generated SEA files through `run_agent`; always-on channel gateways are scheduled as the channel CLI's tick command (no tokens when idle); duplicate job creation deduped. | `sorcar/cron_agent.py` |
| **`run_agent`** | `agent` is optional (empty runs a plain sub-agent, `dummy_sea.py`); accepts every `sorcar.run` option: `chat_id`, `system_prompt`, `tools`, `model_config`, `use_worktree`, `auto_commit`, `use_web_tools`, `classify_tasks`, `use_memory`, `is_parallel`, `append_basic_tools`, `append_to_system_prompt`, `append_to_prompt`, `tool_profile`. | `sorcar/agent_dispatch.py` |
| **Worktrees** | Deferred worktrees auto-merge once the parent branch is committed; conflicts go to the merge SEA before falling back to the manual message; the main checkout's `node_modules` directories are symlinked into each worktree. | `server/merge_flow.py`, `sorcar/git_worktree.py` |
| **Cost levers** | Cache-aware compaction gate (compact only when the drop is at least 25% of the context and the session is not near a hand-off; replaying 213 Claude sessions removed 70% of compactions and 34% of the compaction-related input bill) and a prompt-cache keep-alive that pings the cached prefix every 4 minutes during tool calls allowed to run 300 s or longer. RelentlessAgent stops after 2 zero-progress continuation sessions. | `core/context_compaction.py`, `core/prompt_cache_keepalive.py`, `sorcar/relentless_agent.py` |
| **Chat client** | In-transcript "Question" panel replaces the floating `ask_user_question` modal; "Working directory" panel in the "…" menu with a picker and a shared recent-directories list; autocomplete and `@`-mentions stay active while a task runs; steer-mode text enters autocomplete history; file picker prefers source files over data dumps and scans up to 1,000,000 entries; the two newest panels stay open while streaming; unified spinner/tick/cross status icons; history panel highlights and scrolls to the active task and shows "launched N minutes ago"; Inject (tricks) panel gained search and quick-add. | `vscode/media/main.js`, `main.css`, `src/*.ts` |
| **Remote web app** | Service-worker offline app shell (`/sw.js`) keeps the app on screen through flaky connections and reloads fully on reconnect; loopback (`127.0.0.1`) and LAN URLs shown beside the Cloudflare URL; "Update when idle" action on the update toast. | `vscode/media/sw.js`, `server/web_server.py` |
| **`rsorcar`** | Remote web-app password defaults to this machine's own (one password for both); API keys are copied only to a remote that has none, so a re-deploy never rolls back a rotated key. | `rsorcar` |
| **Naming** | Channel modules renamed `<service>_agent.py` → `<service>_sea.py`; bundled Sorcar-extending SEAs live in `src/kiss/agents/seas/` with the lowest command precedence. | `third_party_agents/`, `agents/seas/` |
| **Prompt assets** | Two new tips (`/ask`, slash commands and `~/.kiss/SEAS.md`), two new tricks (10 total); `SYSTEM.md` gained rules on background jobs, batching shell commands, reviewer dispatch wording, flake handling, immediate channel/cron dispatch, and worktree `node_modules`. | `src/kiss/TIPS.md`, `INJECTIONS.md`, `SYSTEM.md` |
| **Reliability** | Lock-ordering deadlocks and stale-read races removed across agents and server; concurrent `VectorIndex.sync()` no longer clobbers fresher memory rows; Slack channel lookup by ID verified via `conversations.info` (after an outage caused by committed conflict markers). | commits `cd1641762`, `28b9620d4`, `145002a95` |

## 3. Architecture

One local daemon, `kiss-web`, hosts every agent and serves every client. The agent is a five-layer class stack; each layer adds one concern. Lines of code below are from `wc -l` on 2026-09-21.

```
  VS Code extension   Web / mobile app     Python API            Chat channels        OpenAI-compat / A2A   Voice
  (TypeScript,        (tunnel, LAN,        (kiss.server.         (Telegram, Slack,    (kiss-oai, kiss-a2a)  (wake word
   webview)            offline shell)       sorcar.run)           email, ...)                                "Sorcar")
        |                  |                   |                      |                      |                 |
        +------------------+-------------------+----------+-----------+----------------------+-----------------+
                                                          v
  +-----------------------------------------------------------------------------------------------------------+
  | kiss-web daemon (src/kiss/server, 31,860 LoC)                                                             |
  |   Unix-socket JSON protocol . WebSocket web server . tab registry (mirrored tabs) . task runner & stop     |
  |   machinery . slash-command registry . autocomplete (ghost text, @paths, /commands) . merge flow +         |
  |   merge-conflict resolver . Explorer/SCM providers . voice wake . talk player . cron scheduler .           |
  |   SEA loader . tools-file loader . stall watchdog . update checker . remote password + cloudflared         |
  |   tunnel + LAN URLs . service-worker shell                                                                 |
  +-----------------------------------------------------------------------------------------------------------+
                                                          v
  +-------------------------------------------------------------------+  +------------------------------------+
  | WorktreeSorcarAgent (1,968): git worktree per task, auto-commit,   |  | Model adapters                     |
  |                             merge / discard / defer                |  | (src/kiss/core/models)             |
  | ChatSorcarAgent     (740):  chat history & digest, sorcar.db       |  |   Anthropic . OpenAI-compatible    |
  |                             persistence, stop/steer                |  |   (2 generations) . Gemini .       |
  | SorcarAgent         (3,418): tools, browser, run_parallel,         |  |   Claude Code CLI . Codex CLI .    |
  |                             run_agent, memory, classifier, talk    |  |   Decisions (Jev) .                |
  | RelentlessAgent     (1,666): auto-continuation, zero-progress      |  |   MODEL_INFO.json (664)            |
  |                             stop, usage ledger                     |  +------------------------------------+
  | KISSAgent           (1,280): native function calling,              |  | Side systems                       |
  |                             budget/step/context limits, retries,   |  |   memoryfield (vector index of     |
  |                             fallback                               |  |   Markdown pages) . Patchright/    |
  +-------------------------------------------------------------------+  |   Playwright browser . Docker      |
                                    v                                    |   manager . MCP client . skills .  |
  +-----------------------------------------------------------------+    |   Muse-auth vault . cache          |
  | 44 channel SEAs . /ask . /merge . /sh . cron agent . user SEAs  |    |   keep-alive                       |
  | from ~/.kiss/SEAS.md . MCP servers . skills . tools files       |    +------------------------------------+
  +-----------------------------------------------------------------+
```

*Figure 1. Every surface talks to one daemon; the daemon builds an agent from the five-layer stack and gives it tools. Channel agents, the cron agent and the new bundled `/ask`, `/merge` and `/sh` agents are all Sorcar Extension Agents dispatched by the `run_agent` tool, which the slash-command rewriter invokes for you.*

## 4. Interfaces and prompt surfaces

| Surface | What it does | Where |
|---|---|---|
| **VS Code extension** | Sidebar chat or editor-tab chats (`kissSorcar.editorTabsMode`); commands: open panel, new conversation, open settings, stop task, generate commit message, git commit, focus chat/editor, run selection, insert selection into chat, show history. Auto-starts the daemon; installs its own dependencies (`DependencyInstaller.ts`); update checker with snooze; macOS launchd integration. | `src/kiss/agents/vscode/` |
| **Web / mobile app** | The same chat UI served by the daemon over WebSocket; optional cloudflared tunnel plus password (`remote_password`); loopback and LAN URLs listed beside the tunnel URL **NEW**; service-worker offline shell keeps the app on screen through a dropped connection and reloads on reconnect **NEW**; desktop mode adds a Task Info sidebar (tokens, cost, steps, elapsed, machine, work dir, budget, live `tmp/PROGRESS.md`) and an activity bar with Explorer, Source Control and Tasks views. | `src/kiss/server/web_server.py` (9,754 LoC), `media/sw.js` |
| **Slash commands** **NEW** | A prompt starting with `/xxx` is rewritten into an immediate `run_agent` call on `xxx_sea.py` with the rest of the line as the sub-task. Typing `/` opens a Commands popup listing every command with its description. Precedence: bundled channel agents, then folders from `~/.kiss/SEAS.md` (bottom line wins), then `src/kiss/agents/seas/`. Examples: `/slack what is new in #general?`, `/ask which tests failed?`, `/sh git status --short`, `/merge`, `/cron every weekday at 9 run the nightly report`. | `sorcar/sea_commands.py`, `server/web_server.py` |
| **Python client API** | `kiss.server.sorcar.run(prompt, …)` blocks until a daemon task finishes and returns `TaskResult` (text, success, cost, tokens, steps, chat_id). Options: model, work_dir, scope_work_dir, chat_id, use_worktree, auto_commit, max_budget, model_config, use_web_tools, classify_tasks, use_memory, is_parallel, timeout, stop_on_timeout, sock_path, parent_task_id/parent_tab_id, tools file, system_prompt, append_to_system_prompt, append_to_prompt, append_basic_tools, extension_agent_path, tool_profile. | `src/kiss/server/sorcar.py` (re-exports `agents/sorcar/daemon_client.py`) |
| **`sorcar` terminal command** | Runs a SorcarAgent in the current directory without the daemon: `sorcar -t "task"` or `sorcar -f task.txt`; flags for model, budget, work dir; answers `ask_user_question` in the terminal. | `sorcar_agent.py:main` |
| **`kiss-web` daemon CLI** | `kiss-web`, `--workdir DIR`, `--url` (print the public URL). | `web_server.py:main` |
| **Chat channels as inbound surfaces** | Gateway mode: a poll tick drains new messages from Telegram, Slack, Discord, email, WhatsApp, … and runs each as a task; thread continuity, delivery ledger, per-channel model/budget, `--allow-users`, pairing handshake (`--pairing`, `--approve`, `--list-pending`). An always-on gateway is now scheduled by the cron agent as the channel CLI's tick command, so an idle poll costs no tokens **NEW**. | `third_party_agents/_channel_cli.py` |
| **OpenAI-compatible server** | `kiss-oai` exposes the daemon behind an OpenAI chat API so Open WebUI, LibreChat or an `openai` SDK script become chat surfaces. | `openai_compat_sea.py` |
| **A2A agent** | `kiss-a2a` publishes an agent card at `/.well-known/agent-card.json`, accepts JSON-RPC `message/send` and `tasks/get`, and can call peer agents outbound. | `a2a_sea.py` |
| **Voice** | Wake word "Sorcar" via the mic button, speech-to-text, spoken replies through `talk`; also steers a running task. | `server/voice_wake*.py`, `media/voice.js` |
| **Per-channel CLIs** | `kiss-slack`, `kiss-gmail`, `kiss-whatsapp`, `kiss-telegram`, … 41 channel commands plus `kiss-cron`, `kiss-web`, `sorcar`, `check`, `generate-api-docs`, `swedefend-eval`: 49 entry points. | `pyproject.toml [project.scripts]` |

## 5. Agent runtime

### KISSAgent (`src/kiss/core/kiss_agent.py`)

- Native function calling against every adapter; tool schemas are built once per tool set and rebuilt on model switch.
- Hard limits per run: `max_budget` (USD), `max_steps`, and context length (`CONTEXT_LIMIT_FRACTION` of the model's window); raises `BudgetExceededError` / `ContextWindowExceededError`.
- Retries with backoff; non-retryable errors (bad key, permission denied, low credit, model gated) trigger a one-shot **fallback model** registered in `MODEL_INFO.json`, copying the conversation so no context is lost.
- **Tool-output compaction** (`context_compaction.py`): once the context passes 100k tokens, old tool results longer than 2,000 chars are replaced by 200-char stubs, keeping the 20 newest; repeated every 50k of growth. **NEW** A cache-economics gate (`should_compact`) applies a compaction only when it drops at least 25% of the context and the session is not about to hand off, since every compaction invalidates the prompt cache.
- **NEW** **Prompt-cache keep-alive** (`prompt_cache_keepalive.py`): for tool calls allowed to run 300 s or longer, a background thread pings the cached prefix every 4 minutes so the next real call still hits the cache after a long build or test run.
- Read dedupe and outlines: re-reading an unchanged range returns a one-line note; files over 2,000 lines return an outline (config flags `read_dedupe`, `read_outline_lines`).
- **Implicit finish**: a model that stops calling tools is wrapped into a finish result instead of erroring.
- Per-tool-call interrupt (`tool_interrupt.py`): the Stop button on one panel aborts that tool call while the task continues; the model sees `USER_INTERRUPTED_MESSAGE`.
- Streamed responses that go quiet are aborted and retried (`stream_abort.py`, `stream_stall_timeout`, 180 s).
- Image and PDF attachments travel with the prompt (`Attachment`; HEIC/HEIF converted, oversized images re-encoded).
- Prompt caching on Anthropic (`cache_control: ephemeral`); usage/cost tracked per response from catalog prices.
- Non-agentic single-shot mode (used by the classifier, commit-message generation and TTS).

### RelentlessAgent (`agents/sorcar/relentless_agent.py`)

- Auto-continuation: when a session exhausts its context the agent calls `finish(is_continue=True)` with a structured progress summary and a fresh sub-session resumes from it (up to `max_sub_sessions`, default 10,000). Prior-session summaries are capped and re-injected.
- **NEW** **Zero-progress stop**: two consecutive continuation sessions that call no tool but `finish`, or repeat the previous summary verbatim, end the task instead of looping (`MAX_ZERO_PROGRESS_SESSIONS = 2`).
- Append-only **usage ledger** with epochs so spend from sub-agents, abandoned children, classifier calls and TTS is counted exactly once even across stop-interrupts.
- Optional `docker_image`: Bash/Read/Write/Edit execute inside a container (`docker_manager.py`, `docker_tools.py`) with streaming output, per-exec kill tokens and output caps.
- Host settings (IP, machine, OS, PID, user id, chat/task ids) are placed in the system prompt.

### SorcarAgent (`sorcar_agent.py`)

- Builds the tool set (section 6), the browser, the memory tools, the classifier, `run_parallel` fan-out and `run_agent` dispatch.
- **Tool profiles**: `full`; `review` (Bash, Read, run_commands_parallel, memory reads, decide, summary); `shell` (Bash, Read, run_commands_parallel); **NEW** `bash` (Bash only, used by `/sh`). Reviewer children get `review` automatically.
- **Review quota** (`fanout_guard.py`): at most 3 review rounds per task tree; "at most N% of the budget for reviewing" in the prompt clips reviewer budgets; sub-agent budgets are split from the parent's remainder with a $0.50 floor.
- Sub-agents open their own tabs, are persisted with `parent_task_id`, and their spend folds into the parent.
- Auto-commit with an LLM-generated commit message (`commit_message.py`) after non-worktree tasks.

### ChatSorcarAgent and WorktreeSorcarAgent

- Chat continuation: earlier tasks of the same `chat_id` are replayed as context, with a digest for older entries (`chat_history_digest`).
- Every event (system prompt, prompt, thinking, text, tool call, tool result, usage, result) is persisted to `sorcar.db` for live mirroring, resume, replay and `/ask`.
- Worktree isolation per task (section 12).

## 6. Built-in tools

Tools the model can call in a standard run, grouped by concern. Names are the exact function names registered in `SorcarAgent._get_tools`.

| Group | Tools | Notes |
|---|---|---|
| Files and shell | `Bash`, `bash_job` **NEW**, `Read`, `Write`, `Edit`, `run_commands_parallel` | Bash has per-call timeouts and output caps; `Bash(background=true)` detaches the command with `nohup` and returns a job id and log path; `bash_job(job_id, action="wait"\|"tail"\|"kill", timeout_seconds, tail_lines)` follows it; logs go to `tmp/bash_jobs/<job_id>.log` and a task Stop kills the job's process group like a foreground command. Parent-repo path guard rewrites paths into the active worktree; `run_commands_parallel` runs shell commands in threads with no LLM. |
| Browser | `go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser` | Chromium driven through Patchright **NEW** (installed Google Chrome preferred; headed on an Xvfb display on servers), numbered accessibility tree, persistent profile, human-like mouse paths and typing cadence, challenge-vendor detection, watchdog kill of stuck processes; `show_browser` makes the window visible for user login; sign-in pages are handed to the user's default browser instead of being automated. |
| Delegation | `run_parallel(tasks, max_workers, model_name, tool_profile)`, `number_of_cores`, `run_agent(task, agent="", …)` | `run_parallel` spawns LLM sub-agents (nesting capped at 2); `run_agent` dispatches a channel agent, `cron`, any agent-script `.py` file, or (with `agent` empty) a plain sub-agent, and now accepts every `sorcar.run` option (`chat_id`, `system_prompt`, `tools`, `model_config`, `use_worktree`, `auto_commit`, `tool_profile`, …) **NEW**. |
| Decisions | `decide(state, questions)` | Typed `noul`/`choice`/`score` questions answered by OpenRouter's decisions model (Jev) with calibrated probabilities; present only when an OpenRouter key exists. |
| Memory | `memory_search`, `memory_pull`, `memory_read`, `memory_write`, `memory_list`, `memory_refresh`, `memory_delete` | Section 10. |
| User interaction | `ask_user_question`, `talk(language, text, emotion)` | Question panel inside the transcript of every client **NEW** (a plain chat reply also answers it); spoken answer synthesized server-side. |
| Control | `set_model`, `summary`, `finish(success, is_continue, summary_in_html, suggested_next_task)` | `set_model` swaps the model mid-run and carries the conversation over; `summary` collapses the last ten steps in the UI. |
| Skills and MCP | `skill(name)`, `<server>_<tool>` | `skill` appears only when user or project skills exist; MCP tools are generated from each server's schema. |
| Channel tools (dispatched runs) | e.g. `check_slack_auth`, `authenticate_slack`, `post_message`, `read_messages`, `list_messages`, `send_whatsapp_message`, `gdrive_upload_file`, `github_search_repositories`, `cron_job` | Added when the run is a channel or cron agent. |

## 7. Models and multi-model routing

- **Catalog**: 664 models in `MODEL_INFO.json` with prices, context lengths and capability flags (`fc`, `gen`, `emb`, `dec`) across OpenAI, Anthropic, Gemini, Together AI, Z.AI, Moonshot AI, OpenRouter, Claude Code CLI `cc/*` and Codex CLI `codex/*`. Rolling aliases such as `openrouter/~anthropic/claude-opus-latest`.
- **Adapters**: Anthropic native, OpenAI-compatible (Chat Completions and Responses API), Gemini, Claude Code CLI, Codex CLI, Decisions. Reasoning-effort variants (`-low/-medium/-high/-xhigh`) are catalog entries.
- **Mix vendors inside one task**: `set_model` mid-run, `run_parallel(model_name=…)` for sub-agents, per-channel and per-cron model overrides, model picker per tab, live model switch from the UI while a task runs.
- **Routing by prompt**: bundled "tricks" pair a builder model with a read-only reviewer (e.g. `claude-fable-5-1` + `gpt-5.6-sol`); a trick reads `./ROUTING.md` and asks the agent to update the routing strategy after each task.
- **Custom models**: `~/.kiss/MY_MODELS.json` entries (local vLLM/Ollama or any OpenAI-compatible endpoint) appear in the picker; custom endpoint, API key and HTTP headers per run via `model_config` or the Settings panel.
- **Catalog maintenance**: `update_models.py` fetches vendor pricing, tests generation/embedding/function-calling/decisions, and rewrites the catalog; `update_responses_api_support.py` live-verifies Responses API support; an "Update models" button in Settings.
- API keys live in `$KISS_HOME/api_keys.env`, the single key store the daemon parses at startup; settable from the Settings panel or environment.

## 8. Chat client features

Verified against element ids in `media/chat.html` and message types in `media/main.js`.

- **Input**: `@` file/folder mentions with ranked completion (source files ranked above data dumps; scans up to 1,000,000 entries **NEW**); ghost-text autocomplete from input history and frequent tasks, active while a task runs and fed by steer-mode messages too **NEW**; `/` command popup **NEW**; attachments by picker, paste or drag-and-drop (images, PDFs); input history; clear; tricks panel with search and quick-add **NEW**; tips panel; sample-task chips on the welcome screen; "Suggested next" follow-up after each task.
- **Run controls**: model picker with search, per-task budget cap, Stop, per-panel tool interrupt, live steering (send a message into a running task), `<task>…</task>` to queue a follow-up task, model switch during a run, `/ask` questions about the running task answered alongside it **NEW**.
- **Rendering**: streamed text and thinking, collapsible tool panels (the two newest stay open while streaming **NEW**), unified spinner/tick/cross status icons **NEW**, syntax-highlighted code (highlight.js, light/dark theme toggle), inline images produced by tools, clickable file paths that open in the editor (existence verified via `checkPaths`), copy buttons, nested sub-agent panels, collapsed groups after every `summary`, in-transcript Question panel for `ask_user_question` **NEW**.
- **Tabs**: server-canonical tab registry mirrored across every VS Code window and web client on the same workspace; sub-agents open their own tabs; adjacent-task navigation within a chat.
- **History**: search, filters (workspace, running/completed/errors, date range, favorites), resume a chat, open a task from history, the active task highlighted and scrolled into view with a "launched N minutes ago" label **NEW**, frequent-tasks panel, share a chat or selected tasks as a standalone HTML page (`share.js`).
- **Workspace views** (web app): Explorer tree with New File/Folder, Rename, Delete, Cut/Copy/Paste, Find in Folder, Compare; Source Control view with branch, changes, commit graph, `git show`; file open/save from the browser; "Working directory" panel in the "…" menu with a directory picker and a recent-directories list shared by all clients **NEW**.
- **Git actions**: autocommit button and progress, worktree merge/discard/leave prompts, main-tree actions, generate commit message.
- **Settings panel**: work dir, default max budget, worktree on/off, auto-commit, classify tasks, classify with Jev, web tools on/off, memory on/off and memory directory, editor-tabs mode, voice sensitivity and auto-submit, remote password, remote URL, API keys, custom endpoint/headers/model, custom models list, Update button (with "Update when idle" on the toast **NEW**), Update models, Server reset.
- **Status**: bottom bar with steps, tokens, budget, machine; Task Info drawer with task id, chat id, parent, model, date, time, work dir, worktree and parallel flags; daemon status, reconnect handling and update-available notices.

## 9. Voice

- Wake-word listener (`voice_wake.py`, Vosk model auto-downloaded) runs as a daemon child process; fuzzy match on "Sorcar" (Levenshtein), sensitivity levels, silence trimming, speaker identification, language detection.
- Speech-to-text of the captured utterance through a catalog model; transcripts arrive as `Speaker #N says in the language xx that: …` (114 such top-level tasks recorded).
- Spoken replies: the `talk` tool synthesizes MP3 server-side with `gpt-audio-1.5` (emotion hint supported), broadcast to every client tab of the task, with a system-TTS fallback on the daemon (`talk_player.py`).
- Acknowledgement sound (`working-on-it.mp3`) and a listening overlay in the UI; voice auto-submit setting.

## 10. Persistent memory (memoryfield)

- Markdown pages with generated frontmatter under `~/.kiss/memories` (or a custom directory), indexed in SQLite with vector embeddings: OpenAI embeddings when a key exists, otherwise an offline hashed embedder.
- Seven tools (search, pull, read, write, list, refresh, delete); `memory_refresh` re-indexes external edits and reports near-duplicates and stale pages; concurrent `sync()` calls no longer overwrite fresher rows **NEW**.
- A memory protocol in the system prompt: recall before work, record durable knowledge, never store secrets, keep per-session notes in `./tmp/PROGRESS.md`.
- Disabled for Docker runs, `cc/*`/`codex/*` models, runs without the built-in toolset, or runs with a caller-supplied system prompt; toggle with `KISS_USE_MEMORY` or the settings panel.
- `memoryfield/evaluate.py` measures Recall@k and MRR on real past tasks from `sorcar.db`.

## 11. Task classifier

- Runs once before each task to decide (a) whether the task edits files (worktree needed) and (b) whether it is simple (use `SYSTEM_LITE.md`, 5.1 KB, instead of `SYSTEM.md`, 24.6 KB).
- With an OpenRouter key it asks `~typesafe/jev-latest` one typed question (about 0.2 s, $0.00003); otherwise one non-agentic structured-output call on the run's model with a plain-text retry; skipped for CLI models.
- Verdicts are cached on disk; classifier spend folds into the task's totals; benchmark of 415 prompts in `benchmarkings/task_classifier/`.

## 12. Git integration and worktrees

- Each development task runs in a fresh git worktree on a unique branch (`.kiss-worktrees/`); a spare worktree is pre-created in the background (`worktree_pool.py`) so tasks start in well under a second on large repos; the main checkout's `node_modules` directories are symlinked in so JS tooling works without `npm install` **NEW**.
- On success: auto-commit and merge back, or an interactive merge/discard/leave-as-is prompt. **NEW** A conflicting auto-commit merge is handed to the bundled merge agent (`merge_sea.py`, run in-process) before falling back to the manual conflict message; the same agent is available as `/merge`.
- **NEW** Deferred worktrees (left as-is because the parent branch had uncommitted work) are merged automatically once that branch is committed (`merge_flow.py`).
- Failed or stopped tasks leave their work committed on the branch; retired worktrees are cleaned up when no abandoned sub-agent still writes to them; stash-pop and merge-conflict warnings are surfaced; ignored files can be rescued on discard.
- Bash and file tools rewrite parent-repo paths into the worktree and block edits that would escape it (the Bash tool rejects any command naming the main checkout's path).
- Non-worktree tasks get post-task auto-commit with a generated message; the extension exposes Generate Commit Message and Git Commit commands.
- 2,767 task rows carry the `is_worktree` flag; sub-agents share their parent's worktree without setting it (see section 21 for the correction to the previous inventory).

## 13. Scheduled automations

- Natural-language requests are dispatched to the cron agent (`run_agent("cron", …)`, `/cron …`, or `kiss-cron` from the shell). Its `cron_job` tool creates, lists, pauses, resumes, removes or immediately runs jobs stored in `~/.kiss/cron/jobs.json`; creating a job that already exists is deduplicated **NEW**.
- Schedules: 5-field cron expressions, intervals (`every 30m`, `every 2h`) and one-shot timestamps; a scheduler thread in `kiss-web` ticks about once a minute; **NEW** jobs run concurrently, each in its own scratch directory.
- Two job kinds: **prompt jobs** run an unattended Sorcar task (never asks questions, replies `[SILENT]` to suppress delivery), now launched as a generated SEA file through `run_agent` with per-job `work_dir`, `use_worktree`, `auto_commit` and `timeout` **NEW**; **command jobs** run a shell command without a model (whole process tree killed on timeout).
- **NEW** **Gateways**: "poll Slack #sorcar every minute and answer" becomes a command job running the channel CLI's tick, so nothing is spent while the chat is quiet.
- Delivery of results to authenticated channels (`telegram:123456`, `email:user@example.com`, …; 25 of 32 channels), with `until_delivered` retry semantics; outputs kept under `~/.kiss/cron/output`.
- Recorded: 34 `cron_job` calls and 61 unattended cron-launched tasks; four jobs are configured on this machine (a daily check, two one-shot jobs and a Slack gateway ticking every 60 s).

## 14. Third-party agents and credential isolation

44 `*_sea.py` modules in `src/kiss/agents/third_party_agents/` (renamed from `*_agent.py` in this release), each a Sorcar Extension Agent whose `tools()` returns auth tools plus the backend's public methods once authenticated, and each reachable as the slash command `/<service>`. Configuration lives under `~/.kiss/third_party_agents/<service>/`; authentication happens by chatting.

| Kind | Agents |
|---|---|
| Messaging and device channels (32) | BlueBubbles, DingTalk, Discord, Email (IMAP/SMTP), Feishu, Gmail, Google Chat, Home Assistant, iMessage, IRC, LINE, Matrix, Mattermost, Microsoft Teams, Nextcloud Talk, Nostr, ntfy, Phone Control, QQ, Signal, SimpleX, Slack, SMS, Synology Chat, Telegram, Tlon, Twitch, Webhook, WeCom, WeiXin, WhatsApp, Zalo |
| Service APIs (9) | Brave Search, Firecrawl, GitHub, Google Calendar, Google Docs, Google Drive, Google Sheets, Notion, PostgreSQL |
| Infrastructure (2) | A2A protocol agent, OpenAI-compatible server |
| Sorcar itself (1) **NEW** | `ask_sea.py`: `/ask` answers questions about the running task from its persisted events, offline, with `SYSTEM_LITE.md` and no worktree |
| Bundled in `agents/seas/` **NEW** | `merge_sea.py` (`/merge`), `sh_sea.py` (`/sh`), `dummy_sea.py` (the plain sub-agent behind `run_agent` with no `agent`) |
| Extra | `govee.py` smart-light CLI (on/off, brightness, color, color temperature) |

- **Dispatch**: the top-level session names the service and calls `run_agent` at once (the system prompt forbids exploring the agent's source first); channel names match forgivingly; multi-account workspaces via `KISS_CHANNEL_WORKSPACE`; dispatched sessions cannot dispatch further; a channel or cron sub-agent never edits source files or runs test suites.
- **Gateway mode** for channels with an inbound stream (section 4) and **scheduled delivery** (section 13).
- **Muse auth** (Linux): credentials for 24 connectors live in a vault owned by a local auth daemon; the agent holds only surrogate tokens swapped at the network edge; every request is host-allowlisted, classified read/write and checked against an allow/deny/ask policy with an audit log; OAuth device grants (GitHub, Twitch, Microsoft Teams), Nextcloud Login Flow v2, Matrix OAuth device grant, Signal QR linking; CLI `python -m kiss.agents.third_party_agents.muse_auth` with `status`, `enroll`, `import`, `grant`, `revoke`, `audit`, `clear`, `daemon`, `stop`, `export`; opt out with `KISS_MUSE_AUTH=0`.
- **NEW** **Browser hand-off** (`_browser_handoff.py`): consent, device-code and bot-token pages open in the user's own default browser (`$BROWSER`, `open`, `xdg-open`) with the URL and code echoed in chat; the agent never asks for a password or 2FA code.

## 15. Extension points: slash commands, SEAs, MCP, skills

- **NEW** **Your own slash commands**: put `xxx_sea.py` files in any folder and list the folder in `~/.kiss/SEAS.md` (one path per line, `~` and environment variables expanded, `#` comments). `/xxx` exists within two seconds, no restart; a later line overrides an earlier one and any listed folder overrides the bundled `agents/seas/`. Documented in `website/kisssorcar.github.io/docs/sea-commands.md`.
- **Sorcar Extension Agents**: a Python file whose top-level getters (`prompt()`, `model()`, `max_budget()`, `system_prompt()`, `tools()`, `use_worktree()`, `tool_profile()`, …) compute run parameters on the daemon, type-checked and applied atomically; `llm_call_hook` rewrites outgoing messages and `tool_call_hook` can veto tool calls. `run_agent` runs any SEA file by path.
- **MCP servers**: discovered from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, `<project>/.mcp.json`; stdio, HTTP and SSE transports; OAuth 2.1 with dynamic client registration and PKCE, tokens under `~/.kiss/mcp_auth/`; tools exposed as `<server>_<tool>` and filtered by `mcp_permissions` wildcard rules; save/remove servers programmatically.
- **Connector catalog** (`connectors/`): 16 curated privacy-first MCP servers (fetch, time, memory, sequential-thinking, deepwiki, context7, github, google, slack, twilio-sms, whatsapp, brave-search, notion, postgres, firecrawl, playwright) with `enable.py`/`verify.py` CLIs; credentials read from the shell, never stored.
- **Agent Skills**: loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills` and bundled skills; a single `skill(name)` tool loads `SKILL.md` and lists bundled resources; permission rules per skill.
- **Tools file**: `tools="/path/my_tools.py"` whose `get_tools()` returns extra callables; `append_basic_tools=False` restricts the agent to `finish` plus those tools.
- **Prompt customization without code**: `~/.kiss/MY_TASK_TEMPLATES.md` (sample-task chips), `~/.kiss/MY_INJECTION.md` (tricks), `~/.kiss/MY_MODELS.json`, `~/.kiss/SEAS.md`, per-repo `SORCAR.md`, optional `ROUTING.md`; 12 bundled sample tasks, 10 bundled tricks and 22 tips.

## 16. Persistence, accounting, observability

- **`~/.kiss/sorcar.db`** (WAL SQLite): `task_history` (task, result, chat_id, model, work_dir, version, tokens, cost, steps, flags, start/end, favorite, parent_task_id, owner, max_budget), `events` (ordered JSON per task), `model_usage`, `file_usage`, `frequent_tasks`, `replayed_journals`. Journals protect against crashes; `lost_and_found` shows a past recovery. `/ask` reads this database to answer questions about a live task.
- **Live accounting**: cost, tokens, steps and budget appear in every tool result, the status bar and the Task Info sidebar; sub-agent, classifier and TTS spend fold into the parent task.
- **Cost KPIs**: `scripts/cost_report.py` computes token-cost KPIs (peak context, hand-offs, compaction, cache hit rate) over a database; `cost_levers_experiment.py` runs A/B experiments of the cost levers (`read_dedupe`, `tool_output_compaction`, `tool_profiles`, `review_budget_fraction`, `chat_history_digest`, `dispatch_path_rewrite`, `context_limit_fraction` in `config.py`). Two 24- and 72-hour "waste audits" over this database drove the cache keep-alive, compaction gate, batching rules and zero-progress stop in this release.
- **Trajectory viewer**: `kiss.viz_trajectory.server` is a Flask app that renders saved trajectory YAML files under `.kiss.artifacts`; `/api/jobs` endpoints in the web server expose benchmark job trajectories.
- **Shareable chats**: any chat or task selection exports to a self-contained HTML page.
- **Daemon health**: GIL-independent stall watchdog, daemon health checks and restart verification in the extension, logs in `~/.kiss/kiss-web-*.log`, active-task queries (`scripts/check-kiss-web-active-tasks.py`).
- **Database tooling**: `sync_db.py` (one-way merge of task and event rows), `carry_over_tables.py`, `db_fingerprint.py`, `relocate_work_dir.py`, `running_tasks.py`, `scripts/sync-task-db.sh`.

## 17. Installation, deployment, operations

- **One-line install** (`install.sh`, 63 KB): bootstraps Homebrew/Xcode CLT, git, node, uv, Python 3.13, the browser driver, builds and installs the VSIX, launches VS Code; interactive or `--non-interactive`; immune to terminal signals; logs to `~/.kiss/install.log`. Also `pipx install kiss-agent-framework` for the daemon and CLIs only.
- **Remote deployment** (`rsorcar user@host`): checks SSH, installs prerequisites, verifies no task is running and disk space suffices (with a bind-mount helper to move `$HOME`), copies SSH identity, syncs the checkout through `origin`, ships the task database, sets the remote work dir, starts `kiss-web` and waits for the public cloudflared URL. **NEW** The remote password defaults to this machine's own so both share one; API keys are copied only to a remote that has none, so a re-deploy never rolls back a rotated key.
- **Docker**: `sorcar-docker [PORT] [--rebuild]` builds the image from `Dockerfile`, runs code-server with the extension in a container and opens the browser; per-task `docker_image` isolation for tools.
- **Updates**: Settings "Update" button re-runs the installer; the update toast offers "Update when idle" so a running task finishes first **NEW**; the extension checks for new versions with a snooze shared between daemon and clients; `scripts/release.sh` packages releases.
- **Servers without a display** **NEW**: the browser runs headed on an Xvfb virtual display when available, which is what defeats most bot checks; without Xvfb it falls back to headless.
- **Auth helpers**: `install-api-keys.sh`, `install-github-auth.sh`, `collect-github-auth.sh`, `install-ssh-identity.sh`.
- **Platforms**: macOS and Linux (x86_64, aarch64, arm64); Windows bash discovery exists in the tools; Muse auth is Linux-only.

## 18. Long-horizon research drivers and benchmarks

The system prompt contains procedures that turn one chat message into a multi-hour autonomous loop:

- **AI discovery / auto research / optimization** (`SYSTEM.md` §workflow): profile baseline, web-search state of the art, write `./tmp/ideas.md`, judge ideas pairwise, implement and evaluate end to end, log to `./tmp/explored-ideas.md`, compose winners, stop only when the metric goal holds on a held-out check.
- **Adversarial testing**: one sub-task tries to break the system with adversarial tests and workloads while another fixes it. **Adversarial training**: generate adversarial datasets until a model stops overfitting.
- **GEPA prompt optimization** and **repository optimization** as bundled sample prompts; recorded runs include xxHash (5,583 steps, $393), SQLite, LZ4 and the HydraKV and Bespoke OLAP case studies described in `papers/kisssorcar/kiss_sorcar.tex`.
- **Paper writing and review**: `templates/write_paper_prompt.md` and `RECIPES.md` give prompts for venue-conformant LaTeX papers with citation verification and AI-slop gates, and for deep conference reviews; the repo holds five paper directories (KISS Sorcar, SE KISS Sorcar, SorcarCCL, SweDefend, HydraKV) written this way.
- **Deep work, planning, file-browsing rules**: read-before-modify, concrete per-file plans for 3+ file changes, end-to-end tests with 100% branch coverage and no mocks, parallel test splitting by core count, single lint/typecheck at the end, three-item pre-finish check; **NEW** batching of independent shell commands into one call, background jobs for anything over a minute, reviewer dispatch wording, and a flake rule (a failure that passes in isolation and once on re-run is named and not chased).
- **Web research protocol**: a confidence gate that mandates search whenever facts may be stale, a ten-site research procedure, and reports for answers over ~800 words.
- **Benchmarks** (`benchmarkings/`): Harbor agent for data-eng-bench (103 tasks, 309 trials, 69.58% accuracy, pass@3 0.728, $1,143 total); `harnesstax` runners for Terminal-Bench 2.0 (30 sampled tasks) and SWE-bench Lite with official graders; task-classifier benchmark; `projects/swedefend` (a prompt-injection defense pipeline with `swedefend-eval`); **NEW** a 14-host bot-protection probe for the stealth browser (4/14 → 11/14 passed).

## 19. Developer tooling and tests

- `uv run check --full`: syntax, ruff lint, pyright/mypy type check, extension typecheck and lint, with every failed stage listed at the end; JS side `npm run check` (tsc, eslint, stylelint, htmlhint, node tests).
- 1,260 Python test files with 9,576 test functions under `src/kiss/tests/` (up from 1,228 / 8,669), mirroring every package (core, models, memoryfield, sorcar, server, third-party agents, seas, vscode, viz, scripts, swedefend); 324 JS test files for the extension in `agents/vscode/test/`.
- `generate-api-docs` builds `API.md` (84 KB) from the source by AST introspection; `redundancy_analyzer.py` finds redundant tests via branch coverage contexts.
- Type hints throughout (`py.typed`), pyproject-driven tooling, GitHub Actions workflow under `.github/`.

## 20. What the trajectories show

The task database records every task run on this machine and on machines whose databases were synced into it (73 distinct top-level work directories; the main checkout accounts for 5,462 of the 5,783 top-level tasks; 46 distinct model names). Rows before April 2026 are legacy imports; the six months from April to September 2026 hold the working history. All figures were re-queried on 2026-09-21.

- **5,783** top-level tasks · 18,525 sub-agent rows · 2,904 chats
- **8.63 M** persisted events (5.26 M thinking deltas, 499K tool calls, 478K tool results)
- **242** top-level tasks longer than one hour; 23 with 1,000+ steps; 40 costing $100+
- **8,056** steps in the longest task ($762, a repo-wide race-condition audit); costliest task $845 with 456 M tokens over 7,016 steps

**Tool calls recorded in sorcar.db**

| Tool | Calls | Group | Related tools and notes |
|---|---:|---|---|
| `Bash` | 235,419 | files/shell |  |
| `Read` | 148,422 | files/shell |  |
| `Edit` | 40,259 | files/shell |  |
| `summary` | 22,757 | control and interaction |  |
| `finish` | 17,568 | control and interaction |  |
| `go_to_url` | 8,738 | browser |  |
| `Write` | 8,187 | files/shell |  |
| `set_model` | 4,187 | delegation and model switching |  |
| `run_parallel` | 4,045 | delegation and model switching |  |
| `memory_search` | 1,637 | memory and decisions | `memory_pull` 828, `memory_write` 589, `memory_read` 221 |
| `screenshot` | 610 | browser |  |
| `click` | 468 | browser | `get_page_content` 177, `type_text` 140, `press_key` 132, scroll 117 |
| `run_commands_parallel` | 415 | files/shell | 78 on 2026-09-20: five-fold growth in one day after the batching rules |
| `ask_user_question` | 342 | control and interaction |  |
| `talk` | 254 | control and interaction |  |
| `run_agent` | 217 | delegation and model switching | channel tools such as `list_messages` 108, `cron_job` 34, `check_gmail_auth` 12 |
| `decide` | 114 | memory and decisions | 59 on 2026-09-20 |
| `bash_job` | 14 | files/shell | added 2026-09-21 |

*Figure 2. 498,673 tool calls. Shell and file tools dominate, as expected for a coding agent; `set_model` (4,187) and `run_parallel` (4,045) show that multi-model, multi-agent work is routine. The new cost rules are visible already: `run_commands_parallel` went from 78 to 415 calls and `decide` from 59 to 114 in the day since the previous inventory. The 1,524 `code_graph` calls belong to a tool that no longer exists.*

**Top-level tasks per month, 2026**

| Month | Apr | May | Jun | Jul | Aug | Sep (to 21st) |
|---|---:|---:|---:|---:|---:|---:|
| Top-level tasks | 513 | 1,807 | 888 | 883 | 843 | 847 |

*Figure 3. Sustained daily use since April 2026: about 30 top-level tasks per day (135 in the day since the previous inventory), each often fanning out into several sub-agents (3.2 sub-agent rows per top-level task on average).*

### Models actually used

| Model | Task rows | Model | Task rows |
|---|---:|---|---:|
| gpt-5.6-sol | 7,946 | claude-opus-5 | 505 |
| claude-fable-5 | 7,353 | claude-opus-4-8 | 391 |
| gpt-5.6-sol-xhigh | 2,972 | gpt-5.5 | 159 |
| claude-opus-4-7 | 2,655 | claude-sonnet-4-5 | 52 |
| claude-opus-4-6 | 1,098 | codex/gpt-5.5 · cc/opus | 43 · 37 |
| claude-fable-5-1 | 839 | gpt-6-astra-high · gpt-5.5-xhigh · openrouter/moonshotai/kimi-k3 | 35 · 24 · 22 |

The split is the builder/reviewer pattern from the tricks: Anthropic models build, OpenAI `gpt-5.6-sol` reviews in read-only sub-agents. Thirty further models appear in small numbers, including Gemini, Kimi, Sonnet, Haiku and the CLI-backed `cc/*` and `codex/*` namespaces. The 88 `claude-fable-5-1` rows added since the previous inventory match the current default trick.

### Kinds of work recorded

Keyword counts over the 5,783 top-level task texts (a task can fall in several rows; counts recomputed on 2026-09-21 except where marked):

| Kind of task | Tasks | Examples from the history |
|---|---|---|
| Software changes (fix, bug, implement, refactor, add) | 1,980 | "Extend Muse-auth to the remaining credentialed connectors…" (2,289 steps); "major refactoring of the project to significantly simplify the implementation" (2,281 steps) |
| Testing | 1,318 | "run all tests… split by the number of test methods into cores − 2 and run all splits in parallel" (a frequent task, up to 3,519 steps) |
| Code review and audits | 988 | "Find all race conditions, deadlocks, obvious bugs, missing wiring, redundancies…" (8,056 steps, $762); the 24-hour agent-waste audit that produced this release's cost rules |
| Git operations | 663 | "git push origin", "get rid of the last commit to main", "checkout main", merge-conflict help |
| Web research | 631 (2026-09-20 pass) | trip planning to Yosemite with hotel availability, product comparisons, market reads, STT-for-Indian-languages survey |
| Reports | 406 | 67 HTML reports in `reports/`: audits, optimization reports, architecture notes, comparisons, this inventory |
| Papers and LaTeX | 402 | KISS Sorcar, SE KISS Sorcar, SorcarCCL, SweDefend, HydraKV papers; ICSE/ACL/OpenReview reviews |
| Deployment and remote machines | 317 | `rsorcar` deploys, VS Code server checks, disk-space remediation |
| Messaging (Slack, Gmail, WhatsApp, Telegram, iMessage, SMS, email, Discord) | 178 | Slack workspace authentication, Gmail checks, WhatsApp pairing, SMS sends, Slack gateway pollers |
| Shopping, travel, price comparison | 123 (2026-09-20 pass) | non-stick cookware comparison, Yosemite trip, due-diligence research |
| Benchmarking and optimization | 120 (2026-09-20 pass) | xxHash, SQLite, LZ4, DuckDB and TPC-H optimization runs; Terminal-Bench and SWE-bench jobs |
| Voice tasks ("Speaker #N says…") | 114 | spoken questions and steering through the wake word |
| Blog posts, LinkedIn posts, slides | 94 | SQLite and LZ4 optimization blogs, lecture PPTX and one-slide decks |
| Unattended cron jobs | 61 | VS Code release watcher with spoken notification; Slack gateway ticks; classroom-team checks |
| Docker | 40 | code-server container launches, Docker-isolated tool runs |
| Home devices (lights, Govee, Home Assistant) | 25 (2026-09-20 pass) | Govee light control |
| Slash-command and `/ask` runs **NEW** | 5 · 9 | first uses since the commands landed on 2026-09-21 |

## 21. Caveats and gaps

- **Correction to the 2026-09-20 inventory**: it stated that 22,023 of 24,063 task rows ran with worktree isolation. That number is not reproducible from the database: `is_worktree = 1` holds for 2,767 rows (3,484 counting rows whose work dir lies under `.kiss-worktrees`). Sub-agents work inside their parent's worktree without carrying the flag, so the true share of isolated work is higher than 2,767 rows but cannot be read off a single column. It also counted 50 CLI entry points; the scripts table has 49 and was unchanged between the two versions.
- `SYSTEM.md`'s "Desktop Apps" rule refers to screenshot, keyboard and mouse tools, but the only `screenshot`/`click`/`press_key` tools registered operate on the browser page, not the desktop. Desktop control is available only through channel agents such as Phone Control or through shell commands.
- Cron is not a built-in tool of a normal run: it works through `run_agent("cron", …)` or `/cron`. The `skill` tool is registered only when a user or project skill directory exists; bundled skills alone do not add it.
- Slash commands are resolved by the daemon: the `sorcar` terminal command and the Python API do not rewrite `/xxx` prompts (they can call `run_agent` or pass `extension_agent_path` instead).
- Memory, the classifier and prompt caching are unavailable for `cc/*` and `codex/*` runs, which hand the whole task to the external CLI.
- The `summary`-every-ten-steps rule is enforced by the prompt only; there is no mechanical check.
- The stealth browser passed 11 of 14 probed bot-protected hosts; Cloudflare Turnstile still rejects on some, and Xvfb is needed on a server for the headed mode that does most of the work.
- Muse-auth isolation is a process boundary, not an OS boundary: an agent allowed to run arbitrary shell commands could read the vault unless tool permissions deny it (stated in `muse_auth/__init__.py`).
- The database contains artifacts: 64 task rows with an empty model, a minimum timestamp of 2009 (a clock artifact on an imported row), 1,524 calls to a retired `code_graph` tool, and a `lost_and_found` table from a past recovery.
- Counts re-verified against the tree for this version: 664 models, 44 + 3 SEAs, 49 scripts, 1,260 / 9,576 tests, 10 tricks, 12 sample tasks, 22 tips. The 24 Muse-covered connectors and 25 delivery-capable channels are taken from the README and the third-party README.

Sources: `README.md`, `src/kiss/SYSTEM.md`, `src/kiss/SYSTEM_LITE.md`, `src/kiss/TIPS.md`, `src/kiss/INJECTIONS.md`, `src/kiss/SAMPLE_TASKS.md`, `RECIPES.md`, `pyproject.toml`, `src/kiss/core/*`, `src/kiss/agents/sorcar/*` (including `sea_commands.py`, `web_stealth.py`, `cron_agent.py`, `agent_dispatch.py`, `useful_tools.py`), `src/kiss/agents/seas/*`, `src/kiss/server/*`, `src/kiss/agents/third_party_agents/{README.md,ask_sea.py,_browser_handoff.py}`, `connectors/README.md`, `src/kiss/agents/vscode/{package.json,media/chat.html,media/main.js,media/sw.js}`, `website/kisssorcar.github.io/docs/sea-commands.md`, `benchmarkings/`, `reports/anti-bot-blocks-2026-09-20.html`, `reports/agent-waste-audit-2026-09-21.html`, `install.sh`, `rsorcar`, `sorcar-docker`, `git log d684b3193..85d9f3cda`, and SQL queries against `~/.kiss/sorcar.db` on 2026-09-21.
