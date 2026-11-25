# KISS Sorcar Feature Inventory

Version 2026.11.7 · HEAD `d522176885` (6,001 commits) · compiled 2026-11-10 · supersedes the 2026-09-30 inventory

- **Verification**: Every count and feature in this document was checked against the source tree at `d522176885`, the packaged assets under `src/kiss/`, and the development machine's task database on 2026-11-10.
- **Change markers and dates**:
  - Items marked **NEW** did not exist at the previous inventory's commit `d450c0be9a` (2026-09-30).
  - Items marked **CHANGED** existed there and behave differently now; the 1,075 commits between the two are summarised in section 2.
  - A date such as 10-09 is the author date of the commit that made the change, in 2026.
- **Home directory**: `~/.kiss/` stands for the KISS home directory, resolved in order:
  1. `$KISS_HOME` when it is set.
  2. Otherwise, the directory under the user's home that the build's `brand.json` names (`.kiss` in the stock build).
- **Source of truth**: Where the code and a document disagree, the code wins and the disagreement is listed in section 24.

## Contents

1. [At a glance](#1-at-a-glance)
2. [What changed since 2026-09-30](#2-what-changed-since-2026-09-30)
3. [AI discovery and optimisation on real software](#3-ai-discovery-and-optimisation-on-real-software)
4. [Architecture](#4-architecture)
5. [Interfaces and prompt surfaces](#5-interfaces-and-prompt-surfaces)
6. [Agent runtime](#6-agent-runtime)
7. [Built-in tools and tool profiles](#7-built-in-tools-and-tool-profiles)
8. [Models, routing and cost accounting](#8-models-routing-and-cost-accounting)
9. [Sorcar Extension Agents and slash commands](#9-sorcar-extension-agents-and-slash-commands)
10. [Sub-agents and parallelism](#10-sub-agents-and-parallelism)
11. [Task classification, tags and chat summaries](#11-task-classification-tags-and-chat-summaries)
12. [Git worktrees, merging and auto-commit](#12-git-worktrees-merging-and-auto-commit)
13. [Persistent memory and repository knowledge](#13-persistent-memory-and-repository-knowledge)
14. [The kiss-web daemon and remote web app](#14-the-kiss-web-daemon-and-remote-web-app)
15. [Machines: one web app over several computers](#15-machines-one-web-app-over-several-computers)
16. [Chat client features](#16-chat-client-features)
17. [Browser, web research and sign-in hand-off](#17-browser-web-research-and-sign-in-hand-off)
18. [Voice](#18-voice)
19. [Messaging and third-party agents](#19-messaging-and-third-party-agents)
20. [Scheduled automations](#20-scheduled-automations)
21. [Installation, deployment, Docker and release](#21-installation-deployment-docker-and-release)
22. [Developer tooling and tests](#22-developer-tooling-and-tests)
23. [What the trajectories show](#23-what-the-trajectories-show)
24. [Caveats, corrections and stale documentation](#24-caveats-corrections-and-stale-documentation)

## 1. At a glance

- **One agent, many surfaces** **CHANGED**:
  - The same `SorcarAgent` (as its recording subclass `ChatSorcarAgent` wherever a run is stored in `history.db`) runs behind:
    - The web app served by the `kiss-web` daemon.
    - The `sorcar` CLI.
    - The Python client API.
    - Headless SEA runs.
    - 42 messaging and service channels.
    - Cron jobs.
  - The web app is one page, reached in a browser tab, as an installed PWA, in the `kiss-app` desktop window or from an iOS Home Screen icon.
  - The VS Code extension was removed on 10-14.
  - Every client of a daemon mirrors its tab registry (`~/.kiss/tabs.json`) for chat tabs, and sub-agent and browser tabs are synchronised by events, so an open task, a running sub-agent or a browser tab appears in every client.
  - The `sorcar` CLI and a headless SEA run keep the agent in the calling process, without the daemon.
- **Several computers, one web app** **NEW**: A daemon keeps a registry of other computers' daemons (`~/.kiss/machines.json`) and drives them from its own page (section 15):
  - Their chats, files, settings, terminals and browser pages appear on the local surfaces.
  - `run_agent(machine=...)` and `Bash(background=True, machine=...)` send work to them.
  - `/sync` brings every registered machine and the local checkout to the same commit.
- **988 model entries** **CHANGED** in `src/kiss/core/models/MODEL_INFO.json` (section 8):
  - **Capabilities and aliases**: 937 generation, 803 with function calling, 6 embedding, 37 typed decisions; 529 base models and 459 effort aliases.
  - **Providers** **CHANGED** (11-19: Together AI, Z.AI and Moonshot are no longer called directly; Z.AI's and Moonshot's models stay available through OpenRouter):
    - 238 direct API models (OpenAI 99, Anthropic 82, Gemini 57).
    - 514 through OpenRouter.
    - 144 Claude models through Amazon Bedrock.
    - 79 `cc/` (Claude Code) and 13 `codex/` CLI-backed entries, driven as plain models with the KISS system prompt and tools and billed to the Claude or ChatGPT subscription by default.
  - **Routers**: Six router entries (`autorouter`, `bestrouter`, `orautorouter`, `orbestrouter`, `ccxbestrouter`, `brbestrouter`) appear in the model picker and are implemented as SEAs.
- **36 bundled Sorcar Extension Agents** **CHANGED** in `src/kiss/agents/seas/` (37 folders with the shared `base`; section 9):
  - 34 slash commands, 2 hidden (`coding`, `sorcar`) and 6 model-picker entries; 20 were added since 2026-09-30.
  - `src/kiss/agents/third_party_agents/` adds 44 SEA folders (43 commands, 42 of them channel SEAs), and `/cron` is built in, so a registry with an empty `SEAS.md` lists 78 commands.
  - Every SEA defines exactly one `BaseSea` subclass, a bundled one in `<name>/<name>_sea.py`, and every registered command defines `description()`.
- **44 built-in tool functions** **CHANGED** in a full-profile top-level run on the host, in parallel mode with web tools and memory on and the decisions model available, before skill, MCP-server and SEA-supplied tools (section 7):
  - **Tool counts by category**: shell 3, files 3, checkpoints 2, worktree 2, browser 15, memory 7, dispatch 4, MCP sign-in 2, interaction 6 including `finish` and, when the decisions model is available, `decide`.
  - **Tool profiles**: Fifteen tool profile names (`full`, `review`, `assistant`, `bash`, `none` and the ten tool groups `shell`, `edit`, `browser`, `memory`, `agents`, `mcp`, `skills`, `user`, `decide`, `control`), joinable with `+`.
- **54 console scripts** in `pyproject.toml`:
  - Core and developer CLIs: `sorcar`, `kiss-web`, `kiss-cron`, `kiss-app`, `sea`, `check`, `impacted-tests`, `findings-report`, `generate-api-docs` and `swedefend-eval`.
  - Channel CLIs: 44 `kiss-<channel>` CLIs.
- **Prompt assets** (section 5):
  - `SYSTEM.md` 6,237 words, `SYSTEM_LITE.md` 1,074 words, 22 tips and 2 bundled promptlets.
  - `SAMPLE_TASKS.md` (12 sample tasks) still ships, but no code reads it.
- **Codebase** (section 22):
  - 231,293 lines of non-test Python under `src/kiss` (server 45,818; `agents/sorcar` 64,809).
  - 2,073 `test_*.py` files with 14,406 test functions (558,090 lines across the 2,140 Python files under `src/kiss/tests`).
  - The 32,016-line `main.js` of the web app under `src/kiss/webapp/media/` (41,714 lines with the other media scripts and vendored JavaScript) and 407 JS test files.
- **Task database** (development machine, rows dated before 2026-11-10 UTC; section 23):
  - 41,127 task rows (9,685 top-level, 31,442 sub-agent) in 8,614 chats.
  - 23.2 million persisted events and 968,429 tool calls.
  - 1,332 distinct working directories on top-level tasks (1,205 of them per-run cron directories; 2,534 across all rows) and 87 named models.
  - 13,165 of the rows ran on 8 other machines and were copied in by the history sync.
  - Recorded spend on top-level tasks is **$77.6K** across 69.8 billion tokens and 675K steps.

## 2. What changed since 2026-09-30

1,075 commits (931 without the 144 merges) separate the previous inventory at `d450c0be9a` (version 2026.9.27) from `d522176885`:

- Their author dates run from 2026-09-30 to 2026-11-10 (UTC) with a peak of 90 commits on 10-08.
- They bump `src/kiss/core/_version.py` 37 times, from 2026.9.28 to 2026.11.7, 31 of those versions carrying a release tag (`v2026.9.28` through `v2026.11.4`).

### SEA contract and hooks

- A SEA is one `BaseSea` subclass per file (`src/kiss/agents/seas/base/base_sea.py`, 10-05); the launcher folds the methods of the class chain, base first, and `BaseSea` is the root layer of every chat-launched run (10-06).
- **Methods**: `description`, `settings`, `prompt`, `system_prompt`, `tools`, `llm_call_hook`, `tool_call_hook` and `llm_response_hook` (both return a `Verdict`: `ALLOW`, `refuse`, `end_run`, `allow_with_note`), `register_as_model`, `on_picked_as_model`.
- `WorkerSea` and `ChannelSea` replace the `kind` and `channel` settings (10-07).
- 14 settings keys with one precedence rule and `locked` keys; the `model_config` flag `plain_io`; `@tool_schema` and `LLMReply` (10-31).
- The `sea` console script (`sea lint`, `sea docs`), headless runs (`headless.py`, `scripts/run_sea.py`) and help for the slash commands a task names. Details in section 9.
- **Commits**: `08c60f475d` `7762e96571` `ec0ef731ac` `47f5bacdb5` `16c8eb13a3` `e2ff4cd19f` `a17891dc6d` `5cb7f1f98c`

### Tool-call guards

- Shell-rewrite and here-document guards that redirect to `Write` or `Edit`, beside the older read-before-modify rule, which a `Bash` view now satisfies (`edit_guards.py`).
- A quote-aware shell scanner (`shell_scan.py`).
- The destructive-command guard and an install timeout of at least 900 s (`shell_guards.py`).
- A foreground `sleep` over 30 s is refused when it outlasts the call's timeout and otherwise spent waiting on the running background job, if there is one and the sleep can be taken out of the command (one that paces a loop cannot).
- A review sub-agent of its caller's model family is moved to another configured family (`second_opinion_model`) unless the call or the SEA's `model` setting pins its model or the daemon pins every task to its model (`KISS_PIN_TASK_MODEL`).
- A summary-due note every 10 steps. Details in section 6.
- **Commits**: `1509b8cc00` `38a98d0ce1` `efd88240b5` `bf69cb1b67` `79e5c494c1` `f6e4e2bd7f`

### Removals and renames

- **VS Code extension removal (10-14)**:
  - `src/kiss/agents/vscode/` (512 files) was deleted; the front end moved to `src/kiss/webapp`, `VSCodeServer` became `AgentServer`, `vscode_config` became `user_config`, and the host-side `voice_wake` listener went with it.
  - `install.sh` uninstalls leftover copies and the daemon shows a removal notice for 7 days (`vscode_retired.py`).
- The daemon's Unix socket gave way to a token-authenticated local WSS endpoint (10-01).
- Module-level SEA getters gave way to the `BaseSea` class (10-05).
- `sorcar.db` became `history.db` and `SORCAR.md` became `AGENTS.md` (10-03, version 2026.10.2); a legacy file is adopted on first use.
- `WorktreeSorcarAgent` was removed (10-10), the `dummy` SEA became the hidden `sorcar` SEA (10-05), the `run_tasks_parallel` engine was removed (10-07) and `benchmarkings/` left the repository (130,969 files).
- Direct support for Together AI, Z.AI and Moonshot AI was removed (11-19): `TOGETHER_API_KEY`, `ZAI_API_KEY` and `MOONSHOT_API_KEY` are no longer settings, the three routes left the provider registry and their 127 entries left `MODEL_INFO.json`. Z.AI's and Moonshot's models stay available through OpenRouter (section 8).
- **Commits**: `4700ea15a9` `bbee1689b1` `b609fc4a21` `e76e7a0ffa` `4859c0f7ed` `52f15f1521` `c115cb7d2b` `b9ed868023`

### Machines and remote execution

- `/machine` registers other daemons in `~/.kiss/machines.json` with a pinned certificate; a `user@host` machine is enrolled once over ssh and then reached through an `ssh -L` forward with a client certificate (`machines.py`: `enroll_over_ssh`, `SshTunnel`, `MachineLink`).
- `run_agent(machine=...)` and `Bash(machine=...)` run work there.
- The chat header has machine and working-directory drop-downs:
  - A tab bound to a machine relays its task, sub-agent tabs, Explorer, Source Control, Terminal and Browser tabs (`remote_tabs.py`, `remote_views.py`).
  - The model picker, Settings and sidebar panels follow the selected machine.
- CHATS merges the registered machines' tasks, and favourite, rename and delete on a remote row go to its machine.
- `/sync` runs on every registered machine, and `/install` runs `install.sh` on the ssh-registered ones, all or those it names, that hold a clone of the private development repository. Details in section 15.
- **Commits**: `8a464b1ad5` `69349e8fbd` `76e9c90135` `3b9c311fc6` `6e22023167` `c02fcd2aed` `193750eac7` `f2be976174`

### Models and providers

- `MODEL_INFO.json` grew from 706 to 988 entries (counted on 11-19, after the entries of the Together AI, Z.AI and Moonshot routes were removed):
  - `bedrock/` 0 to 144 (Bedrock Claude with ambient AWS credentials).
  - `cc/` 16 to 79.
  - `openrouter/` 412 to 514.
  - 459 alias entries (`alias_of`) that pin an effort or thinking level (`-low` to `-max` for Claude, `-low`, `-medium` and `-high` for Gemini).
- `cc/` and `codex/` are plain models:
  - KISS tools reach the Codex CLI over an MCP bridge (`codex_mcp_bridge.py`) and the Claude Code CLI as text.
  - A turn bills to the subscription unless an API key pays.
  - The Claude Code CLI receives `--effort`.
- Fail-over prefers a model of another family and keeps the same-family twin as the last resort (`model_families.py`).
- Added a vision bridge that describes images and PDFs to text-only models (`vision_bridge.py`), and Claude Fable 5.2. Details in section 8.
- **Commits**: `b446cecf58` `1bb9ec1678` `b62541d867` `c58226df32` `2883814728` `ad050271a9` `e1e85ec39a` `79e5c494c1`

### Pricing and cost accounting

- **Rate coverage**: Cost calculation prices:
  - Long-prompt tiers (`prompt_tiers`, on 107 entries).
  - Anthropic fast mode and US-only inference (`price_modifier_factor`).
  - The 10% premium of Bedrock geo and in-region profiles.
  - Gemini image output and audio tokens at their own rates.
- **Billing and attribution fixes**:
  - Correct cache-write billing (OpenAI 12-hour writes, streamed Anthropic writes) and the cache-read rate of Fable 5.2 (0.025x of the input price, 11-10).
  - Keep sub-cent costs from the usage record to the UI.
  - Bill a call interrupted by Stop.
  - File side calls (classifier, vision bridge, commit message) under the model that made them.
  - Attribute Spend by model through the `model_costs` column.
- **Commits**: `7fb08a0a16` `13127873ff` `0c48894feb` `b03e2241e0` `82faa699f8` `e02e6b5357` `d5d65d9d49` `11a1c80124` `d522176885`

### Tool output and context management

- **Output truncation**: A `Bash` result keeps 8,000 characters by default (`tool_output_default_chars`):
  - On the host, the rest goes to `<scratch dir>/tool-out/` **CHANGED** (11-18) and a whole-file `Read` and `memory_list` are cut to the same budget.
  - In a container or behind an exec bridge, `Bash` truncates without a spill file and `Read`, which runs through it, is cut the same way even for a line range.
- **Auto-detach**: A foreground host `Bash` whose timeout exceeds 120 s continues as a background job after 120 s (`bash_detach_seconds`), except an `until` poll or a call that detaches at its own timeout (`on_timeout`).
- **Context controls**:
  - `compaction` is a run option, a SEA setting and a Settings checkbox.
  - The agent's bulk clearing of a Claude model's old tool calls (`context_edit_trigger_tokens`, 250,000 at Claude Fable 5.2's cache prices) and reasoning-turn retention (`reasoning_keep_turns`, off by default) are configurable.
  - Anthropic calls use fine-grained tool streaming.
  - A compaction removes Claude's signed reasoning from the conversation whichever adapter holds it, and a request a provider rejects for such reasoning is sent again without it (11-20; section 6).
  - A session hand-off is one tool-less call over a scripted digest (`handoff_digest.py`).
  - Memory recall is injected into the task message.
- **Commits**: `1509b8cc00` `79e5c494c1` `f6e4e2bd7f` `97aef43e22` `1e8f6b146b` `8a78456a95`

### Cost levers

- `routine_turn_effort` (off by default) lowers effort on routine turns.
- Runs that cannot edit files get the short `SYSTEM_LITE` prompt.
- **Task classifier** (section 11):
  - Priority order:
    1. Asks the Jev decisions model first by default (10-03) when an OpenRouter key is configured.
    2. Uses the LLM classifier as the fallback.
  - Counts any task that could change a repository file as development unless it is git-only (10-02) and returns only `is_simple` since 10-09.
- A reviewer's budget is 1/8 of what the calling task has spent so far, at least $10 (`review_budget`).
- With prompt caching on, Bedrock Claude requests cache the conversation through a breakpoint that moves to the last cacheable block every turn (10-21, section 8).
- **Commits**: `1509b8cc00` `d66f2ea32d` `30b388acb0` `ff7ff8b742` `8609e6ecac`

### Tools

- **Eleven built-in tools added**:
  - `agent_job` (10-04).
  - `create_worktree` and `commit_and_merge` (10-09).
  - `checkpoint` and `rollback` (10-28).
  - Browser tools `evaluate_script`, `list_console_messages`, `list_network_requests`, `wait_for`, `attach_to_chrome` and `use_my_chrome` (10-28).
- **New parameters**: `run_agent(wait, machine, description)`, `Bash(machine, on_timeout, until, poll_seconds)`, `bash_job(until)`, `Edit(edits, exact)`, and `description` on `run_parallel` and `run_commands_parallel`.
- **Tool profiles**: Went from five fixed names to fifteen that join with `+` (10-03); `review` gained the browser tools and `talk`. Details in section 7.
- **Commits**: `c65da38f14` `20eb36ec25` `b3720dad55` `ffc4493eeb` `8e414731ed` `38a98d0ce1` `4aa120f8ff` `ba14685e20`

### Sub-agents and worktrees

- **Sub-agents** (section 10):
  - `run_agent` takes its options in the SEA settings vocabulary under one precedence rule:
    1. Call arguments.
    2. The SEA's `settings()`.
    3. The caller's settings.
    4. User defaults.
  - `wait=false` returns a job that `agent_job` tails, waits for (`wait_any` since 11-09) or kills.
  - `run_parallel` is one `run_agent` job per task.
  - A result carries a `ran:` line that says what ran.
  - The spend of a job that settles after its caller's task row was saved is added to that row and its finished ancestors (10-07).
- **Worktrees** (section 12):
  - Worktrees are created on demand, one per repository, by `create_worktree(path)`, and `commit_and_merge()` merges them all.
  - A `Write` or `Edit` of a checkout path is refused until its repository has a worktree, and after that it is remapped into the work dir's own worktree or, for a repository isolated elsewhere, refused with the path to use.
  - The SEA settings `use_worktree` and `auto_commit` are gone.
- **Commits**: `c65da38f14` `c115cb7d2b` `16c8eb13a3` `a1ca5b4029` `ffc4493eeb` `52f15f1521` `70e82184c5` `ba14685e20`

### Bundled SEAs added and updated

- **Thirteen SEAs added** to `src/kiss/agents/seas/` (17 to 30, not counting the shared `base` folder):
  - Slash commands: `/discover` (AI discovery loop, section 3), `/devtools`, `/machine`, `/sync`, `/sync_history_and_memory`, `/install`, `/recommendations`, `/good_morning`, `/write_patent` and `/review_patent`.
  - Model-picker routers: `orautorouter`, `orbestrouter` and `ccxbestrouter`.
- `examples/rl_episode/` holds an RL-style example SEA.
- **Updates to existing SEAs**:
  - `/write`, `/review_paper` and `/revise_and_review_paper` moved to `bestrouter` (10-26), where `/discover` and `/review_patent` ran too until all five went back to the caller's model (11-16).
  - `/write_paper` and both paper-review SEAs were aligned with 20 ICLR 2027 reviews (11-07: a contribution step before the outline, four more `check_paper` gates, rating caps).
- **Commits**: `54821cff2c` `37efcd2c28` `aa06ed068c` `fd3c253ccf` `8d12393095` `c12af5d5c1` `764055463a` `148aec09af`

### Routers

- `bestrouter` answers simple questions on `claude-opus-5-5-medium`, works on `claude-fable-5-2-max` and reviews with `gpt-6-astra-xhigh` (simple-question split 11-03; the three best-routers' reviewers at `xhigh` effort 11-10, in `9878ef72f1`, which is later than `d522176885`).
- `autorouter` routes with `claude-opus-5-5-medium` over the tiers `small`, `medium` and `frontier` (its router model and `orautorouter`'s went from max to medium effort on 11-03).
- `orautorouter` and `orbestrouter` use OpenRouter models only (10-18).
- `ccxbestrouter` uses `cc/` models with a `codex/` reviewer (11-10).
- **Commits**: `fb53e9245d` `d78350221e` `ffe9e86f40` `4790f56261` `84b9cace06` `619e551468` `d0bb8e5047`

### Daemon transport, TLS and passwords

- Same-machine clients reach the daemon through its WSS listener with a per-start token in `~/.kiss/sorcar-local.json` (10-01).
- The listener sniffs the first byte and serves plain HTTP to loopback clients (10-15).
- The remote password is generated at install (20 characters), stored as `remote_password` in `config.json`, checked on every frame and editable in Settings with show/hide and refresh buttons.
- Enrolled machines authenticate with client certificates (`tls/trusted-clients/`).
- `install.sh` defers a restart while a task runs and the daemon finishes it when idle (`deferred_restart.py`).
- A lost `cloudflared` tunnel is adopted again, `/webclip.mobileconfig` serves a signed iOS Web Clip profile, and the daemon supervises a brand's `services.d` helpers and gzips text responses (`http_speed.py`). Details in section 14.
- **Commits**: `b609fc4a21` `ff16b5e867` `8b21af4183` `d93d4f203e` `daa030172a` `1217765d98` `751cad3582` `558f857658`

### Web app layout and panels

- **Layout**: No row of chat tabs:
  - The chat on screen is the one picked in the Chats panel.
  - A group strip lists its sub-agent tabs.
  - The desktop page splits into a chat pane and a content pane with a draggable divider.
- **New subpanels**:
  - Sidebar: Machines, Schedule, Recommendations and Skills (11-20) subpanels; Apps was renamed Connectors (11-20).
  - Settings: Bedrock, Claude Code and Codex subpanels.
- **Skills subpanel** (11-20): Lists the skills the agent can use, searches the skills.sh registry and adds a found skill after a security check of its files (sections 7 and 16).
- **Projects** (11-21): Groups chats on this machine and gives their tasks settings and API keys of their own, edited on a Projects page and listed as foldable groups in the Chats panel (sections 11 and 16).
- **Default working directory** (11-21): Settings option (`default_work_dir`) that sets the folder the page switches to when a chat is opened, overridden by a folder picked in the chat header (section 16).
- **Task Progress** (renamed from **Task update**, 11-19): Is the `/ask` agent's answer (10-03), for tasks on other machines too, and runs only while the panel polls:
  - Once the task is 60 s old, then 10 minutes after the last run ended, or after half of the time that run estimated to be left, at least 60 s and at most 10 minutes (11-20).
  - Or at once on refresh.
  - The answer ends with an estimate of the time left, which the panel shows and counts down (11-20).
- **Task Info** (11-20): The Chat id, Task id and Parent task values are buttons that insert the id into the composer, and the gap between the rows was halved.
- **Task Info tags** (11-21): A Tags row lists a finished task's tags and edits them (remove with ×, add a name of the vocabulary, at least one kept). An edited set is stored with `tags_edited` set, so later derivations and `--refresh` keep it.
- **Sidebar and composer changes** (11-21): Recommendations is a single card with no header, position button, refresh button or status line, never folded and hidden when empty, pinned to the bottom of the sidebar, showing one of twenty suggestions at a time for 30 s with the next one sliding in, and scrolling a suggestion taller than a third of the window. Schedule and Machines show a clickable example command. A skill review offers **Cancel** next to **Add anyway**. The right strip's buttons are left-aligned, and the user name sits further right on phones. A Save button in the composer keeps its text in `saved_input.txt`. A link to a file no tab can show opens in the default application on the daemon's machine and as a download elsewhere.
- **Sidebar changes** (11-23): Recommendations has a header with a chevron at its top left that folds the card away and brings it back, plus `<` and `>` buttons beside the card to step to the suggestion before or after the one on show. The Settings button moved from the strip under the Chats list to the first position on the strip under the right sidebar, directly left of Terminal. While the user scrolls the Chats list, it loads a full panel's height of rows at a time, ahead of the view.
- **UI additions**: Swipe between screens on phones, Auto mode (dark from 7 pm to 7 am) and High contrast, refresh and Reload from disk buttons with a page reload after a daemon restart, and the commit hash next to the version in Settings. Details in sections 14 and 16.
- **Commits**: `4691d3a0d6` `29cd0b5525` `d655783f6d` `7e3e85ee9c` `c94088ebb8` `a185d2a96f` `90bde13051`

### Explorer, Source Control and terminal

- Explorer and Source Control moved into the right panel (the activity bar is gone):
  - Explorer has multi-select (09-30).
  - Source Control folds worktree groups, counts the task's worktree in its badge and opens Changes and Graph files as diffs.
  - A terminal's command history outlives its tab.
- Details in sections 12, 14 and 16.
- **Commits**: `4691d3a0d6` `1530b6e683` `eb7c469177` `c28aa99166` `a60f987579`

### Chat

- A message sent mid-task can carry attachments, and images named by file path load in results, answers and previews.
- The Thoughts panel was renamed `Thinking`.
- Added a slash-command popup ranked by recent and frequent use.
- Share chat was renamed Save chat (10-19).
- Added a victory sound when a task finishes. Details in section 16.
- **Commits**: `eb7c469177` `260b1dc697` `3c515908c5` `48a867c34a` `4cabf55ebc` `9d0c3c1468`

### Task database and CHATS panel

- **Schema additions**:
  - New `task_history` columns: `success`, `is_cron`, `model_costs` and `machine`.
  - New `chat_summaries` columns: `user_summary` (chat rename), `last_finished`, `is_favorite` (chat favourite, 11-16) and `project_id` (the chat's project, 11-21).
  - `deleted_tasks` and `deleted_chats` tables so that a deletion survives a sync.
  - `projects` table in `history.db` for projects and their settings overrides (11-21).
- **Tags and metadata**: Tags are capped at three and a trajectory that holds a credential gets the `secret` tag; the task settings record the `host`.
- **CHATS panel**: Gained a Running section, a WORKDIR filter, a Cron filter, sorting by completion time, delete buttons for chats and tasks and a favourite star for chats (11-16).
- **Legacy DB**: A stale pre-2026.10.2 `history.db` no longer blocks the rename (`legacy_task_db.py`). Details in sections 11 and 16.
- **Commits**: `3d1bab5080` `514386ad6e` `5b61c9d5f7` `474a8ca8e9` `6257d2491e` `333861d441` `eb7c469177` `bf3170db3b`

### Standalone app, install and branding

- `kiss-app` opens the web app in its own window (a Chromium-family browser in `--app` mode with its own profile) and installs a launcher (`.desktop` on Linux, `.app` on macOS, a Start Menu shortcut on Windows); `/manifest.webmanifest` describes the app.
- `install.sh` is the single installer:
  - Copies the app into `~/.kiss/kiss`.
  - Byte-compiles it.
  - Registers and starts `kiss-web`.
  - Reports problems in the web app.
- The Docker image runs `kiss-web` on `debian:bookworm-slim` instead of code-server.
- A brand names its state directory with `home_dir` in `brand.json` (10-03), and an executable `signin` hook in the state directory drives a sign-in window (10-20). Details in sections 14 and 21.
- **Commits**: `32daae72f2` `4700ea15a9` `5647d55977` `90a4ff7261` `872893f903` `dd4c3bf79d` `9a14321a68` `075748c4ac`

### `rsorcar` and release

- `rsorcar` no longer copies private ssh keys by default (`SORCAR_SKIP_SSH_COPY`, agent forwarding instead) and deploys while tasks run: it skips only the `kiss-web` restart, and the history sync holds back unfinished tasks and transmits deletions.
- `sync_db` reconciles `task_history` columns between releases instead of refusing.
- `scripts/release.sh` publishes one `Release X.Y.Z` commit per release to the public repository (`build_release_chain`), without the paths in `scripts/exclude.json`. Release commits and tags carry the releaser's identity as mapped by `.mailmap`, and a `Co-authored-by` trailer for each outside contributor that `scripts/credits.txt` lists for that version; these trailers are what GitHub's contributor list is built from.
- **Commits**: `e55180eb69` `c6595e6bad` `62310fac3b` `30582cb5ce` `09afbb036f`

### Memory and knowledge

- `/sync_history_and_memory` merges `history.db` and `memories/` across this machine and every ssh-registered machine in one run (`peer_sync.py`, `memory_delta.py`, `ssh_peer.py`):
  1. Pulls from every peer.
  2. Then pushes the union back.
- Running tasks are held back for the next run, and a page changed on two machines at times too close to order is reported as a conflict and left as it is on each.
- The run probes each peer for its state directory (`home_probe`), so installs of different brands sync too.
- The memory pages most relevant to a task are recalled automatically and listed in the task message. Details in section 13.
- **Commits**: `1509b8cc00` `aa06ed068c` `285c8e87c9` `3c2ef2e698`

### Cron

- A repeating job more than 10 minutes overdue is rescheduled without running (`MISSED_RUN_GRACE_SECONDS`), so a daemon restart no longer fires every missed job.
- Two default jobs are seeded (recommendations and a good-morning greeting).
- Schedule rows have edit, run-now, enable/disable and delete actions and a `+` button.
- Prompt jobs get a run-context block, a connector pre-flight check, back-off after repeated failures and a `covered_until` watermark.
- A messaging-gateway tick is refused as a prompt job, jobs follow the daemon to another state directory (`adopt_previous_home_jobs`), and a `running_pid` record stops a second process from starting a job that is still running. Details in section 20.
- **Commits**: `238e6c9c09` `3e85e38a98` `29cd0b5525` `6c0a5cce8c` `54821cff2c` `ada83228de` `d122c59c85` `7887bfb71d`

### Browser and Chrome bridge

- **Chrome bridge**: `chrome_bridge.py` and the unpacked MV3 extension in `chrome_extension/` (opt-in) let the agent drive the user's own Chrome over a loopback WebSocket with a mutual challenge-response; `/devtools` replaces the browser-devtools skill.
- **Web stealth**: Recognises AWS WAF, Stack Overflow and Reddit blocks, presses the PerimeterX hold bar, waits 45 s on Anubis and falls back to Crossref when Semantic Scholar shows a puzzle.
- **Browser lifecycle and streaming**: Falls back to bundled Chromium when the default browser cannot start, relaunches after the Playwright driver dies, disables WebAuthn in unattended runs and streams frames with per-viewer flow control (`frame_flow.py`). Details in section 17.
- **Commits**: `c0052aac5b` `5aa0c9fcef` `84cbbcf8af` `4aa120f8ff` `2d93669920` `a591b37674` `7f8afb0320` `fef15c3be8`

### Voice

- **Wake word**:
  1. Went from "Hey Sorcar" to "Abracadabra Sorcar" and then "Abracadabra" (both 10-17).
  2. Returned to "Hey Sorcar" (11-08, `WAKE_PHRASE` in `server/speech.py`).
- **Recognition and audio**:
  - A grammar-free recogniser must confirm the wake word, and only final recogniser results trigger it.
  - Received audio is kept in `~/.kiss/voice-debug/` (the last 200 recordings) and transcription retries with varied audio.
- The host-side listener was removed with the extension; voice runs in the browser page. Details in section 18.
- **Commits**: `43ea4d338c` `6fd83c9b43` `651e865fd2` `ef4f27d196` `0f117f71d0` `11ac6435de` `378d3b8f69`

### Channels and auth

- Gmail `reply_to_message` has `draft_only` and drafts reply in-thread.
- The daemon's `ntfy.sh` post of its tunnel URL carries the machine name (section 14).
- Claude Code and Codex connections are subscription-aware, each with a Settings subpanel.
- A replaced MCP OAuth session stays published until its replacement is ready.
- `ChannelSea` is the base class of channel agents. Details in section 19.
- **Commits**: `3125a9ba58` `1eaa2c10f8` `a85cab22f6` `b446cecf58` `16c8eb13a3` `c58226df32` `f0a1702270`

### Benchmarks (Terminal-Bench harness)

- `src/kiss/benchmarks/terminal_bench/` (1,182 lines, 10-25) runs the coding SEA under Harbor:
  - The Harbor agent `SeaHarborAgent`.
  - A private daemon without browser, memory or task classification.
  - The launcher `run.py` (by default 5 attempts per task, 40 concurrent trials, a $100 cap per trial).
  - The scorer `summarize.py`.
- An exec bridge (`exec_bridge.py`) runs the tools on a remote sandbox, the container `Bash` has background jobs (`container_jobs.py`) and a task's MCP servers reach the agent.
- Result and audit reports are under `reports/`.
- **Commits**: `d50b77d7ec` `e8c8a612ff` `554df61c3f` `be211fade3` `03228cbe2d` `86a1dcd6b1` `cc933051f9` `1e8f6b146b`

### Reliability audits

- 27 commit subjects name races or hangs; most are audit sweeps of `core`, `sorcar`, `server` and the web app.
- **Lasting pieces**:
  - `locked` and `locked_call` (`core/locked.py`, 10-13), which take a lock so that an injected Stop cannot leave it held.
  - OpenAI requests aborted on Stop while the connection is still being set up.
  - pid-reuse-safe kills of the Playwright driver.
  - Machine links that redial dead ssh forwards and recover from sleep.
  - Cron runs that do not overlap across processes.
- **Commits**: `7ebd787984` `52da72b6c1` `fcfbaa378e` `6211234f17` `ace893ef52` `74da58bc74` `4e72a49a63` `7887bfb71d`

### Tests and tooling

- `uv run impacted-tests` (10-25) lists the test files whose import graph reaches the changed files, with `--splits`, `--depth`, `--owned` and `--timeout`; splits are balanced by recorded durations since 11-09 (`durations.py`).
- `check` gained `--fix`; `cost_report.py` gained a comparison with a saved baseline (`--baseline`) and absolute windows (`--since`, `--until`).
- Full runs on Linux, macOS and Windows produced a steady stream of cross-platform fixes.
- Tracked Python test files went from 1,360 to 2,073 and `.test.js` files from 369 to 407.
- New scripts: `webclip.py`, `make_victory_sound.py`, `legacy_task_db.py`.
- CI is one workflow, `check.yml`. Details in section 22.
- **Commits**: `912c98c619` `1509b8cc00` `e4e8cdd7d5` `ef8af49289` `f52067fe17` `bd12e57d09` `ba14685e20`

### Papers and website

- **Papers and decks**:
  - `papers/sea/` (the SEA semantics paper for PLDI 2027, with a Lean mechanisation).
  - `papers/kisssorcar_icml2027/` (the harness paper for ICML 2027).
  - `papers/kvstorepaper/` (HydraKV, redone for MLSys 2027).
  - A one-hour tutorial deck in `papers/kisssorcar/tutorial/`.
  - An explainer in `papers/explainers/`.
- **Website and docs**:
  - A blog post on the VS Code removal under `website/kisssorcar.github.io/blog/`.
  - The full model catalogue split out into `MODELS.md`.
- **Commits**: `f86fdf0067` `c5759cd584` `a76775fab1` `4c367a4675` `43220a68e0` `0e82bebbd7` `3fecce9b71` `b63d7556cb`

### Added and withdrawn inside the window

- Not in the code at HEAD:
  - SEA composition by `extends` (10-04, replaced by subclassing on 10-05).
  - `install-app.sh` (10-14, folded into `install.sh` the same day).
  - The install-time hashed password (10-15) and the Rotate button (10-16) of the remote password.
  - The main row of the two-row tab bar (10-06).
  - The quoted-evidence check of the coding SEA's finish gate (10-28, dropped 11-05).
  - Framework-written step recaps (10-25, reverted 10-26).
  - The `rsorcar` wait for running remote tasks (10-12, removed 10-20).
  - Arrow-key navigation in the Chats panel (10-10, reverted 10-11).
- **Commits**: `122aa237a7` `cbd0e56f34` `cad5bb85c5` `531b362fe3` `d33a9fc527` `471fab7939` `dc02c4ed86` `39a5ff63aa`


## 3. AI discovery and optimisation on real software

### The drivers

- **Location and execution model**:
  - The discovery loop now lives in the `/discover` SEA **NEW** (`src/kiss/agents/seas/discover/discover_sea.py`, added 10-27; section 9) and no longer in `SYSTEM.md`, where it had been a 7-step block.
  - Only a `/discover` run carries it, on the caller's model (`bestrouter` until 11-16) with the full tool profile, and the loop was rewritten **CHANGED** as pseudocode (`DISCOVER`, `ROUND`, `FINISH`).
  - None of the loop is code: the SEA appends the text to the system prompt and sets the tool profile, and the model carries out the rounds with the ordinary tools.
- **The loop**:
  - **Before the first round**: Fixes a dev/held-out split, the exact evaluation command and a noise floor in `split.md`.
  - **Per-idea state and files**:
    - Every evaluated idea is a git commit kept reachable by `refs/discover/<task id>/<id>`.
    - `pareto.md` holds the frontier of non-dominated points.
    - `explored-ideas.md` holds one row per idea with status `KEPT`, `TIE`, `FAILED`, `DOMINATED` or `OVERFIT` (a `TIE` or `FAILED` idea is never retried).
  - **Next-move selection after each round** (one random draw picks the next move):
    1. Below 0.33: a variant of a frontier point.
    2. Below 0.66: a merge of two frontier points or, when no pair qualifies, a fork of an older point.
    3. Otherwise: fresh ideas from an internet search or from one of two other vendors' models.
  - **Stopping conditions** (the run ends on any of the following):
    - When a point that meets the goal on dev also passes the held-out check (one that fails is evicted as `OVERFIT`),
    - When the budget is spent, or
    - At the round limit (the task's, else 100).
- **Read-only repositories**: A task that forbids changing its repository is worked in a lab copy and gets `best.patch` back.
- **Related rules, sample prompts and SEAs**:
  - `SYSTEM.md` keeps the two short blocks on adversarial testing (one sub-task writes breaking tests or workloads, another fixes what they find) and adversarial training, and the held-out and several-candidates rules of "How to Work".
  - The sample prompts "Sorcar for AI Discovery" and "Sorcar for Optimization" now begin with `/discover`; they and the GEPA optimiser prompt, which does not use the SEA, each add a stopping condition ("Do not STOP until ...") and a report.
  - `/rsi7d` and `/revise_and_review_paper` hand their improvement steps to `/discover` (section 9).
- **Three-round verification check** (`reports/discover-three-round-check-2026-10-30.html`):
  - Run on a toy classifier; found the three files written with the fields the loop asks for and every logged metric reproducible from its commit.
  - Its two findings were fixed the same day:
    - Round commits unreachable after the squash merge.
    - A baseline status that differed between the two files.
- **Provenance and scope of the real-software runs**:
  - What follows is what the earlier block did when it was pointed at real software, taken from the LaTeX in `papers/`, the logs in `projects/`, the reports in `reports/` and the checked-in copies of the blog under `website/kisssorcar.github.io/blog/`.
  - Every run under "Real-software runs" predates `/discover`.
  - Apart from the collective-communication study, which has nine seeds per problem and an enumerator baseline, each is a single run with no control arm under a task prompt written by the framework's author, so these are records, not experiments.

### Real-software runs

- **HydraKV, July campaign**:
  - **Target**: A larger-than-memory key-value store written from scratch (`projects/kv_adversarial/hydra.cc`, 4,467 lines of dependency-free C++17; the README links a public copy, `github.com/ksenxx/hydrakv` release v1.0.0 **NEW**) for YCSB-A: 250 M keys, 100-byte values, Zipf 0.95, 8 GiB memory cap, 64 vCPUs, eight NVMe.
  - **Baseline → result**:
    - Starting engine 1.85 Mops/s → **5.50 Mops/s**, which the main paper puts at ~89% of the 6.2 Mops/s the device allows at the observed 77% cache hit rate.
    - The paper's "5.9× Microsoft FASTER" divides by the 0.93 Mops/s printed in the task text: FASTER was not run in this campaign, and the October rerun in the next entry measured it at 4.48.
  - **Held-out and correctness gates**:
    - 250,000,000/250,000,000 keys validated.
    - Adversarial workload variants: 5.67 (sparse keys), 5.56 (clustered), 3.97 (drifting hot set).
    - Held-out 60/25/15 mix generated after the loop closed: **3.87** (29% below the scored trace).
    - 28 self-asserting e2e tests, most with a restart leg, clean under ASan/UBSan/TSan, 93% line coverage.
    - Three independent audits, each finding bugs the previous round introduced or missed.
  - **Agent effort**:
    - 8 tasks, 22.6 h, $516, 1,959 steps, 7 continuations, 11 review sub-agents.
    - The human wrote the task prompts and 11 steering messages.
- **HydraKV, from-scratch redo** **NEW**:
  - **Target**: `papers/kvstorepaper/`, in the MLSys style and aimed at MLSys 2027 by its commits: one run on Claude Fable 5.1 from 10-24 to 10-26 on a public benchmark kit with the July workload and machine class, prompted to write the paper "fresh without consulting any previous experiments and results and memory".
  - **Baseline → result**:
    - First engine 4.19 Mops/s → **5.99 Mops/s** (median of five) with 1,903 lines of C++ and a write-back cache, on a kit that never scores durability: 1.3× FASTER's 4.48 under the kit's protocol and 1.15× its 5.23 under FASTER's own driver.
    - The user's 10 Mops/s goal was not met; the paper's bounds for this design are 6.8 (device) and 8.3 (CPU).
  - **Held-out and correctness gates**:
    - Variants on the final engine: Zipf 0.90 4.86, drifting 5.68, sparse 6.08, 95% writes 6.96, the kit's validation trace 5.08.
    - After the freeze: YCSB-B 2.25 (FASTER 1.86) and a YCSB-D-derived trace 1.78 (FASTER ahead at 1.92).
    - 43 scored runs passed read-back and the kit's auditor and `validate.sh` read back all 250 M keys.
    - Durability is by `Checkpoint` only: killed 30 s into a run, the engine lost 16.2 to 19.5 M acknowledged upserts (19.2 to 23.1%) with no checkpoint and 3.4 to 13.2 M (4.2 to 16.2%) with one every 7 s.
    - Six e2e tests run under ASan, UBSan and TSan.
    - The loop found four engine bugs that the kit's integrity checks did not catch, and three recovery bugs got past all of it and were found by a reviewer model reading the source.
  - **Agent effort**: 16 sessions, about 32.6 h and $511 (the last ten sessions' figures are end-of-session estimates).
- **Bespoke OLAP TPC-H engine**:
  - **Target**: VLDB'26 artefact, single-threaded, 22 templates (`projects/bespoke_tpch_x4/`), 32-core Linux, SF10.
  - **Baseline → result**:
    - 3,132.2 ms → **90.8 ms (34.5×)** summed over 22 queries; 10 queries under 0.1 ms.
    - SF1: 276.8 → 14.2 ms (19.5×).
    - Ingest: 52.5 → 63.0 s, repaid after 4 suite runs.
    - 3.1× ahead of the 284 ms the authors' own multithreaded follow-up reports (a README figure without a run log).
  - **Held-out and correctness gates**:
    - Seed 999 excluded from development: **85.1 ms**, tracking seed 42 query by query.
    - DuckDB as differential oracle on six (SF, seed) configurations.
    - 265-case guard/fallback audit against the pristine artefact.
    - Read-only reviewer found an OpenMP race in Q20 and an `int64_t` accumulator in Q1 (baseline uses `__int128`, overflow near SF 8,300), neither visible to the oracle.
  - **Agent effort**:
    - 3 tasks, 2.4 h, $90, 475 steps.
    - Development: 1.3 h, 162 steps, $43, zero steering.
- **LZ4** v1.10.0 CLI:
  - **Target**: Multithreaded file mode, 16 cores / 32 threads, 1.7 GB corpus in tmpfs.
  - **Baseline → result**:
    - Level -1 at 32 threads: 1,519 → **3,810 MB/s (2.51×)**, 2.57× at 64, 1.88× at 8.
    - With 16 to 64 threads: 1.21–1.27× at level -3 and 7–16% at -6 and -9.
    - Scaling: 3.6× → 9.1×.
    - Faster than pzstd (2,788–2,888) and pigz (1,314) at 32 threads.
  - **Held-out and correctness gates**:
    - Compressed output byte-identical to stock at every tested level (-1, -3, -6, -9).
    - 712-case oracle matrix; stock binary decodes patched output.
    - Breaker campaign found 2 bugs (a >300 s hang on a 1-byte file, a SIGBUS on mid-run truncation); reviewer found 4 more.
  - **Agent effort**: ~7.5 h, under $600.
- **DuckDB** development tree:
  - **Target**: Upstream commit `e500d7786`, 32-core VM, five suites.
  - **Baseline → result**:
    - Geometric means: TPC-H 1.152×, TPC-DS 1.184×, IMDB/JOB 1.215×, h2oai 1.234×, ClickBench 1.237×.
    - Largest single query: ClickBench q28 ~1.73× (1.76× re-measured after the review fix).
  - **Held-out and correctness gates**:
    - IMDB/JOB, h2oai and ClickBench held out of all PGO training.
    - Interleaved A/B because run-to-run drift reached 30%.
    - 6,377-test suite with zero attributable failures.
    - ~15,500 differential hardening statements plus a 43,162-row fuzz corpus and a 6,000-statement randomised corpus byte-identical to stock.
    - Reviewer found 5 issues including a NULL-handling regression.
  - **Agent effort**: Not stated.
- **SQLite** 3.54.0 trunk:
  - **Target**: 32-core GCP VM, four benchmarks.
  - **Baseline → result**:
    - speedtest1 2.06×, TATP 1.90×, SSB 1.30×, kvtest 1.25×.
    - **geomean 1.59×** (1.54× at `synchronous=FULL`).
  - **Held-out and correctness gates**:
    - `testrunner.tcl full`: 0 errors in 1,032,940 tests.
    - Checksum-verified benchmark harness.
    - 37-script differential corpus under ASan/UBSan.
    - Breaker found 2 bugs, a second hardening pass 1.
    - An independent rebuild reproduced **1.50–1.55×** and documented the 2.5% shortfall.
  - **Agent effort**: Under 8 h, under $150.
- **Biomni × TusoAI biology benchmarks**:
  - **Target**: Reconstructed: three perturbation-response tasks, one enhancer–gene task.
  - **Baseline → result**:
    - Naive baseline 78.71–92.20 → a single 157-line method scoring **99.66–99.71** on every sealed test set.
    - No reference method (kNN, random forest, gradient boosting on raw features) clears 99 on all four.
  - **Held-out and correctness gates**:
    - Sealed train/val/test splits, a regenerated dataset (99.70–99.72), shuffled-label collapse (R² → 0, AUC → 50), process isolation with labels only in the parent.
    - 27 adversarial and security tests.
    - Two gate bugs found (a `SystemExit(0)` that passed scoring, `PYTHONHASHSEED`-dependent data).
  - **Agent effort**: One autonomous session.
- **Collective communication above AWS Trainium's closed library**:
  - **Target**: SorcarCCL, a search pipeline inspired by Sorcar, vs OverlayCCL's strategy enumeration, 224 NeuronCores on 7 `trn1.32xlarge` instances.
  - **Baseline → result**:
    - 40 production all-reduce / all-gather / reduce-scatter problems in 8 families: **40/40 confirmed on hardware**; excluding one 117× outlier that the paper puts down to an OverlayCCL compile-time pathology, the speedups run **1.12–4.37×** (median 1.46×, geomean 1.86×), enumerator wins 0:
      - E.g. an MoE routing renormaliser collapsed from 8 all-reduces to 2 (4.06×).
      - A per-block normalisation loop vectorised to `view(B,S)` + `mean(dim=1)` (4.31×).
    - Earlier 143-problem build:
      - 55 anchors, 88 ties, 45 confirmed at 1.06–3.38×.
      - Step time **2.46×** (9.75B Llama-style) and **2.15×** (9.70B GPT-3-class) faster than the enumerator's schedule, which reproduces a textbook baseline the paper calls weaker than the AWS production recipe.
      - Both step-time figures come from 12-step runs at one seed of a training step the authors instrumented with nine collective sites, four of them redundant patterns inserted to exercise a rewrite family, with the rewrites written in by hand and a ZeRO-1 optimiser on a one-seventh shard in place of replicated Adam.
  - **Held-out and correctness gates**:
    - Rank-asymmetric correctness oracle at world sizes 4 and 8 plus bf16.
    - Nine simulator seeds with Mann-Whitney and bootstrap bounds, then one warm-cache hardware pair per problem.
    - No regression on OverlayCCL's own 8 problems.
  - **Agent effort**:
    - Capped at 30 steps and $5 of Claude Sonnet 4.5 tokens per problem.
    - On a 24-problem draw the agent cost an estimated $5.37 against the enumerator's $7.77.

### How the runs went

The loop's value shows in the ideas it measured and threw away as much as in the ones it kept:

- **HydraKV, July campaign**:
  - **Search trajectory**:
    1. First engine (a flat 4-byte-per-key array over 128-byte slots) worked only because the published trace's keys fit in 32 bits, which the main paper calls reward hacking; the first adversarial variant (sparse SplitMix64 keys) OOM-killed it in minutes and forced a fingerprint hash index that scored 1.87 Mops/s.
    2. Diagnostics showed a 72% cache hit rate and one contended I/O queue; two rebuilds regressed to 1.10 and 0.47 before an `fio` measurement of the device (718k random 4 KiB reads/s) led to eight raw `io_uring` rings and an asynchronous read-miss state machine (5.27) and SLRU segmentation (5.50).
    3. Four further cache designs measured 0.78, 0.95, 1.98 and 4.17 Mops/s and were reverted, and no engine change recovered the drifting-hot-set variant (3.97) without hurting the stationary workloads.
  - **Unmet goals and off-mix workloads**:
    - The 10 Mops/s goal set mid-run and the 7.0 stretch goal of the later recovery round were never met: at 50% reads the goal needs an 86% hit rate under the device ceiling, and every admission policy tried saturated near 77%.
    - Off the scored mix the engine is far slower: YCSB-B 2.93, YCSB-C 2.78, write-only 5.12, read-modify-write 0.30, delete-heavy time series 0.08.
- **HydraKV, October redo** **NEW**:
  - **Iteration-by-iteration record**:
    1. Baseline: 4.19.
    2. Read-allocate: 3.06 (27% worse, reverted).
    3. A futex stall removed: 4.51.
    4. One shared CLOCK hand: 5.17.
    5. Submitting each miss at once: 4.97 (discarded).
    6. Relaxed atomic copies for ThreadSanitizer: 5.11.
    7. A write-back cache: 5.99.
    8. In a later session, a 48-bit tag index: 5.99 (no effect).
  - **Replication and stopping**:
    - A second agent instance repeating the first iteration reached 4.38 for $6.
    - The loop stopped after the write-back cache because the session's budget was spent and its own bound said the two remaining ideas could not reach the goal.
  - **FASTER baseline finding** (its largest finding concerns the baseline, not the engine):
    - The kit's FASTER adapter keeps an audit session open, which stalls FASTER's epoch protection, so FASTER reaches at best 1.23 Mops/s through the adapter as published and 4.48 with a 39-line session patch, 4.8× the kit's printed 0.93.
    - The paper asks that the kit's reference number and the ratios built on it be rerun, and the July "5.9×" divides by that same 0.93.
  - **Archived draft between the two campaigns** (`papers/kvstorepaper-prev-2026-10-24/`):
    - Contains a third engine at 7.50 Mops/s, 1.3× the strongest FASTER configuration it measured (5.75).
    - Ran two control seeds without the variant instructions; they scored 5.84 and 7.92 against 7.50 and 5.62 for the protocol seeds, and the draft concluded that "the variants bought no throughput and caught no shortcut".
    - The write-and-review task behind that draft (10-23, 39 h 54 min, $741.62, 65 tasks in the tree) was stopped at round 7 with the reviewer at "weak reject" (`reports/hydra_kv-trajectory-analysis.html`).
- **TPC-H engine**:
  - **Plan vs. agent contributions**: The human supplied a researched plan (5,979 characters: remove the `AffinityGuard`, OpenMP first, precomputed join artefacts second), so the agent ran the implement-and-measure step without the search step; its own contributions were:
    - Per-query choice of artefact:
      - Date-indexed prefix cubes for Q1, Q4–Q8, Q12, Q14, Q15.
      - CSR join indexes for Q9, Q17, Q20.
      - Per-order aggregates for Q10, Q18.
      - Pre-filtered rows for Q13, Q16, Q19, Q21, Q22.
    - A guarded fallback to the baseline algorithm on every fast path.
    - The profile-driven removal of a serial 60-million-row scan in Q19.
    - The 265-case audit.
  - **Q2 and Q10**: Q2 was left alone because its parameters admit no small prefix structure and is the one query slower than baseline (by 0.3 ms); Q10 is 43% of the final total.
- **Blog projects (rejected ideas and build fixes)**:
  - **Rejected ideas**:
    - **LZ4**: `--no-frame-crc` reached 6.4–8.1 GB/s and was rejected as cheating because it changes the output bytes (sequential XXH32 caps the honest path near 3.7 GB/s).
    - **DuckDB**:
      - Row-based aggregate `Combine` rewrite won individual queries (TPC-DS q32 1.088×, q18 1.142×) but moved suite geomeans only 1.006–1.011×, under the pre-registered 1.05× keep bar, and was kept as a patch file.
      - gcc `-march=native` alone gave 0.91–1.16× with regressions on TPC-DS and IMDB, and gcc with PGO and LTO lost to the clang equivalent on every suite.
  - **LZ4 build fixes** (two of the LZ4 gains were build fixes, not algorithm changes):
    - PGO had silently never applied (the object cache was keyed on the compiler flags, leaving an LTO-only "PGO" binary).
    - Training on level 1 alone cost the HC levels 5–6% until retraining on levels 1, 3 and 9 turned that into +7%.

### What the developer model's own tests missed

In every project, the defects that survived a green test suite were found by a different mechanism:

- A read-only reviewer from a second vendor (`gpt-5.6-sol`),
- A breaker sub-agent writing adversarial inputs, or
- An external audit with its own oracle.

Findings by project:

- **HydraKV, July campaign**:
  - **Second audit** reproduced four bugs on hardware:
    - An upsert/delete deadlock.
    - An OOM on a full disk.
    - A recovery that dropped 480K of 976K keys and reported success.
    - Deletes 160× slower than upserts (244 µs vs 1.5 µs).
  - **Third audit and repair review**:
    - The third audit showed that the slow-delete fix had made deletes non-durable: 5,000 of 5,000 deleted keys came back after a kill in the minimal repro.
    - The repair was a delete-intent log written before acknowledgement (final `Delete` 1.0 µs).
    - The reviewer's second round on that repair found a linearisability race between the registry mark and the cache sweep that no fault had triggered.
  - **Multi-workload task**:
    - Found that values over 101 bytes had no disk path at all (100% stale reads at 1 KB values).
    - Found that clean shutdown lost data on every workload, a bug that predated the task.
- **HydraKV, October redo** **NEW** (repeats the pattern at smaller scale):
  - **Loop findings** (four bugs that the kit's integrity checks did not catch):
    - A key lost when two writers claimed one free index entry (an aborted scored run).
    - A GC relink that dropped a newer version (a stress test).
    - An evictor acting without its stripe lock (reading the code).
    - 16 threads parked on one futex (`perf`).
  - **Source-reading reviewer findings**: Three recovery bugs were found only when a reviewer model read the source, two in round 15 of the paper's review and the third in round 16:
    - An empty value read as padding.
    - A segment unlinked early.
    - A dropped checkpointed tombstone.
- **Jitskit reference store concurrency audit** **NEW** (`projects/jitskit_refkv_audit/`, 10-22; the closest related system):
  - Lists ten issues, all reproduced, among them:
    - A checkpoint that stops the world (hot-key read latency 0.026 ms → 1,084 ms).
    - A memory budget that counts value bytes only (RSS 14× the budget).
    - Throughput that falls from 1.58 Mops/s at 4 threads to 0.82 at 16.
- **TPC-H engine and blog projects**:
  - **TPC-H engine**: The DuckDB oracle passed while Q20 had a data race (both racing stores wrote 1) and Q1 accumulated in a narrowed type.
  - **LZ4 reviewer**: Found silent corruption when stdin arrived at a non-zero offset.
  - **DuckDB reviewer**: Found a NULL-handling change from a hoisted rewrite validation.
  - **SQLite reviewer**: Found the engine changes clean and the benchmark harness at fault:
    - Failures passed silently through shell pipelines.
    - Checksums were printed but not enforced.
    - One fixture was nondeterministic.
  - **Blog breaker campaigns**: Found two real bugs each in LZ4, SQLite and Tuso and none in DuckDB.
- **KISS Sorcar paper summary**: Its summary of the list ("a delete-durability regression, an aliasing fingerprint, an OpenMP race, a narrowed accumulator") is also its argument for review by a model from another vendor.

### Where the agent went wrong

- **Two failure modes are recorded and both changed the prompt**:
  - **Layering fixes onto a slower branch**: In the July HydraKV's bug-hunting task, the fixes were layered onto a rejected slower branch and the engine sat at 3.6 Mops/s until the human asked "did you start with most performant variant of the engine?", which prompted a diff against archived revisions.
  - **Retrying a marked-failed idea**: In the same project, the agent once retried an idea its own log had marked failed, which is why the loop, then and now, says such an idea is never retried.
- **HydraKV (July)**:
  - The agent's phrase "production-hardened" is contradicted by its own record (three audits, each finding new bugs).
  - The testing rule's 100% branch coverage was not reached (93.3% of lines and 95.4% of branches at the README's last measurement).
- **TPC-H engine**: The Q1 accumulator was a violation of a stated hard constraint.
- **Biology benchmark**: The feature engineering was informed by the documented data generator, which the post says outright.
- **SQLite verification**: The sqlite-optimized verification found that "full test suite green" held only for the source changes under default build options:
  - The changed runtime defaults fail 26 upstream tests that assert the old defaults.
  - The run with WAL by default was aborted mid-suite and a WAL-default-only build reproduced the `attach2` (26 of 71) and `autovacuum` (23 of 339) failures.
  - Test builds disable computed-goto dispatch, so the suite exercises the fallback path.
- **HydraKV (October redo)**: The kit's validation trace was no longer out of sample once the agent had seen its scores, which the paper states.

### The agent as harness

- **Terminal-Bench 2.0 (HarnessTax comparison)**:
  - **Evaluation scope**: The same coding rules, cut to 727 words, were run on the 30 Terminal-Bench 2.0 tasks and seven models of the HarnessTax study (630 attempts).
  - **30-task sample**:
    - **75.6%** of attempts solved against Pi 70.0%, Codex CLI 65.7% and Claude Code 65.1%, the best point estimate on all seven models (74.1% if solves past 100 turns are counted as failures), inside a pooled interval of 63.8 to 85.9 that covers the other three means.
    - Paired rerun of Pi 0.87.1 on Claude Fable 5: +11.1 points (95% interval +3.3 to +20.0) at five cents more per attempt.
  - **59 unsampled tasks**:
    - The same frozen prompt scored 79.1% on the 59 tasks the study did not sample.
    - Main paper's paired Pi rerun on 57 of them: +10.5 points (+2.9 to +18.7) at 63 cents more per attempt.
    - ICML 2027 paper **NEW**, which counts every failed attempt on all 59 as a failure: +6.2 (−3.4 to +15.3) at 61 cents more.
  - **What was switched off**: Discovery, adversarial testing, memory and the reviewer were all switched off for these runs, so they measure the loop, six tools and the rules, not the procedures above.
- **Terminal-Bench 4.0** **NEW**:
  - **Setup**: The paper's and the reports' name for these runs (66 tasks sized for hours, five attempts each, 330 per run); runs through a Harbor bridge added on 10-25 (`src/kiss/benchmarks/terminal_bench/`; its launcher names the dataset `terminal-bench/terminal-bench` without a version, section 22), which drives the coding SEA with the browser, web search and memory off.
  - **November runs (ICML paper)**:
    - `claude-opus-5-5` at maximum effort: **64.2%** (212 of 330; $10,813; median attempt $28.90 and 131 minutes), 0.6 points under the 64.8% the leaderboard lists for Claude Code on the same model and effort at about $4,700.
    - `gpt-5.6-sol` at xhigh: 32.1% (106; $4,075) against Codex's 37.3%.
    - `deepseek-v4.1-flash` at maximum effort: 30.3% (100; $930).
  - **October runs (kept under `reports/`)**:
    - Opus at default effort: 49.7% ($977).
    - Opus at maximum effort: 59.1% ($5,492).
    - `gpt-5.6-sol`: 33.9%.
  - **Finish-gate false accepts and cross-vendor review**:
    - Most failures are success claims the coding SEA's finish gate accepted: the hidden tests passed for 164 of 250 accepted claims on Opus, 104 of 313 on GPT and 97 of 287 on DeepSeek, and 84 of the Opus run's 86 wrong claims had the gate's second verifier green as well.
    - A reviewer from another vendor did not rescue the GPT run: in the 29 tasks whose failures were read, the Opus reviewer named the actual defect in 2 and pushed about ten attempts into the change that then failed.
  - **Trajectory and integrity audits**:
    - The paper's audit of every passing trajectory found:
      - No access to the hidden tests or reference solutions and no web search for the benchmark.
      - One reward hack: DeepSeek rebuilt a hidden reference model by brute-forcing its seed (30.0% without it).
      - Two GPT passes that did not meet the task on its merits (31.5% without them).
      - Eight borderline Opus passes (61.8% if all are dropped).
    - The integrity report on the Opus and DeepSeek runs (`reports/terminal-bench-4-integrity-2026-11-07.html`) also names two holes in the harness:
      - The second verifier's script sat world-readable in the agent's container while gate refusals quoted its expected values (Opus opened the script in 200 of 330 attempts).
      - The agent's log directory was mounted into the container.

### Research artefacts built the same way

- **July papers** (two papers sitting beside these runs):
  - **SWEDefend**:
    - Written under a prompt ending "Do AI discovery to get better results" as a defence against SWExploit-style backdoored program-repair patches.
    - Its combined pipeline at confidence threshold 0.9 catches 97.96% of 49 malicious cases with 0 of 100 benign vetoes where a naive fail-closed judge sat at ~20% false positives.
    - Reports its own defeat by an adaptive attacker (3 of 3 seeds by the third iteration).
  - **Cleverest+**:
    - A fixed-budget (4,3,3) three-member portfolio with sanitiser-signature oracles.
    - Solved 6 of 6 issues on a three-subject mini-benchmark in 10 trials per issue (60 in all), with no run yet on the 72-commit benchmark it targets.
- **Two new papers (plus the redo HydraKV paper's review rounds)**:
  - **SEA semantics paper** **NEW** (`papers/sea/`, a PLDI 2027 paper by its commits):
    - Writes the layering, locking and dispatch rules of extension agents (section 9) as an 817-line Python reference with fifteen Lean theorems.
    - Tests the shipped code against it: 47,920 and 95,549 generated checks at two seeds with no disagreement, and 400 live runs through the real runner that all agree.
    - Writing the rules down exposed four corner cases no shipped agent had reached, since repaired, and six of the fifteen disagreements recorded along the way were errors in the reference itself.
    - Prompt-injection experiment: 91 of the 160 runs without the verdict hook had a forbidden effect; with the hook all 302 forbidden calls were refused.
  - **ICML 2027 paper** **NEW** (`papers/kisssorcar_icml2027/`): Carries the Terminal-Bench 4.0 runs above and the review study below.
  - **Redo HydraKV paper**: Went through 17 write-and-review rounds (`reports/hydra_kv-revise-review-log.md`), and its last review recommends "weak accept".
- **Additional material in `papers/`**:
  - A one-hour tutorial deck on the harness and SEAs **NEW** (`kisssorcar/tutorial/`, with a `/discover` part since 11-09).
  - A one-page research proposal that applies AI discovery to modelling the Neuron compiler **NEW** (`kisssorcar/agenda/`).
  - A plain-language explainer of an external paper (`explainers/`).

### The loop turned on the agent itself

- **One-week speed audit** (2,786 tasks in the task database, measuring where 110 hours of user waiting went):
  - 38% in the agent's own LLM round trips (on `claude-fable-5`, a trivial step took 4.4 s at under 25k tokens of context and 9.8 s above 300k).
  - 33% waiting on `run_parallel`.
  - 25% of the total in reviews, counted inside the `run_parallel` and `run_agent` waits; in the parent tasks with three or more review rounds, 0 of 135 rounds came back clean (the same "run all tests" prompt took 0.2 h with 27 parallel splits and 7.9 h with a 12-round review).
- **The cost-lever work that followed**:
  1. Removed the mandatory first read of `SORCAR.md` (renamed `AGENTS.md` on 10-03) and the shell-wrapper sub-agents (306 and 258 in the 24-hour baseline before the change, none in the filtered sample of 52 tasks after it; that sample held no reviewer tree, so the baseline's 7 trees whose reviewers took over half the spend have no after figure).
  2. Reached a 0.970 prompt-cache hit ratio.
  3. Found that its own context compaction was cache-hostile (57 of 57 compactions missed the cache, net +$130), which produced the cache-aware gate and the keep-alive ping (replay: input bill 0.66× of production).
- **Six months of the agent developing its own repository** (section 23) gave the controlled result behind the rules, over six task grids:
  - **Full prompt vs. lite prompt**: Changed how the agent works (a test file written or changed in 90 of 114 cells against 28) but not the hidden-test pass rate (+0.9 points on the held-out grid, interval −2.8 to +4.1).
  - **Exploratory review arms** (two of the four review arms gained on that grid):
    - Second-vendor reviewer: 4.7 points (+1.2 to +9.4) at 2.5× the cost.
    - Stronger same-vendor reviewer: 3.3 (+0.8 to +6.0) at 3.4×.
  - **Ten-week two-vendor deployment**: The reviewer made 3 writes in 2,164 calls, all to `tmp/`, took 9.3% of task cost and reported roughly 1,500–2,000 findings, 53% of the coded sample functional bugs.
- **Matched-checkpoint review and guard ablation (ICML 2027 paper)** **NEW** (repeats the review experiment from matched checkpoints, 12 tasks and 36 cells per arm):
  - **Review by a second vendor's model with a bounded repair**: Gained 9.2 points of hidden-test pass rate over the checkpoint and 4.2 over an extra pass by the developer model (+1.6 to +6.9, although the task-level sign test does not reject, p = 0.227).
  - **Same-model reviewer**: Gained no more than the extra pass (+4.7 against +5.0).
  - **Stronger same-vendor model simply continuing the work**: Came within 0.3 points of second-vendor review (−3.8 to +3.2).
  - **Tool guards switched off**: Changed nothing measurable; they fired in 1 of 24 cells and the pass rate moved by −1.1 points (−4.3 to +1.7).
- **Terminal-Bench 4.0 was then used to measure the harness's own mechanisms** **NEW**:
  - **Coding SEA finish-gate A/B** (24 of its tasks, 11-04 to 11-05, `reports/gate-redesign-ab-2026-11-04/`):
    - **Comparison**: Old gate (a same-model reviewer, up to three reviews, a second agent writing a blind verifier) against the current one (one review by a model of another family with 240 s to answer; section 9).
    - **Results**: 31 of 46 finished attempts passed under the old gate (67.4%) and 41 of 64 under the new (64.1%); the cost per attempt fell from $22.54 to $12.78; the time the gate held the agent fell from 27.3 to 2.6 minutes on average.
  - **Seven harness changes A/B** (same 24 tasks, 11-07, `reports/harness-improvements-ab-2026-11-07/`, running the changed harness twice against a before arm):
    - **Before arm**: 28 passes in 46 attempts at a mean $15.08 per attempt.
    - **As proposed**: Passed 33 of 47 at $29.24.
    - **With context-edit settings retuned**: Passed 33 of 48 at $12.95 (paired over the 23 tasks both arms completed, a cost ratio of 0.84 with an interval of 0.65 to 1.06) in 58.1 agent minutes against 80.2, a figure the report calls confounded because that run had the host to itself; the report says 48 attempts per arm cannot show a pass-rate change of a few points and claims none.
  - **Eight-run analysis behind the seven changes** (`reports/kiss-sorcar-harness-improvements-2026-11-07.html`; 2,505 trials, 493,657 model turns, $30,900):
    - 61–66% of the spend of the four maximum-effort runs fell on steps with more than 150k tokens of context.
    - DeepSeek copied a context-compaction stub back into its own tool calls 1,152 times in 257 of 330 trials.
    - The Opus and DeepSeek runs held 944 and 1,579 foreground sleeps of more than 30 s.

### Provenance

- **Macros, scripts and checked-in evidence**:
  - The headline figures above are `\newcommand` macros, in most papers recomputed by a script beside the LaTeX:
    - Scripts: `kisssorcar/ks_numbers.py` and `ks_tb2.py`, `sesorcar/se_numbers.py`, `sea/make_numbers.py`, `kisssorcar_icml2027/icml_numbers.py` and `tb4_november.py`, and the Collective Algebra `compute_numbers.py`.
    - Checked-in evidence: `kisssorcar/evidence/`, the 5,132-file `kisssorcar/ablation/`, `projects/kv_adversarial/DISCOVERY_LOG.md`, `projects/bespoke_tpch_x4/results/`, the 858-file `sorcarccl-production-set-main/`, and the redo's 776-file `kvstorepaper/artifact/`.
  - Macro counts by `grep -c '\\newcommand'`:
    - Main paper: 325 macros.
    - `sesorcar/macros.tex`: 2,495.
    - `sea/numbers.tex`: 2,239.
    - `kisssorcar_icml2027/macros.tex`: 799.
    - Redo's `hydra_kv.tex`: 656, with a comment naming each macro's log and no script.
  - Repository count: `git ls-files papers` counts 9,084 files in 14 directories.
- **Cross-source discrepancies** (where sources disagree, the numbers above follow the latest paper):
  - **HydraKV process record**: 8 tasks / 22.6 h / $516 in `kiss_sorcar.tex` but 6 / 17.2 h / $352 in `ks_assistant.tex` (which also attributes the starting engine to Claude Code, not to a Sorcar task).
  - **HydraKV held-out drop**: "29%", "more than a quarter" and "a third" in three places.
  - **Stale references in `kiss_sorcar.tex`**: Still describes the discovery loop as a `SYSTEM.md` block and still points to `papers/kvstorepaper/` for the July standalone paper, which the redo replaced on 10-25.
  - **Archived Hacker News draft** (`papers/kvstorepaper-prev-2026-10-24/social/hackernews.md`): Still says six prompts, ~3,970 lines and 5.51 Mops/s.
  - **Terminal-Bench 2.0 unsampled rerun and review interval**: The two papers that report the rerun on the unsampled tasks headline different accountings of it (+10.5 on 57 tasks, +6.2 on 59), and the second-vendor review interval starts at +1.2 in one and +1.1 in the other.
  - **Redo HydraKV recovery bugs**: The redo paper says its three recovery bugs were found one per review round where its review log (`reports/hydra_kv-revise-review-log.md`), followed above, has two in round 15 and one in round 16.
  - **Collective Algebra builds**: Exists twice with 55 vs 40 anchors and the end-to-end step-time result only in the older build.
  - **Cleverest+ comparison table**: Still calls its portfolio allocation "proposed, not yet wired" in one row that its results section contradicts.
- **Paper gates and build**: The `/write_paper` SEA's `check_paper` gate (AI-slop and consistency checks) and `build_paper` (pdflatex + bibtex) run on every revision (section 9; 87 of the 591 slash-command tasks in the database are paper tasks).

### Website

- **Checked-in copy contents** (`website/kisssorcar.github.io/`, 49 tracked files; figures from the checked-in copy, the live site was not read):
  - Home page (`website/kisssorcar.github.io/index.html`, 3,917 lines, with the Terminal-Bench 2.0 table and the paired-Pi numbers) and its Markdown twin.
  - 12 docs pages (`docs/benchmarks.md` **NEW**, the Terminal-Bench 2.0 comparison).
  - 8 blog pages:
    - The six posts used above, and
    - Two **NEW** pages on the 10-14 removal of the VS Code extension: a 23.5 MB shared-chat export and its trajectory analysis of 48 agents over 4 h 45 min.
  - Six paper PDFs under `assets/`.
  - `privacy.html`, `llms.txt` / `llms-full.txt` and a 25-URL `sitemap.xml`.
- **Where the copy lags the repository**:
  - `llms.txt` and the home page list the same six PDFs (`kiss_sorcar`, `se_kiss_sorcar`, `writing_paper`, `hydra_kv`, `swedefend`, `cleverest_plus`) and none of `fable_sol`, `ks_assistant`, `sesorcar`, Collective Algebra, the SEA paper or the ICML paper.
  - `website/kisssorcar.github.io/assets/hydra_kv.pdf` is still the July paper.
  - The two VS Code pages are in the sitemap but in neither blog list.
  - No page mentions Terminal-Bench 4.0.

## 4. Architecture

```
 ┌────────────────────────────────┐ ┌─────────────────────────────────┐ ┌───────────────────────────┐
 │ Web app page                   │ │ Same-machine API clients        │ │ No daemon: the agent runs │
 │ browser tab · installed PWA    │ │ sorcar.run() Python API         │ │ in the caller's process   │
 │ kiss-app window · iOS Web Clip │ │ cron scheduler · run_agent tool │ │ sorcar CLI                │
 │ auth: password; none for a     │ │ 44 kiss-<channel> CLIs          │ │ headless.run_sea()        │
 │ page opened at a loopback URL  │ │ auth: token, sorcar-local.json  │ │ scripts/run_sea.py        │
 └────────┬───────────────────────┘ └────────┬────────────────────────┘ └────────────┬──────────────┘
          │ WebSocket /ws on one port (8787) │ plain on loopback, TLS elsewhere      │ run_sea runs a
          │ 103 JSON commands (sorcar.py API)│ one object per frame, events by tab   │ SorcarAgent;
 ┌────────▼──────────────────────────────────▼────────────────────────┐              │ run_agent and
 │ kiss-web daemon  src/kiss/server (38 modules, 45,818 lines)        │              │ run_parallel
 │ RemoteAccessServer: HTTP + WebSocket listener, auth, tunnel, TLS   │              │ need a daemon
 │ ServerApi -> AgentServer: TabRegistry (~/.kiss/tabs.json), replay, │              │
 │   sub-agent tab lifecycle, task_runner, merge_flow, autocomplete   │              │
 │ WebPrinter (json_printer): events to a task's tabs and history.db  │              │
 │ tabs: browser_tab + frame_flow · terminal_tab · explorer           │              │
 │ panels: sidebar_panels (Schedule, Machines, Skills, Connectors,    │              │
 │   Spend, Recommendations) · task_update · file_index · speech      │              │
 │ machines: remote_tabs · remote_views · machine_stream forward a    │              │
 │   tab's commands to the kiss-web daemon of a registered machine    │              │
 │ background: cron scheduler thread · SEA registry watcher (2 s) ·   │              │
 │   services.d helpers · stall_watchdog · deferred_restart           │              │
 └────────┬───────────────────────────────────────────────────────────┘              │
          │ one agent per task or sub-agent                                          │
 ┌────────▼──────────────────────────────────────────────────────────────────────────▼──────────────┐
 │ ChatSorcarAgent ⊂ SorcarAgent ⊂ RelentlessAgent, which runs one KISSAgent per session            │
 │ tools (44 + finish): shell 4 · edit 6 (create_worktree, commit_and_merge, checkpoint,            │
 │   rollback) · browser 15 · memory 7 · agents 4 (run_agent, agent_job, run_parallel,              │
 │   number_of_cores) · mcp 2 · skill · ask_user_question, talk · decide · summary,                 │
 │   set_model; plus the tools of connected MCP servers and of the run's SEA                        │
 │ prompt: SYSTEM.md (SYSTEM_LITE.md for a simple task or a run that cannot edit) + SEA             │
 │   system_prompt() + memory protocol + Task Settings + ~/.kiss/AGENTS.md                          │
 │ tool-call guards · usage ledger · context compaction · prompt-cache keep-alive ·                 │
 │   fallback model · tools on the host or in a Docker container (docker_image)                     │
 └────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
          │
 ┌────────▼─────────────────────┐ ┌──────────────────────────────┐ ┌────────────────────────────────┐
 │ Models (988 entries)         │ │ State in ~/.kiss             │ │ Extension points               │
 │ Anthropic · OpenAI · Gemini  │ │ history.db · config.json     │ │ SEAs (seas/, SEAS.md folders,  │
 │ Amazon Bedrock · OpenRouter  │ │ api_keys.env · tabs.json     │ │ third_party_agents/): tools()  │
 │ cc/ and codex/ CLIs          │ │ memories/ · cron/jobs.json   │ │ and hooks · skills (SKILL.md)  │
 │ MY_MODELS.json overrides     │ │ machines.json · tls/         │ │ MCP servers (mcp.json)         │
 │ 6 router SEAs in the picker  │ │ AGENTS.md · AUTOROUTER.md    │ │ promptlets (MY_INJECTION.md)   │
 │                              │ │ sorcar-local.json            │ │ white-label brand (brand.json) │
 └──────────────────────────────┘ └──────────────────────────────┘ └────────────────────────────────┘
```

### Figure 1. The data path from clients to models

- Every client of the daemon sends commands of one catalogue (`src/kiss/server/sorcar.py`, `API`, 98 commands) over the same WebSocket listener and receives the same types of JSON event.
- Delivery of a task's events is scoped by tab:
  - One copy per subscribed tab, stamped with its `tabId`, goes to every page (which filters by tab) and to the same-machine clients that addressed that tab or none.
  - The copies of a tab closed on every page go to those clients alone.
  - Another daemon's machine link receives replies and global events but no task events.
- Client authentication:
  - A web app page signs in with the remote password, which only a page opened at a loopback address (`http://127.0.0.1:8787` or `localhost`) and reaching the daemon directly does not need (section 14).
  - A same-machine program presents the per-start token of `~/.kiss/sorcar-local.json`.
- Where the agent runs:
  - The daemon runs the agent for every client except the `sorcar` CLI and the headless `run_sea`, which run it in their own process:
    - The CLI runs it as a `ChatSorcarAgent` and `run_sea` runs it as a plain `SorcarAgent`.
    - In both, `run_agent` and `run_parallel` still submit to a daemon and fail without one.
  - A `kiss-<channel>` CLI starts a private in-process daemon on an ephemeral loopback port and submits to it through `sorcar.run`.

### What the figure compresses

- **One transport**:
  - The Unix socket `~/.kiss/sorcar.sock` was removed 10-01.
  - Local clients use the listener browsers use, and since 10-15 that listener speaks plain HTTP to loopback peers and TLS to everyone else (section 14).
- **One client codebase**:
  - The VS Code extension was removed 10-14.
  - The page `media/chat.html` with `main.js` (32,016 lines) runs in a browser tab, as an installed PWA, in the `kiss-app` window and from an iOS Home Screen Web Clip (section 5).
- **Layers**:
  - `src/kiss/server` (38 modules, 45,818 lines) is the daemon.
  - `src/kiss/agents/sorcar` (50 modules, 64,809 lines) holds the agent, its tools and the synchronous client `daemon_client.py`, kept in that layer so that `run_agent` and the cron scheduler submit tasks without importing `kiss.server`.
  - `src/kiss/core` (29 modules, 13,544 lines) and `src/kiss/core/models` (15 modules, 20,164 lines) hold the base agent and the model adapters.
- **Catalogue**:
  - 84 commands at the previous inventory, 103 now.
  - 21 left, among them:
    - The ten that the VS Code extension host consumed and the daemon discarded.
    - The wake-word and frequent-task commands.
    - The commit-message generator.
  - 40 arrived:
    - Four for terminal tabs.
    - Four for machines.
    - Three for the sign-in window.
    - Three for Claude Code and Codex status and login.
    - Three for the Skills subpanel (11-20).
    - Five for projects (11-21).
    - Two for the browser tab.
    - `createPrAction` for the pull-request button, and `setPinned` and `setChatPinned` for pins (11-20).
    - `getTaskUsage`, `setChatSummary`, `deleteTask`, `deleteChat`, `openExternal`, `getReplayImage`, `cronAction`, `getRecommendations`, `dismissNotice`, `setChatFavorite`, `getTaskEvents`, `openInBrowser` and `stageHtmlView`.
  - `ServerApi.dispatch` validates every command against the catalogue and answers an invalid one with an `error` event.
  - The page calls the catalogue through `media/api.js`.
- **Machines**: A tab bound to a registered machine has its commands forwarded to that machine's daemon and that daemon's frames relayed back (section 15).
- **Left the figure**:
  - `WorktreeSorcarAgent` (folded into `SorcarAgent` 10-10).
  - The spare-worktree pool and its prewarm (removed 10-09, section 12; the daemon still prewarms provider imports and the model registry).
  - The server's two voice-wake modules (the page detects the wake phrase, section 18).

## 5. Interfaces and prompt surfaces

### Web app

- **How it is reached**:
  - `http://127.0.0.1:8787` on the daemon's machine, `https://<host>:8787` on the LAN, or the Cloudflare tunnel the daemon opens once a remote password is set.
  - The LAN and tunnel addresses require the password, on the daemon's own machine too, and only a page opened at the loopback address (`127.0.0.1` or `localhost`) needs none.
- **Notes**:
  - One page for every device, `chat.html` with `main.js` (32,016 lines) under `src/kiss/webapp/media/`.
  - Installable PWA with an offline shell (`sw.js`).
  - Light theme by default.
  - Reconnects in place with jittered exponential backoff and a 45 s half-open detector.
  - Daemon side in section 14, page features in section 16.

### Desktop window

- **How it is reached**: `kiss-app`, with `--url`, `--wait`, `--foreground`, `--print-command`, `--install-launcher`, `--uninstall-launcher` and `--reveal-launcher`.
- **Notes**:
  - The same page in a Chromium-family browser in app mode, with a launcher on Linux, macOS and Windows.
  - Added 10-16 (section 14).

### iOS Home Screen

- **How it is reached**: The "iPhone Home Screen" item of the page's "..." menu downloads `/webclip.mobileconfig`.
- **Notes**:
  - A signed Web Clip profile that opens the page full screen.
  - Added 11-03 (section 14).

### `sorcar` CLI **CHANGED**

- **How it is reached**: `sorcar -t "task"` or `-f file` (exactly one), with `-m model`, `-b budget` and `--work-dir`, whose default resolves in order:
  1. `$KISS_WORKDIR`.
  2. The current directory.
- **Notes**:
  - Runs the agent in the terminal process, without the daemon.
  - Since 10-10 the agent is `ChatSorcarAgent` (`SorcarAgent` before), so:
    - The run is recorded in `history.db` with its result, usage and events.
    - It can create worktrees; any it leaves unmerged are named on stderr.
  - `ask_user_question` is answered on the terminal.
  - On a terminal it shows the live console view; with stdout piped it prints only the YAML result.
  - Exit status 0 on success, 1 on failure, 2 on a usage error.

### Python client API **CHANGED**

- **How it is reached**: `from kiss.server import sorcar; sorcar.run(task, ...)` returning `TaskResult(text, success, cost, tokens, steps, chat_id, task_id, settings, reconnects)`.
- **Notes**:
  - Blocks until the task ends on the running daemon, found through `~/.kiss/sorcar-local.json` (`endpoint_file` or `KISS_SORCAR_LOCAL` name another file; a socket path before 10-01).
  - 37 keyword options, 27 at the previous inventory:
    - 9 left or were renamed (the worktree and auto-commit switches, the inline tool list and the socket path among them).
    - 19 arrived.
  - Among the 37:
    - `sea_path` (a file defining a `BaseSea` subclass whose `settings()` configures the run and whose `tools()` returns its toolset, loaded and run in the daemon).
    - `sea_command`, `tool_profile`, `docker_image`.
    - `machine` (the task runs on a registered machine, section 15).
    - `attachments`, `budget_cap`, `compaction`, `cancel`, `timeout=3600`, `stop_on_timeout`.
    - `reconnect_seconds=300` (a dropped connection is re-established and the task's events replayed; `reconnects` counts them).
  - `settings` is the run's `task_settings` payload (section 6).

### Headless SEA run

- **How it is reached**: `headless.run_sea(sea, task, ...)`, or `scripts/run_sea.py <sea> "<task>"` from a checkout.
- **Notes**:
  - One SEA in the calling process, run as a plain `SorcarAgent`, with no daemon and no page.
  - Its `run_agent` and `run_parallel` tools still need a running daemon.
  - Added 10-09 as a script and 10-31 as a module (section 9).

### Slash commands

- **How it is reached**: `/<name> <text>` at position 0 of a prompt.
- **Notes**:
  - 72 commands in a stock checkout, registered from four sources.
  - `/<name> help` and `/<name> check` answer without running the SEA (section 9).

### Channels

- **How it is reached**: `kiss-<channel>` CLIs, `run_agent(agent="slack", ...)`, always-on gateways scheduled by cron.
- **Notes**: 44 agents (section 19).

### Scheduler

- **How it is reached**: `kiss-cron`, the `/cron` command, `run_agent(agent="cron", ...)` and the Schedule panel.
- **Notes**: Section 20.

### Voice

- **How it is reached**:
  - In-page wake phrase "Hey Sorcar".
  - `talk()` playback in every open page and, for a task a same-machine API client started, on the daemon machine's speakers.
- **Notes**: Section 18.

### Removed surfaces and console scripts

- **VS Code extension** (listed in this section in the previous inventory; removed 10-14):
  - A client that authenticates with the local token and sends neither a `User-Agent` nor an `Origin` header is taken for an old copy of the extension and is answered with a notice naming the web app's address.
  - When `install.sh` finds and uninstalls a copy, every page shows a toast about it for 7 days, or until its "Do not show this" box is ticked (`dismissNotice`).
- **Root `sorcar` script vs console script**:
  - The root `sorcar` script of the repository still launches `kiss.agents.sorcar.worktree_sorcar_agent`, a module deleted 10-10, and so fails.
  - The `sorcar` console script is the working entry point.
- **Console scripts in `pyproject.toml`**: 53 console scripts declared, 50 at the previous inventory:
  - The 44 `kiss-<channel>` CLIs.
  - `sorcar`, `kiss-web`, `kiss-app`, `kiss-cron`.
  - `sea` (`sea lint` and `sea docs`, section 9).
  - `check`, `impacted-tests`, `generate-api-docs` and `swedefend-eval`.

### Prompt assets shipped in `src/kiss/`

- `SYSTEM.md` (6,237 words) **CHANGED**, 3,971 words at the previous inventory:
  - Sections covered:
    - Identity and rule precedence.
    - Visibility constraint.
    - Tool usage rules.
    - Context and continuation.
    - Periodic summaries.
    - Voice.
    - Web research protocol (10 sites, 1 for real-time data).
    - Code style.
    - Pre-flight checks.
    - A four-step "How to Work" procedure (added 09-30).
    - Adversarial testing and training.
    - Deep work.
    - Complex task planning.
    - File browsing.
    - Desktop apps.
    - Testing.
    - Pre-finish verification.
    - Sorcar-specific rules.
  - The 7-step AI discovery loop left the file 10-27 and now ships with the `/discover` SEA (section 3).
  - With prompt caching on (the default), the Anthropic adapter sends the prompt in two parts:
    1. The static part is sent under its own `cache_control` breakpoint, so runs with the same tools and static part read one cache entry.
    2. The per-task tail (work directory, process id, the "Task Settings" block, `AGENTS.md`) follows as a block with no breakpoint of its own and is cached with the run's conversation instead.
- `SYSTEM_LITE.md` (1,074 words):
  - The reduced prompt for a task the classifier marks as simple and for a run that cannot edit files (sections 6 and 11).
  - `/ask` ships its own `_ask_system_lite.md` (515 words) plus an answering playbook.
- **Brand and version placeholders**:
  - Both prompts and `TIPS.md` carry brand placeholders that `render_brand` fills from `brand.json` when the file is read:
    - The identity sentence and the state directory in the two prompts.
    - The product name and the state directory in `TIPS.md`.
  - `SYSTEM.md` also carries a version placeholder, filled with the package version of `src/kiss/core/_version.py`, the same for every brand.
  - A white-label build therefore changes no prompt file (section 14).
  - The repository and website addresses moved from `SYSTEM.md` into the brand's `identity` on 10-18.
- `INJECTIONS.md` **CHANGED**:
  - 2 promptlets since 10-17, 6 at the previous inventory.
  - Listed after those of `~/.kiss/MY_INJECTION.md`, which is created with one starter promptlet and is editable in place from the Inject panel.
- `TIPS.md`:
  - 22 tips (23 at the previous inventory).
  - Shown once per version in each browser profile until the reader opts out, which creates `~/.kiss/TIPS_DISABLED`.
- `SAMPLE_TASKS.md` (894 words, 12 sample tasks): Still shipped, but no code has read it since the welcome page's sample-task chips were removed on 09-26.

### User files that shape a run

Files a user keeps in `~/.kiss/` (`$KISS_HOME` when set; a white-label brand names another directory with `home_dir`) that shape a run:

- `AGENTS.md`:
  - Standing instructions appended to every system prompt except that of a `plain_io` run, managed by `/remember` and `/forget`.
  - The file was `SORCAR.md` until 10-03, and a file left under the old name is adopted on first use (section 6).
- `SEAS.md`: Extra SEA folders, one per line, a later line beating an earlier one (section 9).
- `MY_INJECTION.md`: The user's promptlets, above.
- `MY_MODELS.json` (section 8):
  - Model entries that override or extend the bundled catalogue.
  - Top-level keys starting with `_` are comments.
- `mcp.json` and `skills/`: MCP servers and skills, read together with their project-level and cross-client counterparts (section 7).
- `config.json` and `api_keys.env`: Settings and credentials, written by the Settings panel (sections 6 and 16).
- The personal task-template file (MY_TASK_TEMPLATES.md), auto-created there until 09-26, is read by nothing.

## 6. Agent runtime

- **Runtime layers**:
  - `KISSAgent` runs one model conversation.
  - `RelentlessAgent` chains such sessions until the task is done.
  - `SorcarAgent` adds the tools, the system prompt and the guards.
  - `ChatSorcarAgent` adds chat persistence.
- **Related sections**: Tools are in section 7, models in section 8, SEA hooks in section 9, sub-agent dispatch in section 10 and worktrees in section 12.

### KISSAgent (`src/kiss/core/kiss_agent.py`, 3,017 lines)

#### Loop, ending, recovery and hooks

- **Loop**:
  - Signature: `run(model_name, prompt_template, arguments, system_prompt, tools, is_agentic, max_steps, max_budget, model_config, printer, verbose, attachments, print_prompts, llm_call_hook, tool_call_hook, llm_response_hook, compaction)`, with defaults of 10,000 steps and $10.
  - Each step executes in order:
    1. Compacts the conversation when due.
    2. Calls the model.
    3. Runs the returned tool calls in order (calls after `finish` in the same turn do not run).
  - Each step's tool result ends with a usage footer reporting steps, context tokens, the step's tool-output tokens, the model call's cost, its output tokens and stop reason, total tokens and budget spent against the cap.
- **Ending**:
  - `finish`.
  - The step or budget bound.
  - The context limit (`KISS_CONTEXT_LIMIT_FRACTION`, 0.7 of the model's window), which hands the task to the next RelentlessAgent session.
  - 3 consecutive errors.
  - 12 consecutive stalled streams, counted separately **NEW**.
  - A second consecutive text-only turn ends the run as an implicit `finish` with `success: true`; two empty replies raise an error instead.
- **Recovery** **NEW**:
  - **Output-token cap retry**: A reply cut by the output-token cap is retried once at effort `low` (11-01), unless the model's catalogue entry has no `thinking` level or the request already ran at `low`, `minimal` or `none`.
  - **Refusals, content-filter stops, unretried cap cut-offs, or repeated empty replies**: A refusal, a content-filter stop (a provider's content-policy HTTP error counts as a content-filter stop), a cap cut-off that gets no such retry or repeated empty replies follow this fallback order:
    1. Switch the run to a model of another family (11-01).
    2. When no other family is configured, switch to the `get_fallback_model()` answer of the model's own family.
    3. When no switch is possible, retry a refusal once with a message restating the task's context (10-26).
  - **Non-retryable errors**: Authentication, permission, missing or invalid API key, `not_found_error` or "credit balance is too low" errors switch to `get_fallback_model()`:
    1. The catalogue's declared fallback.
    2. Else the model's OpenRouter twin when `OPENROUTER_API_KEY` is set.
  - **Single-switch cap**: A run switches model this way once, and the switch is printed as `Switched model from X to Y after <reason>` (section 8).
  - **Rejected reasoning** (11-20): When a provider rejects a request because a reasoning block is "bound to a different conversation", the agent removes Claude's reasoning from the stored conversation (on the Anthropic adapter the newest reply's blocks too) and sends the request again on the next step:
    - The rejection is not counted among the consecutive errors; before, three of them ended the session.
    - With no reasoning left to remove, it is handled like any other error.
- **Hooks**:
  - Direct hooks:
    - `llm_call_hook(new_messages)` before each model call.
    - `tool_call_hook(name, args)` before each tool call.
    - `llm_response_hook(reply)` after each reply **NEW** (10-31), which receives an `LLMReply` (`llm_reply.py`: `text`, `tool_calls`, `usage`, `step`, `budget_used`, `total_tokens_used`).
  - Hooks installed by the layers above: `pre_step_hook`, `tool_call_guard`, `helper_charge_hook`, `budget_check_hook`, `budget_status_hook` and `context_reset_hook`.
  - What a hook may answer is under "Verdicts and guards"; the SEA methods of the same names are in section 9.

#### Context compaction and bulk tool-call clearing

- **Tool-output compaction** **CHANGED** (`context_compaction.py`, 686 lines):
  - **Trigger conditions** (before a model call, replacing old tool results and tool-call arguments with stubs):
    - Once the context has passed 100K tokens and then every further 100K, or at any step once it has passed the target: the larger of 100K tokens and 30% of the model's window (11-01; 100K when the window is unknown).
    - Never sooner than 10 steps after the previous compaction (10-25).
    - None runs once the context has reached 80% of the hand-off limit.
  - **Stubbing rules**:
    - Results longer than 500 characters and older than the six most recent are stubbed with a 200-character preview, never those of `Edit`, `Write` or `finish`.
    - Arguments are stubbed with a 120-character preview (11-01), never those of `finish` or of `summary`, an agent's own recap of the steps being stubbed (11-13); the stub of an `Edit` or `Write` argument points at the file.
  - **Cache-miss payback gate**: A compaction is applied only when it pays for the prompt-cache miss it causes: it must drop 25% of the context below the target and 10% above it, each raised to a payback floor derived from the model's cache-read price (28% at Anthropic's 5% cache reads, 15% at OpenAI's 10%).
  - **Stub resolution**: Each stub carries an id under which the run keeps the original:
    - A call whose argument is a whole stub runs on the original text.
    - A partial or unknown stub is refused.
  - **DeepSeek-family models**: Models whose API wants every earlier turn's reasoning passed back lose whole turns instead: all but the newest 10 (11-08).
  - **Claude reasoning under another adapter** **NEW** (11-20): Every compaction, a whole-turn removal included, removes Claude's signed reasoning from the stored conversation whichever adapter holds it (`Model.drop_stale_reasoning`):
    - The thinking blocks of a session that `set_model` or a fail-over handed from a Claude model to another adapter, which leaves them out of its requests while a switch back sends them again.
    - The reasoning items of a Claude model on the Responses API (those whose `format` starts with `anthropic`, as OpenRouter's do), which are sent back as they are.
    - Before, only the Anthropic adapter removed such reasoning, and the next request that carried it to Claude after a compaction lost the block or, on Claude Fable 5.2, was refused.
  - **Events and switches**:
    - Every compaction is logged as a `compaction` event (`step`, `context_tokens`, `dropped_tokens`, `results`, `arguments`, `reasoning_blocks`, `target_tokens`; a whole-turn removal adds `turns`) **CHANGED** (11-20: a whole-turn removal reports `reasoning_blocks` too).
    - `compaction=False` on `run()`, the SEA setting or the global `compaction` setting turns it off (11-05).
- **Bulk clearing of a Claude model's old tool calls** **NEW** (11-11, replacing the server-side clearing of 11-08):
  - **Scope and base thresholds**:
    - A Claude model on the Anthropic API (not Bedrock, which keeps the compaction thresholds above) has its old tool calls and its old reasoning cleared by the agent, in the stored conversation.
    - Past 250,000 tokens of context, the inputs and results of all but the last 20 tool calls are stubbed and the old thinking blocks are removed in one edit, and only when that removes at least 50,000 tokens, so the cached prefix is re-written only then.
  - **Price-ratio scaling and hand-off cap** **CHANGED** (11-11):
    - The two numbers hold for a model whose cache write costs fifty cache reads (Claude Fable 5.2 and 5.1) and are multiplied by the square root of another model's ratio over fifty:
      - 177,000 and 35,000 for Opus and Sonnet 5.5 at 25.
      - 125,000 and 25,000 for the other Claude models at 12.5.
    - The trigger is at most 80% of the hand-off limit (112,000 on a 200,000-token window, which was never cleared before).
    - The minimum counts the reasoning that goes (a reply's output tokens less its visible text).
  - **50-step payback floor** **CHANGED** (11-13):
    - A clearing must also remove the share of the context that pays its cache write back within 50 steps (`payback_floor`: 44% on Claude Fable 5.1 and 5.2, 28% on Opus and Sonnet 5.5, 15% on the other Claude models).
    - One that would remove less is planned again on the next step, and a session in which none reaches that share grows to its hand-off limit and restarts there.
  - **Character-to-token sizing**: A text is sized at 2.55 characters a token on a Claude model and 4 on the other families (`model_families.chars_per_token`): the recorded steps give 2.52 on Claude Fable 5.2, where the 4 used before read 37% low.
  - **Replay evaluations**:
    - Replayed through the agent's own code on the 103 Claude Fable 5.2 sessions of 11-12 and 11-13 (20,582 calls), this makes 326 clearings and 62 hand-offs where the 50,000-token minimum alone made 617 and 16, and costs 13% less (`reports/559a2cb1-clear-payback-recaps-ratio.html`).
    - Replayed on the 19,174 Claude Fable 5.2 calls of 11-11, this clears once per 78 calls, 176,000 tokens at a time, and 3 sessions reach the hand-off limit instead of 23 under 200,000 and 100,000 with the reasoning left out of the count.
  - **Retirement of server-side `clear_tool_uses`**: Anthropic's server-side `clear_tool_uses` edit is no longer requested: its boundary moved by a result or two between requests, and each move re-wrote the prompt cache without freeing anything (312 times on 11-11).
  - **Thinking blocks across Claude versions**:
    - Claude 5.1 and later, which drop a thinking block sent back after an earlier message changed or (Fable 5.2) refuse the request, lose every thinking block at a clearing.
    - An earlier model keeps those of its newest reply.
    - A model whose name the version grammar does not read (a name of the user's own catalogue) is taken for an earlier one until a provider rejects one of its blocks; from then on it loses every block too (11-20).
  - **Server-side reasoning retention (`KISS_REASONING_KEEP_TURNS`)**:
    - `KISS_REASONING_KEEP_TURNS` (0, the default, asks for nothing) has two provider-specific forms, while other adapters ignore the setting:
      - **Anthropic API**: The value is the number of turns whose thinking the server keeps, sent as a further edit on requests with thinking on.
      - **Responses API adapter**: Sends an OpenAI model from GPT-5.6 on `reasoning.context: current_turn` for any positive value, unless the run's `model_config` sets `reasoning.context` itself.
    - While a model's requests carry either form, the agent starts a new turn every 5 steps (`KISS_REASONING_TURN_STEPS`) with a short user note.
  - **Overrides and events**:
    - With `KISS_CONTEXT_EDIT_TRIGGER_TOKENS=0` the agent compacts such a model by the thresholds above.
    - The compaction switches above turn the bulk clearing off too, and not the reasoning edit.
    - A clearing is a `compaction` event (`reasoning_blocks` counts the thinking blocks removed); what the server cleared under the reasoning edit is a `context_edits` event.

#### Turn scheduling, native tools, billing and interrupts

- **Routine turns** **NEW** (10-25):
  - A turn is flagged `routine` in its `llm_call` event when it is the turn after:
    - A refused call.
    - A `summary` or memory tool (`memory_delete` excepted).
    - A `bash_job` or `agent_job` wait that ended with the job still running.
    - An unchanged `Read` on the host.
  - When a turn makes several calls, the last one other than `finish` decides.
  - `KISS_ROUTINE_TURN_EFFORT` (empty by default) runs such turns at a lower reasoning effort on models whose catalogue entry has a `thinking` level.
- **Prompt-cache keep-alive** **CHANGED** (`prompt_cache_keepalive.py`, 233 lines):
  - Runs under a thread that pings the model's prompt cache every 240 s for:
    - Fan-out tools (`run_parallel`, `run_commands_parallel`, `run_agent`).
    - `ask_user_question`.
    - `bash_job` and `agent_job` waits.
    - Any tool call with a timeout of 300 s or more.
  - The number of pings now follows the model's prices (10-25): `int(0.8 × (cache-write − cache-read) / cache-read)`, at least 1 and at most 60 (4 hours), or 10 when the prices are unknown.
  - Each ping is billed and recorded as an `llm_call` event.
- **Native tool calling** **NEW** (10-17):
  - `codex/*` models receive the tools as an MCP server over bridge threads.
  - `finish` and tools that must run between turns (`set_model`) stay on the text protocol (section 8).
- **Per-call cost events**:
  - Every billed model call emits `llm_call` with `model`, `step`, `duration_ms`, `input_tokens`, `output_tokens`, `cache_read`, `cache_write`, `cost` and three newer fields **NEW**: `routine` and `effort` (10-25) and `stop_reason` (10-28).
  - Pricing and attribution:
    - Up to nine token counts are priced (input, output, cache read, 5-minute and 1-hour cache writes, audio in and out, audio cache read, image output).
    - A cost reported by the provider wins over the catalogue price.
    - A failed call that was billed is charged from its partial usage.
    - The spend of a vision-bridge helper is filed under the helper's own model (11-06).
- **Stop**:
  - **Task stop** (`stop_signal.py`, 58 lines): Publishes each task thread's stop event, and a stop request sends no further model request.
  - **Tool interrupt** (`tool_interrupt.py`, 612 lines): Stops one tool call from its panel (nested calls are registered under their parent):
    1. Signals a cooperative event first.
    2. Raises a forced exception in the tool's thread after 1.0 s.
    3. Returns the result `User interrupted the tool call.`
- **Tool errors**: An exception in a tool returns `Failed to call <name> with <args>: <error>` followed by the expected signature and, for an unknown argument, a hint.

### RelentlessAgent (`src/kiss/agents/sorcar/relentless_agent.py`, 2,065 lines)

- **Sessions and parameters**:
  - Runs a fresh `KISSAgent` per session (`<name> Session-N`) until the task finishes, for at most `max_sub_sessions` (10,000) sessions.
  - `run()` takes 18 parameters: KISSAgent's without `is_agentic` and `print_prompts`, plus `work_dir`, `max_sub_sessions` and `docker_image`.
  - Defaults: model `claude-opus-4-6`, budget $200; the app passes its `max_budget` setting instead, default $100.
- **Hand-off** **CHANGED** (10-25):
  - A session of more than one step that ends on the context limit, or on a `KISSError` with no underlying exception, is summarised from a scripted digest (`handoff_digest.py`, 257 lines) of:
    - The task text (3,000 characters).
    - The last 3 `summary` texts (1,500 characters each).
    - Up to 40 written files with the top-level symbols of the Python ones.
    - The last 5 failing tool calls (600 characters of each result).
    - The last 40 tool calls (300 characters of each result).
  - Any other error, or one in a session's first step, ends the run with a failed result.
  - A summariser with no tools but `finish` writes the narrative from the digest (3 steps at most); it replaced a summariser that read files and ran commands.
  - The next session starts with `# Task Progress (Continuation N)` over the summaries of the sessions so far, the oldest dropped first to fit 60,000 characters.
  - The run stops after 2 zero-progress sessions: no tool call other than `finish`, or a summary identical to the previous one.
- **Budget exhaustion**:
  - A sub-agent that runs out of budget returns a partial result that quotes up to 8 of its steps (600 characters each) after the summaries of its earlier sessions.
  - A top-level run raises.
- **Framework instructions**:
  - Every session's system prompt gets (a plain-I/O run gets none of them):
    - `IMPORTANT_INSTRUCTIONS`.
    - A Task Settings block (model name, maximum budget, starting time, user id, IP address, OS, machine info).
    - `~/.kiss/AGENTS.md`, renamed from `SORCAR.md` on 10-03 (a legacy file is adopted) and read with writer-tolerant IO so a rename in progress cannot abort the task.
- **Usage ledger**:
  - Append-only ledger with epochs and de-duplication by source and sequence number, so sub-agent, classifier, TTS and side-channel spend are counted exactly once, including across stop interrupts.
  - Printer offsets never decrease, so the UI cost never drops.
  - The step footer shows the task's cumulative spend against its cap **NEW** (10-20).
- **Docker and remote execution**:
  - `docker_image=` modes:
    - Container image: starts a container through `DockerManager` (`docker_manager.py`, 1,663 lines) with the task work dir bind-mounted at its host path and set as `WorkingDir`.
    - `container:<name-or-id>`: attaches to a running container without mounts and leaves it running.
    - `exec:<url>`: runs the tools on a remote environment through an exec bridge (`exec_bridge.py`, 488 lines) **NEW** (10-26).
  - Container output streams as `bash_stream` events.
  - Every exec is tagged, so a timed-out or stopped command is killed inside the container together with the processes it spawned.
  - The shell and edit guards below apply in the container as on the host (section 21).

### SorcarAgent (`sorcar_agent.py`, 5,132 lines)

- **Parameters**: `run()` takes 32 parameters: RelentlessAgent's plus `web_tools`, `is_parallel`, `current_editor_file`, `ask_user_question_callback`, `base_system_prompt`, `append_basic_tools`, `inherited_tools`, `use_memory`, `tool_profile`, `live_browser`, `prompt_suffix`, `system_prompt_hook`, `tools_hook` and `task_begun`.
- **Resolved once per run**:
  - **The model**:
    1. The caller's.
    2. Else the last one picked.
    3. Else the default (section 8).
  - **The tool profile**:
    1. The explicit one.
    2. Else `review` for a reviewer sub-agent whose task is not an implementation task (`KISS_TOOL_PROFILES`).
    3. Else `full` (section 7).
  - **The system prompt**:
    - Base prompt: `SYSTEM.md`, or the lite prompt when the classifier calls the task simple or the run cannot edit files (section 11); `base_system_prompt` replaces either, and a SEA's `system_prompt()` returns the final text (section 9).
    - Additions:
      - A restricted-profile note lists only the tools the run really has.
      - `WEB_TOOLS_OFF_NOTE` is added when browsing is off.
      - The memory protocol with a recall for the task text is appended (section 13).
      - Task Settings gain the parallel mode and the scratch dir (`<OS temp dir>/kiss-tmp/<task-id>/` under a chat, `<OS temp dir>/kiss-tmp/` with no task id, `<OS temp dir>/kiss-tmp/kiss-sub-<random>/` for a sub-agent, and `/tmp/kiss-tmp/…` in a container; `kiss-tmp-<uid>` when `kiss-tmp` is taken in a shared `/tmp`, private to the user and emptied on the first use after a reboot) **CHANGED** (11-18; `./tmp/` and `./tmp/<task-id>/` before).
- **Tool set** (sections 7 and 9):
  - Starts from the built-ins of the profile plus the caller's tools; `append_basic_tools=False` leaves only `finish` and the caller's tools.
  - A SEA's `tools()` sees that set without `finish`, which is added after the hook returns and cannot be removed by it (a tool named `finish` that the hook returns takes the built-in one's place, as the `coding` SEA's does).
  - Of two tools of one name in the list `tools()` returns, the later wins.
- **Model switching and `summary`**:
  - `set_model` switches the model between turns: the conversation is handed over and the tool schema rebuilt for the new model.
  - `summary` is a no-op tool; its every-10-steps cadence is no longer left to the prompt alone (see the summary cadence guard below).
- **Follow-up messages**: A message the user sends while the task runs is injected before the next step as `User says: <message>. Take the message into account and finish your task.`, with its attachments in one more user message (attachments since 10-20).
- **Background jobs**: Stay on the agent, so a follow-up prompt in the same chat can still wait for, tail or stop them (section 7).
- **Sub-agents**: A child's default budget share is `remaining / (num_tasks + 1)`; the review fan-out quotas removed on 09-25 have not returned (section 10).
- **Worktree tools**: `create_worktree` and `commit_and_merge` live on `SorcarAgent` itself; `WorktreeSorcarAgent` was removed on 10-10 **CHANGED** and the `is_worktree` setting on 10-09 (section 12).
- **End of run**:
  - Cleanup sequence, each step protected from the failure of the one before:
    1. The browser is closed.
    2. The remaining `run_agent` jobs are stopped.
    3. The marker of an unmerged worktree is kept.
    4. Classifier spend is folded in.
    5. The run hooks are released.
    6. The usage totals are emitted.
  - Exceptions and warnings:
    - An unexpected exception becomes a `success: false` result; `KISSError` and `SeaError` propagate.
    - Warnings collected during the run are emitted as `warning` events.
- **Command line** (section 5):
  - The `sorcar` console script runs a `ChatSorcarAgent` on `-t TASK` or `-f FILE`, with `-m/--model`, `-b/--max-budget` and `--work-dir` (default `$KISS_WORKDIR`, else the current directory).
  - Exits 0 when the result reports success and 1 otherwise.

### ChatSorcarAgent (`chat_sorcar_agent.py`, 1,315 lines)

- **Persistence and parameters**:
  - Persists chats and task chains in `~/.kiss/history.db`:
    - Allocates the chat id and the `task_history` row.
    - Records the budget, the SEA name of the run (`task_history.sea`) and the cron flag.
    - Broadcasts `new_tab`, `tasks_updated` and `task_settings`.
    - Saves the result.
  - `run()` takes `prompt_template` first and 36 parameters in all, 5 of them private to the daemon.
- **Chat history in the prompt**:
  - The tasks of a chat form a tree whose root is the chat's first task:
    - A task run after the user loaded a task from the CHATS list continues that task; any other task continues the chat's last task. The row stores the link in `task_history.chat_parent_id`.
    - The load counts once, for the next task started from that tab, and ends when the tab's transcript is replaced by another task's or by the chat's latest.
    - Clicking a task while the chat is running applies to the next new task sent from that tab. Plain messages typed in the meantime still go to the running task and leave the selection in place, whereas a `<task>` message queues a new task and consumes the selection—unless the daemon refuses the message, in which case the selection returns with the restored draft.
    - A task from before the link was stored, or whose parent was deleted, continues the task just before it in the chat. A sub-agent continues the task that dispatched it.
  - Includes the tasks on the path from the root to the task being continued, at most 10 of them (the first 2 and the newest 8 when the path is longer), under `## Previous tasks and results from the chat session for reference`, followed by `# Task (work on it now)`.
  - With `KISS_CHAT_HISTORY_DIGEST` on (the default):
    - The newest two tasks and results are quoted in full while they fit, and the older tasks become 600/300-character task/result digests, the oldest dropped first until the prefix fits 6,000 characters.
    - When the newest two alone exceed that, they are digested too and the prefix is then cut at 6,000 characters.
- **Bare-path prompts** (`bare_path_task.py`, 112 lines):
  - A prompt that is only an existing path, optionally quoted or starting with `~`, becomes a directive to open it with:
    - `open` (macOS).
    - `rundll32.exe url.dll,FileProtocolHandler` (Windows).
    - `xdg-open` (elsewhere).
  - The rewrite is for a task typed into a chat, the prompt of an `in_chat` cron job included (section 20).
  - The daemon skips it for:
    - A run that names a SEA, such as a `/xxx` command (a model-picker SEA alone does not count).
    - Every sub-agent task, since there the path is the task itself.
    - A run with its own base system prompt.
  - The history row and tab title keep the raw path.
- **Unattended runs**: With `is_unattended` set, `ask_user_question` refuses instead of blocking.
- **Run settings** **NEW** (`run_config.py`, 201 lines, 10-05):
  - The `task_settings` event carries the model, work dir and budget together with the SEA, channel, tool profile, timeout and which settings were inherited or pinned.
  - The same data renders the `ran:` line that opens every `run_agent` and `run_parallel` result (section 10).
- **Transcript digests** (`task_digest.py`, 752 lines):
  - Turns the persisted event stream of a task into numbered, clipped entries and renders it as:
    - A paged transcript (150 entries a page by default, 400 at most).
    - An overview.
    - One entry in full.
    - A 60,000-character context.
  - Read by the task-update agent, `/ask` and `rsi7d` (sections 9 and 16).

### Verdicts and guards

Every tool call passes through two stages in order:

1. The SEA's `tool_call_hook`.
2. The agent's guard, which asks three checks in order and returns the first refusal:
   1. A pending user message.
   2. Live `run_agent` jobs.
   3. The worktree guard.

#### Hook verdicts, refusals and redirects

- **Verdicts** **NEW** (`tool_verdict.py`, 219 lines, 10-07):
  - A hook answers one of:
    - `ALLOW`.
    - `allow_with_note(text)` (the call runs and the text is appended to its result; 11-02).
    - `refuse(text)`.
    - `end_run(result, success=True)`.
  - The `"OK"` allow-spelling was dropped on 10-07.
- **Run-ending verdicts** **NEW** (10-31): `end_run` from `tool_call_hook` or `llm_response_hook` ends the run at once with the given result, shaped like a `finish` result.
- **Refused calls** **NEW** (10-24):
  - The result of a refused call starts `REFUSED: the <tool> call did NOT run and had no effect.`
  - The agent remembers the refused call and appends a re-issue reminder to the first allowed result of another tool; a refused `finish` owes no re-issue.
- **Redirected calls** **NEW** (11-01):
  - A tool can raise `ToolRedirect(name, arguments, note, followup)` to have the agent make another call in its place.
  - The redirected call and its follow-up pass the same hook and guard.

#### Built-in guards

- **Finish guards**:
  - A `finish` issued while a user message is pending is rejected.
  - The first `finish` and the first `commit_and_merge` with `run_agent` jobs still running are rejected once each, listing the jobs, and finishing stops whatever still runs **NEW** (10-05).
  - The first `finish` with an unmerged worktree is refused once, asking for `commit_and_merge`.
  - The general agent refuses `finish` for nothing else: only the hidden `coding` SEA gates its finish, on a verifier script and a review (section 9).
- **Worktree guard**:
  - `Write` and `Edit` of a checkout path that git does not ignore, and `rollback` in a checkout, are refused until `create_worktree`.
  - Exceptions (section 12):
    - A run without that tool (a worker SEA), a run in a Docker container and a repository whose worktree setup failed edit in place.
    - A checkout path of the work dir's own worktree passes because the tools remap it into the worktree.
- **Fan-out guard** **CHANGED** (`fanout_guard.py`, 441 lines):
  - `tasks` must be a JSON array of non-empty strings (a lenient decode repairs raw newlines and lone backslashes).
  - Review tasks and budget cap:
    - A task that reads as a review and not as an implementation runs on the read-only `review` profile.
    - A review dispatched without a budget has the share it inherits from the caller capped at 1/8 of what the calling task has spent so far (its own calls and ended sub-agents combined), with a $10 floor (11-20; an `inherit: false` review inherits no share and has no such cap).
  - Same-family restriction (11-01, section 10): A sub-agent of the caller's own model family is refused unless:
    - The fan-out has two or more tasks over disjoint named pieces,
    - The caller's own task asks for sub-agents,
    - The child's model is of another family, or
    - The child is a review.
- **Shell guards** **CHANGED** (`shell_guards.py`, 231 lines):
  - One module shared by host and container `Bash` and `run_commands_parallel` since 09-30; before that the guards belonged to the coding SEA's container harness.
  - A command that would end the task itself is refused before it runs:
    - `kill -1`, `pkill -f .`, or an `rm` of `/` or the work dir, also through `sudo`, `env`, `nohup`, `timeout` or `sh -c`.
    - The patterns are plain regular expressions on the command text, so a quoted mention is refused too.
  - Package installs and builds (`apt`, `pip`, `uv`, `npm`, `cargo`, `make`, `cmake` and others) get a timeout of at least 900 s.
- **Edit guards** **NEW** (`edit_guards.py`, 1,885 lines, 10-25):
  - **Read-before-edit requirement**:
    - A file must have been shown before it is edited.
    - Bookkeeping is keyed by the repository-relative path, so a file read in the main checkout counts as read in its worktree.
    - A `cat`, `sed -n`, `grep -n`, `nl`, `head` or `tail` of one file counts as a read.
    - An `Edit` refusal carries the 10 lines around the string (section 7).
  - **Shell file-rewrite refusals and conversions**:
    - Shell rewrites of an existing file are refused with the `Edit` call to make instead (`sed -i` with a newline in its script, a Python here-document that reads, `.replace()`s and writes).
    - A `cat` or `tee` here-document into an existing file or into a path only the shell can resolve is refused as well (10-28).
    - A rewrite that converts exactly (a here-document copy with a literal body, a `.replace` rewrite, a one-substitution `sed -i`) is applied as the `Write` or `Edit` it stands for when the run has both tools.
- **Summary cadence guard** **NEW** (`seas/base/base_sea.py`, the root SEA layer):
  - At every 10th step of a run that has the `summary` tool, a summary comes due.
  - Until `summary` is called, every other call except `finish` still runs, with the note `A summary is due ...` appended to its result.
  - It was first a refusal (10-05), was restored on 10-26 and has been a note since 11-02, because each refusal cost a model step to re-issue.

### Configuration knobs

#### Environment-backed `Config` fields (`src/kiss/core/config.py`, 723 lines)

- **Field count and boolean parsing**:
  - `Config` has 34 fields **CHANGED** (11-19, down from 37): 33 read the environment when the process starts, and the 34th, `max_budget`, defaults to 100.0.
  - Boolean flags are off for `0`, `false`, `no` and `off`.
- **Runtime and threshold fields**:
  - **`KISS_CLAUDE_CODE_USE_SUBSCRIPTION`** (default: on): `cc/` models bill the Claude subscription of the `claude` CLI; API keys are kept out of its environment (section 8) **NEW**.
  - **`KISS_CODEX_USE_SUBSCRIPTION`** (default: on): The same for `codex/` models and the ChatGPT subscription (section 8) **NEW**.
  - **`KISS_READ_DEDUPE`** (default: on): `Read` returns a one-line "unchanged" stub for a range still in the context.
  - **`KISS_READ_OUTLINE_LINES`** (default: 2000; 0 disables):
    - On the host, a `Read` of a longer file from line 1 with `max_lines` of 2,000 or more (the defaults) returns a symbol outline when the file has at least 5 recognised definitions (headings in a Markdown file), the first 400 listed and the rest counted.
    - Such a `Read` is an ordinary whole-file read otherwise.
  - **`KISS_TOOL_OUTPUT_COMPACTION`** (default: on): Compaction of old tool outputs and arguments.
  - **`KISS_COMPACTION_START_TOKENS`** (default: 100000): Context size of the first compaction.
  - **`KISS_COMPACTION_STEP_TOKENS`** (default: 100000): Context growth before the next one.
  - **`KISS_COMPACTION_TARGET_FRACTION`** (default: 0.3): Share of the window past which every step considers a compaction, never less than `KISS_COMPACTION_START_TOKENS`; 0 disables the target.
  - **`KISS_REASONING_KEEP_TURNS`** (default: 0): Reasoning retention asked of the provider (0 asks for nothing):
    - On the Anthropic API: the number of turns whose thinking is kept.
    - On an OpenAI model from GPT-5.6 on (Responses API adapter): any positive value keeps the current turn's reasoning only.
  - **`KISS_REASONING_TURN_STEPS`** (default: 5): Steps per reasoning turn while the model's requests carry such a request; 0 never starts a new turn.
  - **`KISS_CONTEXT_EDIT_TRIGGER_TOKENS`** (default: 250000): Context tokens past which the agent clears a Claude model's old tool calls and reasoning, at fifty cache reads a cache write (scaled to the model's prices); 0 leaves the model to the compaction thresholds.
  - **`KISS_CONTEXT_EDIT_KEEP_TOOL_USES`** (default: 20): Tool calls the clearing keeps.
  - **`KISS_CONTEXT_EDIT_CLEAR_AT_LEAST`** (default: 50000): Tokens of tool calls and reasoning a clearing must remove to be applied (scaled like the trigger).
  - **`KISS_ROUTINE_TURN_EFFORT`** (default: empty): Reasoning effort of a routine turn; empty keeps the model's effort.
  - **`KISS_TOOL_OUTPUT_MAX_CHARS`** (default: 50000): Ceiling of `max_output_chars` for one `Bash` or `memory_list` result.
  - **`KISS_TOOL_OUTPUT_DEFAULT_CHARS`** (default: 8000): Default budget (section 7) **NEW** of:
    - A `Bash` result (head and tail are kept; on the host the whole output is also written to a file under `<scratch dir>/tool-out/` **CHANGED** (11-18)).
    - A whole-file `Read` on the host (the lines that fit, then a continuation note; a container `Read` is cut head and tail like a `Bash` result).
    - A `memory_list` listing (cut at a page boundary, with the count of pages left out).
  - **`KISS_BASH_DETACH_SECONDS`** (default: 120): A foreground host `Bash` with a longer timeout, no `until` poll and `on_timeout` left at `kill` continues as a background job after this grace period; 0 disables (section 7) **NEW**.
  - **`KISS_CONTEXT_LIMIT_FRACTION`** (default: 0.7): Share of the window at which a session hands off.
  - **`KISS_TOOL_PROFILES`** (default: on): Reviewer sub-agents get the read-only `review` profile.
  - **`KISS_CHAT_HISTORY_DIGEST`** (default: on): Digest of the chat history in the prompt.
  - **`KISS_DISPATCH_PATH_REWRITE`** (default: on):
    - Sub-agent task text naming the parent repository is rewritten to the active worktree path (paths under the checkout's `.venv`, the environment the worktree shares, stay) when the sub-agent inherits from the caller and runs on this machine.
    - A channel, an `inherit: false` call and a run on another machine get the text as given (section 10).
- **Credential fields (the other 12 fields)**:
  - Each is read from the environment variable of its own name, is empty by default, and is stored by the app in `~/.kiss/api_keys.env`, never in `config.json` (section 8):
    - Four API keys and a workspace id **CHANGED** (11-19): `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`.
    - Amazon Bedrock credentials: `AWS_BEARER_TOKEN_BEDROCK`, or `AWS_ACCESS_KEY_ID` with `AWS_SECRET_ACCESS_KEY` and optional `AWS_SESSION_TOKEN`, and `AWS_REGION`, in whose place the Bedrock adapter uses `us-east-1` while it is empty.
    - Two CLI tokens: `CLAUDE_CODE_OAUTH_TOKEN` and `CODEX_ACCESS_TOKEN` **NEW**.

#### Environment variables read outside `Config`

- **`KISS_HOME`**: State directory, default `~/.kiss`.
- **`KISS_USE_MEMORY`**: Precedence (within the gates of section 13):
  1. A run's own `use_memory` argument wins over both settings below.
  2. When non-empty, `KISS_USE_MEMORY` wins over the stored `use_memory` key for the process.
  3. The stored `use_memory` key.
- **`KISS_DISABLE_TASK_CLASSIFIER=1`**: The test suite's switch (section 11).
- **`KISS_WORKDIR`**: The directory the `sorcar` wrapper was invoked from.
- **`KISS_SKILLS_REGISTRY_URL`**, **`KISS_SKILLS_ARCHIVE_URL`** and **`KISS_SKILL_JUDGE_MODEL`** **NEW** (11-20): The base URLs a skill install reads the registry (`https://skills.sh`) and the repository archives (`https://codeload.github.com`) from, and the models that check the skill, separated by commas (section 7). All three are read at each call.
- **`KISS_DISABLE_WORKTREE_POOL`**: Removed on 10-09.

#### `~/.kiss/config.json` (`src/kiss/core/user_config.py`, 1,796 lines **NEW**, 10-14)

- **Storage and permissions**:
  - Before 10-14, the settings lived with the VS Code extension.
  - The file is written atomically with mode 0600 under a lock.
- **`DEFAULTS` (18 keys)**:
  - **`max_budget`** (default: 100.0): Budget in USD of a task started from the app.
  - **`custom_endpoint`**, **`custom_api_key`**, **`custom_headers`** (default: empty): Custom model endpoint (section 8).
  - **`use_web_browser`** (default: true): Browser tools (section 17).
  - **`remote_password`** (default: empty): Remote-access password; empty admits loopback clients only (section 14).
  - **`classify_tasks`** (default: true): Pre-run task classification (section 11).
  - **`classify_with_decisions`** (default: true): Use of the decisions model, which also needs an OpenRouter key and the model's catalogue entry (section 11):
    - The classifier asks it first.
    - A run whose tool profile keeps `decide` gets that tool.
  - **`use_memory`** (default: true): Persistent memory (section 13).
  - **`memory_dir`** (default: empty): Memory pages directory; empty means `~/.kiss/memories`.
  - **`compaction`** (default: true): Global compaction switch, applied to running tasks too; off overrides every run's own option **NEW**.
  - **`chrome_bridge`** (default: false): Start the Chrome bridge server at daemon boot (section 17) **NEW**.
  - **`claude_code_use_subscription`** (default: true): Settings-panel form of `KISS_CLAUDE_CODE_USE_SUBSCRIPTION`.
  - **`codex_use_subscription`** (default: true): Settings-panel form of `KISS_CODEX_USE_SUBSCRIPTION`.
  - **`work_dir`** (default: empty): Work dir setting.
  - **`default_work_dir`** (default: empty): The folder the page switches to when a chat is opened; a pick in the chat header overrides it (section 16) **NEW** (11-21).
  - **`last_model`** (default: empty): The model last picked.
  - **`last_machine`** (default: empty): The machine a new chat's tasks run on; empty means this machine (section 15) **NEW**.
- **Environment mirroring, pass-through keys, and retired keys**:
  - `compaction` and the two subscription switches take their defaults from the environment variables of the same meaning, and `max_budget`, `compaction` and the two switches are mirrored into `Config` by `apply_config_to_env`.
  - Keys outside `DEFAULTS` pass through: `tunnel_token`, `skill_permissions`, `mcp_permissions`, `mcp_upfront_tool_chars` (absent; **NEW** 11-12), `email` and `recent_work_dirs` (30 at most).
  - `demo_mode`, `is_parallel` and `remote_password_hash` are retired and dropped on load.
  - `auto_commit_mode` and `is_worktree` left the settings on 10-09, with the change that introduced the worktree tools (section 12).
  - The voice options, which the previous inventory listed here, are not keys of this file: the web page keeps them in the browser's local storage (section 18).

## 7. Built-in tools and tool profiles

### Shell (3)

- **Tools**: `Bash`, `bash_job`, `run_commands_parallel`.
- `Bash(command, description, timeout_seconds=300, max_output_chars=8000, background, machine, on_timeout, until, poll_seconds=5)`:
  - Runs each call in a fresh shell in the work dir.
  - **Output budget and truncation**:
    - A result keeps 8,000 characters by default (`KISS_TOOL_OUTPUT_DEFAULT_CHARS`), head and tail, and on the host names the file under `<scratch dir>/tool-out/` (`<kiss-tmp>/tool-out/` with no scratch dir) that holds the whole output **CHANGED** (11-18).
    - `max_output_chars` raises the budget to at most 50,000 (`KISS_TOOL_OUTPUT_MAX_CHARS`) **CHANGED**.
  - **Secret redaction** **NEW** (11-11): A secret of the key store that a tool prints is shown as its variable's name before the output is stored, sent or cut (`src/kiss/core/secret_redaction.py`):
    - Covers every value of at least 8 characters whatever its variable is called, the region `AWS_REGION` excepted.
    - A value once read stays covered while the daemon runs, so a key rotated while a command still prints the old one is replaced all the same.
  - **Background and remote jobs**:
    - `background=true` detaches the command like `nohup`, logs to `<scratch dir>/bash_jobs/<id>.log` (`<kiss-tmp>/bash_jobs/<id>.log` with no scratch dir) **CHANGED** (11-18) and returns a job id.
    - `machine=` starts that job on a registered SSH machine (section 15) and streams its output into the local log **NEW**.
  - **Timeouts and auto-detach**:
    - `on_timeout="background"` starts the command as a job and waits `timeout_seconds` for it, so a command that outlives its deadline keeps running.
    - Without it, a foreground host call given more than 120 s (`KISS_BASH_DETACH_SECONDS`, 0 disables) and no `until` poll that is still running after 120 s continues as a job **NEW**.
    - A command killed at its deadline returns what it had printed **NEW**.
  - **Polling with `until`**:
    - `until=<regex>` re-runs the command every `poll_seconds` (at least 1) until its output matches or `timeout_seconds` pass, so a poll costs one call **NEW**.
    - On the host, a run whose output is too long to hold whole (the capture then keeps its first and last 1,000,000 characters) ends the poll unsearched.
- `bash_job(job_id, action, timeout_seconds=300, tail_lines=50, max_output_chars=50000, until)`:
  - Tails, waits for or kills a job.
  - `until=<regex>` ends a wait as soon as the log matches **NEW**.
- **Early exit on unread sub-agent results** **NEW** (11-09): On the host, an `until` poll and a `bash_job` wait also end early once every sub-agent job of the task has finished with a result unread (each such result ends one wait), naming `agent_job('', 'wait_any')` as the call that returns it (section 10).
- `run_commands_parallel(commands, max_workers=0, timeout_seconds=1800, max_output_chars=5000, description)`:
  - Runs a JSON list of commands in threads (`max_workers=0` runs all at once) and reports a tally line, then each command's exit code and output.
  - On the host, a report over 50,000 characters is written to `<scratch dir>/tool-out/` **CHANGED** (11-18) and a digest is returned in its place (each command's status and the last 1,200 characters of each failure) **NEW**.

### Files (3)

- **Tools**: `Read`, `Edit`, `Write`.
- `Read(file_path, max_lines=2000, start_line=1, force)`:
  - **Whole-file reads**:
    - A whole-file read of a file over 2,000 lines (`KISS_READ_OUTLINE_LINES`) returns an outline of its definitions or headings when it has at least five.
    - Any other whole-file read stops at the 8,000-character budget and says where to continue **NEW**.
  - **Deduplicated windows** **CHANGED**: A window whose text an earlier `Read` already returned, from any file or range, comes back as a one-line note naming that step, and `force=True` re-reads it.
  - **Directories, media and missing paths**:
    - A directory returns its listing.
    - An image, PDF, audio or video file of up to 20 MB is attached for the model.
    - A missing path gets the closest existing name.
- `Edit(file_path, old_string, new_string, replace_all, edits, exact)`:
  - `edits` is a JSON list of `{old_string, new_string, replace_all}` objects applied in order, all or none **NEW**.
  - **Whitespace-normalised fallback and miss reporting** **NEW**:
    - When `old_string` is not in the file verbatim and `replace_all` is off, the one place that matches once runs of blanks are collapsed and trailing whitespace is dropped is replaced instead, re-indented to the file (`exact=true` forbids the retry).
    - A miss names the closest region with its line numbers.
  - The result shows the lines around each replacement **NEW**.
- `Write(file_path, content)`: Creates or overwrites a file.

### Checkpoints (2)

- **Tools** **NEW**: `checkpoint`, `rollback`.
- `checkpoint(name)`:
  - Saves every file of the work dir's git working tree that git does not ignore, tracked or not, as a git tree object written through a private index: HEAD, the branch, the index and the stash are never touched.
- `rollback(name)`:
  - Makes those files match the checkpoint again (changed and deleted files rewritten, files created since deleted, ignored files left alone) after saving the replaced state as `before-rollback`.
  - Changes nothing when a file no checkpoint holds is in the way, and refuses a checkpoint taken before `create_worktree` moved the work dir.
- **Names**: Names are 1 to 40 letters, digits, `_` or `-`; `checkpoint` refuses the reserved `before-rollback`.
- **Scope**: Top-level tasks on the host only; a sub-agent edits its caller's tree in place.

### Worktree (2)

- **Tools**: `create_worktree`, `commit_and_merge` (**NEW** as agent-called tools; section 12).
- `create_worktree(path)` isolates the repository of `path` (default: the work dir) on a private branch.
- `commit_and_merge()` commits every worktree and squash-merges it into the branch the task started from.
- Not in Docker.

### Browser (15)

- **Tools**: `go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser`, `evaluate_script`, `list_console_messages`, `list_network_requests`, `wait_for`, `attach_to_chrome`, `use_my_chrome` (section 17).
- **Patchright/Chromium actions and availability**:
  - `click(element_id, action)` clicks or hovers, `type_text(element_id, text, press_enter)`, `scroll(direction, amount)`, `screenshot(file_path)`, `get_page_content(text_only)`.
  - `show_browser(visible)` streams the page into the KISS Browser tab on every surface.
  - Sub-agents get an ephemeral profile.
  - Offered only in profiles that include the `browser` group (`full`, `review`, `browser` and composites) with "Use web tools" on.
- **DevTools-grade observation** **NEW**:
  - `evaluate_script(script)` runs a function in the page (awaited, JSON result capped at 20,000 characters).
  - `list_console_messages(clear)` and `list_network_requests(url_filter, clear)` read per-page ring buffers (200 console entries, 300 requests, with `clear=` for before/after attribution).
  - `wait_for(text, timeout=15, disappear)` waits for visible text (hidden copies and `content-visibility:hidden` ignored, text split across inline markup found, 120 s cap).
- **Two routes into the user's own browser** **NEW**:
  - `attach_to_chrome(cdp_url)` adopts the newest tab of a Chrome started with `--remote-debugging-port` (`close_browser` detaches without closing it; `show_browser` carries only the URL out of it, never its cookies).
  - `use_my_chrome(action, tab_id, url, script, uid, text, timeout=30)` drives the user's everyday default-profile Chrome (which Chrome 136+ no longer exposes over CDP) through the bundled "KISS Sorcar Bridge" MV3 extension, with ten actions:
    - `list_tabs`, `open_tab`, `navigate`, `activate_tab`, `close_tab`.
    - `evaluate`.
    - `screenshot` (saved to `<scratch dir>/chrome-shot-<ns>.png` **CHANGED** (11-18)).
    - `snapshot` of interactive elements with uids.
    - `click` and `type_text` by uid.

### Memory (7)

- **Tools**: `memory_search`, `memory_pull`, `memory_read`, `memory_write`, `memory_list`, `memory_delete`, `memory_refresh` (section 13).
- `memory_search(query, k=5, memory)` returns page names with scores.
- `memory_pull(query, k=3, memory)` returns the matching pages whole within a 24,000-character budget:
  - A first page over the budget is cut to it with a pointer to `memory_read`.
  - A later page that does not fit ends the result, and the pages left out are counted.
- `memory_write(name, content, title, summary)` creates or replaces a page and warns when its searchable text passes the 8,192 bytes the index embeds.
- `memory_list(memory, max_output_chars)` keeps the listing within an 8,000-character default budget **NEW**.
- `memory_refresh(stale_days=30, duplicate_threshold=0.9, memory)` re-indexes and reports stale and near-duplicate pages.
- `memory=` narrows to the general or a domain memory.
- Not registered in Docker runs or in a profile without a memory tool.

### Dispatch (4)

- **Tools**: `run_agent`, `agent_job`, `run_parallel`, `number_of_cores` (section 10).
- `run_agent(task, agent, model, tool_profile, max_budget, timeout, options, wait, machine, description)`:
  - Runs one sub-task as a plain sub-agent or a SEA, on this host or with `machine=` on a registered machine (section 15) **CHANGED**.
  - `wait="false"` returns a job id at once.
- `agent_job(job_id, action, timeout_seconds)` tails, waits for or kills such jobs, its `wait_any` action returning the results of every finished job not yet read **NEW**.
- `run_parallel(tasks, agent, model, tool_profile, max_budget, timeout, options, description)` fans a JSON list of tasks out **CHANGED**.
- `run_parallel` and `number_of_cores()` exist only in parallel mode.

### MCP (2, plus server tools)

- **Tools**: `connect_mcp_server`, `finish_mcp_server_connect`, and the tools of configured MCP servers.
- **Server configuration precedence** (`stdio`, `http` or `sse`; the later file wins):
  1. `~/.kiss/mcp.json`
  2. `<work dir>/.mcp.json`
  3. `<work dir>/.kiss/mcp.json`
- **Tool naming, schema and permissions**:
  - After the `mcp_permissions` wildcard rules, each server tool is offered as `<server>_<tool>` (numbered on a clash with a built-in or earlier name), cut to 64 characters ending in a digest when longer **NEW**, with the server's own input schema declared through `@tool_schema` **NEW**.
- **Deferred servers** (off by default) **NEW** (11-12):
  - Configured by `"alwaysLoad": false` in a server's entry, or an `mcp_upfront_tool_chars` budget in `config.json` under which servers load upfront from the smallest up while their definitions fit.
  - Deferred servers' definitions leave the request and their tools are reached through:
    - `mcp_tool_search`: lists their names in its description and returns definitions on demand.
    - `mcp_tool_call`: runs one whose definition came back to the model in an earlier response of the same conversation; a call sent before that, or together with the search that fetches the definition, gets the definition or a note back and does not run.
    - A search or call for a tool that is not deferred, such as `finish`, is told to call it directly.
  - **Evaluation** (`reports/mcp-tool-deferral.html`):
    - With 11 servers and 205 tools, a 40,000 budget cut a run's prompt tokens by 65% and the right tool was called in 114 of 120 runs against 120 of 120 with everything upfront.
    - On reasoning problems, the answers were right as often, within chance, with the definitions in the request as without, 485 against 489 and 482 of 560.
- **Connection limits and eviction**:
  - 60 s to connect and 300 s per call.
  - Each connect closes other connections in order, sparing those mid-handshake or with a call in flight:
    1. Connections left idle for over 600 s.
    2. The least recently used beyond 8 open.
- **Remote OAuth sign-in**:
  - `connect_mcp_server(name, url, transport)` and `finish_mcp_server_connect(name)` run the OAuth sign-in of a remote server (Notion, Linear, Asana, Zoom or any URL):
    - The sign-in page opens in the Browser tab on every surface, else in the default browser.
    - The server's tools load in the next task.
  - Credentials are kept per server name and URL under `~/.kiss/mcp_auth/` with mode 0600 **CHANGED**.

### Skills (0 or 1)

- **Tools**: `skill` (registered only when a user or project skill exists).
- `skill(name)` loads the body of a `SKILL.md` and lists up to 50 of its resource files, with a note when the listing is cut; the names and descriptions of the available skills sit in the tool's docstring.
- **Discovery precedence** (lowest precedence first):
  1. Claude Code plugins dropped into the git-ignored folder `src/kiss/agents/claude_skills/<plugin>/skills/<name>/` (named `<plugin>:<name>`).
  2. `~/.claude/skills`.
  3. `~/.agents/skills`.
  4. `~/.kiss/skills`.
  5. The work dir's `.claude/skills`.
  6. The work dir's `.agents/skills`.
  7. The work dir's `.kiss/skills`.
- The `skill_permissions` wildcard rules hide denied skills.
- **Skills from the skills.sh registry** **NEW** (11-20): The web app's Skills subpanel (section 16) searches the registry and adds a found skill to `~/.kiss/skills` after a security check (`skill_registry.py`, `skill_rules.py`, `skill_check.py`, `skill_install.py`):
  - **Search**: A query of two characters or more is the registry's search and a shorter one its all-time leaderboard. At most 50 rows come back, the most installed first, without the registry's duplicates, the skills that are not in a GitHub repository and the names that are not valid skill names.
  - **Download**: The repository's archive comes from `codeload.github.com`, and only the skill's own folder is unpacked, into a staging folder inside `~/.kiss/skills` that skill discovery does not see:
    - The archive may be 256 MiB, with 100,000 members and 2 GiB unpacked.
    - The skill may hold 500 files, 5 MiB in one file and 20 MiB in all, in folders at most 32 deep.
    - A link, a device, a sparse file or an unsafe file name refuses the skill, and nothing in it is executed.
  - **Check**: Three layers read the staged files:
    1. Pattern rules over every file (`skill_rules.py`): instructions that override the agent's or speak to the checker, download-and-run and decode-and-run commands, reverse shells, credential exfiltration, destructive commands, start-up persistence, hidden Unicode, compiled programs or archives among the files, a command that runs one of the skill's files that is not text (`runs-unread-file`), and a file whose special characters spell out to more than 10 million characters, which the rules do not read (`inflated-text`).
    2. The audits that skills.sh's partners published, which count for `medium` at most: a failed audit asks for a review, and so do audits that cannot be fetched.
    3. One model call without tools over the text files, `SKILL.md` first, up to 300,000 characters; text that does not fit is a `medium` finding.
    - When the rules already block the skill, the audits are not fetched and the model is not called.
  - **Verdict**: A `critical` or `high` finding blocks the install, a `medium` one asks the user to review the findings, and anything less installs the skill (the staged folder is renamed into place):
    - **Add anyway** installs a reviewed skill only when its files are the ones the review was about (same SHA-256 digest); files that changed since are checked again. A blocked skill is never installed.
    - **Cancel** (11-21) dismisses the review in the page and installs nothing.
    - Nothing is installed when the model cannot be called, when every model declines to read the skill or when its reply cannot be read.
  - **Checking model**: The default model of the first configured provider (`get_default_model`). When it declines to read the skill, the first of the `SECOND_OPINION_MODELS` of another family whose provider is configured reads it, and the skill then needs a review. `KISS_SKILL_JUDGE_MODEL` names the models instead.
  - **Limits**: A skill whose name the agent already has is refused, and at most four installs run at once.

### Interaction (6)

- **Tools**: `ask_user_question`, `talk`, `set_model`, `decide`, `summary`, `finish`.
- `ask_user_question(question)`: Waits for the user's answer, and returns an error at once in an unattended cron run (section 20).
- `talk(language, text, emotion)`: Speaks on every device with the task's tab open (section 18).
- `set_model(model_name)`: Switches the task's model and is refused on a daemon started with `KISS_PIN_TASK_MODEL` **NEW**.
- `decide(state, questions)`: Puts typed questions (`noul`, `choice`, `score`) to the Jev decisions model and is offered only when `classify_with_decisions` is on, `OPENROUTER_API_KEY` is set and the catalogue has the decisions entry (section 11).
- `summary(description)`: The ten-step recap (section 6).
- `finish(success, is_continue, summary_in_html, suggested_next_task)`: Ends the run.

### Tool counts by run mode

- **Full-profile top-level run on the host** (in parallel mode with web tools and memory on and the decisions model available): Offers 44 tool functions:
  - Breakdown: shell 3, files 3, checkpoints 2, worktree 2, browser 15, memory 7, dispatch 4, MCP sign-in 2 and interaction 6.
  - `skill`, the MCP servers' tools and the tools a SEA's `tools()` hook adds (section 9) come on top.
- **Restricted modes**:
  - **Sub-agent**: 42 (no `checkpoint`, `rollback`).
  - **Serial run**: 42 (no `run_parallel`, `number_of_cores`).
  - **Web tools off**: 29.
  - **Docker run**: 33 (no checkpoints, no worktree tools, no memory).

### Guards and rules across shell and file tools

Guards and rules that apply across the shell and file tools:

- **Shell guards** **CHANGED** (`shell_guards.py`; `Bash` and `run_commands_parallel`, on the host and in Docker):
  - A command that kills every process (`kill -1`, `pkill -f .`) or deletes `/` or the work dir is refused.
  - An install or build (`apt`, `pip`, `uv`, `npm`, `cargo`, `make`, `cmake` and similar) gets a timeout of at least 900 s.
- **Waiting and batching** **NEW**:
  - A foreground `sleep` of at most 30 s runs as written.
  - A longer foreground `sleep`:
    - Is refused when it exceeds the call's `timeout_seconds`.
    - Otherwise is spent waiting on the task's most recently started running background job, after which the rest of the command runs.
    - Runs as written when it paces a loop, when it cannot be cut out of the command text, or when there is no such job (where the result names `Bash(until=...)` and `bash_job` as the one-call waits).
  - `run_commands_parallel` refuses any `sleep` over 30 s.
  - A short read-only `Bash` call gets a hint to batch independent commands, at most once in 20 calls.
- **Shell rewrites of files** **NEW** (`edit_guards.py`, read with the quote-aware `shell_scan`):
  - **Applied as the `Write` or `Edit` they stand for when the file was shown in the session** (`ToolRedirect`; the rest of a compound command runs as a follow-up `Bash` call):
    - A `cat`/`tee` here-document copy into an existing file.
    - A Python here-document that reads one file, applies `.replace()` with literal strings and writes it back.
    - A one-substitution `sed -i`.
  - **Refused with the call to make instead**:
    - A `sed -i` with a newline in its script.
    - Any other Python here-document that reads, substitutes and writes a file (`re.sub`, a loop, two files).
    - A here-document into a path only the shell can resolve.
  - **Exemptions**: Scratch files (in the agents' temp directory `<OS temp dir>/kiss-tmp/` when the work dir does not itself lie there, or under `~/.kiss/cron/work`; a `tmp/` folder of the work dir is an ordinary folder) are exempt **CHANGED** (11-18), and a here-document may create a new file at a literal path.
- **Read before modify**:
  - `Edit` and `Write` refuse an existing file that was not shown in the session.
  - Shown means a `Read`, or a `cat`, `sed -n`, `grep -n`, `nl`, `head` or `tail` whose single operand is that file **NEW**.
  - Reads are keyed by repository-relative path, so a file read in the main checkout stays read in its worktree **NEW**.
  - A refused `Edit` returns the region around `old_string`, which then counts as read **NEW**.
  - New files and scratch files (in the agents' temp directory `<OS temp dir>/kiss-tmp/` when the work dir does not itself lie there, or under `~/.kiss/cron/work`; a `tmp/` folder of the work dir is an ordinary folder) are exempt **CHANGED** (11-18).
- **Worktree rules** (section 12): Inside a worktree:
  - A command that names the main checkout's path, other than under its `.venv`, is refused.
  - `uv run <program>` is rewritten to the main checkout's environment binary with the worktree's sources on `PYTHONPATH` when the rewrite is unambiguous (that environment has the program; no `uv` option with a value such as `--with` or `--directory`), and any other `uv run`, `uv sync`, `uv add` and `uv remove` are refused **NEW**.
- **Docker and the exec bridge** (section 21):
  - **Container execution**: The six shell and file tools run inside the container:
    - `Bash` defaults to a 30 s timeout there and has no `machine` and no detach after 120 s.
    - `background`, `on_timeout` and `bash_job` work through `ContainerJobs`, which keeps each job's log, pid and exit code under `/tmp/kiss-jobs` (a sub-agent's under `kiss-jobs/` of its private folder, deleted with it), and `until` re-runs the command inside the container **NEW**.
    - A `Bash` output over its budget is cut with no spill file under `tool-out/`, and a `run_commands_parallel` report is returned whole whatever its length.
    - `Read` returns at most `max_lines` lines from `start_line` with a count of the lines left, cut head and tail like a `Bash` output at its 8,000-character default, with no outline, and answers a directory or a missing path with `Error: File not found`.
    - The file tools have no read-before-modify guard, and an `edits` batch stops at its first failure with the earlier edits applied.
  - **Exec bridge** **NEW**: A `docker_image` of the form `exec:http://127.0.0.1:<port>` runs the same tools over an exec bridge, an HTTP endpoint that executes commands in a remote sandbox.
- **Tool schemas** **NEW**:
  - The schema a model sees is derived from a tool function's signature and docstring.
  - `@tool_schema(parameters=..., description=...)` (`src/kiss/core/tool_schema.py`, re-exported by `base_sea.py`) declares an object JSON schema instead, which is how MCP tools keep their arrays, enums and nested objects.
  - `@unlisted` (same module, **NEW** 11-12) registers a tool the agent runs when called and no model request lists.

### Tool profiles

Tool profiles (`TOOL_PROFILES` and `TOOL_GROUPS` in `sorcar_agent.py`):

| Profile | Tools kept | Used by |
| --- | --- | --- |
| `full` | everything above | default |
| `review` **CHANGED** | `shell` + `browser` groups, `memory_search`, `memory_pull`, `memory_read`, `memory_list`, `decide`, `summary`, `talk` (26 tools); aliases `readonly`, `read_only`, `read-only` | read-only reviewer sub-agents (section 10): named by the caller, or inferred for a sub-agent in a reviewer's sub-tree whose task asks for no change (`KISS_TOOL_PROFILES=0` turns the inference off) |
| `assistant` | `shell` + `user` + `decide` + `control` groups (9 tools) | conversational runs without file edits; one of the three profiles (`shell`, `assistant`, `review`) that `/rsi7d` proposes when a full-toolset SEA used fewer tools |
| `none` **NEW** | nothing built in (`finish` plus what the SEA's `tools()` hook returns) | `/ask`, `/install`, `/machine`, `/good_morning`, `/memory_consolidate`, `/memory_quiz`, `/recommendations`, `/sync_history_and_memory`, `/timeline` (all locked) |
| `bash` | `Bash` | `/sh` (locked), `/remember`, `/forget`, `/task_update` |
| `shell` | `Bash`, `bash_job`, `Read`, `run_commands_parallel` | `/skillopt` |
| `edit` **NEW** | `Edit`, `Write`, `create_worktree`, `commit_and_merge`, `checkpoint`, `rollback` | `/sync` runs on `shell+edit+agents` (locked) |
| `browser` **NEW** | the fifteen browser tools | |
| `memory` **NEW** | the seven `memory_*` tools | |
| `agents` **NEW** | `run_agent`, `agent_job`, `run_parallel`, `number_of_cores` | |
| `mcp` **NEW** | `connect_mcp_server`, `finish_mcp_server_connect` and, with them, the configured MCP servers' tools | |
| `skills` **NEW** | `skill` | |
| `user` **NEW** | `ask_user_question`, `talk` | |
| `decide` **NEW** | `decide` | |
| `control` **NEW** | `summary`, `set_model` | |

- **Profile composition** **NEW**:
  - A profile name may join any number of the fifteen keys with `+` (`tool_profile="shell+edit+memory"`), and `resolve_tool_profile` keeps the union of their tools.
  - `full` in a composite means everything, and an unknown key is refused with the list of keys and the closest spelling.
  - The name is accepted everywhere a profile is named:
    - `sorcar.run(tool_profile=...)`.
    - The `tool_profile` argument of `run_agent` and `run_parallel`.
    - The `tool_profile` key of a SEA's `settings()`, where `"locked": ["tool_profile"]` makes a call that asks for another profile fail instead of widening it.
- **Per-run intersections and prompt selection**:
  - Per-run switches still intersect with the profile:
    - "Use web tools" off removes the browser.
    - Docker removes the checkpoint, worktree and memory tools.
    - A sub-agent has no `checkpoint` or `rollback`.
    - Serial mode removes `run_parallel` and `number_of_cores`.
    - A missing skill, MCP server or decisions model removes `skill`, the server tools or `decide`.
  - `finish` is always added.
  - A restricted run gets `RESTRICTED_PROFILE_NOTE` in its system prompt listing the tools it has.
  - A run whose profile holds none of the `edit` tools uses `SYSTEM_LITE.md` without being classified (section 11).

## 8. Models, routing and cost accounting

- **Codebase layout and related sections**:
  - `src/kiss/core/models/` holds the catalogue (`MODEL_INFO.json`, 12,463 lines), its loader, router and price calculator (`model_info.py`, 3,122 lines) and one adapter per transport.
  - `src/kiss/scripts/update_models.py` (3,596 lines) maintains the catalogue and regenerates [MODELS.md](MODELS.md), the per-provider list.
  - The SEA contract behind the picker routers is section 9; the picker and Settings controls are section 16.

### Catalogue and custom models

- **Catalogue**:
  - 988 entries **CHANGED** (11-19; 706 on 09-30, 1,115 before the 127 entries of the Together AI, Z.AI and Moonshot routes were removed):
    - 937 generation, 803 with function calling, 6 embedding and 37 typed-decision (`dec`) entries; 7 OpenRouter entries and 1 Gemini entry have every capability flag off.
    - 529 are base models and 459 are effort aliases of one (`alias_of`).
  - The prefix of a name selects its provider; a name of none of them (`glm-4.7`, `kimi-k3`, `Qwen/Qwen3.7-Max`) is an unknown model **CHANGED** (11-19):

    | Provider | Name prefix | Credential | Entries | Base | Aliases |
    |---|---|---|---:|---:|---:|
    | OpenRouter | `openrouter/` | `OPENROUTER_API_KEY` | 514 | 373 | 141 |
    | Amazon Bedrock | `bedrock/` | AWS credentials | 144 | 32 | 112 |
    | OpenAI | `gpt`, `o1`, `o3`, `o4`, `text-embedding`, `codex`, `computer-use` | `OPENAI_API_KEY` | 99 | 51 | 48 |
    | Anthropic | `claude-` | `ANTHROPIC_API_KEY` | 82 | 18 | 64 |
    | Claude Code CLI | `cc/` | the `claude` executable | 79 | 18 | 61 |
    | Gemini | `gemini-` | `GEMINI_API_KEY` | 57 | 24 | 33 |
    | Codex CLI | `codex/` | the `codex` executable | 13 | 13 | 0 |

  - **Families**:
    - **OpenAI**: `gpt-*` 94, `o3` 2, `text-embedding-*` 3.
    - **Anthropic**: Opus 37, Sonnet 19, Fable 18, Haiku 8.
    - **OpenRouter**: 74 vendor segments (`openai` 97, `anthropic` 78, `qwen` 40, `google` 34), 40 rolling `openrouter/~vendor/*` entries and all 37 `dec` entries; Z.AI's and Moonshot's models are reachable only here (`openrouter/z-ai/*`, `openrouter/moonshotai/*`).
    - **Bedrock**: 17 `us.` and 15 `global.` Claude inference-profile ids.
    - **Claude Code CLI**: `cc/opus`, `cc/sonnet`, `cc/haiku` and 15 `cc/claude-*` names.
- **Entry fields and loading**:
  - **Required fields**: `context_length`, `input_price_per_1M` and `output_price_per_1M` are required and must be finite, non-negative numbers.
  - **Flags**: `gen`, `fc`, `emb` and `dec`.
  - **Optional fields**: cache-read and cache-write prices, audio input and output prices, `image_output_price_per_1M`, `prompt_tiers` **NEW**, `thinking`, `alias_of`, `fallback`, `use_responses_api`, `extended_thinking` and `adaptive_thinking`.
  - **Context window cap**: The updater stores 500,000 for any context window of 1,000,000 tokens or more (587 entries).
  - **Loading precedence** (read once at import):
    1. `KISS_MODEL_INFO_PATH`, when set, decides on an installed copy and in a git checkout alike: the file it names is read, or the bundled file when it names none that exists.
    2. Without it, an installed copy reads `~/.kiss/MODEL_INFO.json` (seeded by the installer) when that file exists, and every other run reads the bundled file.
    3. An unreadable or invalid override or user copy falls back to the bundled one.
- **Custom models**:
  - `~/.kiss/MY_MODELS.json` (seeded with a documentation block and an example) overrides or adds entries:
    - An entry with an `endpoint`, and optionally `api_key` and `headers`, is an OpenAI-compatible model served from that endpoint.
    - A new entry whose name resolves in the catalogue copies that entry's prices and limits; otherwise it gets 128,000 tokens at $0.
    - A malformed entry is skipped with a warning.
  - **Live updates from Settings**:
    - Settings writes the file under a file lock (`saveMyModel`, `deleteMyModel`).
    - A name added there is selectable and priced without a daemon restart **CHANGED** (11-03) and an endpoint, key or header edit applies from the next run.
    - Every other field of a name the daemon loaded at start (a bundled name, or a custom entry present then), prices and limits included, keeps its loaded value until the restart.
  - **Config endpoint**: One more endpoint can be set in `config.json` (`custom_endpoint`, `custom_api_key`, `custom_headers`) and is listed as `custom/<last path segment>`.

### Provider adapters and CLI-backed models

- **OpenAI-compatible adapter**:
  - OpenRouter and OpenAI share `OpenAICompatibleModel` (Chat Completions) and `OpenAICompatibleModel2` (the Responses API); so does every custom endpoint **CHANGED** (11-19: the Together AI, Z.AI and Moonshot routes were removed).
  - **Transport selection**:
    - A tool-bearing request to `api.openai.com` goes to `/v1/responses`, because Chat Completions there rejects `tools` together with `reasoning_effort`; only the model config's `use_responses_api` forces either transport.
    - The catalogue flag of the same name (true on 533 entries) puts every generation request of a model on the Responses API, but only at the default endpoint of the vendor the name routes to; at any other endpoint (a custom gateway, another vendor's) it is ignored and the Chat Completions adapter is used, except that an endpoint model in `MY_MODELS.json` whose name none of the OpenAI-compatible vendors routes (a `claude-` or `gemini-` name included) follows its own entry's flag.
  - **Vendor-specific rules**:
    - At an endpoint that neither vendor owns (a custom gateway), the adapter keeps the effort and learns from the endpoint's reply whether tools accept it.
    - OpenRouter Claude routes, the rolling `~anthropic` aliases included (10-18), get a top-level `cache_control` so prompt caching applies.
    - A harbor-style `openai/`, `anthropic/` or `google/` prefix on a directly routed name is stripped.
- **Anthropic adapter**:
  - `AnthropicModel` sends an effort level as `output_config.effort` **NEW** (10-07; `low`, `medium`, `high`, `xhigh`, `max`), taken in precedence order from:
    1. The model config's `reasoning_effort`.
    2. Its `output_config.effort`.
    3. The catalogue `thinking` of the alias or base entry.
  - `ANTHROPIC_WORKSPACE_ID` serves identity-linked keys.
  - Anthropic has no embeddings API.
- **Amazon Bedrock** **NEW** (10-16):
  - `bedrock/<id>` runs a Claude model through the `AnthropicBedrock` client in `AWS_REGION` (default `us-east-1`); the id is Bedrock's own model or inference-profile id, or its ARN, with an optional effort suffix.
  - **Credentials** are:
    - A bearer token (`AWS_BEARER_TOKEN_BEDROCK`),
    - An access-key pair with an optional session token, or
    - An ambient boto3 source that can be detected without a network call (a profile in `~/.aws`, a web-identity token, ECS container credentials; EC2 instance metadata is not detected).
  - **Pricing**:
    - An id is priced as, in fallback order:
      1. Its `claude-*` twin.
      2. That model's OpenRouter entry.
      3. A sibling profile.
    - That price is taken times 1.1 for a geographic (`us.`, `eu.`, `apac.`, `jp.`, `au.`) or in-region profile of Claude 4.5 or later; `global.` profiles and older models bill the base rate.
  - **Prompt caching**: Bedrock rejects a top-level `cache_control`, so the adapter moves a cache breakpoint to the last cacheable block each turn (10-21).
- **Gemini adapter**:
  - **Thinking levels** **NEW** (11-05):
    1. The model config's `reasoning_effort` is sent as `thinking_level` (`xhigh` and `max` become `high`) and replaces a level or budget set in its `thinking_config`.
    2. Without it, a configured `thinking_level` or `thinking_budget` is kept.
    3. Only when neither is set does the catalogue `thinking` of the entry apply (`high` for `gemini-4-argon-high`).
  - Thought summaries are on unless the config sets `include_thoughts` to false.
  - Image output tokens are split from text output and billed at the image rate **NEW** (10-24); audio input bills at the audio rate.
- **CLI-backed models** **CHANGED** (10-18):
  - `cc/*` and `codex/*` used to hand the whole task to the Claude Code or Codex CLI, which ran its own tools on the host with permission checks bypassed; both are now plain models driven turn by turn with the KISS system prompt and KISS tools.
  - **`cc/*`**:
    - Runs `claude --print --safe-mode --strict-mcp-config --tools "" --no-session-persistence`: no native tools, MCP servers, `CLAUDE.md`, skills, plugins or hooks.
    - Tool calls travel as text (the `tool_calls` protocol, and Claude's own `<invoke>` form since 11-07) and the CLI is stopped at the end of the first complete run of calls.
    - An effort alias is passed as `--effort` (11-08) and thinking text is shown (11-07).
  - **`codex/*`**:
    - Runs `codex exec --json --skip-git-repo-check` with `--ephemeral`, `--ignore-user-config` and `--sandbox read-only`, and with the shell, web search, apps, plugins, skill search, memories and hooks switched off.
    - The KISS system prompt is passed as `model_instructions_file`; reasoning summaries are shown (11-08).
  - **Codex MCP bridge** **NEW** (10-17, `codex_mcp_bridge.py`):
    - Every tool but `finish` and a tool that must run between turns (`set_model`) is served to Codex as the MCP server `kiss`.
    - `ToolBridge` listens on an ephemeral loopback port, Codex spawns the stdio proxy `python -m kiss.core.models.codex_mcp_bridge <port>`, each request carries a secret from `KISS_MCP_BRIDGE_TOKEN`, and calls run one at a time in the agent's own process.
    - Codex may call several tools in one turn; the calls are replayed into the conversation as ordinary tool messages, so a later model switch keeps a valid transcript.
  - **Billing** **NEW**:
    - **Subscription mode (default)**: Both CLIs default to the subscription (`claude_code_use_subscription`, `codex_use_subscription`); the API-key variables are hidden from the CLI, the turn costs $0 (every `cc/*` and `codex/*` entry is priced $0) and a Codex CLI logged in with an API key is refused.
    - **API-key mode**: With the switch off and an API key present, a turn is priced at the catalogue rate of the API model behind it, cache tiers included (10-19):
      - **`cc/*` turn**: Priced request by request, so long-prompt rates and Anthropic's fast-mode and US-inference modifiers apply per request; usage the stream did not attribute to a request (all of it in a multi-request turn without per-request usage) is priced from the summed counts at short-prompt rates with the turn's modifiers. For a model the catalogue lacks or a turn that used several models, the CLI's own `total_cost_usd` is billed when the result carries one.
      - **`codex/*` turn**: Priced from the turn's summed usage, without OpenAI's processing-tier multipliers and with long-prompt rates only when it made a single request; a model the catalogue cannot price stays at $0 with one warning.
- **Typed decisions and embeddings**:
  - A `dec` entry answers `noul`, `choice` and `score` questions through OpenRouter's Decisions API (`POST /api/alpha/decisions`) instead of generating text; it backs the `decide` tool (section 7) and the Jev classifier (section 11).
  - `get_embedding` serves the 6 `emb` entries (3 OpenAI, 3 Gemini).

### Effort aliases, pricing, fail-over and vision bridge

- **Effort aliases** **CHANGED**:
  - A `<base>-<level>` entry copies its base entry's prices and limits and sets `thinking`; the base entry holds the highest supported level, capped at `high`.
  - The 459 aliases are 109 `-high`, 108 `-low`, 107 `-medium`, 75 `-xhigh` and 60 `-max`; on 09-30 only OpenAI, OpenRouter, Together and Moonshot names had aliases (118).
  - **Ladders**:
    - **OpenAI**: `low`/`medium`/`high`/`xhigh`.
    - **Claude**: Up to `low` to `max` depending on the generation (10-07), mirrored onto `bedrock/`, `openrouter/anthropic/claude-*` (10-28) and `cc/claude-*` names.
    - **Gemini**: `low`/`medium`/`high` (11-05).
    - **Kimi K3** (through OpenRouter): `low`/`high`/`max`.
    - **Grok 4.3 and 4.5**: `low`/`medium`/`high` (6 aliases).
    - **GLM-5.2** (through OpenRouter): `high`/`max` (2 aliases).
    - **Grok 3 Mini**: The updater also holds a `low`/`high` ladder for Grok 3 Mini, which has no catalogue entry.
- **Prices**:
  - `calculate_cost` prices input, output, cache-read, cache-write (a default and a long lifetime: 5 minutes and 1 hour on Anthropic, 30 minutes and 12 hours on OpenAI), audio and image-output tokens separately.
  - **Cache prices**: Cache prices an entry states are kept and the others are derived per provider (for an `openrouter/` entry that states a read price only the long-lifetime write is derived; its unstated default write bills at the input price):
    - **Anthropic**: Reads at 0.1x the input price (0.05x for Opus 5.5 and Sonnet 5.5, 0.025x for Fable and Mythos 5.1 and, since 11-10, 5.2), writes at 1.25x and 1-hour writes at 2x.
    - **OpenAI** (its `openrouter/openai/` and rolling `openrouter/~openai/` routes included): Reads at 0.1x for most GPT-5 and GPT-6 names, 0.05x for the GPT-6 point releases (GPT-6.1 and, since 11-18, GPT-6.2), `gpt-astra-latest` (since 11-18), `gpt-sol-latest` and `gpt-terra-latest`, 0.25x or 0.5x for older families and the full input price for `-pro` names, with free writes before GPT-5.6 and writes at 1.25x (30-minute) and 2.5x (12-hour) for GPT-5.6, GPT-6 and the rolling aliases that track them.
    - **Gemini**: Reads at 0.1x (0.05x for the `gemini-4-*` models, direct or through OpenRouter, since 11-13).
  - **Long prompts** **CHANGED**:
    - `prompt_tiers` (10-30; 107 OpenRouter entries, imported from OpenRouter's `pricing.overrides`) replace an entry's prices for prompts strictly longer than `min_prompt_tokens`.
    - Entries without tiers use a name-based uplift on the whole request:
      - **2x input and cache and 1.5x output**:
        - Above 272,000 prompt tokens for `gpt-5.4` (not mini or nano), `gpt-5.5`, `gpt-5.6-*`, `gpt-6-*` and the GPT-6 point releases (`gpt-6.1-*` and, since 11-18, `gpt-6.2-*`).
        - Above 200,000 for `gemini-3.1-pro`, `gemini-2.5-pro` and Claude Sonnet 4.5 (10-29).
      - **5x every rate**: Above 100,000 for Claude Haiku 5.5.
  - **Modifiers** **NEW**:
    - **Anthropic**: Fast mode (`usage.speed` `fast`) bills 2x and US-only inference (`usage.inference_geo` `us`, Claude 4.6 and later) 1.1x; the two stack and neither applies on Bedrock (10-21).
    - **OpenAI**: Processing tiers are read from the response's `service_tier`: 0.5x for Flex, 2x for priority or fast with per-family exceptions (2.5x for `gpt-5.5`), 6x for ultrafast (10-08), 3x for `gpt-6.2-*` until March 1, 2027, Pacific time (11-18).
  - **Provider-reported and interrupted calls**:
    - OpenRouter calls are billed from the provider-reported `usage.cost` (BYOK upstream cost is added only when `is_byok`); a reported cost always replaces the catalogue estimate.
    - A call that fails or is stopped mid-stream is billed for the usage its adapter had received by then; with none, nothing is billed.
- **Default model and fail-over**:
  - **Default and fast model selection**:
    - With no model named, `get_default_model` takes the first configured provider in the order (`claude-opus-5-5`, `gpt-6.1-sol-medium`, `gemini-3.8-flash`, and so on):
      1. Anthropic
      2. OpenAI
      3. Gemini
      4. OpenRouter
      5. Bedrock
      6. Claude Code CLI
      7. Codex CLI
    - `get_fast_model` picks the cheap counterpart (`claude-sonnet-5-5`, `gpt-6-luna`).
  - **Fail-over rules** (a run fails over at most once; with no candidate there is no switch; on switch, the conversation is copied over and a warning names both models):
    - **Non-retryable provider error**: Switches to:
      - The entry's declared `fallback` (12 entries: Fable 5 falls back to Opus 4.8), or
      - With an OpenRouter key, the model's OpenRouter twin, at the same effort level when that alias exists there and on the base twin otherwise.
    - **Safety refusal, content-filter stop, output-cap cut-off or repeated empty turn**: Prefers a model of another family **NEW** (11-01, `model_families.py`), trying in order:
      1. The declared `fallback` when it is of another family.
      2. The first configured of 10 second-opinion models that is of another family (the list is headed by `gpt-6-astra`, `claude-opus-5-5` and `gemini-3.1-pro-preview`).
      3. The same-family `fallback` or twin.
    - The same family table picks reviewer sub-agents of another family (section 10).
- **Vision bridge** **NEW** (11-01, `vision_bridge.py`):
  - For a model that cannot read images, each image or PDF a tool returns is described by one call to the first configured of five vision models (`claude-sonnet-5-5`, `gpt-6-luna`, `gemini-3.5-flash-lite` and two OpenRouter twins).
  - The description replaces the attachment, and the call is charged to the task under the vision model's name.

### Credentials, model picker and picker routers

- **Credentials and Settings**:
  - Keys live in `~/.kiss/api_keys.env` (one `export KEY=value` line each, mode 0600), loaded at daemon start and applied to the running process when saved, so a newly configured provider's models are listed at once; an empty value deletes the key.
  - **Settings subpanels** (five collapsible subpanels):
    - **API Keys** **CHANGED** (11-19): Anthropic, Anthropic workspace id, OpenAI, Gemini, OpenRouter.
    - **Custom Models**.
    - **Bedrock** **NEW**: Bearer token, access key id, secret access key, session token, region.
    - **Claude Code** **NEW**: Login state from `claude auth status`, the subscription switch, a `claude setup-token` token.
    - **Codex** **NEW**: Login state from `codex login status`, the subscription switch, "Sign in with ChatGPT" device-code login, an access token.
  - With a registered machine selected, the Settings sheet reads and writes that machine's configuration, custom models and CLI login state (section 15).
- **Model picker**:
  - The daemon lists the function-calling models whose provider is configured:
    - Grouped by vendor (Anthropic with `cc/`, Bedrock, OpenAI with `codex/`, Gemini, OpenRouter) and sorted by descending price within a group.
    - Each row displays its input and output price and its use count (table `model_usage`).
    - Routers head the list, followed by custom models.
  - A pick sets the tab's model and the daemon default (`last_model` in `config.json`).
  - With a registered machine selected, the picker shows that machine's list **NEW** (10-30).
- **Picker routers**:
  - Five SEAs register as models (`register_as_model`); a picked router is the first-applied layer of every run of its tab, and the tab runs on the router's starting model:

    | Router | Starts on | Main model or tiers | Reviewer | |
    |---|---|---|---|---|
    | `bestrouter` | `claude-opus-5-5-medium` | `claude-fable-5-2-max` | `gpt-6-astra-xhigh` | **CHANGED** |
    | `orbestrouter` | `openrouter/anthropic/claude-opus-5.5-medium` | `openrouter/anthropic/claude-fable-5.2-max` | `openrouter/openai/gpt-6-astra-xhigh` | **NEW** |
    | `ccxbestrouter` | `cc/claude-opus-5-5-medium` | `cc/claude-fable-5-2-max` | `codex/gpt-6-astra-xhigh` | **NEW** (11-10) |
    | `autorouter` | `claude-opus-5-5-medium` | 10 small, 11 medium, 6 frontier candidates | chosen by tier | **CHANGED** |
    | `orautorouter` | `openrouter/anthropic/claude-opus-5.5-medium` | 6 small, 8 medium, 3 frontier, all `openrouter/` | chosen by tier | **NEW** |

  - **The three best-routers**: Share one prompt:
    - **Model roles**:
      - The starting model answers simple questions and queries.
      - The main model does every other task.
      - The reviewer checks the main model's work read-only through a non-blocking `run_agent(tool_profile="review")` on at most 75% of the task budget; a second round gets the first round's findings and the diff since then.
    - **Sub-agent dispatch rules** **NEW** (11-11): A task that mostly dispatches sub-agents may dispatch, wait and log on the starting model, but the prompt has it call `set_model` to the main model before it triages a failed test run, acts on a sub-agent result that reports a failure or a warning, or writes the final summary.
    - **History and Codex reviewer sizing**:
      - On 09-30 `bestrouter` ran everything on `claude-fable-5-1` and reviewed through `run_parallel`; the starting model arrived 11-03, Fable 5.2 on 11-09 and the reviewers' `-xhigh` suffix on 11-10 (`9878ef72f1`, later than this inventory's HEAD).
      - `codex/gpt-6-astra-xhigh` is not a catalogue entry: the Codex adapter runs `codex/gpt-6-astra` with `model_reasoning_effort="xhigh"`, and the name is sized and priced as that entry.
  - **The two auto-routers**: Share another prompt:
    - **Workflow**:
      1. Split the task into units with a mechanical acceptance check.
      2. Classify each with `decide` (tier, and whether it is guarded work, which always goes to frontier).
      3. Take the tier's first candidate whose provider is configured with `pick_model`.
      4. Dispatch it with `run_agent(model=...)` and verify the check.
      5. On failure, escalate one tier (two after a reasoning failure, at most twice per unit).
      6. Log every dispatch and outcome to `~/.kiss/MODEL_DECISIONS.md` with `log_decision`.
    - **Prompt budget rules** (which no code enforces):
      - An estimate prices one model call and a sub-agent is budgeted at about 50 of them.
      - A pick whose estimate exceeds 25% of the remaining budget is split, or dropped one tier when the classifier allows (guarded work stays on frontier), or put to the user first.
    - **Pricing tools, evidence and history**:
      - `model_menu` and `estimate_cost` price candidates from the catalogue.
      - `observed_call_costs` reports what each model's calls cost, read from the `llm_call` events of the tasks launched in the last 7 days (argument `days`; the newest 5,000 tasks at most), so a call is dated by its task's launch, not by its own time.
      - Per-model evidence in `~/.kiss/AUTOROUTER.md` (capped at 2,500 characters, rewritten weekly by `/rsi7d`) is included in the prompt.
      - On 09-30 the orchestrator was the first runnable frontier candidate and the tiers held 8, 10 and 2 candidates; it is now fixed and frontier has six.
  - **`on_picked_as_model` hook** (15 s cap):
    - Picking a router fires its `on_picked_as_model` hook.
    - The auto-routers' hook ensures the weekly cron job that runs `/rsi7d autorouter`:
      - An enabled job is left as it is, a paused one is resumed, and a missing one is created for Saturday 01:00 Pacific with a $30 budget, $25 of it for the nested `/rsi7d` run.
      - Model fallback order when creating a missing job:
        1. `claude-fable-5-2-max`
        2. The first runnable frontier candidate
        3. The first runnable function-calling model in the picker's order
        4. The default model

### Cost accounting, model updates and cost reporting

- **Cost pipeline**:
  - Per call, the adapter reports up to nine token classes (input, output, cache read, cache write, 1-hour cache write, audio input, audio output, cached audio, image output).
  - `KISSAgent` prices them with `calculate_cost` times the response's price multiplier unless the provider reported a cost, and commits tokens and cost as one snapshot.
  - `RelentlessAgent` keeps an append-only usage ledger (`_UsageLedger`, folded every 200 records) from which budget, tokens and steps are derived:
    - A retried source counts once.
    - `reset_usage()` opens a new epoch so a previous run's late spend cannot land in the current one.
  - **Sub-task and out-of-loop attribution**:
    - Classifier, merge-agent, TTS, side-channel and late-arriving sub-agent spend fold into the owning task via `_net_totals` and `_add_late_task_usage`, which also credits finished ancestors; a sub-task stopped on timeout still charges its caller (`StoppedOnTimeoutError`).
    - Each task row stores `model_costs` **NEW** (10-23): the per-model split of its own calls plus the charges of model calls made outside the agent loop (commit message, speech synthesis, `decide`; 11-05).
    - The Spend subpanel (all-time total, daily heatmap, cost-by-model bars; `getSpendReport`) splits each top-level task's cost across the models its tree used, so each day's sum still equals its listed tasks.
    - A sub-agent that ran on another machine is represented by a `remote_subtasks` record **NEW** (section 15).
- **Live accounting**:
  - Cost, tokens, steps and remaining budget are printed in every tool result and shown in the status bar and Task Info; the daily cost-calculation audit is a cron job on the development machine.
  - **Tool-result footer**: Also gives the step's tool-output size, the cost of the step's model call, and its output tokens and stop reason **NEW** (10-25); `Budget: $spent/$max` includes sub-agents.
  - **Events and display formatting**:
    - Every agent-loop model call that reports usage sends an `llm_call` event (`model`, `step`, `duration_ms`, token counts and `cost`, plus `routine`, `effort` and `stop_reason` **NEW**) to the run's printer, and the daemon's printer persists it; a run without a printer (a quiet `KISSAgent` run) emits none.
    - A call made outside the loop leaves a `gate_stage` record with `stage` `charge` instead.
    - `usage_info` events carry the exact `cost_usd` beside a four-decimal string; the page shows two decimals, four for a positive amount below half a cent, and `<$0.0001` for one below $0.00005 (10-09).
    - The Result panel header lost its tokens and cost 10-12.
  - **Default budgets**: Defaults to $100 per task in Settings (`max_budget`) and to $200 for a run that states none.
- **Model updates**:
  - The "Update Models" button in Settings runs `python -m kiss.scripts.update_models --model-info ~/.kiss/MODEL_INFO.json` as a detached process, one at a time, logging to `~/.kiss/update_models.log`; the running daemon keeps the catalogue it loaded until it restarts.
  - **Vendor listings and pricing**:
    - The script lists models from OpenRouter (models and decisions), Gemini, Anthropic, OpenAI and, since 10-21, Bedrock, and Codex slugs from the Codex repository's `models.json`; `cc/` names are derived from the Anthropic listing.
    - Only the OpenRouter listing carries prices (with context lengths, which Gemini's gives too); the Anthropic, OpenAI and Bedrock listings give ids alone.
    - A new direct OpenAI, Anthropic or Gemini entry therefore copies the prices of its OpenRouter twin or, without one, is added at $0 with the comment `NEW: needs pricing` and bills nothing until a later run finds a price.
    - A direct entry that has a price is never repriced, so a vendor's price change reaches it only through a hand edit.
    - It never lists MiniMax models, nor OpenAI `-pro` models since 10-18.
  - **Probing and atomic write**:
    1. The script probes each new model for generation, embeddings, decisions, function calling, the Responses API and the highest effort level it accepts, except new `cc/`, `codex/` and `bedrock/` candidates, which are not probed (a CLI name is entered as a function-calling generator, a Bedrock id copies its Claude twin's flags).
    2. It then writes the alias entries and replaces the file atomically.
  - **Options**:
    - `--model-info PATH`, `--dry-run`, `--scrub-only`, `--verbose`.
    - `--skip-test`: new models are not probed; their capabilities are taken from the vendor listings.
    - `--test-existing`: re-probes the stored entries of configured providers except aliases, deprecated names and `bedrock/` ids, `cc/` and `codex/` entries included, with or without `--skip-test`.
  - **Repository sync and schedule**:
    - Run against the repository catalogue, it also rewrites the totals in `README.md` and the lists in `MODELS.md` (split out of the README 09-30).
    - On the development machine a daily cron job runs it at 06:00.
- **`cost_report`** **CHANGED** (10-25):
  - `python -m kiss.scripts.cost_report [--db PATH] [--hours 24] [--since T] [--until T] [--baseline FILE] [--json]` reads `task_history` and `events` of `~/.kiss/history.db` and prints 43 KPIs, among them:
    - Cost per model call, median and 90th-percentile cost of a task tree, repeated file reads, steps and cost by context-size bucket, the reviewer share of a tree, sub-agents used as shell wrappers, cache hit ratio and refused tool calls by reason.
  - **New options and normalised counts**:
    - `--since`, `--until` and `--baseline`, which prints each KPI beside an earlier run's, are new.
    - Three counts (refused tool calls, `summary` calls made by the model, tool outputs of 10,000 characters or more) are also given per 1,000 model calls.

## 9. Sorcar Extension Agents and slash commands

A Sorcar Extension Agent (SEA) is a Python file that defines exactly one class of its own deriving from `kiss.agents.seas.base.base_sea.BaseSea` (`base_sea.py`, 356 lines) and overrides only the methods it changes:

- **Layout and loading**:
  - The layout `<name>/<name>_sea.py` is the convention that registers a SEA as a slash command (below).
  - Any other `.py` file with such a class runs as a SEA when loaded by path (`run_agent(agent=<path>)`, `run_sea`).
- **Contract history**: The class-based contract replaced the module-level getters of the previous inventory: `base_sea.py` landed 10-05 and the base classes 10-07. **NEW**

### Methods

Every method has a pass-through default:

- `settings(self, settings)`: Returns the run's settings dict (`settings | {...}`); it is data only.
- `prompt(self, task)`:
  - Returns the prompt body; `{task_id}` in it is replaced by the calling task's id.
  - A method that changes the text to an empty one is an error.
- `system_prompt(self, system_prompt)`: Receives the assembled system prompt and returns the run's:
  - Its own text to replace it.
  - `system_prompt + more` to append.
- `tools(self, tools)`: Receives the built-in toolset plus the caller's tools and returns the run's.
- **Hooks**:
  - `tool_call_hook(self, name, args)` runs before every tool call.
  - `llm_call_hook(self, new_messages)` runs before every model call.
  - `llm_response_hook(self, reply)` runs after every model reply.
- `description(self)`:
  - Returns the one sentence `/<name> help` prints.
  - A registered command must return a non-empty one.
- `register_as_model(self)` and `on_picked_as_model(self, work_dir)`: Serve model-picker SEAs (section 8).

### Chaining and root layer

- **Chaining**:
  - A SEA extends another by Python inheritance (`class MySea(ShSea)`, or `class MySea(sea_class("sh"))`).
  - The launcher applies each method of every class of the chain base first (`sea_commands._chain` over the MRO), so methods never call `super()`:
    - `settings()` starts from `{}`.
    - `prompt()` starts from the task.
    - `system_prompt()` and `tools()` start from the assembled prompt and toolset.
  - `description()`, `register_as_model()` and `on_picked_as_model()` are ordinary inheritance: the most derived definition is the one called.
  - Each result is type-checked and a wrong type is a `SeaError` naming the SEA and the method.
- **Root layer**:
  - `BaseSea` heads every chain, and a plain chat prompt runs the bare `BaseSea`, so editing `base_sea.py` customises every run (the daemon imports it once; SEA files are re-executed on every run).
  - Its one rule of its own is the `summary` cadence (`SUMMARY_EVERY_STEPS = 10`): once a summary is due, every call but `summary` and `finish` still runs, with a reminder appended to its result.
  - The reminder replaced a refusal on 11-02.

### Verdicts, response hook and tool schemas

- **Verdicts** (`src/kiss/core/tool_verdict.py`, 219 lines): `tool_call_hook` returns a typed `Verdict`:
  - `ALLOW`: Lets the call run.
  - `allow_with_note(text)`: Lets it run and appends `text` to its result.
  - `refuse(text)`:
    - Stops the call; the model reads `REFUSED: the <tool> call did NOT run and had no effect. <text>`.
    - The first refusal in the chain wins.
  - `end_run(result, success=True)`: Ends the whole run with `result` as its outcome: the call and the remaining calls of the same turn do not run. **NEW** (10-31)
- **Response hook** **NEW** (10-31):
  - `llm_response_hook` receives an `LLMReply` (`text`, `tool_calls`, `usage` with tokens, cost and stop reason, `step`, `budget_used`, `total_tokens_used`) after every model reply and before any tool call of that reply runs.
  - It returns `ALLOW` or `end_run(...)`; a refusal is a `SeaError`.
- **Tool schemas**:
  - `@tool_schema(parameters, description="")` declares the JSON schema of a tool's parameters (an object schema, else `ValueError`); the declared schema is what every model sees, in native tool definitions and in the text catalogue of CLI-backed models alike. **NEW** (10-31)
  - `@unlisted` leaves a tool out of both while the agent still runs it when called; the deferred MCP tools use it. **NEW** (11-12)

### Settings and precedence

`sea_settings.SETTING_TYPES` defines 14 keys:

- **Per-run parameters**: Eleven are per-run parameters: `work_dir`, `model`, `chat_id`, `max_budget`, `model_config`, `use_web_tools`, `auto_classify`, `use_memory`, `compaction` (added 11-05), `tool_profile`, `docker_image`.
- **Dispatcher keys**: Three belong to the dispatcher:
  - `timeout`:
    - Seconds a `run_agent` or `run_parallel` call waits (call's value, else the setting, else 3,600).
    - On expiry the run is not stopped; the call returns its job id (section 10).
    - A `/<name>` command has no such timer.
  - `locked`:
    - Keys an explicit call argument or option may not change.
    - A different value is refused with an error, never silently overruled.
  - `hidden`:
    - The literal `True` in the class's own `settings()` keeps the SEA out of the command list and out of `run_agent` agent names; it stays loadable by path and as a base class.
    - The registry reads the literal from the source, so a computed value is ignored.
- **Precedence rule** (`sea_settings.PRECEDENCE_RULE`): One precedence rule covers every setting of a `run_agent`, `run_parallel` or `/<name>` run (highest to lowest):
  1. An explicit call argument or option.
  2. The SEA's `settings()`.
  3. What the calling task passes on (for `/<name>`, the chat panel's settings).
  4. The user's defaults.
- **Precedence qualifiers and exception**:
  - `None`, and `""` for string keys, mean "no override".
  - A relative `work_dir` is a path under the calling task's directory.
  - The rule has one exception: the global `compaction` setting, when off, turns compaction off for every run whatever a SEA or a call passes; the per-run `compaction` default is on (section 6).

### Base classes

Base classes replaced the `kind` and `channel` settings on 10-07:

- `BaseSea`:
  - **Defaults it lays under the SEA's own keys**: none.
  - **Behaviour**: An ordinary session.
- `WorkerSea`:
  - **Defaults it lays under the SEA's own keys**: `auto_classify`, `use_web_tools`, `use_memory` off.
  - **Behaviour**: A tool-bound run on the caller's tree as it is; `create_worktree` and `commit_and_merge` are dropped from its toolset.
- `ChannelSea`:
  - **Defaults it lays under the SEA's own keys**: the worker defaults plus `work_dir = ~/.kiss/channel_work`, all locked.
  - **Behaviour**: Six behaviours in `sea_settings.CHANNEL_BEHAVIOURS`:
    - Worker.
    - Scratch directory.
    - No inheritance from the calling task.
    - A channel workspace held for the run's lifetime.
    - The channel preamble.
    - Listing among the installed channels (section 19).

### Removed keys and `model_config` flags

- **Removed keys**:
  - `sea_settings.REMOVED_SETTINGS` lists 14 keys that are now a `SeaError` naming the replacement, among them `kind`, `channel` and `allow_fan_out` (removed 10-07), `extends` (replaced by inheritance 10-05), `preset`, `inherit`, and `use_worktree` and `auto_commit` (removed 10-09: the agent isolates its own edits with the worktree tools, section 12).
  - `classify_tasks` and `model_name` are refused as renamed (`auto_classify`, `model`).
- **`model_config` flags**:
  - A `model_config` replaces, as a whole, the configuration that would otherwise apply, the settings panel's custom endpoint included; since 11-03 the daemon instead adds a dict holding only framework-only keys (the 7 of `model.FRAMEWORK_ONLY_CONFIG_KEYS`, among them `plain_io`) to that endpoint's configuration, so the endpoint stays.
  - `plain_io: true` **NEW** (10-31) leaves the framework's additions out of what the model reads:
    - The system prompt is the SEA's `system_prompt()` return alone (the memory protocol stays while the run has the memory tools, and a CLI-backed model keeps its text tool catalogue).
    - Tool results carry no usage line.
  - `base_url` (with `api_key`):
    - Routes every model call of the run to an OpenAI-compatible endpoint.
    - `model` must still be a catalogued name so calls are priced.

### Headless runs

`kiss.agents.sorcar.headless.run_sea(sea, task, *, model="", work_dir="", max_budget=None, save_trajectories=True, verbose=False, printer=None)` (203 lines) runs a SEA in the calling process, without a daemon or a UI **NEW** (10-31):

- Runs through the daemon's own evaluation path, and returns a `SeaRunResult` (`result`, `success`, `error`, `budget_used`, `total_tokens_used`, `total_steps`, `trajectory_paths`).
- Its `model`, `work_dir` and `max_budget` arguments rank below the SEA's settings: each applies only when the SEA sets none.
- `run_agent` and `run_parallel` need the daemon, so a SEA meant for headless runs leaves the `agents` tool group out of its profile.
- `scripts/run_sea.py` (73 lines) is the command-line form.
- **Example**:
  - `examples/rl_episode/rl_episode_sea.py` (220 lines) uses run-ending verdicts, the response hook, a declared tool schema and `plain_io` together as a guess-the-number episode.
  - `wire_check.py` next to it checks that the model reads the system prompt byte for byte.

### Lint and generated docs

Console script `sea` (added 10-05) **NEW**:

- `sea lint [paths] [--fix] [--registered]` (`sea_lint.py`, 1,022 lines):
  - Defines 13 rule codes: `broken`, `renamed-key`, `removed-key`, `base-class`, `verdict`, `redundant-key`, `no-description`, `unknown-model`, `lock-without-value`, `hidden-not-literal`, `stale-docstring`, `home-literal`, `stale-prose`.
  - Refuses:
    - A file that does not load or breaks the contract.
    - Renamed and removed keys.
    - A `tool_call_hook` that returns a literal `None` or string instead of a `Verdict`.
    - A key that only repeats a base class's value.
    - A registered command without `description()`.
    - A `model` that is neither catalogued nor a picker SEA.
    - A `locked` key with no value.
    - A `hidden` that evaluates to `True` without being written as the literal `True`.
    - A docstring that mentions a removed getter.
    - A home path written as a literal in code.
    - Stale sentences about SEA semantics in the generated documentation pages.
  - `--fix` rewrites four of them (`renamed-key`, `base-class`, `removed-key`, `verdict`).
- `sea docs [--check]` (`sea_docs.py`, 274 lines):
  - Regenerates six tables (`precedence`, `settings`, `bases`, `channel`, `options`, `commands`) inside the blocks that the HTML comments `sea-docs: NAME` and `/sea-docs` delimit in three files:
    - `src/kiss/server/README.md`
    - The website's `sea-commands.md`
    - The website's `cli.md`
- `uv run check` runs both (section 22).

### Registry and slash commands

`src/kiss/agents/sorcar/sea_commands.py` (1,905 lines) keeps the registry:

- **What registers**:
  - A folder `<name>/` becomes `/<name>` only when it holds `<name>_sea.py`.
  - Loose `*_sea.py` files, the `base` folder and hidden SEAs do not register.
- **Sources**, highest precedence first:
  1. `third_party_agents/`.
  2. The folders listed in `~/.kiss/SEAS.md` (a later line beats an earlier one; `#` comments, `~` and environment variables are handled).
  3. The bundled `seas/`.
  4. The built-in `/cron` (`cron_agent.CronAgentSea`, section 20), added as the fourth and lowest source on 10-04.
- **Watcher**:
  - A thread rescans every 2 s and open chats receive the new command list.
  - A lookup that misses triggers an immediate rescan.
- **Running** **CHANGED**:
  - `/xxx text` at the first character of a prompt runs the SEA in the tab's own run: its settings, system prompt, tools and hooks apply to that very run, and the chat keeps the raw `/xxx text` as the display prompt.
  - The previous inventory's rewrite into a `run_agent` directive was removed on 10-04.
- **Not a command**:
  - An unknown name runs as typed.
  - A bare `/xxx` runs the SEA only when its `prompt()` turns the empty task into text (`/sync` does); otherwise it too runs as typed.
- **`help` and `check`**:
  - `/xxx help` answers with `description()`.
  - `/xxx check` answers with a diagnostic: the description, class chain, effective settings, model, tools added, methods defined and a sample prompt. **NEW** (`check`, 10-04)
  - Neither runs the SEA nor calls a model; a broken SEA answers `/xxx is broken: <error>`.
- **Picker SEAs**:
  - A model-picker SEA picked on a tab is the first-applied layer of every run of that tab whose model names it (`task_runner._picker_sea`, `sea_commands.sea_layers`).
  - A `/xxx` command typed there runs its own SEA on top:
    - One whose `system_prompt()` appends keeps the routing protocol.
    - One that replaces the prompt (`/sh`) drops it.
  - A `run_agent` child inherits the real model the picker resolved to, not the picker layer.
  - A `model` setting that names a picker is resolved to the picker's own `model` setting, without the picker's protocol.
- **Help for named commands** **NEW** (10-30):
  - A plain task the user wrote gets, after its text, a `# Slash commands the user named` block with one `` - `/xxx`: <description> `` line per command its text or the chat's earlier tasks mention (`sea_commands.named_commands_help`).
  - Paths, URLs and link targets do not count, and a SEA run or a sub-agent gets no block.
- **Loading**:
  - Every SEA file of a run is executed once by the one loader `sea_settings.execute_python_file`.
  - `sea_commands.load_sea` instantiates the one `BaseSea` subclass it defines; none or several is an error.
- **By name**:
  - `run_agent(agent=...)` resolves names through the same registry, ignoring case, spaces, hyphens and underscores.
  - An empty name or a generic one (`general`, `assistant`, `sorcar`, ...) runs the hidden `sorcar` SEA (section 10).

### Bundled SEAs

`src/kiss/agents/seas/` holds 37 folders: `base` and 36 SEAs, of which 34 are commands, 2 are hidden and 6 are model-picker entries:

- Below, "caller's" means the setting is left to the tab or the calling task.
- W marks a `WorkerSea`; all others derive from `BaseSea`.
- Timeouts apply to `run_agent` and `run_parallel` callers only.

#### `/ask` **CHANGED**

- **Purpose**:
  - Answers a question about a task from its digested trajectory in two or three sentences.
  - Side channel in a running task's tab, and the source of the Task Info panel's update (section 16).
- **Profile and own tools**: W, `none` (locked): `task_context`.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/autorouter` **CHANGED**

- **Purpose** (picker entry): Splits a task into units and runs each on the cheapest tier (small, medium, frontier) that passes its check, escalating on failure and logging to `MODEL_DECISIONS.md`.
- **Profile and own tools**: caller's, web and memory on: 5 routing tools.
- **Model**: `claude-opus-5-5-medium` (the first runnable frontier model at the previous inventory).
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/bestrouter` **CHANGED**

- **Purpose** (picker entry):
  1. Starts on a simple model.
  2. Switches to `claude-fable-5-2-max` for everything else.
  3. Has `gpt-6-astra-xhigh` review read-only on at most 75% of the budget.
- **Profile and own tools**: caller's.
- **Model**: `claude-opus-5-5-medium`.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/ccxbestrouter` **NEW**

- **Purpose** (picker entry): The bestrouter on CLI-backed models:
  - `cc/claude-fable-5-2-max` main.
  - `codex/gpt-6-astra-xhigh` reviewer.
- **Profile and own tools**: caller's.
- **Model**: `cc/claude-opus-5-5-medium`.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `coding` (hidden) **CHANGED**

- **Purpose**:
  - Unattended Sorcar inside a Docker container for benchmark trials (`ContainerHarness`, JSONL trajectories); `write_trial_sea` generates a per-trial SEA that runners load by path.
  - Its `tool_call_hook` is the only finish gate in the tree (on with the trial's `finish_gate`):
    - The first edit is refused, twice at most, until the agent has written its interpretation ledger.
    - `finish` (the SEA's own tool, with an `evidence` argument) is checked by the agent's verifier script (900 s cap) and one read-only review (240 s an attempt).
    - The first refusal buys one round whatever refused it, a later round is refused only while a script fails, and 3 rounds are the most.
  - The gate's readers are `gate_readers.py` (714 lines) and `shell_changes.py` (481 lines).
- **Profile and own tools**: W.
- **Model**: per trial.
- **Budget**: per trial.
- **Timeout**: 3,600 s.

#### `/devtools` **NEW**

- **Purpose**:
  - Drives a web app as a developer would:
    1. Act.
    2. Verify with `wait_for`, `list_console_messages`, `list_network_requests` and `evaluate_script`.
  - Replaced the bundled `browser-devtools` skill.
- **Profile and own tools**: caller's, web on.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/discover` **NEW**

- **Purpose**: The AI discovery loop (section 3):
  - One idea per round as a git commit.
  - A Pareto frontier.
  - A held-out check.
  - 100 rounds by default.
- **Profile and own tools**: `full`.
- **Model**: caller's (`bestrouter` until 11-16).
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/discover_viz` **NEW** (11-17)

- **Purpose**: An HTML animation of the Pareto frontier of a `/discover` run, built from its persisted trajectory (section 3):
  - The points appear one by one under their parents.
  - A tooltip on a point shows its `pareto.md` fields.
  - Every round says how its idea was picked and whether the point was added to the frontier or discarded.
- **Profile and own tools**: W, `shell+edit+browser`, web on: `discover_trajectory`, `render_discover_viz`.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/forget`

- **Purpose**: Removes a standing instruction from `~/.kiss/AGENTS.md`.
- **Profile and own tools**: W, `bash`: `forget_instruction`, `list_instructions`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 3,600 s.

#### `/git_extract_knowledge`

- **Purpose**:
  - Indexes every tracked file and commit of a repository into its domain memory and an FTS5 block store.
  - `update` and `ask` modes.
  - Schedules a daily refresh (section 13).
- **Profile and own tools**: W, `full`: 10 knowledge tools.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/good_morning` **NEW**

- **Purpose**:
  - Greets the user and shows the stored recommendations word for word.
  - Run by a default cron job at `0 8 * * *` in a new chat (section 20).
- **Profile and own tools**: W, `none` (locked): `stored_recommendations`.
- **Model**: caller's.
- **Budget**: $0.50.
- **Timeout**: 300 s.

#### `/install` **NEW**

- **Purpose**: Runs `./install.sh --non-interactive` over ssh, concurrently, on every machine registered with an ssh login, or on those it names, in that machine's clone of the private development repository (sections 15 and 21).
- **Profile and own tools**: W, `none` (locked): `install_on_machines`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 4,200 s.

#### `/machine` **NEW**

- **Purpose**: `add`, `remove` and `list` for the registry of remote machines, `~/.kiss/machines.json` (section 15).
- **Profile and own tools**: W, `none` (locked): `machine_add`, `machine_remove`, `machine_list`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 3,600 s.

#### `/memory_consolidate` **NEW** (11-18)

- **Purpose**: Merges near-duplicate memory pages without losing a fact.
  - A *family* is a connected group of pages of one memory whose embeddings are at cosine similarity 0.9 or more (`kiss.core.memoryfield.consolidate.families`, section 13).
  - A family is examined only when its bodies fit in one page together (8,192 bytes less 700), none of its pages was written by a merge, all of them are UTF-8, no merge of its pages was undone, and the archive does not record it as left alone with the same pages and text.
  - Each run examines up to 10 families (`--limit N`), the most similar first, with up to four model calls each (a family stops at the first check it fails):
    1. The task's model writes the page that replaces the family, or answers `SEPARATE`.
    2. The judge model writes up to 5 questions per page (20 in all) from the originals without seeing the merged page.
    3. The judge says for each question whether the merged page still gives the answer.
    4. The judge reads the originals beside the merged page and lists every fact it no longer finds. The questions only sample the facts: on real merges with one line taken out they noticed the loss in about one page of five, the side-by-side reading in more than nine of ten.
  - The judge is `--judge <model>`, else the strongest configured model of another family, else the task's model. A family passes only when the merged page fits in a page, answers every question and misses no fact. One run at a time examines a memory.
- **Writes**:
  - Nothing unless the task text holds `--apply`: the dry run reports each family with the page it would write.
  - With `--apply`, a merge copies every member into `<memory>/.archive/<time>-<page>/` with a `merge.json` record, writes the merged page under the newest member's name with `writer: consolidation`, and deletes the other members. Each write and delete happens only while the page is still the text the job read.
  - With `--apply`, a family that is left as it is because its pages are separate records, the merged page is too large, it did not answer every question or a fact was found missing is recorded in `<memory>/.archive/left-alone.json` and skipped until one of its pages changes. A family whose examination ended in a reply that could not be used is not recorded, so another run examines it again.
  - `--list` prints the recorded merges and writes nothing; `--restore <merge>` works with or without `--apply`: it writes back each page the merge deleted where no page of that name exists, and the page the merge replaced only while the merged page is still byte for byte what the merge wrote. `--memory <name>` narrows the job to one memory.
- **Profile and own tools**: W, `none` (locked): `consolidate_memory`, which takes no argument and reads the options from the user's own words of the task text (not from the run context of a scheduled run, which quotes the previous report), so neither the model nor an earlier report can switch writing on. It starts no model call once 10% or less of the task's budget is left.
- **Model**: caller's.
- **Budget**: $10.
- **Timeout**: 3,600 s.

#### `/memory_quiz` **NEW** (11-18)

- **Purpose**: Checks that what finished tasks learned can be found in memory.
  - Questions the user's own top-level tasks (no sub-agent, no `/ask`) that ran on this machine, ran no SEA and no cron job, succeeded with a result of at least 2,000 characters within the last 2 days (`--days N`) and are not yet in the log; at most 20 per run (`--limit N`). `--task <id prefix>` instead names one task that succeeded on this machine.
  - Per task, the task's model writes up to 5 questions about durable knowledge in the report (from the last 3,000 characters of the task and the first 10,000 of the report), each with the report sentence that states the answer. A question is used only when the report holds that sentence and the judge model finds its answer and fact stated in the report.
  - Memory is searched as a new task in the task's work directory would search it, and the judge model (chosen as for `/memory_consolidate`) reads the first 5 results. The outcome is a **hit** when they give the answer, a **retrieval gap** when the pages the task or its sub-tasks wrote or the next 10 results give it, and a **construction gap** when nothing does.
- **Writes**:
  - A task whose examination is complete is appended to the log `<memory root>/.quiz/log.jsonl`: one JSON line per question with its outcome, or a single line with the outcome `no questions` when its report holds nothing durable to ask about. A task in the log is not questioned again unless `--task` names it. A task whose questions could not be written or judged is not logged, so another run asks again; a run that completes no task appends nothing. One run at a time questions a memory.
  - Nothing else unless the task text holds `--write`: then the facts of each task's construction gaps go into one new page `quiz-<first 12 characters of the task id>`, in the repository's domain memory when the task ran in one and the general memory otherwise, with `writer: quiz`, no `verified` time, a `review_after` 21 days ahead and, for each fact, the passage of the report it came from. An existing page of that name is never overwritten.
- **Profile and own tools**: W, `none` (locked): `quiz_memory`, which takes no argument and reads the options as `/memory_consolidate` does. It starts no model call once 10% or less of the task's budget is left.
- **Model**: caller's.
- **Budget**: $10.
- **Timeout**: 3,600 s.

#### `/merge`

- **Purpose**:
  - Resolves git merge conflicts and stages the result without committing.
  - Also run in-process when `commit_and_merge` conflicts (section 12).
- **Profile and own tools**: W.
- **Model**: caller's.
- **Budget**: $5.
- **Timeout**: 3,600 s.

#### `/orautorouter` **NEW**

- **Purpose** (picker entry): The autorouter with every tier candidate an `openrouter/` entry, so one OpenRouter key serves the session.
- **Profile and own tools**: caller's, web and memory on: 5 routing tools.
- **Model**: `openrouter/anthropic/claude-opus-5.5-medium`.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/orbestrouter` **NEW**

- **Purpose** (picker entry): The bestrouter on OpenRouter:
  - `openrouter/anthropic/claude-fable-5.2-max` main.
  - `openrouter/openai/gpt-6-astra-xhigh` reviewer.
- **Profile and own tools**: caller's.
- **Model**: `openrouter/anthropic/claude-opus-5.5-medium`.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/recommendations` **NEW**

- **Purpose**:
  - Reads the recent tasks and the signed-in apps and stores twenty suggested next tasks in `~/.kiss/recommendations.json`.
  - Run by a default cron job at `0 */6 * * *`.
- **Profile and own tools**: W, `none` (locked): `recent_tasks`, `connected_apps`, `save_recommendations`, plus 3 read-only memory tools while the global `use_memory` setting is on (section 13).
- **Model**: caller's.
- **Budget**: $2.
- **Timeout**: 900 s.

#### `/remember`

- **Purpose**: Appends a standing instruction to `~/.kiss/AGENTS.md`.
- **Profile and own tools**: W, `bash`: `remember_instruction`, `list_instructions`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 3,600 s.

#### `/review_paper` **CHANGED**

- **Purpose**:
  - Workflow:
    1. Reviews a paper (PDF, `.tex`, `.md`, `.txt`) for a venue with a related-work search, seven 1-10 scores and word-limit and AI-slop gates.
    2. A second model, when the task names one (`gpt-5.6-sol` by default until 11-16), then checks the review read-only on at most 50% of the budget, through `set_model` when the run cannot dispatch a reviewer; with none named the review is checked on the run's own model.
  - Since 11-07:
    - The review starts from a one-sentence statement of the paper and the causal evidence for its central claim.
    - A single run with no baseline or ablation caps the rating at reject.
    - The prompt carries an ICLR 2027 calibration.
    - `check_review` adds an evidence-words line that never fails.
- **Profile and own tools**: caller's, web on: `read_paper`, `check_review`.
- **Model**: caller's (`bestrouter` from 10-26 to 11-16).
- **Budget**: caller's.
- **Timeout**: 7,200 s.

#### `/review_patent` **NEW**

- **Purpose**:
  1. Examines a patent draft or published patent as an examiner would and writes a review shaped like an Office action, gated by `check_patent_review`.
  2. A second model, when the task names one (`gpt-5.6-sol` by default until 11-16), then checks the review read-only on at most 50% of the budget, through `set_model` when the run cannot dispatch a reviewer; with none named the review is checked on the run's own model.
- **Profile and own tools**: caller's, web on: `read_patent`, `check_patent`, `check_patent_review`.
- **Model**: caller's (`bestrouter` until 11-16).
- **Budget**: caller's.
- **Timeout**: 7,200 s.

#### `/revise_and_review_paper` **CHANGED**

- **Purpose**: Loops `/write_paper` and `/review_paper` (6 rounds and 1,000-word reviews by default) until strong accept or no further improvement.
- **Profile and own tools**: `full`: `writer_task`, `reviewer_task`, `loop_status`.
- **Model**: caller's (`bestrouter` from 10-26 to 11-16).
- **Budget**: caller's.
- **Timeout**: 86,400 s.

#### `/rsi7d` **CHANGED**

- **Purpose**:
  - Mines the last 7 days of SEA runs in `history.db` for mistakes and cost sinks and hands each SEA to `/discover`, which edits the SEA's script.
  - Refreshes the autorouter's evidence and, with permission, improves `SYSTEM.md`, `AGENTS.md` and Sorcar's code.
- **Profile and own tools**: caller's, memory on, web off: 27 tools.
- **Model**: caller's.
- **Budget**: $2,000.
- **Timeout**: 3,600 s.

#### `/sh`

- **Purpose**: Runs one shell command in the tab's work dir and returns its verbatim output.
- **Profile and own tools**: W, `bash` (locked).
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/skillopt`

- **Purpose**: Optimises the text of a `SKILL.md`, a SEA prompt or a module constant against a JSON eval set and writes `<target>.proposed`.
- **Profile and own tools**: W, `shell`: `optimize`, `status`.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `sorcar` (hidden) **CHANGED**

- **Purpose**:
  - The plain Sorcar sub-agent that an empty `agent` argument of `run_agent` runs.
  - `/dummy` in the previous inventory, renamed and hidden 10-05.
- **Profile and own tools**: caller's.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/sync` **NEW**

- **Purpose**:
  - Commits the checkout's changes, merges `origin/main` (or a named branch) and pushes in one `git_sync` call.
  - With registered machines:
    1. Syncs each of them first.
    2. Syncs this checkout.
    3. Syncs each again.
- **Profile and own tools**: W, `shell+edit+agents` (locked): `git_sync`, `remote_url`.
- **Model**: caller's.
- **Budget**: $2 × (1 + 2 × machines).
- **Timeout**: 3,600 s.

#### `/sync_history_and_memory` **NEW**

- **Purpose**: Brings `history.db` and `memories/` of this machine and every SSH machine to the same state in one tool call (section 15).
- **Profile and own tools**: W, `none` (locked): `sync_history_and_memory`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 3,600 s.

#### `/task_update` **CHANGED**

- **Purpose**:
  - Reports what a task has done from its persisted transcript.
  - No longer the Task Info panel's source (10-03).
- **Profile and own tools**: W, `bash`: `task_transcript`.
- **Model**: caller's.
- **Budget**: $1.
- **Timeout**: 3,600 s.

#### `/timeline` **NEW** (11-20)

- **Purpose**: An SVG Gantt chart of a task's sub-agents, drawn from the task history and shown in the result (`/timeline <task id or its first six or more characters>`):
  - One bar per sub-agent in start order, coloured by role; the model names the roles from how the sub-agents' task texts open, and the chart is refused while a sub-agent has no role.
  - Marks the task updates and `/ask` questions, the messages sent to the running task, and every stretch of over 10 minutes that the task spent in one tool call while no sub-agent ran, with what it was waiting on.
  - A sub-agent with no recorded end (still running, or killed) is drawn pale, up to the end of the run.
  - Writes `reports/timeline-<first 8 characters of the task id>.svg` in the work dir unless the task names another path.
- **Profile and own tools**: W, `none` (locked): `timeline_agents`, `draw_timeline`.
- **Model**: caller's.
- **Budget**: $2.
- **Timeout**: 600 s.

#### `/write` **CHANGED**

- **Purpose**: Concise professional prose with the vocabulary of machine text banned.
- **Profile and own tools**: caller's.
- **Model**: caller's (`bestrouter` from 10-26 to 11-16).
- **Budget**: caller's.
- **Timeout**: 3,600 s.

#### `/write_paper` **CHANGED**

- **Purpose**:
  - Writes or revises a LaTeX paper with `check_paper` (AI-slop and consistency gates) and `build_paper` (pdflatex and bibtex).
  - A new paper gets an independent read-only review by a reviewer model, and each confirmed finding is fixed before the final rebuild.
  - Since 11-07:
    - The contribution is written down before the outline (research question, central claim, the controlled comparison that tests it).
    - `check_paper` gates concessions in the abstract (one scope sentence at most), engineering trivia in the abstract and introduction, and hedged or question headings, and lists the concessions in the body.
- **Profile and own tools**: caller's, web on: `check_paper`, `build_paper`.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 21,600 s.

#### `/write_patent` **NEW**

- **Purpose**:
  - Drafts a utility patent application in Markdown in the order of 37 CFR 1.77, with `check_patent` (claim and drafting gates) and `build_patent` (`.docx`, and `.pdf` when `pdflatex` is installed).
  - Before the final build, either:
    - Starts `/review_patent` on a second model (`gpt-5.6-sol` unless the task names another) as a non-blocking `run_agent` job, collects it with `agent_job` and addresses each substantiated finding, or
    - Reads the draft on that model itself through `set_model` when it cannot dispatch.
- **Profile and own tools**: caller's, web on: `check_patent`, `build_patent`.
- **Model**: caller's.
- **Budget**: caller's.
- **Timeout**: 21,600 s.

#### Additional SEAs and state files

- `third_party_agents/` adds 44 SEA folders in the same layout (section 19).
- `src/kiss/agents/sorcar/cron_prompt_sea.py` (81 lines) is the SEA every prompt cron job runs under (section 20); it sits outside a `<name>/` folder and is no command.
- **Files these SEAs keep under `~/.kiss`**:
  - `AGENTS.md`
  - `SEAS.md`
  - `AUTOROUTER.md`
  - `MODEL_DECISIONS.md`
  - `recommendations.json`
  - `machines.json`
  - `memories/<repo>/knowledge.sqlite3`
  - `.archive/` in each memory directory (`/memory_consolidate`'s merges and `left-alone.json`)
  - `memories/.quiz/log.jsonl` (`/memory_quiz`)
  - `channel_work/`
  - `cron/jobs.json` (the weekly rsi7d, daily knowledge, recommendations and good-morning jobs)
  - Read access to `history.db`

## 10. Sub-agents and parallelism

One dispatcher (`src/kiss/agents/sorcar/agent_dispatch.py`, 3,422 lines, with the guards of `fanout_guard.py`, 441 lines) serves the `agents` tool group (`run_agent`, `agent_job`, `run_parallel` and `number_of_cores`):

- A sub-task runs on the daemon as a nested sub-agent of the calling task, with its own task row and its own tab.
- Its cost, tokens and steps are folded into the caller's, including spend that arrives after the caller's row has closed (section 8).

### Dispatch, options, precedence and inheritance

- **`run_agent`** **CHANGED**:
  - `run_agent(task, agent="", model="", tool_profile="", max_budget="", timeout="", options="", wait="", machine="", description="")` runs one SEA (section 9) on one task; every argument is a string.
  - Returns YAML: a `ran:` line, then `success` and `summary` (of a run of several sessions, the last session's summary only).
- **Agent resolution** **CHANGED** (`resolve_agent`): three rules lead to one SEA path:
  1. Empty, or one of 10 generic labels (`general`, `agent`, `sorcar`, `kiss`, `analysis`, `analyst`, `assistant`, `default`, `llm`, `model`), is the plain sub-agent, the hidden `sorcar` SEA.
  2. A path (a `.py` suffix or a separator, relative to the calling task's work dir) is that file.
  3. Anything else is a registered slash-command name with case, spaces, `-` and `_` ignored: a bundled SEA, a `SEAS.md` folder, an installed channel or `cron`.
     - `reviewer`, `review`, `codereview` and `codereviewer` are refused with a pointer to `tool_profile="review"`.
     - Any other unknown name, `worker` included, is an error that lists the registered names, with a did-you-mean suggestion when one of them is spelled closely.
- **`options`** **NEW**: a JSON object in the SEA settings vocabulary:
  - `agent_dispatch.OPTION_TYPES` has 12 keys:
    - `work_dir` (relative to the calling task's), `chat_id`, `model_config` and `docker_image`.
    - The booleans `use_web_tools`, `use_memory`, `auto_classify` and `compaction` (`false`: the sub-task never compacts its context).
    - Four call-only keys: `inherit`, `workspace` (the account of a multi-account channel), `add_to_prompt` and `add_to_system_prompt`.
  - Refused and unknown keys:
    - `model`, `tool_profile`, `max_budget` and `timeout` are arguments and are refused inside `options`.
    - A renamed key (`classify_tasks`, `model_name`, `append_to_prompt`, `append_to_system_prompt`) is refused with its new name.
    - A removed one (`use_worktree`, `auto_commit`, `allow_fan_out`, `is_parallel`, `kind`, `channel` and others) is refused with the reason.
    - Any other unknown key is refused with the known keys and, when one is spelled closely, a did-you-mean suggestion.
  - A run setting passed as a keyword argument gets a hint that shows the `options` object to pass.
- **Precedence** **NEW** (`sea_settings.PRECEDENCE_RULE`, one rule for every dispatch):
  - **Priority order**:
    1. An explicit argument or `options` key wins.
    2. Then the SEA's `settings()`.
    3. Then what the calling task passes on.
    4. Then the user's defaults.
  - **Locked keys**: a call that passes a different value for a key the SEA lists in `locked` is refused, never silently overruled; the daemon checks the locks again when it applies the run (`sea_apply.apply_run`).
- **Inheritance** **CHANGED** (`inherit_from_parent`, the one table behind `run_agent` and `run_parallel`):
  - **Inherited from the caller**:
    - The caller's model (and its `model_config` when it runs the caller's launch model and the SEA picks none), a budget share and the chat.
    - The `base_system_prompt` and `add_to_system_prompt` text of the caller's run, and the appended prompt text.
    - The extra tools, `use_web_tools`, `use_memory` and `compaction`.
    - The Docker container or exec bridge the caller runs in, and the caller's worktrees, which it edits instead of isolating the repositories again (section 12).
  - **System prompt assembly**: the prompt that the caller's SEA returned from `system_prompt()` is not passed on; a sub-task:
    1. Starts from the inherited base or its own default (section 11).
    2. Appends the inherited text.
    3. Applies its own SEA's method.
  - **Exceptions and opt-outs**:
    - A sub-task on another machine (`machine=`) does not take the chat, the container or exec bridge, the extra tools or the worktrees, which exist only on the caller's machine.
    - A channel SEA inherits nothing and runs in `~/.kiss/channel_work` (`cron` in `~/.kiss/cron/work`); `inherit: false` opts any other SEA out.
    - A sub-task of a scheduled run keeps the run's no-questions rule and is recorded as a cron task too (section 20).
  - `allow_fan_out`, `use_worktree` and `auto_commit` are no longer inherited: the three settings were removed on 10-07 and 10-09.

### Budgets, timeouts, background jobs and clean-up

- **Budget** **CHANGED**:
  - A sub-task whose `max_budget` neither the call nor the SEA names gets the caller's remaining budget, less what the caller's other still-running jobs hold (the siblings of its own `run_parallel` call included), divided by the number of the call's tasks still without a budget plus one: half for one `run_agent`, a third each for a `run_parallel` of two.
  - A budget that the call or the SEA names is capped at the same unreserved remainder (`budget_cap`, applied by `sea_apply.cap_budget`); with nothing left the sub-task is refused.
  - A review's share is at most `max($10, task_spend / 8)` — one eighth of what the calling task and its ended sub-agents have spent so far, $10 at least (`fanout_guard.review_budget` of `agent_dispatch.task_spend`, 11-20); the cap applies on remote machines too, never raises the inherited share, and does not apply to a budget named by the call or the SEA.
- **Timeouts and background jobs** **CHANGED**:
  - `timeout` bounds the call, not the sub-task.
  - Wait fallback order:
    1. The `timeout` argument.
    2. Else the SEA's `timeout` setting.
    3. Else 3,600 s (300 s on 09-30).
  - At expiry the sub-task keeps running as a job and the call returns a notice with the job id (`agent-` plus 8 hex digits).
  - `wait="false"` returns the notice as soon as the sub-task's tab exists, at most 30 s later.
- **`agent_job`** **NEW**: `agent_job(job_id, action="tail", timeout_seconds="")` follows a job:
  - `tail` says whether it runs, or returns its result once it has finished.
  - `wait` blocks until it finishes or `timeout_seconds` pass.
  - `kill` stops it.
  - `wait_any` (11-09) waits on several ids, or on every job of the task when `job_id` is empty, and returns every finished result not yet read plus the names of the jobs still running.
  - A task sees only its own jobs.
  - On the host, a `Bash(until=)` poll and a `bash_job` wait end early once every job of the task has finished with a result unread (section 7) **NEW** (11-09).
- **End of the caller's run**:
  - `finish` and `commit_and_merge` are each refused once while jobs are still running, with their ids; a job still running when the run ends is cancelled.
  - Stop on the caller cancels the sub-task its call is waiting on, and what a stopped sub-task spent is still charged to the caller.
- **Clean-up when a sub-agent ends** **NEW** (11-11; under `kiss-tmp/` since 11-18):
  - A sub-agent gets a private folder inside the agents' temp directory, `<OS temp dir>/kiss-tmp/kiss-sub-<random>` on the host and `/tmp/kiss-tmp/kiss-sub-<8 hex>` in an attached container **CHANGED** (11-18), which is:
    - The `Scratch dir` of its Task Settings.
    - The `TMPDIR`, `TMP` and `TEMP` of every command its tools start (on Windows `TMPDIR` only: Git Bash takes `/tmp` for all of a user's shells from the `TMP`/`TEMP` of whichever started first, so a private folder there left those shells without `/tmp` once it was deleted).
    - The place of its job logs and spilled output, separate from the parent task's scratch dir.
  - When its run ends (finish, failure, Stop or the caller's kill), clean-up runs in order:
    1. Its background jobs are killed.
    2. Every process whose start environment still carries `KISS_SUBAGENT_DIR=<folder>` is killed (`processes.kill_env_marked`, which also finds a process that left its group through `setsid` or `nohup`).
    3. The folder is deleted.
  - A top-level task keeps `<OS temp dir>/kiss-tmp/<task-id>/` (`<OS temp dir>/kiss-tmp/` with no task id; `/tmp/kiss-tmp/…` in a container) and its jobs **CHANGED** (11-18; `./tmp/<task-id>/` before).
  - Not covered: a process started with a scrubbed environment or as another user, exec-bridge runs, and the folder of a sub-agent whose daemon died.

### Fan-out guards, review rules and sandboxes

- **Same-model rule** **NEW** (11-01, by model and effort since 11-11, `fanout_guard.same_model_conflict`):
  - A sub-task that would run on the caller's own model at the caller's own reasoning effort (whatever route serves it, a Bedrock id being the Claude model behind it; the effort is the `reasoning_effort` of the model configuration, else the catalog effort of the name) is refused.
  - The refusal names a model to pass that the rule lets through as the sub-task would run it (`model_families.other_worker_model`), chosen in fallback order:
    1. The strongest model of another family.
    2. Else the caller's model at another effort.
    3. Else a strong model of its own family.
  - Four cases pass:
    - A review.
    - A fan-out of two or more tasks that each name a piece of their own (paths, directories, dotted names, URLs, backticked spans; scratch paths — `tmp/`, `/tmp/`, `kiss-tmp/` and `kiss-sub-…` — and a piece named by every task of a fan-out of three or more do not count **CHANGED** (11-18)).
    - A caller whose own task asks for sub-agents (it mentions sub-agents, sub-tasks, `run_agent`, `run_parallel`, parallel agents, adversarial testing or training, a second opinion or a reviewer).
    - Another model, or another effort of the same model, named by the call or by the SEA, whatever its vendor.
  - Channels, `inherit: false` calls and runs on another machine are not checked.
- **Second-opinion reviewer** **NEW** (11-01):
  - A review whose model neither the call nor the SEA names, and that would run on the caller's family, is moved to the first of the 11 `SECOND_OPINION_MODELS` of another family that has a credential (`gpt-6-astra`, `claude-opus-5-5` and `gemini-3.1-pro-preview` lead the list).
  - The `ran:` line says so, or says that no other family is configured.
- **Inferred review profile** **CHANGED**:
  - With no `tool_profile`, a sub-task of a reviewer, and a task that reads as a review, get the read-only `review` profile (section 7) unless the task contains one of 19 change verbs such as fix, implement, write or refactor.
  - A task reads as a review when it matches the review words of `fanout_guard.py`: review, audit, critique, inspect, regression, read-only, bug-hunt, vulnerability, adversarial, or a search for bugs, defects, flaws, issues or mistakes.
  - The `ran:` line then shows `tools=review(inferred)`.
  - `KISS_TOOL_PROFILES=0` turns the inference off.
- **Sandboxed and pinned callers** **NEW** (10-30):
  - A caller that runs in a Docker container or behind an exec bridge may only start plain sub-agents in the same sandbox: `inherit: false`, `machine`, `docker_image`, `work_dir`, any other SEA, `use_web_tools: true`, `use_memory: true` and a task that begins with a registered slash command are refused.
  - A daemon started with `KISS_PIN_TASK_MODEL` (set by the Terminal-Bench harness, `src/kiss/benchmarks/terminal_bench/daemon.py`) keeps every sub-task on the caller's model: a call or a SEA that names another model or a `model_config` is refused, so is a sub-task that does not inherit from its caller (`inherit: false`, a channel), and so is `set_model`.

### Remote dispatch, configuration line and path rewrite

- **Other machines** **NEW** (10-12): `machine=<registered name or host>` runs the sub-task on that machine's daemon, and the `ran:` line ends `machine=<name>` (section 15).
- **`description`** **NEW** (10-12): a short title shown as the header of the call in the chat, on `run_agent`, `run_parallel` and `run_commands_parallel`.
- **The `ran:` line** **NEW** (`run_config.run_config_line`):
  - Shows the sub-task's effective configuration on one line, for example `sh model=gpt-5 tools=bash budget=$1.00 timeout=3600s inherited=model,chat_id,max_budget pinned=use_web_tools(True->False)`.
  - Names the SEA (`sub-agent` for the plain one, `(channel)` after a channel), the model, the tool profile, the budget, the wait, the keys taken from the caller and the values the SEA's settings replaced.
  - A SEA reached by path that is also a command gets `(also agent="name")`.
- **Path rewrite**: task text that names the caller's main repository is rewritten to the caller's worktree for an inheriting sub-task on the same daemon (`dispatch_path_rewrite`, on by default; `KISS_DISPATCH_PATH_REWRITE=0` turns it off).

### Parallel tools, sub-agent tabs and usage counts

- **`run_parallel`** **CHANGED** (10-07):
  - `run_parallel(tasks, agent="", model="", tool_profile="", max_budget="", timeout="", options="", description="")` is one `run_agent` call per task of a JSON array, started together and waited for together; the `max_workers` argument is gone.
  - The SEA is resolved and checked once and every task is planned before any starts, so an error starts nothing.
  - The result is a YAML list in input order, and a child still running at `timeout` is returned as a job notice.
  - A report over `tool_output_max_chars` (50,000) is written to `<scratch dir>/tool-out/<tag>-agents.txt` in the caller's scratch dir and replaced by a digest of one line per child (11-02; returned whole when the caller's tools run in a sandbox **CHANGED** (11-18)).
  - `tasks` is parsed leniently (raw newlines and lone backslashes are repaired) and a `$(...)` substitution is rejected.
  - `run_parallel` and `number_of_cores` are offered only to a run in parallel mode:
    - The web client always asks for it.
    - `is_parallel=False` in the Python API's `run()` (section 6) turns it off.
    - The sub-tasks of a sequential caller do not get it either.
  - The prompt asks for at most two levels of nesting; the code sets no limit.
- **`run_commands_parallel`**:
  - `run_commands_parallel(commands, max_workers=0, timeout_seconds=1800, max_output_chars=5000, description="")` runs shell commands concurrently in threads with no model in the loop (section 7).
  - On the host, a report too long for the context is written under `<scratch dir>/tool-out/` **CHANGED** (11-18) and replaced by a digest.
- **Sub-agent tabs**:
  - A sub-agent's tab has the deterministic id `{parent_tab_id}__sub_{task_id}`.
  - It is open on a client exactly while the sub-agent runs **CHANGED** (11-20): `new_tab` or `openSubagentTab` opens it and `subagentDone` closes it when the sub-agent finishes. Before, the user could also close it (`closeSubagentTab` closed it on every client and the daemon remembered the close), and unfolding a finished sub-agent's panel reopened its tab on every surface (10-30).
  - A finished sub-agent has no tab: a replay sends `subagentDone` for it, so a client that missed the finish closes the tab too, and after a reconnect the sub-agent tabs a client still holds are checked again.
  - The daemon ends the tab by itself: it drops its own state for the tab when it sends `subagentDone`, and a client sends no `closeTab` for a sub-agent tab.
  - The tab has no close button, and a `run_agent` or `run_parallel` panel neither opens nor closes tabs.
  - Sub-agents of a task on another machine are relayed the same way (section 15).
- **Recorded usage**:
  - 31,442 of the 41,127 recorded task rows are sub-agents; 7,644 of the 9,685 top-level tasks ran in parallel mode.
  - 5,609 `run_parallel`, 3,066 `run_agent` and 872 `agent_job` calls are recorded.

## 11. Task classification, tags and chat summaries

### Pre-run task classification

- **Pre-run classifier** **CHANGED** (`task_classifier.py`, 1,030 lines):
  - Before a run, one call decides a single verdict, `is_simple`: the task involves neither software development nor Internet search.
  - A simple task runs on `SYSTEM_LITE.md` (53 lines) instead of `SYSTEM.md` (209 lines); a failed or disabled classification means the full prompt.
  - The need for a worktree is no longer classified (`is_development` removed 10-09): the agent calls `create_worktree` itself (section 12).
  - A run that cannot edit files (no built-in toolset, or a profile without the edit tools such as `review` or `shell`) gets the lite prompt without a classification (10-25).
  - The verdict is cached in `~/.kiss/task_classifier_cache.json` (2,000 entries, 7 days, keyed by criteria, model, model config and task), and the classifier's spend is folded into the run's totals.
  - The tab shows "Classifying task…" while the call runs; the "Preparing worktree…" phase was removed on 10-09.
- **Switches**:
  - `KISS_DISABLE_TASK_CLASSIFIER=1` turns the classifier off.
  - A run's `auto_classify` (a SEA setting, an `options` key of `run_agent`, an argument of the Python API) overrides the `classify_tasks` config key, which is on by default.
  - The "Classify tasks before running" checkbox was removed from Settings on 10-15; a value stored in `config.json` still counts.
- **LLM classifier**:
  - Makes one non-agentic call on the run's model, whose prompt asks for a bare JSON verdict.
  - **Provider attempts**:
    - For Gemini and the OpenAI-compatible providers (OpenAI, OpenRouter) the call also enforces the verdict's JSON schema, and one prompt-only attempt follows when it yields no verdict.
    - Every other provider, Anthropic included, gets a single prompt-only attempt.
  - **Limits and timeouts**:
    - The task is cut to 20,000 characters, and each attempt is configured with an output cap of 1,000 tokens (`max_tokens`), a `stream_stall_timeout` of 60 s and a nominal budget of $1, which is checked before the single request and cannot stop it.
    - The `cc/` and `codex/` CLI backends apply neither the cap nor that timeout: their CLIs get no output limit, and a turn is abandoned after `timeout` seconds of silence (a `model_config` key, 300 by default).
- **Jev (decisions model)** **CHANGED**:
  - When `decisions_tool_available()` holds, the classifier asks `openrouter/~typesafe/jev-latest` instead; the three conditions are:
    - The `classify_with_decisions` config key (on by default since 10-03).
    - An `OPENROUTER_API_KEY`.
    - A catalogue entry marked `is_decisions_supported`.
  - Jev answers one typed `choice` question with five kinds (`development`, `git_only`, `internet`, `simple` and `ambiguous`):
    - `git_only` and `simple` count as simple.
    - A choice with a probability under 0.6 counts as `ambiguous`.
  - The call has a 5 s timeout; when it fails, the LLM classifier answers and both spends are charged.
  - The same three conditions offer the agent the `decide(state, questions)` tool (section 7).
  - The "Use Jev (decisions model)" checkbox was removed from Settings on 10-15.

### Tags, secret scanning and history columns

- **Tags** **CHANGED** (`task_metadata.py`, 644 lines):
  - When a task finishes, regexes (no model call) write up to 3 comma-separated tags to `task_history.tags` (`MAX_TAGS`; 6 until 10-12).
  - The vocabulary has 22 tags (`ALL_TAGS`), ordered as:
    1. `work` or `personal` first (`work` by default).
    2. Then `secret`, `chore` and `question`.
    3. 15 activity tags (`coding`, `testing`, `debugging`, `review`, `research`, `paper`, `writing`, `docs`, `data`, `devops`, `messaging`, `browsing`, `shopping`, `finance`, `scheduling`).
    4. `subagent` and `failed`.
  - The scope tag, `secret` and `failed` are never dropped by the cap; the others fill the remaining slots in that order.
  - The Chats panel's Tag filter, applied in SQL, offers the vocabulary without `subagent`; a task row no longer lists its tags **CHANGED** (11-12; section 16).
  - The Task Info panel's Tags row shows a finished task's tags and edits them (`getTaskTags`, `setTaskTags`, event `taskTags`) **NEW** (11-21): a set the user stored is marked `tags_edited` and no derived write (task end, journal recovery, backfill) replaces it; the tags of a running task cannot be edited.
- **Secret scanner** **NEW** (10-12, `has_secret_value`):
  - A task was already tagged `secret` when its text names a secret (a password, API key, credential, private key, credit card, `.env` and similar); it is now also tagged when its trajectory holds a credential-like value.
  - The trajectory is the task text, the result and every persisted event, tool arguments and outputs included.
  - Matched values:
    - A private-key block, an `sk-` key, a GitHub, AWS, Slack or Google token, a JWT, a `Bearer` token, a `scheme://user:password@` URL, and a password, secret, token or API-key assignment.
    - Placeholders such as `<token>`, `${TOKEN}`, `changeme` and `****` do not count.
  - The tag is a label and a filter; nothing else reads it.
- **SEA column**:
  - `task_history.sea` holds the file stem of the SEA that ran the task (`bestrouter_sea`, `review_paper_sea`, `cron_agent`, `slack_sea`, ...; empty for a plain run), taken from the run's SEA path and back-filled for sub-agents from their parent's `run_agent` call.
  - The Chats panel's SEA filter offers `All`, `None` and one entry per script seen; a task row no longer names its SEA **CHANGED** (11-12).
- **`task_history` columns** **CHANGED**:
  - A newly created table has 26 columns; four were added since 09-30:
    - `success`: the `finish` verdict, NULL after a crash or a Stop (10-11).
    - `is_cron`: a run the scheduler started, or a sub-task of one (10-17).
    - `model_costs`: JSON of cost and tokens per model, read by the Spend panel (10-23).
    - `machine`: the short host name of the machine a synchronised row ran on, empty for this one (11-08).
  - `history_columns.py` (250 lines, standard library only, 10-17) adds missing columns to an older database, so the daemon and `sync_db` run the same code (section 15).
  - It drops none, so a table created with the 09-30 schema ends with 28 columns: `is_worktree` and `auto_commit_mode`, which left the schema on 10-09, stay in it.
  - `is_cron` is back-filled from the cron marker text or a `/cron/runs/` work dir, and `machine` from each task's first events, once per database.

### Chat summaries, rename, favourite, deletion and backfill

- **Chat summaries** **CHANGED** (`chat_summary.py`, 210 lines):
  - `chat_summaries(chat_id, summary, last_launched)` stores a title of up to 8 words per chat (target 6-8, derived from the chat's first five top-level tasks, sometimes shorter or empty).
  - History shows it as the group header when present and the first task's text otherwise (4,927 rows on the development machine, 4,764 of them with a non-empty summary and none with a title typed by the user).
  - The title is built without a model: the tasks' leading words in order, a slash command's name as words, greetings and politeness dropped, URLs reduced to their host.
  - The table has five more columns now: `last_finished` (the end instant of the chat's newest task, 0 while it runs; 10-16), `user_summary`, `is_favorite` (11-16), `is_pinned` (the chat's pin, 11-20) and `project_id` (the chat's project, 11-21).
  - The row is refreshed when a task of the chat finishes, by the backfill, and by `sync_db` after a merge.
- **Chat rename** **NEW** (10-15):
  - A title the user types is stored in `user_summary` and wins over the computed one on every recompute; an empty title brings the computed one back.
  - The command is `setChatSummary`, sent by the pencil button or a double-click on the title in the Chats panel (section 16); every surface refreshes.
- **Chat favourite** **NEW** (11-16):
  - A star the user gives a chat is stored in `is_favorite` and stays through every recompute of the row.
  - The command is `setChatFavorite`, sent by the star button of the chat's header in the Chats panel (section 16); every surface refreshes.
  - A deploy that replaces the database puts the stars back, as it does the typed titles (`carry_over_tables`).
- **Projects** **NEW** (11-21):
  - A project groups chats on this machine and gives their tasks settings of its own (`src/kiss/agents/sorcar/projects.py`; page and sidebar in section 16); `projects(id, name, description, settings, is_pinned, created_at, updated_at)` in `history.db` stores the projects and `chat_summaries.project_id` stores a chat's project (empty when in none).
  - `settings` is JSON and holds only what differs from the app's settings, so a field left alone keeps following the Settings sheet; a project can override eight `config.json` keys (`default_work_dir`, `max_budget`, `use_web_browser`, `use_memory`, `compaction`, `chrome_bridge`, `claude_code_use_subscription`, `codex_use_subscription`; `projects.py` `CONFIG_KEYS`), the `victory_sound` preference and the API keys.
  - Precedence for a run is an option the command itself carries, then the chat's project, then `config.json` (a chat on a registered remote machine is in no project); the project's keys and switches reach the run through a per-run `Config` overlay (`kiss.core.config.run_overlay`), so two tasks of two projects run at the same time, each with its own keys, masked in tool output like the app's own.
  - Five commands manage projects (`getProjects`, `saveProject`, `deleteProject`, `setProjectPinned`, `setChatProject`), with events `projects`, `chatProject`, `projectSaved` and `projectError`; `submit` and `run` accept `projectId`, and `getHistory` accepts `project` and pages that project's chats 200 at a time.
  - Deleting a project keeps its chats and takes them out of it.
  - Keys stay on the machine: the `projects` list is sent to this daemon's own pages and never broadcast to linked daemons, `scripts/sync-task-db.sh` takes the `projects` table and the chats' `project_id` out of a whole-database snapshot before it travels (section 21), and a deploy that replaces the database carries both across (`carry_over_tables`), making the file private (mode 0600) before it writes keys.
- **Chat and task delete** **NEW** (10-20):
  - `deleteTask` removes a task, its sub-tasks and their events and recomputes the chat's summary.
  - `deleteChat` removes every task of the chat and its summary row.
  - Both are refused while a task they would remove is running.
  - Every removed task id is recorded as a tombstone in `deleted_tasks`, and `deleteChat` also records the chat in `deleted_chats`, so the deletion reaches synchronised machines (section 15).
- **Backfill**:
  - `uv run python -m kiss.scripts.backfill_task_metadata [--db PATH] [--refresh]` (`src/kiss/scripts/backfill_task_metadata.py`, 72 lines) runs while the daemon runs:
    1. Adds missing columns.
    2. Then fills rows without tags and chats without a summary in batches of 500, and sub-agent rows without a SEA one parent task at a time (from the parent's `run_agent` calls).
  - `--refresh` recomputes every derived tag and every derived summary and keeps the user's titles and the tags a user edited in Task Info (`tags_edited`).

## 12. Git worktrees, merging and auto-commit

- **Core modules**:
  - `GitWorktreeOps` (`src/kiss/agents/sorcar/git_worktree.py`, 4,319 lines) is the git plumbing.
  - `SorcarAgent` (`sorcar_agent.py`) carries the two worktree tools and the merge engine.
  - `src/kiss/server/merge_flow.py` (929 lines, 2,212 before 10-09) serves the buttons.

### Worktree isolation and guards

- **Agent-driven isolation** **CHANGED** (10-09; folded into `SorcarAgent` 10-10):
  - The launcher no longer creates a worktree; every run (web app, CLI, Python API, cron job) starts in the caller's work dir, and the agent isolates its own edits with two tools of the `edit` group (section 7): `create_worktree(path="")` and `commit_and_merge()`.
  - The worktree protocol in `SYSTEM.md` and `SYSTEM_LITE.md` has it call the first once per repository before its first change to a file git does not ignore, and the second once the work is verified; a task that edits no repository file gets no worktree.
  - Removed 10-09:
    - The "Use worktree" and "Auto commit" settings.
    - The `use_worktree` and `auto_commit` SEA settings (now refused by name).
    - The classifier's `is_development` verdict (it decides only `is_simple`, section 11).
    - The "Preparing worktree…" launch phase and the automatic commit and merge after a successful task.
  - `WorktreeSorcarAgent` was removed 10-10.
- **`create_worktree`**: under the repository's locks:
  - `git worktree add -b kiss/wt-<epoch>-<8 hex>` checks a new branch out at `<repo>/.kiss-worktrees/kiss_wt-<epoch>-<8 hex>` from the checkout's current branch; the branch's git config records the branch to merge back into (`kiss-original`) and the owning process (`kiss-owner-pid`, `kiss-owner-identity`).
  - `copy_dirty_state` mirrors the checkout's uncommitted state (staged, unstaged and untracked files, deletions, renames) into the worktree and commits it as `kiss: baseline from dirty state` (`kiss-baseline`), so the task builds on what the user sees and the merge later applies only what the task added.
  - Each ignored `node_modules` directory whose parent directory exists in the worktree is symlinked in.
  - A file named `.venv` is written in the worktree and into every directory with a tracked `pyproject.toml` (where the repository un-ignores the name, a directory that ignores itself instead; only when the main checkout has a `.venv` to share) so that no `uv` (typed, inside `bash -c` or in a script) builds the worktree a Python environment of its own (`GitWorktreeOps.block_own_environment`; a typed `uv run` is rewritten to the main checkout's environment, whose interpreter the Task Settings of every agent and the first line of every pytest run name) **NEW** (11-11).
  - `.kiss-worktrees/` goes into the repository's `info/exclude`, `core.untrackedCache` is set to `true` when unset, and the work dir moves into the worktree at the same sub-directory offset (event `worktree_created`).
- **One worktree per repository** **NEW** (10-10):
  - `create_worktree("<path in another repository>")` isolates that repository as well; the work dir stays put and the agent edits the files under the returned directory.
  - A task holds at most one worktree per repository (`SorcarAgent.worktrees`), and `commit_and_merge` merges each into its own repository's branch.
- **No worktree**:
  - Outside a git repository, on a detached HEAD, in a repository without commits, on a git failure or with a dirty tracked submodule, the tool returns the reason and the task edits in place.
  - A Docker run is not offered the tools.
  - `WorkerSea` and `ChannelSea` runs (section 9) drop them: a worker works on the caller's tree, and a channel SEA works in `~/.kiss/channel_work` by default (section 19).
- **Write guard** **NEW** (`_guard_worktree`): while the run has the tools:
  - `Write` or `Edit` of a checkout path that git does not ignore is refused until the repository has a worktree, and the refusal names the `create_worktree` call to make.
  - A path of a repository whose worktree the work dir is not in is refused with the worktree path to use; an absolute checkout path of the work dir's own worktree is remapped into it.
  - Scratch files (`_is_scratch_path`), ignored paths (`git check-ignore`), paths inside a worktree and repositories whose worktree setup failed pass **CHANGED** (11-18).
  - `rollback` is refused while the work dir is in a checkout (checkpoints, section 7).
  - The first `finish` with an unmerged worktree is refused once.
  - `Bash` is outside this guard, except for a shell rewrite that it applies as a `Write` or an `Edit` (section 7), which is judged like one: any other `Bash` write goes straight into the checkout, and the worktree rules of `Bash` (section 7) apply only inside a worktree.
- **Sub-agents**:
  - A sub-agent inherits the worktrees of its calling tasks (`inherit_worktrees`, `isolated_repos`, wire field `isolatedRepos`), edits them in place and never creates a second worktree for the same repository; the calling task commits and merges.
  - Checkout paths in a sub-agent's task text are rewritten to the worktree, except those under the checkout's `.venv`, the one environment the worktree runs on (section 10).

### Merging, conflicts and unmerged worktrees

- **`commit_and_merge`**:
  - For each worktree:
    1. Stage and commit everything with a generated message, and anything that arrives during the commit in a second one (`kiss: commit late-arriving changes`).
    2. Remove the worktree and move the work dir back.
    3. Under the repository's locks, stash the checkout's own uncommitted changes (`git stash push --include-untracked`), check out the original branch when the checkout sits on another, squash-merge, pop the stash and delete the task branch.
  - The first call while a `run_agent` job is still running is refused once (section 6).
  - When a step fails after the directory was removed, the branch is checked out again at the same path (`GitWorktreeOps.reattach`) so the agent can fix the problem and call the tool again.
  - Each outcome is broadcast as `worktree_result`.
- **Squash merge**:
  - The work lands as one commit on the original branch, with the task branch's last commit message and a `User prompt:` and a `Result:` block ensured for each of the two values the agent holds (an HTML result is converted to Markdown):
    - `commit_and_merge` runs before the task has a result, so its commit carries the prompt alone.
    - A merge from the bar after the task carries both.
    - A `SorcarAgent` outside a chat (`run_sea`, section 9) holds neither.
  - Merge mechanism:
    - A worktree made from a clean checkout is merged with `git merge --squash`.
    - One with a baseline is merged as a synthetic commit (`git commit-tree <branch>^{tree} -p <baseline>`) applied by one `git cherry-pick --no-commit`, with `-X theirs` only while the original branch still sits at the baseline's parent.
  - Outcomes (`MergeResult`):
    - `SUCCESS`, `CONFLICT` and `CHECKOUT_FAILED`.
    - `MERGE_FAILED`: the squash commit failed, for example on a rejecting hook; the tree is reset and the branch kept.
    - `STASH_FAILED`: the checkout's changes could not be stashed; nothing is touched and the blocking files are listed.
  - The stash is popped after `SUCCESS` and `CHECKOUT_FAILED` and stays in `git stash` after a failed merge, whose message gives the manual steps.
  - A merge that would overwrite a file git ignores in the checkout (a never-committed `.env`) is refused as a conflict **NEW** (11-04).
  - A `kiss-scratch` merge driver keeps the incoming version of `PROGRESS.md` and `src/kiss/INJECTIONS.md`.
- **Conflict resolution** (`merge_conflict_resolver.py`, 246 lines; in `src/kiss/server/` until 10-09):
  - On `CONFLICT`, `commit_and_merge` and the bar's merge button re-apply the branch with the conflicts left in the tree and run the bundled `/merge` SEA (`src/kiss/agents/seas/merge/merge_sea.py`, a `WorkerSea` with a $5 budget and its own system prompt) as a sub-agent in the checkout, on the parent's model unless the SEA names one, its cost attributed to the parent task.
  - The merge is committed only when `HEAD` has not moved, no path is unmerged, no `<<<<<<<` or `>>>>>>>` line remains in the files that conflicted and something is staged; otherwise the tree is restored and the result gives the manual commands.
  - `/merge <task>` typed into an idle tab runs the same SEA on the tab's work dir.
- **Unmerged worktrees**:
  - A task that ends, finished, failed or stopped, with a worktree still pending marks its branch preserved, and the daemon broadcasts `worktree_done` when the worktree differs from its baseline commit or, without one, from its merge base with the original branch (committed, staged, unstaged or untracked changes; a worktree git cannot read counts as changed); a verified clean one is discarded.
  - The chat shows a bar with three buttons, sent as `worktreeAction`:
    - "Auto-commit and merge" (`merge`): the same commit, squash and conflict resolver, with the merge agent's cost added to the task's row.
    - "Do nothing" (`nothing`): branch, directory and uncommitted changes stay as they are.
    - "Discard" (`discard`, after a confirmation): worktree and branch are deleted.
  - The bar comes back when the session is reloaded, acts only while no task runs on the tab, and for a tab bound to a machine is forwarded to it (section 15).
  - A new task on the tab, a new chat or closing the tab retires a pending worktree: its changes are committed to its `kiss/wt-*` branch, the directory is removed and the branch stays unmerged, named in a warning.
  - After a daemon restart no bar is shown; the branch and directory remain for a manual merge.
  - The CLI names the unmerged branch and directory on stderr; a library caller uses `agent.merge()`, `agent.discard()` or `agent.leave_as_is()`.
- **Git-ignored files**:
  - A worktree's git-ignored files (build output, caches, a `.env`) are in no commit, and nothing copies them into the checkout.
  - A merge, a retirement, a discard and the orphan reclaim remove them with the worktree directory.

### Commits and Source Control

- **Commit messages** (`commit_message.py`, 172 lines):
  - Makes one non-agentic call to the fast model (section 8) on the staged diff, in conventional-commit form, followed by a `User prompt:` and a `Result:` block under the same condition as on the squash commit; without a model or on a failed call the subject is `kiss: auto-commit agent work`.
  - The diff is capped at 200,000 bytes: a larger one is sent as a truncation note with the `--shortstat` totals, the head of the `--stat` list and the head of the patch.
  - Emptiness is tested with `git diff --cached --quiet`, and the call's cost goes to the task.
- **What auto-commit means now**:
  - No setting and no step after the task.
  - The name survives in the commit `commit_and_merge` makes, in the bar's "Auto-commit and merge" button and in the wire names of the Git Commit button (`autocommitAction`, `autocommit_done`).
  - Edits made outside a worktree are not committed when the task ends; the Git Commit button and `/sync` (both below) commit them on request.
- **Git Commit** (a button of the strip under the right sidebar **CHANGED** 11-20; before, an entry of the chat's footer menu):
  - Stages everything (`git add -A`) in the folder the chat header shows for the tab, generates a message from the diff alone and commits.
  - A tree that changed while the message was generated is not committed (the reply asks to press the button again), and one commit runs per tab at a time.
  - When the header names a registered machine, the command is forwarded and that machine commits its own folder **NEW** (10-18; 900 s timeout, section 15).
- **Create pull request** **NEW** (11-20; the button next to Git Commit; `src/kiss/server/pr_flow.py`, wire names `createPrAction` and `pr_done`):
  - Pushes the commits of the checked-out branch that the remote's default branch lacks, from the folder the chat header shows for the tab, and opens a pull request for them. Uncommitted changes are never part of it, and the reply says so when there are any.
  - The remote is the branch's own, else `origin`, else the only one; the default branch is the remote's `HEAD`, else `main`, else `master`.
  - Pressed on the default branch itself, it pushes the commits to a new branch `pr/<words of the title>-<short sha>` instead.
  - The title and body are the commit message when one commit is ahead, and are generated from the diff otherwise.
  - For a repository on github.com with a logged-in `gh`, the open pull request of the branch is reported as updated or a new one is created; without `gh`, or when it fails, the reply links to GitHub's page that opens the pull request with title and body filled in. For any other host the reply reports the push alone.
  - Nothing is pushed from a folder without a repository, a detached `HEAD`, a repository without a remote or default branch, a branch with nothing new, or a checkout that moved while the description was generated.
  - One request runs per tab at a time, under the repository lock, and a registered machine named in the header pushes its own folder.
- **Source Control data** (`src/kiss/server/explorer.py`, 1,081 lines; the view is in section 16):
  - `git_status` lists the changes of every worktree of the repository, main checkout first, and reads them with `git --no-optional-locks status` so the view's poll never takes `index.lock` from a running task **NEW** (10-30).
  - The view's badge counts only the worktree the tab's task runs in **NEW** (10-20).
  - A file under Changes opens as a side-by-side diff of `HEAD` against that worktree's file **CHANGED** (11-07), as a file of a commit in the graph does against the commit's parent (`git_file_diff`).
  - The graph's context menu is the only writer (`git_action`: detached checkout, new branch, tag, cherry-pick, under the repository lock).
  - For a tab bound to a machine, `gitStatus`, `gitLog`, `gitShow` and `gitAction` are answered by that machine **NEW** (10-14, section 15).

### Concurrency, recovery, sync and history

- **Locks and timeouts**:
  - Worktree setup (with its reclaim and sweep), re-attachment after a failed merge, the merge into the checkout (from the stash to the branch deletion) and a discard each take a per-repository lock in the process and then a file lock on `kiss-reclaim.lock` in the git directory (`flock`; `msvcrt.locking` on Windows), so the daemon and a CLI run cannot interleave them; a wait of more than 300 s for another process ends in an error instead of a hang **NEW** (10-19).
  - The steps of `commit_and_merge` before the merge are outside the file lock: the commit in the task's worktree takes neither lock and the removal of its directory takes only the in-process one.
  - Each git call has a 300 s timeout and ignores inherited `GIT_DIR`-style variables.
  - A tab with a merge, discard or disposal in flight refuses a new run ("Cannot run a task while a merge is in progress. Wait for it to finish first."), and a claim whose thread died is released at the next check.
- **Orphan recovery** **CHANGED**:
  - There is no pass at daemon start any more; each `create_worktree` first runs `reclaim_orphaned_worktrees` and `sweep_orphaned_state` on its repository.
  - Worktree reclaim:
    - Takes a `kiss/wt-*` worktree whose process died mid-task: it commits the changes (`kiss: reclaim orphan worktree`), squash-merges the branch into its recorded original branch and removes it, but only while the checkout is clean, not detached and on that branch; a failed merge keeps the worktree.
    - Never touches a worktree stamped by another live process (pid plus start time and executable, so a recycled pid does not count), one of a live task in the same process, or one marked preserved; every worktree left unmerged at the end of a run gets that marker (a file under `kiss-preserve/` in the git directory since 10-13).
  - State sweep:
    - Deletes `kiss/wt-*` branches that have no worktree and no commits of their own, their config sections, stale markers and unregistered `kiss_wt-*` directories that hold no `.git` entry.
    - A task that has removed its worktree directory and not yet merged keeps its branch: the sweep skips one that a live task of the same process holds or another live process stamped **NEW** (10-30).
  - Task rows have their own recovery: the `owner` column of `task_history` holds the creating process's token, and at daemon start rows of dead owners that still carry `Agent Failed Abruptly` are rewritten to `Task terminated unexpectedly (process killed)` or restored from the result journal (section 14).
- **`/sync`** (section 9):
  - Commits the checkout's uncommitted changes, fetches and merges `origin/main` (or a named ref) and pushes, forcing nothing.
  - With registered machines, it syncs each one before and after (section 15).
- **Removed 10-09**:
  - The worktree pool (`worktree_pool.py`, `prewarm_worktree_pool`, `KISS_DISABLE_WORKTREE_POOL`; the reclaim discards a spare left by an older release).
  - The main-tree admission gate (a non-worktree task blocking a merge, "A worktree merge is in progress", deferred merges).
  - The `is_worktree` and `auto_commit_mode` columns of `task_history`, which a database created earlier keeps unwritten.
- **Recorded usage**:
  - Recorded since the two tools arrived on 10-09: 580 `create_worktree` calls in 525 tasks and 675 `commit_and_merge` calls in 502 tasks.
  - The `is_worktree` and `auto_commit_mode` columns that earlier inventories counted are no longer written, and a database created today does not have them; on the development machine they stop at 3,532 worktree rows (the last on 10-09) and at 4,585 auto-commit tasks (the last on 10-10) among the 7,337 top-level tasks launched up to that one.

## 13. Persistent memory and repository knowledge

`kiss.core.memoryfield` (`pages.py`, `index.py`, `statements.py`, `tools.py`, `consolidate.py`, `evaluate.py`; 3,907 lines with `__init__.py`):

### Storage, indexing and tools

- **Pages**:
  - A flat directory of Markdown files with YAML frontmatter (`title`, `uuid`, `summary`, `created`, `updated`), names `^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`, and an 8 KB guideline (larger pages are saved with a warning and only the first 8,192 bytes are embedded).
  - A write keeps the stored `uuid` and `created`, refuses an empty body and publishes atomically; a delete writes `.tombstones/<name>` before it removes the page, so a sync cannot resurrect it.
  - Conditional write and delete **NEW** (11-18): `MemoryDir.write(..., expected=)` and `MemoryDir.delete(name, expected=)` act only while the stored page is still the text the caller read (`""` for a page that must not exist), compared under the memory's lock; otherwise they raise `PageChangedError` and change nothing. The maintenance SEAs use it so a page another task stored meanwhile is never overwritten or deleted.
  - Consolidation **NEW** (11-18): `consolidate.py` is the model-free part of `/memory_consolidate` (section 9). `families()` groups the near-duplicate pairs of a memory into connected families and marks the ones to leave alone; `commit()` copies the members into `<memory>/.archive/<time>-<page>/`, writes the merged page under the newest member's name and deletes the others, each step conditional; `restore()` writes the archived pages back with a fresh `updated` time, so they outlive the tombstones on other machines; `leave_alone()` records a family examined and not merged. The `.archive/` directory is read by no page listing, index or machine sync.
  - Writes and deletes of one memory are serialised across processes by the lock file `<memory>.lock` beside its directory (60 s wait, then the tool returns an error) **NEW** (10-18).
  - Symlinked pages, index files and tombstone directories are refused; sync and editor debris (`.DS_Store`, `desktop.ini`, `Thumbs.db`, `*~`, `*.sync-conflict-*`) is ignored.
- **Index**:
  - One SQLite file per embedder inside the memory directory (`text-embedding-3-small.sqlite3` when `OPENAI_API_KEY` is set, otherwise the offline 1,024-dimension `hashed-bow-v1.sqlite3`), schema v3 with a read-only fast path.
  - The embedded text is title, summary and body, without `uuid` or timestamps.
  - Sync and embedding updates:
    - Git-style racy-clean stat keys (size and mtime, plus ctime and inode on POSIX; 3 s window) let a sync skip unchanged pages unread (racy files and `memory_refresh` verification still hash).
    - Embeddings are batched (256 inputs / 250,000 bytes per batch through `Model.get_embeddings`).
    - Rows are written by compare-and-swap on a per-row `revision`, and a batch that finds the index cleared and rebuilt since its snapshot is skipped.
  - Search and deduplication:
    - Search is a cached numpy float32 matrix scan with exact `math.sumprod` rescoring and a 64-entry query cache; pages whose cosine similarity is not above 0 are dropped.
    - Near-duplicate detection reports pairs at cosine 0.9 or more.
- **Tools**:
  - Available tools:
    - `memory_search(query, k=5, memory)`: returns score, name, title and summary per hit.
    - `memory_pull(query, k=3, memory)`: returns the matching pages whole within a budget of 24,000 characters per call (a first page over the budget is cut to it, and a later page that does not fit ends the result; section 7).
    - `memory_read`.
    - `memory_write`: rejects bad names, and warns on oversize pages and on ephemeral names such as `round-3`.
    - `memory_list(memory, max_output_chars)` **NEW** (10-25): cut at the tool-output budget with a count of the pages not shown; the budget defaults to 8,000 characters and its ceiling to 50,000 (`KISS_TOOL_OUTPUT_DEFAULT_CHARS` and `KISS_TOOL_OUTPUT_MAX_CHARS` of section 6).
    - `memory_delete`.
    - `memory_refresh(stale_days=30, duplicate_threshold=0.9, memory)`: re-hashes every page, and lists at most 20 near-duplicate pairs and 20 stale pages per memory.
  - Each memory is synced and searched on its own worker thread; the task thread waits for them interruptibly and a cancelled sync starts no further embedding batch, so Stop returns at once **NEW** (10-30).
  - The `memory` tool group holds all seven and the `review` profile keeps the four read-only ones (section 7).

### Recall, domains and switches

- **Protocol and recall**:
  - A run with memory gets `MEMORY_PROTOCOL` in its system prompt (four rules, plus a fifth that lists the domain memories) and, appended to its task, the block `# Memory pages relevant to this task`: the five best pages for the task text, one line each with name, date, title and summary (`MemoryTools.recall`; a task over 4,000 characters is searched by its first and last 2,000; an empty or failed search adds nothing) **NEW** (10-25).
  - Rule 1 no longer demands a `memory_search` as the first turn **CHANGED** (10-25): the model reads the listed pages it needs, searches only for a sub-topic the list does not cover and, when a page updated in the last 30 days answers a factual question, re-verifies only the facts that change (prices, versions, dates) on at most three sites.
  - The other rules are as before:
    - Record durable knowledge one topic per page under about 8 KB with sources.
    - Store no secrets, credentials or raw transcripts and keep per-round notes in `<scratch dir>/PROGRESS.md` **CHANGED** (11-18; `./tmp/PROGRESS.md` before).
    - Call `memory_refresh` when results look redundant or outdated.
- **Location and domains**:
  - The general memory is `~/.kiss/memories/`, or the directory named by the `memory_dir` key of `~/.kiss/config.json`.
  - A run whose work directory is inside a git repository also gets that repository's domain memory `<memory root>/<slug of the main checkout's directory name>/` (`general` maps to `general-repo`), shared by its worktrees (`git rev-parse --git-common-dir`) and addressed as `<memory>/<page>`; searches span every memory unless `memory=` narrows them.
  - Two repositories whose directories have the same name share one domain memory.
  - On the development machine (pages created before 2026-11-10 UTC): 949 general pages, `kiss/` 1,805 pages, and 76 pages in 18 further domain memories.
- **Switches and gates**:
  - Switch precedence (highest to lowest):
    1. A per-run `use_memory` (`SorcarAgent.run`, `daemon_client.run`, the `useMemory` wire field, a SEA's `use_memory` setting).
    2. A non-empty `KISS_USE_MEMORY` (`0`, `false`, `no`, `off` disable; anything else enables).
    3. "Use persistent memory" in the settings panel (config key `use_memory`, default on).
  - SEA and sub-agent defaults:
    - Worker SEAs default to off; the `rsi7d`, `autorouter` and `orautorouter` SEAs turn it on, and `/recommendations` adds read-only `memory_search`, `memory_pull` and `memory_read` over the general memory alone while the switch is on **NEW** (10-12; section 9).
    - Sub-agents inherit the parent's override unless `options.use_memory` is given, and a sandboxed task may not start a sub-task with it on (section 10).
  - Four hard gates beat every switch **CHANGED**:
    - No basic tools.
    - A tool profile without a memory tool (since 10-26: `assistant`, `bash`, `none`, and any group combination without `memory`).
    - A Docker, attached-container or exec-bridge run (`docker_image` set).
    - A caller-supplied `system_instruction`.
  - The `cc/`/`codex/` gate was removed 10-18, so those models get the memory tools like any other.

### Repository knowledge and standing instructions

- **Repository knowledge** (`/git_extract_knowledge <path or clone URL>`, with `update <repo>` and `ask <question>` modes; section 9): two tiers in the repository's domain memory:
  - Tiers:
    - The agent writes curated pages (`overview`, `domain-glossary`, `domain-concepts`, `architecture`, `conventions`, `history`, `module-<slug>`, `faq`, `recent-changes`) and a tool writes `knowledge-lookup`.
    - The FTS5 block store `knowledge.sqlite3` holds kinds `repo, dir, file, chunk, symbol, commit, change, tag, branch, author, note` (`note` mirrors the curated pages).
  - Indexing is deterministic and incremental:
    - Blob hashes.
    - The set difference between `git rev-list --all` and the stored commits (every branch and tag; commits dropped by a rewritten history are deleted).
    - Children-first writes.
    - Chunks of at most 80 lines and 6,000 characters (a longer line is split into numbered parts).
    - Symbol patterns for 11 language families.
    - Patches in 8,000-character `change` blocks.
    - One commit's diff cut at 64 MB.
  - Binary files (a NUL byte in the first 8 KB), generated files (`.min.js`, `.min.css`, `.map`) and text files over 64 MB get a metadata-only `file` block, without `chunk` or `symbol` blocks.
  - Search is BM25 with title 4x / key 2x and history rank factors 0.6 (`change`) and 0.8 (`commit`), running an all-words pass then an any-word pass, returning at most 100 hits.
  - Ten tools and a nine-command CLI (`python -m kiss.agents.seas.git_extract_knowledge.git_extract_knowledge_sea index|status|search|read|pages|read-page|write-page|delete-page|schedule`) serve the SEA, its fan-out sub-agents and any shell; the SEA itself runs with the `full` tool profile and no memory tools.
  - Checkouts, schedule, and sync scope:
    - Clone URLs are checked out under `~/.kiss/knowledge/checkouts/`.
    - The cron job `git-knowledge daily update: <slug>` refreshes the memory at 04:00 America/Los_Angeles on the pinned `claude-fable-5-1` ($25 + $2 relay, 4 h timeout; section 20).
    - The block store stays on its machine: neither sync below copies `*.sqlite3` files.
- **Standing instructions**:
  - `~/.kiss/AGENTS.md` is not a memory page; it is appended to the system prompt of every `SorcarAgent` run except a `plain_io` one, and a caller-supplied `system_instruction` replaces the composed prompt, standing instructions included (section 6).
  - It is edited by `/remember` and `/forget` (section 9) through `agents_md.py` (196 lines):
    - One bullet per instruction under `# User instructions`.
    - Duplicates recognised regardless of case and spacing, and hand-written text kept byte for byte.
    - Updates written under `AGENTS.md.lock` with an atomic replace.
  - The file was renamed from `SORCAR.md` **CHANGED** (10-03); one left under the old name is adopted on first use.

### Cross-machine sync and evaluation

- **Memory sync on deploy**:
  - `rsorcar` step 4c (section 21) runs `scripts/sync-memory.sh user@host` (268 lines, usable on its own): pages and tombstones travel both ways as tar over ssh, without the `*.sqlite3*` files, and `merge_memory_pages.py` (505 lines, standard library only, run on the remote by its system `python3`) merges them:
    - The newest `updated` wins, and mtime breaks a tie for hand edits.
    - Same-second conflicts keep each side's copy and are named on stdout.
    - A tombstone removes an older page and yields to a newer one.
    - Replacements are atomic.
  - Clock skew between the machines is measured first and widens `--tolerance`; each side's `memory_dir` is honoured; the merge holds the memory's mutation locks **CHANGED** (11-08).
  - Step 4d syncs the autorouter's notes `AUTOROUTER.md` and `MODEL_DECISIONS.md` with `sync_kiss_notes.py` (sections 8 and 21).
- **Sync across machines** **NEW** (`/sync_history_and_memory`, 10-25; section 15):
  - The memory half of the SEA compares SHA-1 fingerprints of pages and tombstones with every SSH machine, sends only the entries that differ as a gzip'd tar, and merges them by the same `merge_memory_pages` rules under the memory's locks:
    1. First from every peer into this machine.
    2. Then from this machine into every peer.
  - The tolerance is the spread of the clocks measured at connect; writes closer together than that are conflicts, kept on both sides and named.
  - Peer home resolution order:
    1. The directory whose endpoint file names the registered daemon's port, looked for under this install's brand directory, then `~/.kiss`, then every other `~/.*` (11-04), so a white-label install and a stock one sync with each other.
    2. When no endpoint file matches (a stopped daemon removes its file) or the port is unknown, the home is the first of those two directories that holds a `history.db`, else the first that exists, else the first.
  - The helper scripts travel as text and run under the peer's system `python3`; a dry run only counts what would travel.
  - It syncs `<home>/memories` on both sides and ignores `memory_dir`.
- **Recall evaluation**:
  - `python -m kiss.core.memoryfield.evaluate` builds a memory from the most recent successful top-level tasks of `~/.kiss/history.db` (`--limit 300`, `--llm-probes 40`) and reports Recall@1/3/5 and MRR for vector, hashed, BM25 and hybrid (reciprocal rank fusion) retrievers.
  - `src/kiss/scripts/memory_search_eval.py <memory dir>` scores a memory's own question set `eval/questions.yaml` (76 questions for the `kiss/` memory on the development machine) and exits 1 when Recall@5 is below `--min-recall5` or a gold page is missing.
  - The public-benchmark harness `benchmarkings/memoryfield/bench_public.py` (LoCoMo, LongMemEval-S) left the repository with `benchmarkings/` (removed 09-29 by the author-local date of `b9ed868023`, 09-30 UTC and inside the window of section 2).

## 14. The kiss-web daemon and remote web app

- **Codebase**: `src/kiss/server/` (38 modules, 45,818 lines; `web_server.py` 11,812).
- **Daemon responsibilities** (one daemon per machine):
  - Owns the agents, the tab registry, the cron scheduler (section 20), the SEA slash-command watcher (section 9), the browser and terminal tab services, the file index and the daemon side of voice (section 18).
  - The spare-worktree pool it used to prewarm was removed 10-09 (section 12).
- **Entry points**: `kiss-web` (`--url`, `--trust-ca`, `--workdir`) and `kiss-app`.
- **Related sections**:
  - The page the daemon serves is section 16.
  - One page over the daemons of several machines is section 15.

### Transport, authentication and command catalogue

- **Transports** **CHANGED**:
  - One port (8787, every interface) carries the page, its media and the WebSocket `/ws`.
  - The first byte of a connection picks the protocol:
    - A TLS handshake gets TLS.
    - Anything else is plain HTTP, served to loopback peers only (a 400 for everyone else); loopback has been plain HTTP since 10-15, where TLS used to be mandatory.
  - The Unix socket `~/.kiss/sorcar.sock` was removed 10-01: local clients (among them the Python client API of section 5, `run_agent` sub-tasks and cron prompt jobs, which all go through `daemon_client.run`) read `~/.kiss/sorcar-local.json` (mode 0600; `url`, `token`, `ca`, `pid`; a new 256-bit token at every daemon start) and authenticate with the token (`agents/sorcar/local_endpoint.py`).
  - Frame splitting and slow-reader handling:
    - A frame over 65,536 characters goes out as `part` frames to a client that declared `parts`, each later part held until the bytes that client has not yet taken are down to 65,536 (local clients are not held) **NEW**.
    - A peer that reads nothing for 30 s is dropped.
  - Every 15 s the daemon:
    - Pings each client.
    - Closes one that stays silent for 10 s.
    - Follows a successful ping with a `heartbeat` frame for the client's half-open detector.
- **Authentication** **CHANGED**:
  - Three kinds of client (`AuthKind`):
    - `local`: the endpoint token from a loopback address or a registered machine's link with an enrolled client certificate (section 15).
    - `loopback` **NEW**: a page opened at a loopback address (`http://127.0.0.1:8787` or `localhost`) and reaching the daemon directly, admitted without a password.
    - `remote`: the `remote_password` of `~/.kiss/config.json`.
  - Password generation and storage:
    - `install.sh` generates a 20-character password when none is stored and keeps it across reinstalls **NEW** (`python -m kiss.core.remote_password --show` or `--rotate`).
    - It is stored as typed and compared in constant time.
  - Every frame of a `remote` client carries the password and is checked against the stored one, so a changed password shuts out the other sessions at their next frame **NEW**.
  - With no password stored the daemon serves only loopback peers and machine links with an enrolled client certificate: other peers get a 403 and no tunnel is started.
  - Five wrong passwords from one address within 60 s lock that address out (`auth_locked` with `retry_after`).
- **Command catalogue**:
  - `sorcar.py` `API` lists 103 commands (49 forwarded unchanged to the backend, 54 with their own `ServerApi` handler).
  - Dispatch pipeline:
    1. `validate_command` refuses an unknown name or a missing required field.
    2. `ServerApi.dispatch` routes the command.
  - The page's whitelist (`media/api.js` `SORCAR_API_COMMANDS`) holds 97 of them; the other six are:
    - `auth` and `ping` (sent by the page's transport itself).
    - `run` (local clients).
    - `activeTasksQuery` (the installer).
    - `browserList` and `getTaskUsage` (another machine's daemon, section 15).
  - Since 09-30 the catalogue gained 40 commands and lost 21, among them `voiceWakeStart`/`voiceWakeStop`, `getFrequentTasks` and `generateCommitMessage`.

### Tabs, sidebar and files

- **Tab registry and replay**:
  - `TabRegistry` persists the top-level chat tabs atomically to `~/.kiss/tabs.json` (cap 512) and broadcasts a `tabs_state` snapshot after every change, which every client mirrors.
  - A chat id is bound to at most one tab (the newest bind displaces the older).
  - Unregistered tabs:
    - Sub-agent tabs are not registered: they open on `openSubagentTab` and close on `subagentDone`.
    - File, directory and terminal tabs stay local to the client.
  - Events are recorded and persisted per task and fanned out only to the tabs subscribed to that task (`json_printer.py`), so one chat's events never reach another chat's view.
  - Initial `ready` handshake and deferred replay:
    - `ready` answers the connecting client only (registry, models, input history, configuration, SEA commands, browser tabs, tips, promptlets).
    - Since 10-12 it replays the transcript of the one tab that client shows and marks every other bound tab `replay_deferred`, fetched by a scoped `resumeSession` when the tab is opened **NEW**.
  - A replay is bounded, and embedded pictures travel as handles the page fetches with `getReplayImage` **NEW**. The bounds:
    - The text of a `thinking_delta`, `text_delta` or `system_output` event, when longer than 200,000 characters, is cut to its first and last 100,000 joined by a note counting the omitted ones.
    - A 16 MiB budget applies for the serialised events, past which the oldest are left out with a note while the newest is always kept.
  - A transcript is loaded in the pieces the reader opens (`server/lazy_replay.py`, 11-18) **NEW**, for a page that sends `lazy: true` with `ready` or `resumeSession`:
    - A finished task is sent as its settings, a `trajectory` stub with the panel count, its result cards and its last usage; a running task is sent its outline at once.
    - `getTaskEvents` fetches the outline (the summaries and every panel no summary folded) when the Trajectory panel is opened and, with `seg`, the panels one summary folded; a tab bound to another machine fetches over the machine link.
    - The layout is computed from the event rows without their bodies, and a finished task's first piece is kept in the database, so opening the chat again reads the task row, that piece and the few rows it names.
    - A history click is broadcast to every page that shows the tab, so it comes in pieces only when all of them load that way.
  - Replay snapshots are taken under the delivery lock with a reserved send slot, so no event is duplicated or lost.
- **Browser tab**:
  - `BrowserTabService`:
    - Launches the daemon machine's browser with a persistent profile (`~/.kiss/browser-tab-profile`).
    - When the browser cannot start, Playwright's bundled Chromium is used instead (11-03) **NEW**.
    - Streams each page as a `browser__*` tab (1280×800, JPEG quality 60, at most 15 frames/s, 90 s open timeout).
    - Relays clicks, keys and viewport changes (`browserOpen`, `browserClose`, `browserNavigate`, `browserInput`, `browserViewport`).
  - Per-viewer flow control (`frame_flow.py`, 10-31) keeps at most 2 frames unacknowledged per viewer (`browserFrameAck`, 5 s grace) and holds only the newest meanwhile, so a slow link shows a late page instead of losing the tab **NEW**.
  - Agents receive the service as `live_browser`:
    - `show_browser()` moves the agent's headless session into such a tab.
    - Third-party sign-in pages open there through `browser_handoff.set_browser_tab_opener` (section 17).
  - For a chat bound to a registered machine the page, or the shell, opens on that machine (section 15).
- **Terminal tab** **NEW** (10-05):
  - `terminal_tab.py` runs the user's shell in one pty per tab on the daemon's machine, chosen in fallback order:
    1. `$SHELL`
    2. `bash`
    3. `/bin/sh`
  - Protocol messages:
    - Commands: `terminalOpen`, `terminalInput`, `terminalResize`, `terminalClose`.
    - Events: `terminalOpened`, `terminalData`, `terminalExit`, `terminalError`.
  - A shell outlives a dropped connection by 60 s, so the tab re-attaches after a reconnect.
  - `terminal_rc.py` (10-09) starts bash and zsh with start-up files under `~/.kiss/terminal/` that load the user's own and append each command to the history file at once, so terminal tabs share history.
  - Windows has no pty: the tab reports an error.
- **Sidebar data**:
  - `getCronJobs` with `cronAction` **NEW**: Schedule (`create`, `update`, `run_now`, `pause`, `resume`, `remove`; section 20).
  - `getAppsStatus`: Connectors, titled Apps until 11-20 (every channel module probed in one subprocess, cached 30 s).
  - `getSkills`, `searchSkills` and `installSkill` **NEW** (11-20): Skills (section 7). Each may name a `workDir`, the folder whose project skills are listed and count as installed (the page sends the working directory its Explorer shows; without one the daemon uses its own):
    - `getSkills` is answered with `skills`: the name, description and source of every skill the agent can use, sorted by name.
    - `searchSkills` (`query`, `generation`) is answered with `skillSearch`: the `query`, the registry's rows (`id`, `name`, `source`, `installs`, `url`, and `installed` for a name the agent already has) or an `error`, and the command's `generation`, by which the page shows the answer to its newest search only.
    - `installSkill` (`id`, optional `force`) is answered with `skillInstall`: the `id`, a `status` of `installed`, `review`, `blocked` or `error`, the skill's `name`, a `message` and the `findings` (`severity`, `rule`, `message`, `file`, `line`). A `skills` event follows an install.
    - A search and an install are answered from a background task, so the connection's next commands do not wait for the registry or the check.
  - `getSpendReport`: Spend (total, per day, per model from persisted `cost`/`tokens`, sub-agent rows excluded).
  - `getRecommendations` **NEW**: twenty suggested tasks from the last `/recommendations` run, a default 6-hourly cron job.
  - `getMachines` with `machineAction` **NEW** (section 15).
  - `getHistory` **CHANGED** (11-22): Chats, in pages of `limit` rows (50 when none is named, 500 at most), which the reply echoes. A list that does not fill its panel asks for as many rows again as it holds, and a refresh asks in one request for as many rows as the list has asked for so far. **CHANGED** (11-23): once the user has scrolled the list, it asks for its next rows as soon as less than one panel's height of rows is left under the visible part (before: within 50 px of the list's end, 50 rows at a time); each request asks for as many rows as fill the panel, estimated from the room the rows already held take (50 at least, 500 at most from the daemon), and when a reply still leaves less than a panel's height under the view (rows hidden by a filter or folded into a chat or project), the next request follows at once until a full panel's height lies under the view or the list ends. When the daemon grants fewer than a refresh asked for (more than 500 rows, or a daemon that sends 50 whatever it is asked for), the page asks for the rest and builds the list once all are in, so the list is never emptied and filled again in sight. The "Loading..." row shows only at the end of a list the user scrolled through and in a list that has no task to show yet.
  - `getTaskUpdate` **CHANGED** 10-03, **CHANGED** 11-19, **CHANGED** 11-20:
    - **Task Progress** (formerly Task update) is the `/ask` agent's answer to "What has this task done so far, and what are its partial results?".
    - First asked when the task is 60 s old and again once the answer is 600 s old, sooner when it estimated the time left (below), or on Refresh; only a poll starts a run.
    - **Time left** (11-20): The question (`UPDATE_QUESTION`) also asks the agent to end its answer with `Time left: <low>-<high> min, most likely <best> min.`, or with `Time left: unknown.` when no estimate is possible, as when the task waits for the user:
      - `_split_estimate` takes that sentence out of the answer only when it is the answer's last sentence; the same words anywhere else stay in the text.
      - The `taskUpdate` reply carries `etaAt`, `etaLowAt` and `etaHighAt`: the most likely, earliest and latest finish in epoch milliseconds, counted from the moment the question was asked, and 0 when there is no estimate (`unknown`, numbers that are not low ≤ best ≤ high, or more than 365 days).
      - The range is the agent's own, widened by `ETA_RANGE_WIDENING` (1.75): the low end divided by it and the high end multiplied.
      - With an estimate, the next run is due after half of the time left, no sooner than `FIRST_UPDATE_DELAY_S` (60 s) and no later than `UPDATE_INTERVAL_S` (600 s); without one, or after a failed run, after `UPDATE_INTERVAL_S`.
      - Measured on 11-20 at 70 checkpoints of 35 finished tasks, each task's context rebuilt as it was at the checkpoint: the most likely value was off by a median factor of about 1.9 on `claude-fable-5-2-max` and `claude-opus-5-5-medium`, the range as stated held the real time left at 50% and 46% of the checkpoints, and the widened range at 89% and 90%. `gpt-6.1-sol` and `gemini-4-carbon` underestimate more: their widened range held at 57% and 51% of 47 of those checkpoints.
    - Capped at $1 and run as a side-channel sub-agent whose spend is charged to the task.
    - The `/task_update` SEA remains as a chat command.
- **Files**:
  - `openFile` answers the requesting connection with `fileContent`:
    - Text up to 2,000,000 bytes; an HTML page above the text cap opens read-only instead of being refused (10-26) **NEW**.
    - PDFs and images up to 24 MiB as base64.
    - A directory as a text listing.
  - `saveFile` replaces a file atomically and refuses with `conflict` when it changed on disk since it was opened, unless `force`.
  - Explorer and Source Control:
    - `explorer.py` serves `listDir` (capped at 2,000 entries), `gitStatus`, `gitLog`, `gitShow` and `gitAction`.
    - `fs_actions.py` serves `fsAction` (`newFile`, `newFolder`, `rename`, `delete`, `copy`, `move`, `findInFolder`, `compare`).

### TLS, file index, prompt shortcuts and branding

- **TLS**:
  - Connections from other machines use a certificate issued by a machine-local CA (`kiss/core/tls_certs.py`, moved out of `server/`; files in `~/.kiss/tls/`):
    - An ECDSA P-256 root valid 3,650 days.
    - A server certificate valid 820 days naming `localhost`, the host name and the current loopback and LAN addresses.
  - Re-issuance and reload:
    - The server certificate is re-issued when missing, within 30 days of expiry, not signed by the current CA or lacking a current LAN address.
    - The watchdog checks every 15 s and loads the new one without dropping connections.
  - CA distribution:
    - `kiss-web --trust-ca` installs the CA into NSS databases, the macOS login keychain or the Windows user store.
    - Other devices fetch it from `GET /ca.crt`.
  - The same directory holds the client certificate and `trusted-clients/` of machine links **NEW** (section 15).
- **File index**:
  - Persistent home-wide index (`~/.kiss/file-index`) warmed at startup:
    - Depth 12; 1,000,000-entry cap, directories included.
    - Skips dot-directories, `node_modules`, `__pycache__`, `venv`, `site-packages` and the literal, non-negated entries of every `.gitignore` on the way down, and collapses bulk data directories.
    - Refreshed by directory mtime once 60 s stale.
  - `@` picker ranking (20 suggestions, up to 5 places kept for home):
    1. Previously mentioned paths.
    2. `./` matches from the work dir.
    3. `~/` matches from the rest of home.
  - Query prefixes:
    - A query starting with `./` searches the work dir only.
    - One starting with `~` searches the whole home directory only.
  - Since 10-31 **NEW**:
    - The contents of a directory the OS refuses to list, or whose `.gitignore` cannot be read within 5 s, are left out until the daemon restarts instead of blocking the picker.
    - The home index skips `~/Library` on macOS.
    - After a restart the index is published from the persisted listings before the rescan.
- **Prompt shortcuts**:
  - Path-only prompts:
    - A single-line prompt that is only the path of an existing regular file, typed in a tab with no running task and no pending question, opens the file (`promptOpened`, then `fileContent`) instead of starting a task.
    - A directory name still starts one.
  - The daemon also serves:
    - Inject promptlets (`addTrick`, `editTrick`, `deleteTrick`), in this order:
      1. `~/.kiss/MY_INJECTION.md`
      2. The bundled `INJECTIONS.md`
    - Ghost-text completion (`complete`).
    - The 22 tips of `TIPS.md` (`tipsOptOut` writes `~/.kiss/TIPS_DISABLED`).
    - An input history of 500 entries (section 5).
- **Branding and brand services**:
  - `kiss/core/brand.py` reads `webapp/media/brand.json` (`product_name`, `short_name`, `tagline`, `identity`, and `home_dir` **NEW** 10-03: the state directory under `$HOME`, `.kiss` for the stock brand); missing keys fall back one by one.
  - Template and manifest rendering:
    - `render_brand` fills the double-braced placeholders `PRODUCT_NAME`, `SHORT_NAME`, `TAGLINE`, `IDENTITY`, `HOME_DIR` and `VERSION` in the prompt files and tips.
    - The page template receives `brand.css`, `PRODUCT_NAME`, `TAGLINE`, `HOME_DIR`, `BRAND_JSON` and a title naming the machine.
    - `/manifest.webmanifest` **NEW** names the installable app after the brand.
  - `install.sh` overlays a git-ignored `.brand/` directory onto the installed media for white-label builds (section 21).
  - Two hooks extend the daemon without code changes:
    - Executables in `~/.kiss/services.d/` are started once the local endpoint exists and restarted with a back-off of 1 s doubling to 60 s (`services.py`, 10-29) **NEW**.
    - An executable `~/.kiss/signin` makes every page show a sign-in window, one button per provider, until it reports the user signed in (`signin.py`, 10-20; `signinLogin`, `signinPaste`, `signinCancel`) **NEW**; without the hook, or with a broken one, nothing is shown.

### HTTP, remote reach, lifecycle and export

- **HTTP** **CHANGED**:
  - Since 10-29 `http_speed.py`:
    - Gzips text responses of 1,024 bytes or more (cached by content hash up to 64 MiB).
    - Marks `/media/<file>?v=<hash>` as `Cache-Control: public, max-age=31536000, immutable`; everything else stays `no-store`.
  - Service worker (`media/sw.js`, cache `kiss-shell-<version>`):
    - Precaches the shell, serves `/media/*` cache-first and serves `/` network-first, falling back after 4 s to the cached shell marked offline.
    - Never intercepts `/ws`, `/api/*`, `/trajectories` and the voice model.
  - `GET /api/daemon` **NEW** returns the daemon's instance id and version, and a page that reconnects to a different instance reloads itself (10-23).
  - `GET /webclip.mobileconfig` **NEW** (11-03) serves an iOS Web Clip profile, signed through the local CA, that puts the web app on the home screen.
  - Trajectory visualiser (`/trajectories`, `kiss/viz_trajectory`): its data routes `/api/jobs` and `/api/jobs/<job>/trajectories` now ask anyone but a direct loopback peer or an enrolled machine link for the remote password as an HTTP Basic credential **NEW**.
- **Remote reach**:
  - An optional `cloudflared` tunnel publishes the daemon:
    - A quick tunnel with a random `*.trycloudflare.com` URL, or
    - A named tunnel with a fixed URL when `CLOUDFLARE_TUNNEL_TOKEN` (or `tunnel_token` in `config.json`) is set.
  - The active URL is:
    - Written to `~/.kiss/remote-url.json` (`kiss-web --url` prints it).
    - Broadcast as `remote_url`.
    - Posted, while a tunnel is in use, to the machine's `ntfy.sh` topic with the machine name appended.
  - A tunnel the daemon lost track of is found and adopted again, which keeps the URL stable across restarts (10-03) **NEW**.
- **Restart and update handling**:
  - The daemon polls PyPI hourly and broadcasts `update_available`.
  - Update and model commands:
    - `runUpdate`: runs the bootstrap `~/.kiss/kiss_ai/scripts/install.sh` detached (log `~/.kiss/update.log`), which syncs the clone and hands over to its `install.sh` (section 21), refuses new task submits while the installer runs and reports a failed installer to the window that asked.
    - `updateWhenIdle`: waits until no task runs.
    - `snoozeUpdate`: hides the offer for 24 h.
    - `updateModels`: refreshes `~/.kiss/MODEL_INFO.json` (section 8).
  - `serverReset` (Settings, Reset Server) sends the daemon SIGTERM so that the LaunchAgent or systemd unit respawns it on the same port and tunnel.
  - Deferred restart when busy (`deferred_restart.py`, 10-30) **NEW**: `install.sh` never restarts a daemon with a task in flight:
    1. `install.sh` leaves `~/.kiss/.kiss-web.restart-pending`.
    2. The daemon runs `install.sh --restart-only --non-interactive` itself once idle.
  - Sticky notifications report:
    - An install that finished with problems (`~/.kiss/.install-problems`, shown for 7 days) **NEW**, and
    - The retirement of the VS Code extension (removed 10-14; `dismissNotice` hides it) **NEW**.
- **Desktop app window** **NEW** (10-16):
  - `kiss-app` (`standalone_app.py`) opens the web app in a Chromium-family browser in app mode under its own profile (`~/.kiss/webapp-profile`).
  - `--install-launcher` registers a `.desktop` entry on Linux or an application bundle in `~/Applications` on macOS with the brand's name and icon (section 21).
  - The daemon counts app-window connections (`/ws?appWindow=1`, `GET /api/app-window`) and on macOS quits a browser left without a window.
- **Resilience**:
  - A watchdog runs every 15 s:
    - Restarts a dead tunnel.
    - Rewrites a deleted `remote-url.json`.
    - Reacts to changed network addresses.
    - Re-binds the macOS loopback alias.
    - Pings every WebSocket.
    - Refreshes the TLS certificate.
    - After the host slept, drops the dead links to registered machines.
  - A stall watchdog dumps every thread's stack to stderr after 60 s of a blocked GIL (`faulthandler`, disarmed at shutdown).
  - SIGTERM stops in-flight tasks before the event loop unwinds.
  - Limits:
    - Prompts of 1,000,000 bytes.
    - 32 attachments.
    - 64 MiB per frame.
    - 64 executor threads.
- **Chat export** **CHANGED**:
  - The chat menu's Share chat became Save chat on 10-19 (section 16).
  - `shareChat`:
    - Writes a self-contained page (`main.css`, `brand.css`, both highlight themes and `share.js` inlined) to `reports/chat-<title-slug>-<chat id>.html` under the tab's work dir and answers `share_done`.
    - Records on the page's last line the version and git commit of the daemon that wrote it, `unknown` for one it cannot determine **NEW** (11-12).
  - `shareChatTasks` supplies every task of the chat, oldest first, with its sub-agents nested and its metrics (tokens, cost, steps, start and end time), within a 24 MiB reply budget that drops the oldest tasks first and flags `truncated`.
  - `share.js` gives the saved page collapsible event panels, nested `run_parallel` groups, a sub-agent tab bar and a light/dark theme.

## 15. Machines: one web app over several computers

**NEW**: the whole section is new since 09-30; registered machines first appeared on 10-10.

A daemon keeps a registry of other computers' daemons and drives them from its own web app: their chats, files, settings, terminals and browser pages appear on the local surfaces, and a registered machine's own web UI is never opened.

- **Modules**:
  - `src/kiss/agents/sorcar/machines.py` (2,301 lines: registry, certificate pins, SSH enrolment and forwards, `MachineLink`, merged history).
  - `src/kiss/server/remote_tabs.py` (1,335), `remote_views.py` (824), `machine_stream.py` (148) and the forwarders of `sidebar_panels.py` (950).
  - Four SEA folders under `src/kiss/agents/seas/`: `machine/` (146), `sync/` (824), `install/` (203) and `sync_history_and_memory/` (1,238).

### Registration and links

- **Registry**:
  - `~/.kiss/machines.json` (mode 0600) holds one entry per machine with `name`, `url`, `password` and `ssh`; a name matches `^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$` and defaults to the address's host.
  - Editing surfaces:
    - `/machine add [name] <address> [password]`, `/machine remove <name>` and `/machine list` (a $1 worker whose only tools are `machine_add`, `machine_remove` and `machine_list`; section 9), and
    - The **+** form and the delete button of the Machines subpanel (`machineAction`).
  - `add_machine` registration sequence (so a wrong address or password changes nothing):
    1. Stages the certificate pin.
    2. Completes one authenticated handshake.
    3. Only then publishes the pin and the entry.
  - The daemon reads the file's size and mtime every 2 s (`MACHINES_REGISTRY_POLL_SECONDS`): a change retires the links and ssh forwards of the machines whose entry changed and pushes a fresh `machines` list to every connection.
- **URL machines**:
  - `https://host:port` (or `host[:port]`; the port defaults to 8787) is registered with that daemon's remote password.
  - Registration pins the remote's local CA (`GET /ca.crt`), or its leaf certificate when it serves none, as `~/.kiss/machines/<name>.pem` (`pin_certificate`).
  - Every later connection verifies the chain against that file alone, with host-name verification off because a daemon's generated certificate names its LAN addresses.
- **SSH machines**:
  - `user@host[:port]` (or `ssh://[user@]host[:port]`) needs no password.
  - Enrolment and connection steps:
    1. One `ssh` login (`ENROLL_COMMAND`) finds the home directory of the daemon running there, copies this daemon's CA into its `tls/trusted-clients/` and prints back the daemon's port and CA, which is pinned.
    2. Every later connection goes through one `ssh -N -L` forward per login (`SshTunnel`, reopened when it dies) to `127.0.0.1:<port>` on the machine and presents the client certificate `~/.kiss/tls/client.pem`, which the remote admits like its own local token whatever its remote password; its daemon port need not be open to the network.
  - SSH login requirements and settings:
    - The login must work non-interactively (`BatchMode=yes`), allow TCP forwarding and land in a POSIX shell.
    - An unknown host key is accepted on first use.
    - `~/.ssh/config` applies, or the file `$KISS_SSH_CONFIG` names.
  - Revoking a machine means removing its ssh access there.
- **Links**:
  - `MachineLink` is a persistent authenticated WebSocket per machine and lane, dialled on first use and again after a drop.
  - A link answers one request at a time, so slow traffic has lanes of its own beside the default one: `panels` (sidebar polls, Task Progress, pictures, share), `events` (a folded panel's events, `getTaskEvents`), `skillSearch` and `skillInstall` (a skill search and a skill install, 11-20), `actions` (Git and file actions), `commit`, `login` and `status`.
  - Timeouts:
    - A request's reply times out after 8 s unless its forwarder allows more (`REQUEST_TIMEOUT`), and each `part` frame or `more` piece of a long reply restarts the wait.
    - Three waits have budgets of their own:
      - Waiting for the link's turn: the same time again at most.
      - A dial's handshake: 8 s at most.
      - The opening of an ssh forward: 30 s (`SSH_TIMEOUT`).
  - Sockets negotiate permessage-deflate, receive a frame above 64 KiB in `part` frames and are pinged every 20 s.
  - Back-off and sleep recovery:
    - After a failure the link reports the machine down without dialling for 5 s, doubling to 30 s (`FIRST_BACKOFF`, `DOWN_BACKOFF`).
    - A rejected password waits the full 30 s so the remote's brute-force lockout is not tripped.
    - `forget_after_sleep` drops the cached sockets and ssh forwards once the host has slept 30 s or more (`SLEEP_GAP`).
  - Request correlation (`generation`):
    - Each request carries a fresh `generation`, and a reply that echoes one must echo this one.
    - The five replies of `CORRELATED_REPLIES` (`models`, `history`, `myModelsData`, `machines`, `historyDeleted`) must echo it when the remote's `auth_ok` carries `echoesGeneration`, so a list the remote pushed for another client's edit is not taken for the answer.

### Remote tabs, views and UI panels

- **Chat header drop-downs**: every chat header has a machine drop-down and, next to it, a working-directory drop-down:
  - **Machine drop-down**:
    - Lists this machine and every reachable registered machine, labelled by the host name its daemon reports when it was registered under the default name.
    - A remote pick shows green, and red once a `machines` reply has shown it down.
    - The pick for a new chat is one global setting (`setMachine` stores `last_machine` in `config.json` and broadcasts `machineChanged`), so the surfaces agree; a tab that shows a chat keeps that chat's machine.
    - Picking another machine than the chat's opens a new chat on that machine, as **New chat** does **CHANGED** (11-20; before, the pick was where the chat's next prompt went). The chat left keeps its machine and its running task and stays in the Chats panel.
  - **Working-directory drop-down**:
    - Lists the picked machine's directories (`getMachineWorkDirs` returns its `workDir`, `recentWorkDirs` and `maxBudget`).
    - A pick there sends `setWorkDir` with `machine`, which that machine's daemon adopts.
- **Bound tabs** (`remote_tabs.py`):
  - A `submit`, or the API's `run`, that names a machine binds its tab to it; `tabMachine {tabId, machine}` tells every surface the binding.
  - Stream forwarding (`RemoteTabs` keeps one authenticated stream per bound tab):
    - Forwards `submit`, `run`, `stop`, `userAnswer`, `appendUserMessage`, `interruptTool` and `worktreeAction` unchanged under the same `tabId`, so the remote daemon runs, queues and stops the tab's tasks as for a page of its own.
    - A forwarded `submit` drops the local `workDir`, so the task runs in the machine's working directory.
    - Every frame the remote stamps with the tab id is handed to the local clients as it is; the task, its events and its history row live on the remote only.
  - The stream also:
    - Relays the task's sub-agent tabs (`new_tab`, `subagentDone`).
    - Replays the chat to a surface that connects or reloads.
    - Opens the file of a path-only prompt from the machine.
  - Tab lifecycle:
    - A machine's chat shows in one tab (a resume into a second tab closes the first on every surface).
    - `closeTab` is forwarded while the task keeps running there.
    - A lost socket ends the tab's running state with an `error` naming the machine.
- **Terminal and Browser** (`remote_views.py`):
  - In a chat bound to a machine the Terminal and Browser buttons open a shell, or a page of the browser, on that machine.
  - One `RemoteViewStream` per machine forwards the four terminal and six browser commands and relays the events stamped `machine`, with browser tab ids namespaced `browser__N@machine`.
  - Stream persistence and reconnect:
    - Stays open while a tab is bound to the machine, so a page the machine's agent opens for the user (`show_browser`, a sign-in hand-off; section 17) reaches the local surfaces.
    - A stream lost meanwhile is reopened after 3 s (`REOPEN_DELAY`), and one that opens lists the pages the machine's browser already shows.
  - Every browser frame is acknowledged (`browserFrameAck`), so at most two frames of each page are in flight on the stream, and to each local connection watching the page.
  - A lost stream ends its terminals with a `terminalError` naming the machine and closes its browser tabs on every surface.
- **Forwarded requests**:
  - 42 commands are answered by the machine's daemon when they carry `machine` (`MACHINE_FORWARDERS`), and the page stamps them with the machine the active tab's header shows, so the sidebar, the model picker and the Settings sheet follow the chat in view:
    - Explorer, Source Control and editor (`listDir`, `gitStatus`, `gitLog`, `gitShow`, `gitAction`, `fsAction`, `openFile`, `saveFile`, `checkPaths`).
    - Settings sheet (`getConfig`, `saveConfig`, `getMyModels`, `saveMyModel`, `deleteMyModel`, `getClaudeCodeStatus`, `getCodexStatus`, `startCodexLogin`).
    - Model picker (`getModels`, `selectModel`).
    - `@` picker (`getFiles`, `recordFileUsage`).
    - Schedule, Machines, Skills, Connectors, Spend and Recommendations subpanels (ten commands):
      - A forwarded skill search is waited for 30 s longer than the link's 8 s, and a forwarded skill install 900 s longer (11-20).
      - A machine whose daemon does not know the Skills commands answers with an error, which the subpanel shows.
    - Task Progress (`getTaskUpdate`), a replayed transcript's pictures (`getReplayImage`), a folded panel's events (`getTaskEvents`), chat sharing (`shareChatTasks`) and the seven CHATS writes below.
  - Remote UI behaviour:
    - The Explorer of a bound chat shows the machine's working directory alone and hides **Add Folder to Explorer...**, and a file opened from it saves back to the machine.
    - A remote machine's Settings sheet is headed `Settings of <label>` and hides this daemon's URLs and the **Update**, **Reset Server** and **Update Models** buttons.
    - **Git Commit** runs in the machine's folder (`machine_autocommit`, waited for up to 900 s).
- **Machines subpanel**:
  - `getMachines` returns `machines_report` (each row with `name`, `label`, `address`, `ip`, `port`, `local`, `status` and `error`), ordered:
    1. This machine first.
    2. Every registered one pinged in parallel over its link.
  - The report is cached for 60 s, or for 10 s while any machine is unreachable.
  - Row and header controls:
    - A row shows a status dot, the label, the address and its reachability, a copy-address button and a delete button with inline confirmation (disabled on the local row).
    - The header's **+** opens the add form (optional name, address, password for a URL machine).
  - A machine registered elsewhere, by `/machine add` or on another surface, unfolds the subpanel on every surface.
- **Merged CHATS**:
  - `getHistory` runs `merged_history`:
    - Every registered machine is asked for its rows from the first to the end of the requested page, in one request **CHANGED** (11-22; one per 50 rows before, and still so of a machine whose daemon does not echo `limit`; one per 500 rows of a longer page) (same query and tag, its own rows only, `machine: "__local__"`, so two daemons registered with each other do not echo each other).
    - The union is ranked with one copy per task id and cut to that page, ordered:
      1. Running tasks first.
      2. Newest first.
  - Fan-out timeout:
    - The merge waits 2 s (`HISTORY_FANOUT_WAIT`).
    - A slower machine contributes the page it last answered with, and a `tasks_updated` follows when the rows it then brings differ.
  - Remote row display and actions:
    - Every row's **...** menu names the machine it ran on (a registered machine's label or this daemon's host name, **CHANGED** 11-19), and a machine filter (all, this machine, each registered machine) appears once a machine is registered.
    - A click resumes the chat in a local tab bound to the machine.
    - The favourite star, title edit and delete buttons of a remote row, the star of its chat (11-16), and the pin of either (11-20), act on that machine's `history.db` (`setFavorite`, `setChatSummary`, `setChatFavorite`, `setPinned`, `setChatPinned`, `deleteTask`, `deleteChat`).
  - A task that a history sync copied in is listed under the machine it ran on: `task_history.machine` holds that machine's short host name, matched to a registered machine through `~/.kiss/machine_hosts.json`.

### Remote agents, multi-machine SEAs and brand hub

- **Agents across machines**:
  - `run_agent(machine=<name or host>)` runs the sub-task on that machine's daemon with its tools, files and models (section 10):
    - `work_dir` is a path there, and paths in the task are not rewritten.
    - A command name, or the path of a SEA registered as a command here, runs the machine's copy of that SEA (`remote_sea`).
    - The plain sub-agent (an empty or generic `agent`, or the path of the hidden `sorcar` SEA) is sent as a run without a SEA.
    - Any other `.py` path must exist at the same path there.
    - Naming this machine itself is an ordinary local call.
  - Sub-task inheritance and constraints:
    - Inherits the model, budget share and prompt suffixes, but not the caller's chat, container or extra tools.
    - Opens its tab under the caller's on every surface.
    - A sandboxed caller is refused.
    - `run_parallel` takes no `machine`.
  - Other remote execution entry points:
    - The Python API's `run(machine=)` forwards a top-level run the same way (`daemon_client.py`).
    - `Bash(background=True, machine=<SSH machine>)` runs a command there through `ssh` with its output streamed into the local job log (section 7); a URL machine is refused.
  - Spend accounting:
    - The spend of a remote sub-task is folded into its local parent and recorded in `remote_subtasks`.
    - `sync_remote_subtask_usage` later asks the machine (`getTaskUsage`, 500 task ids per request) for spend charged after the fold, for runs of the last 7 days, with a sweep of every recorded run at most hourly.
- **Keeping machines in step** (SEAs, section 9):
  - `/sync` (budget of $2 × (1 + 2 × machines); `/sync this machine only` skips the machines) commits, merges `origin/main` and pushes in sequence:
    1. On every registered machine one at a time (`run_agent(agent="sync", machine=...)` on that machine's checkout, found by the repository's URL).
    2. Here.
    3. On every machine again.
  - `/sync_history_and_memory` brings `history.db` and `memories/` of this machine and every SSH machine (or only those its message names) to the same state over one multiplexed ssh connection per machine:
    - Runs in three waves:
      1. Probe.
      2. Pull.
      3. Push.
    - History travels as the rows the other side lacks (task rows matched by `id`, events by `(task_id, seq)`), and tasks still running are held back until they finish.
    - Memory pages travel by newest `updated` stamp with tombstones (section 13).
    - URL machines are reported as skipped.
    - Deletions travel as well: the tombstones of `deleted_tasks` and `deleted_chats` (section 11) make the receiving database drop the deleted tasks' rows and events (a task still running there keeps its row and events until a later sync, and a tombstone older than the task row does not apply) and the summary row of a deleted chat left without tasks; section 21 has the other merge rules.
    - Clock-spread conflict rule: two writes of a page no further apart than the spread of the machines' clocks, measured in the first wave, are not ordered (pages with equal stamps are compared by file modification time, under the same rule): each side keeps its own copy and the page is reported as a conflict.
    - A message that asks for a dry run or a preview changes no history or memory and reports what would travel between each machine and this one as it stands.
  - `/install` runs `./install.sh --non-interactive` on every SSH machine concurrently, or on the machines it names, in the clone of the private development repository it finds in the login's home by its `origin` URL.
  - `./rsorcar user@ip-address` deploys a checkout to a host and syncs history and memory with it without registering it (section 21).
- **A brand's hub**:
  - A white-label brand can let any device reach a daemon's web app through its own hub, without a tunnel.
  - For its relay the daemon runs every executable file in `~/.kiss/services.d/`, dot files excepted, for its own lifetime, restarting one that exits after 1 s, doubling to 60 s while it keeps failing (`services.py`; section 14).
  - Each task's `task_settings` event records the `host` it ran on, so a hub sync sends only the tasks of the sending machine.

## 16. Chat client features

- **Single-page frontend** (`src/kiss/webapp/media/`): One page serves every client:
  - **Core files**:
    - `chat.html` (837 lines, 41 template placeholders)
    - `main.js` (32,016 lines)
    - `main.css` (5,280 lines)
    - `remote-codex.css` (1,061 lines)
  - **Feature scripts** (one script per feature next to them):
    - `voice.js` (1,379 lines)
    - `transport.js` (1,332)
    - `contentContextMenu.js` (865)
    - `pdfView.js` (774)
    - `share.js` (703)
    - `panelCopy.js` (654)
    - `tips.js` (628)
    - `browserTab.js` (552)
    - `terminalTab.js` (412)
    - `mobileSwipe.js` (211)
- **Surfaces**: The VS Code extension was removed on 10-14, so a browser tab, the phone's home-screen icon and the `kiss-app` window (section 21) all show this page.
- **Related sections**:
  - What the daemon does behind it: section 14.
  - Other computers: section 15.
  - The page's 407 `*.test.js` files in `src/kiss/webapp/test/`: section 22.

### Layout, header and transcript

- **Layout and tabs**:
  - At 900 px or wider, the CHATS sidebar is docked on the left of the chat and the task-info sidebar on its right.
  - **Split panes** **NEW** (10-08): Chat and content are split panes:
    - The content pane appears on the right of the chat while a file, browser or terminal tab is open and has its own tab row.
    - The task-info sidebar then has no column of its own: it slides off the window's right edge when a tab is shown in the pane or the pane is pressed, and a drawer tab on that edge brings it back on top of the pane.
    - The divider drags between 20% and 80% of the width (50% at first; arrow keys move it, a double-click resets it).
    - The width and a folded content pane are remembered in the browser (`kiss-chat-pane-pct`, `kiss-content-pane-folded`).
  - **Tab strips**:
    - There is no row of chat tabs: a chat is opened from CHATS, and a tab strip shows above the chat only when the chat has sub-agent tabs to switch to.
    - On a phone, the file screen has the strip to itself: it lists the open file, browser and terminal tabs and nothing else, and shows for a single one too (11-12).
  - **Tab badges, menu and lifecycle**:
    - A tab carries a spinner while its task runs, then a green tick or a red cross, a `?` badge while the agent waits for an answer and a dot for unsaved edits.
    - A file, browser or terminal tab has a close button and a menu offering Close, Close Others and Close All; a chat's or a sub-agent's tab has neither **CHANGED** (11-20; before, every tab had both, and the menu offered Close Inactive too).
    - A sub-agent's tab closes when the sub-agent finishes.
- **Chat header** **NEW** (10-11, 10-12): Two drop-downs above the transcript, machine and working directory:
  - **Machine drop-down**:
    - Lists this computer and the registered machines (section 15).
    - The pick is one setting shared by every client (10-19) and decides where the next task, the Terminal and Browser tabs, Git Commit and Create pull request run.
    - A pick of another machine opens a new chat there **CHANGED** (11-20).
  - **Working directory drop-down**:
    - Entries in order:
      1. The directories selected as the working directory on that machine so far, most recent first (`recent_work_dirs` of `~/.kiss/config.json`, up to 30, plus this computer's current directory; the folders that earlier tasks ran in are not offered).
      2. `Choose ...`, which opens an in-page folder picker on the picked machine.
    - `No working directory` shows until one is chosen.
- **Transcript**:
  - **Task, Thinking and text**:
    - A task starts with a Task panel that scrolls with the thread **CHANGED** (10-07).
    - Reasoning streams into a Thinking panel (called Thoughts until 10-08), and text renders as sanitised Markdown with highlighted code.
    - Prompt and System Prompt panels start folded, and panels carry a copy button and a timestamp.
  - **Tool-call panels**:
    - Each tool call is a panel named after the tool.
    - It starts folded to a one-line preview (the description of a Bash, `run_agent`, `run_parallel` or `run_commands_parallel` call, the path of a Read or Write) **CHANGED** (10-08; before, it started open), except that a Question starts open. A `run_agent` or `run_parallel` panel starts folded like the others **CHANGED** (11-20; before, it started open and folding or unfolding it closed or opened its sub-agent tabs).
    - It has a Stop button that interrupts that one call, except that a `finish` panel has no Stop button.
    - Bash output streams into the panel and a failed result is marked Failed.
    - **NEW** (10-08): The panel of the call in flight pulses and counts the elapsed time.
  - **Auto-folding, digests and Trajectory**:
    - **Live sweep**: While a task runs, a sweep folds every panel older than the two newest (`STREAM_OPEN_PANELS`), except a panel the user opened; it never folds the Task panel, the Result, Thinking, questions, user messages, `/ask` answers and panels with pictures.
    - **Summary digest**: A `summary` call works differently: it gathers the panels since the previous summary (or the Task panel) under one folded digest, Thinking included, and leaves questions, user messages, `/ask` answers and other panels with pictures outside it.
    - Both rules are **CHANGED** (10-06, 10-07; before, the sweep also folded Thinking and answered questions, and a digest took in questions and pictures).
    - **Trajectory** **NEW** (10-07): When the task ends, its event panels fold into a single Trajectory panel ("N events") between the task and the result.
    - **NEW** (11-18): In a chat loaded from history the Trajectory panel is fetched when it is opened, and each summary digest inside it fetches the panels it folded when opened in turn; Save chat first loads what the page still lacks.
  - **Result panel and Suggested next**:
    - The Result panel shows the summary as sanitised HTML with a `Status: FAILED` or `Status: Continue` line when it applies.
    - Tokens and cost left its header on 10-12 **CHANGED** and are read in Task Info.
    - A "Suggested next" bar puts the agent's follow-up prompt into the composer.
  - **Links and inline images**:
    - File paths in the text are links that open the file in the content pane, resolved on the chat's machine.
    - **NEW**: A web link opens in a new browser tab, or in the default browser when the page runs in the app window (10-20), and image paths in results and answers load inline (10-24).
  - **Scrolling and navigation**:
    - Scrolling up stops the auto-follow and a "New output ↓" button returns to the end.
    - Scrolling past either end loads the neighbouring task of the chat.
    - An opened chat lands at its end when it is running and on its task panel otherwise **CHANGED** (10-27).
- **Questions**: An `ask_user_question` call is a Question panel:
  - Until it is answered:
    - Its header reads "(answer in the chat box below)".
    - The composer is the answer box ("Type your answer and press Enter") with the half-typed prompt parked and restored afterwards.
    - The page switches to the asking tab and the tab shows the `?` badge.
  - There is no toast and no system notification **CHANGED** (10-06).
  - An answer being typed survives a page reload.

### Composer, promptlets, footer and model picker

- **Composer**:
  - **Keys, completion and touch gestures**:
    - Enter sends and Shift+Enter breaks the line on the desktop; on a phone or tablet the composer sets `enterkeyhint="enter"` and Enter breaks the line while the Send button submits **CHANGED** (11-19).
    - Up and Down walk the prompt history, starting from an empty box.
    - Tab accepts the grey ghost completion, which is requested 300 ms after typing stops (two or more characters, cursor at the end).
    - On a touch screen, a right swipe accepts it and vertical swipes walk the history.
  - **Autocomplete lists**:
    - **Typing**: Opens a list grouped History, Frequent, Suggestions (the promptlets) and From editor.
    - **File picker**: `@` opens the file picker (Frequent, Files, Home), answered by the chat's machine **NEW** (10-29).
    - **Slash commands**: `/` lists the slash commands of section 9 for a `/word` at the start of the text or after any whitespace, ranked **NEW** (10-18):
      1. Recent (the 3 run last, `_RECENT_COMMANDS`).
      2. Frequent (others run at least twice, `_FREQUENT_COMMAND_MIN_RUNS`).
      3. The rest under Commands.
    - **Keyboard navigation while one of these three lists is open** (the keys go to it first):
      - Its first row starts selected, and Up and Down move the selection.
      - Tab accepts the selected row (the first when Up has cleared the selection).
      - Escape closes the list.
      - Enter inserts the selected file of the `@` picker but otherwise closes the list and sends the text as typed.
  - **Attachments and image encoding**:
    - Images, HEIC and HEIF included, and PDFs attach through Attach files, a paste or a drop on the composer.
    - The browser re-encodes a picture as a JPEG of at most 1,568 px on its longer edge when it is:
      - HEIC or HEIF,
      - of a type other than JPEG, PNG, WebP or GIF, or
      - a JPEG, PNG or WebP larger than 1,500 KiB.
    - A GIF is sent as it is.
    - A picture the browser cannot decode (HEIC anywhere but Safari 17 or later, by the code's note) is dropped with an error chip.
  - **Running-task controls and voice overlay**:
    - While a task runs, a sent message goes to the running agent as a steering message, with its attachments since 10-20 **CHANGED**; a prompt the daemon refuses comes back into the composer.
    - A Stop agent button sits next to Send for as long as the task runs: it stops the whole task, where a tool panel's Stop interrupts one call, and once pressed it pulses until the agent has stopped.
    - A "Listening ..." overlay covers the box during voice capture (section 18).
- **Promptlets and tips**:
  - **Inject sheet**:
    - The Inject button opens a sheet with a search box, a box that adds a promptlet to `~/.kiss/MY_INJECTION.md`, and the list:
      1. The user's own entries first.
      2. The 2 bundled in `src/kiss/INJECTIONS.md`.
    - A click inserts the text at the cursor.
    - Every row has a copy button, and the user's rows have an in-place editor and a two-step delete.
  - **Tips popup** (22 tips in `src/kiss/TIPS.md`; Previous and Next, arrow keys, Copy buttons on code):
    - Opens once per browser after each update and from Settings.
    - Its opt-out is stored in the browser and as a `TIPS_DISABLED` marker on the daemon.
- **Footer and "..." menu**:
  - **Footer controls** (under the composer):
    - The Chats and Task info drawer buttons (shown only in a narrow window on a device without touch).
    - **New chat** (a chat-bubble icon with a `+` inside its top right, **CHANGED** 11-19; an untouched new chat is reused).
    - Inject.
    - The model picker.
    - **...**, whose 4 entries are Voice trigger, Save chat, Attach files and iPhone Home Screen **CHANGED** (11-20, down from 11: Git Commit, Browser and Terminal are buttons of the strip under the right sidebar; Dark mode or Light mode, High contrast and Auto mode are buttons of the strip under the Chats list; Settings, one of them until 11-23, is now the first button of the strip under the right sidebar).
  - **Menu actions**:
    - **Save chat** (Share chat until 10-19):
      - Writes the whole chat as one standalone page, `reports/chat-<title-slug>-<chatId>.html` in the working directory.
      - A chat over the daemon's reply budget (section 14) is saved with its most recent tasks only, under a warning, and a page too large to upload is refused with an error.
    - **iPhone Home Screen** **NEW** (11-03):
      - Loads `/webclip.mobileconfig`, which Safari on a phone passes to the profile installer.
    - The Working directory entry is gone: it is the chat-header drop-down.
- **Model picker**:
  - A pill in the footer opens a searchable list ordered as:
    1. Recently Used first, most used on top.
    2. One group per vendor.
  - Each row shows its input and output price per million tokens; the five router entries are SEAs and show a cost label.
  - The pick becomes the chat's model and the default of the machine in the chat header, whose list this is **NEW** (10-30).
  - When the agent calls `set_model`, the pill shows the new model until the task ends.
  - Models and routing are section 8.

### CHATS panel

The left sidebar, a drawer on a phone:

- **Top bar and Filter and group menu**:
  - A top row holding only a refresh button that reloads the whole page, above the **Projects** row and the search box **CHANGED** (11-21; button **NEW** 10-23; the Chats header row is gone).
  - A search box with placeholder "Search prompts..." **CHANGED** (11-21; was "Search chats...").
  - A **Filter and group** button to the right of the search box **NEW** (11-20; it replaces the Flat list or Group by chat toggle and the folded Filters drawer). It opens a menu with one row per setting; a row shows the setting's current value and opens a submenu whose chosen options carry a tick:
    - **Status**: Running, Errored and Succeeded (all three on).
    - **Favorites**: All, or Only favorites (the starred tasks and every task of a starred chat, **CHANGED** 11-16).
    - **Cron tasks**: Hidden (the default, so tasks started by scheduled jobs stay out of the list) or Shown **NEW** (10-17).
    - **Date**: a From and To date range that one button clears.
    - **SEA** (All, None, each SEA), **Tag**, **Machine** (**NEW** 10-10; shown once a machine is registered) and **Work dir** (**NEW** 10-12, in place of the workspace-only chip).
    - **Group by**, under a separator: Project (the default, **NEW** 11-21; the same as Chat while no project exists), Chat, or None for the flat list; the choice is kept in the browser.
  - While a filter narrows the list the button wears a dot and its name says how many are on.
  - The menu works from the keyboard: ArrowDown on the button opens it, the arrows move, Right opens a submenu on its first option, and Left or Escape closes it.
- **Querying and filtering**:
  - Search, tag, machine and working directory are answered by the daemon, 50 tasks on the first page and on scroll a full panel's height of rows at a time (**CHANGED** 11-23; see `getHistory`); the other filters act on the rows already loaded.
  - A Work dir pick is a folder of this computer and lists this daemon's tasks only, so while one is picked the Machine row is reset to All and disabled.
- **Sorting and date sections**:
  - Chats sort by completion time **CHANGED** (10-14) under sections in order:
    1. A Running section **NEW** (10-07).
    2. A Pinned section **NEW** (11-20) that lists each pinned item itself, running or not **CHANGED** (11-23): a pinned task as a row of its own, a pinned chat as its panel with all its tasks, and a pinned project as its panel (11-21).
    3. A Projects section **NEW** (11-21) with the projects that are not pinned, shown when Group by is Project; chats in no project stay under the day sections.
    4. Today.
    5. Yesterday.
    6. One section per earlier day.
- **Chat group header and actions**:
  - A chat is one group whose header is a single ellipsized line showing the stored chat summary (section 11) or the first task, a fold button and a **...** menu button at the right and the last task's status mark **CHANGED** (11-19; mark **NEW** 10-07):
    - A spinner while it runs.
    - A `?` while it waits for an answer (10-06).
    - A red cross once it failed.
    - A green tick once it finished successfully **CHANGED** (11-20; before, only a task that finished while the page was open had one).
  - Header actions:
    - Clicking the header opens the chat **CHANGED** (10-08), and double-clicking the title renames it **NEW** (10-15).
    - The fold button left of the **...** button shows or hides the chat's tasks **CHANGED** (11-20; it was the first item of the menu).
    - The **...** button opens a popover menu **CHANGED** (11-19) with named items (**Favourite chat** **NEW** 11-16, **Pin chat** or **Unpin chat** **NEW** 11-20, **Edit the chat summary** where Enter saves, Escape cancels and an empty box restores the computed summary, and **Delete chat** **NEW** 10-20, confirmed inside the menu), followed under a hairline by the relative time of the latest task and the machine the header opens the chat on.
- **Task rows**:
  - A task row is a single ellipsized line **CHANGED** (11-21; three lines until 11-19, fold button removed on 11-21) with the same status mark, the task text and a **...** button whose popover menu lists **Favourite task**, **Pin task** or **Unpin task** (**NEW** 11-20), **Copy task to clipboard**, **Delete task** (**NEW** 10-20, not while the task runs), the launch age and the machine name (the tag and SEA labels left on 11-12).
  - Nothing is listed under a task row and a running row no longer repaints live usage **CHANGED** (11-21; chat rows and project rows keep their fold buttons): tokens and cost, working directory, model, machine, budget, agent and the chat, task and parent ids are shown in the Task Info view of the right panel, and the parallel or sequential mode of a task is no longer shown.
- **Registered machines**:
  - Chats of every registered machine are merged in **NEW** (10-10, 10-24), open in a tab bound to that machine (10-13), and their rename, favourite, pin and delete are sent to the machine that owns them (11-08).
- **Pins** **NEW** (11-20):
  - A pin is stored by the daemon (`is_pinned` on the task's row and on the chat's summary row), so every client shows it; it survives a deploy as a star does.
  - The daemon lists pinned rows right after the running ones and ahead of every other task, most recent first, so an old pinned task is loaded before newer tasks that are not pinned.
  - The **...** menu of every item under Pinned reads **Unpin ...** **CHANGED** (11-23), also after the daemon restarts and with every chat folded: a pinned task's row is taken out of its chat's panel (unless that chat is pinned too), and a pinned chat's panel out of its project's panel. Until 11-23 a pinned task listed its whole chat, whose menu read **Pin chat**, a pinned chat or task of a project stayed inside the project's panel, and a pinned item that was running was listed under Running only.
  - **Unpin ...** puts the item back: the task into its chat's panel, the chat under its project or its day, the project under Projects. A chat left without a task is not listed, and an unfolded project left with nothing to list reads "Listed under Pinned" (nothing while a filter hides what is pinned of it).
  - The keyboard focus stays with an item that a pin moved, on its **...** button; when the item goes back into a folded chat or project, the header of that chat or project takes the focus.
  - With Group by set to None the list has tasks only, so the Pinned section lists the pinned tasks.
- **Projects** **NEW** (11-21):
  - A **Projects** row between the top refresh-button row and the search box opens a Projects page in a tab (`src/kiss/webapp/media/projects.js`): a **New project** button, then one settings panel per project (storage and overrides in section 11).
  - A panel holds the project's name, description, default working directory with a **New folder** button, max budget, the seven check boxes of the Settings sheet and its API key groups, then **Cancel** and **Save**; a new panel is pre-filled with the app's current settings, and saving a new project or pressing its **Cancel** closes the Projects tab and takes the content window off screen (in the split desktop layout the content pane folds away with its other tabs still open in it; in the stacked phone layout the chat comes back), while saving or cancelling an existing project's panel leaves the page open.
  - A pin button ("Pin project" or "Unpin project") in a panel's top right corner pins the project in the sidebar and is drawn like the pin of a project page in Claude Code Desktop (a borderless icon button with a pushpin tilted 45 degrees, an outline while not pinned and solid while pinned, with a soft fill under the pointer), and a saved project's panel has a **Delete** button (asking "Delete this project? Its chats are kept.").
  - With Group by set to Project, a project is a foldable row with the same fold button as a chat row and a **...** button whose popover menu lists **Pin project** or **Unpin project** and **Project settings** **CHANGED** (11-23; the button used to open the settings); unfolding it lists its chats (200 at a time, with **Show more chats**), each of which unfolds to its tasks, while a chat with a running task stays in the Running section at the top.
  - Clicking a project's name starts a new chat in it.
  - The chat header has a Project drop-down listing "Default" first **CHANGED** (11-21; was "No project") and then every project to put the chat on screen into a project or take it out (chats of this machine only; a chat on a registered remote machine is in no project).
  - On a page served by an older daemon process started before Projects existed and still running after an in-place upgrade of the page files, the page ignores an "Unknown command" answer to the four reads it issues on its own (`getProjects`, `getInputDraft`, `getTaskTags` and, since 11-22, the Skills subpanel's `getSkills`), showing every other error as before **NEW** (11-21). The Skills subpanel of such a page says that the server runs an older version and is to be restarted.
- **Bottom strip** **NEW** (11-19):
  - Sits below the scrolling Chats list with the daemon's OS username on the left and buttons on the right: Dark/Light mode, High contrast, Auto mode, Decrease font size and Increase font size.
  - **CHANGED** (11-20): the buttons are 32 px with 18 px icons (24 px and 16 px before) and are the only place for these actions, which left the `...` menu; Terminal and Browser moved to the strip under the right sidebar.
  - **CHANGED** (11-23): Settings moved to the strip under the right sidebar.
  - **CHANGED** (11-23): High contrast and Auto mode are on/off buttons. Each keeps its name and, while on, shows its icon in the accent colour on a filled chip with a line around it, also under the pointer. A second press of Auto mode switches it off and keeps the palette on screen (before, only Dark/Light mode left it); the High contrast button was renamed Low contrast while on and looked the same in both states. On a touch screen a tapped button no longer keeps the hover fill.

### Right sidebar

- Ten sections, up from five on 09-30 **CHANGED** (Skills is the tenth, 11-20), each collapsible and resizable with its state kept in the browser.
- **Look** **CHANGED** (11-20): set like the Chats sidebar.
  - One UI font at two sizes: a row's text (a Task Info row, the Task Progress report, the name of a job, machine, app, model, file or commit, a recommendation) has a chat summary's size and weight, and what is secondary to it (a job's schedule line, a badge, a count, an age, an app's state, a group or form label) a day label's size; the Source Control branch and status letters left the monospace font.
  - The Task Progress report keeps to those sizes whatever it is written in: a heading is a small grey label over its part, bold text and a table's header are the medium weight, and code keeps the monospace font at the label's size.
  - A row is a chat row's box: its corner and a fill under the pointer that ends 8px from the panel's edges, with a list's scrollbar at the panel's edge.
  - A recommendation keeps its border and a form's field its box, both now the Chats search field's.
- **Bottom strip** **NEW** (11-20): below the sections, buttons of the same size as those under the Chats list:
  - **Settings** opens the settings sheet; it is the strip's first button, directly left of Terminal (**CHANGED** 11-23; before, the last button of the strip under the Chats list). On a phone or in a narrow window it closes the task-info drawer first.
  - **Terminal** and **Browser** open their tabs on the chat header's machine (moved from the `...` menu and the Chats strip).
  - **Git Commit** (moved from the `...` menu) commits the folder and machine of the chat header (10-18) and reports "Staging changes…", "Generating commit message…" and "Committing…" on one line; the flow is section 12.
  - **Create pull request** pushes the chat's branch and opens its pull request; the outcome is a notification with a button that opens the pull request or the page that finishes it; the flow is section 12.
  - While a commit or a pull request is under way its button is disabled, pulses and says why.
- **Folding behaviour**:
  - The first page load after an install or update, once per browser, leaves only Recommendations open **NEW** (10-12), opening it again if it was folded.
  - A click on the header of a folded section opens it and folds every other section, while a click on an open section folds only that one **CHANGED** (10-11; before, a click toggled its own section alone). Recommendations is apart from this since 11-23: only its own header folds or opens it, alone.
  - A task that starts in the chat on screen opens Task Info and Task Progress and folds the other sections **NEW** (10-09), leaving Recommendations as it is; in the split layout it folds the content pane away too.
- **Machine routing and polling**:
  - Schedule, Machines, Skills, Connectors, Spend and Recommendations show the machine picked in the chat header **NEW** (10-18) and are re-read every 30 s while the page is visible.
  - A re-read replaces the rows in place when its answer arrives. A change of the chat's machine or folder keeps the old rows on screen, out of reach, until the new answer replaces them; "Loading…" shows only in a view that has shown nothing yet **CHANGED** (11-20; before, a folder broadcast from another window or a machine binding that dropped out and came back blanked the views).
- **Sections**:
  - **Task Info**:
    - **Shows**:
      - Tokens, Cost, Steps, Time, Machine, Workdir, Max budget, Date, Base model, Chat id, Task id; Parent task and Agent when they apply.
      - Time is `Running 2m 5s` while the task runs and the time it took (`2m 5s`) once it is over, however it ended **CHANGED** (11-12; it read `Done (2m 5s)`, `Error` or `Stopped` after a live end and `Ready` for a finished task opened from the history list), and `—` when the task's row records no end.
      - The Parallel mode row left on 10-12.
      - The rows are 5px apart, half of the gap before **CHANGED** (11-20).
    - **Actions** **NEW** (11-20): The Chat id, Task id and Parent task values are buttons:
      - A click, Enter or Space inserts the id into the composer at the caret, in place of a selection when there is one.
      - A folded composer unfolds, and below 900 px, where the sidebar is a drawer, the drawer closes first.
      - A click that ends a drag or a double-click that selects the id inserts nothing, so the id can still be copied; an unknown id (`—`) is plain text.
  - **Task Progress** **CHANGED** (10-03; renamed from **Task update** on 11-19; time left 11-20):
    - **Shows**: The /ask agent's answer to what the task has done so far:
      - Cadence (each run capped at $1):
        1. First asked 60 s after the start.
        2. Then asked when the last answer is 10 minutes old, or sooner when that answer estimated the time left: after half of it, 60 s at least (11-20).
      - Read every 5 s while on screen; also for a task on another machine (10-24).
      - **Time left** **NEW** (11-20): A line above the answer words the estimate the answer ended with, such as "About 12 min left (8 to 20 min), done around 10:50 AM" (the range and its measured accuracy are in section 14):
        - The line is rewritten every 15 s from the three finish times of the `taskUpdate` reply, so it counts down between two answers.
        - It reads "Less than a minute left" near the end and "Taking longer than estimated" once the latest time has passed.
        - A time of an hour or more is written as "1 h 20 min", and the finish time is led by its weekday when it falls on another day.
        - An answer without an estimate, or the end of the task, empties the line.
    - **Actions**: Refresh runs it now.
  - **Explorer, Source Control**: See Explorer and Source Control below.
  - **Schedule**:
    - **Shows**: Cron jobs: name, schedule, next or last run, a running or paused badge (section 20).
    - **Actions**:
      - **+** opens an in-place form for a task prompt or a shell command **NEW** (10-12).
      - Edit, run now, pause or resume and delete on each row **NEW** (10-11).
      - Copy.
      - Refresh reloads the jobs.
  - **Machines** **NEW** (10-11):
    - **Shows**: Registered machines: reachability dot, address, a `local` badge (section 15).
    - **Actions**: **+** form (name, address, password); copy, delete; refresh pings again.
  - **Skills** **NEW** (11-20):
    - **Shows**:
      - The skills the agent can use in the working directory the Explorer shows: name, a badge for where the skill comes from, description.
      - After a search, the skills found on skills.sh, the most installed first (the registry's most installed skills for an empty search): name with a link to its registry page, repository and install count.
    - **Actions**:
      - Enter in the search field or its button searches the registry; Clear removes the results.
      - **+** on a found skill has the daemon check it and add it (section 7): the row reads "Checking…", then "Added", "Needs review" with the findings and an **Add anyway** button, or "Blocked" with the findings.
      - A found skill whose name the agent already has reads "Added" and has no **+**.
      - Refresh reloads the list.
  - **Connectors** **CHANGED** (11-20; titled **Apps** before):
    - **Shows**: Messaging and third-party apps with "N of M connected" (section 19).
    - **Actions**: Connect opens a new chat and sends the sign-in task at once; refresh checks the sign-in state again.
  - **Spend**:
    - **Shows**: An all-time total, a heatmap with one cell per day (at least 98 days, weeks as columns) and one bar per model.
    - **Actions**: Arrows page through earlier weeks; refresh reloads the figures.
  - **Recommendations** **NEW** (10-12), **CHANGED** (11-21), **CHANGED** (11-23):
    - **Shows**: A header with a collapse chevron at its top left ("Recommendations") and a card beneath it between `<` ("Previous recommendation") and `>` ("Next recommendation") buttons, pinned to the bottom of the sidebar over the strip of buttons. It advances every 30 s through the twenty tasks the last `/recommendations` run stored (`RECOMMENDATION_COUNT`; section 9), the next card sliding in from the right as the previous one slides out to the left, and waiting while the pointer or keyboard focus is on the card or either step button; while folded the rotation rests, and it starts over when unfolded. The step buttons stand in the panel's side padding (16 px wide on desktop, 32 px on a phone or tablet), as tall as the card, and are hidden while keeping their room when only one suggestion is present. The card is no taller than a third of the window (nor than an equal share of the sidebar), and a longer suggestion scrolls; it has no position button, no refresh button and no status line, and is hidden while there is no suggestion (none stored yet, or the machine's answer is an error).
    - **Actions**: A click on the header folds the card and its step buttons away (leaving only the header over the strip) and a second click brings them back, with the fold state kept in the browser across reloads. Clicking `>` shows the next suggestion (wrapping from last to first) and slides left like the rotation; clicking `<` shows the previous suggestion (wrapping from first to last) and slides the other way (in from the left, out to the right); either click gives the card on show a full 30 s before rotation resumes. A click on the card opens a new chat with the task in the composer, unsent.

### Explorer, Source Control and content tabs

- **Explorer and Source Control**:
  - Both moved from tabs of the left sidebar into the right one on 10-08 **CHANGED**.
  - For a chat bound to another machine, both show that machine's files **NEW** (10-14).
  - **Explorer**:
    - Lazy file tree rooted at the working directory; Add Folder to Explorer... adds more top-level folders, kept in the browser.
    - Its menu has New File, New Folder, Sort (Alphabetical, Created and Modified, ascending and descending, inherited by sub-folders and remembered for the 200 most recent folders) **NEW** (11-19), Cut, Copy, Paste, Rename, Delete, Copy Path, Copy Relative Path, Open to the Side, Find in Folder, Select for Compare and Compare with Selected, Compare Selected for a multi-selection **NEW** (09-30), Set as Working Directory and Remove Folder from Explorer.
  - **Source Control**:
    - Shows:
      - The branch and a count of uncommitted files.
      - The changed files under Merge Changes, Staged Changes and Changes (one foldable block per worktree when the repository has several).
      - A commit graph of up to 24 lanes whose commits unfold to their files.
    - A click on a file row opens a side-by-side diff:
      - A commit's file against the commit's parent.
      - A changed file as the working tree against `HEAD` **CHANGED** (11-07; before, a changed file opened as the plain file and a deleted one not at all).
    - A commit's menu has Open Changes, Checkout (Detached), Create Branch, Create Tag, Cherry Pick, Compare with, Copy Commit Hash and Copy Commit Message.
    - It stages and commits nothing: that is the Git Commit entry.
- **Editors, viewers, browser and terminal**:
  - **Text editor**:
    - A text file opens in Monaco 0.52.2 (loaded from jsDelivr; a read-only plain view when the CDN is unreachable) under a File, Edit, Selection, View and Go menu bar.
    - Ctrl/Cmd+S saves.
    - Closing a tab with unsaved edits asks Save and close, Don't save or Keep editing.
    - A save over a file that changed on disk asks Overwrite or Reload from disk.
    - **NEW** (10-23): Every file tab has a Reload from disk button.
  - **Viewers and diffs**:
    - **Markdown and HTML files**: Open rendered, with an Edit source and Preview toggle; `.html` and `.htm` tabs add an **Open in Browser** button next to Save that opens the file in the OS default browser locally or stages it in daemon memory (up to 8 documents or 64 MiB) at `/view/<token>/<name>` under a `sandbox` CSP **NEW** (11-19).
    - **PDFs**: A PDF opens in the pdf.js viewer (pdfjs-dist 6.3.289: Download link, page box, Ctrl/Cmd+0 fit width, Ctrl/Cmd with `+`/`-` zoom, Home/End, PageUp/PageDown, and Left/Right, which turn the page only while the pages are no wider than the view and pan otherwise).
    - **Images, folders and binaries**: An image opens as a picture, a folder as a listing, and any other binary as a download link.
    - **Diffs**: A diff opens in Monaco's diff editor **NEW** (09-30).
    - **Reports**: A report, that is a Markdown or HTML file that a Write call saved under a `reports` folder by a task or its sub-agents, opens automatically once the top-level task has ended and its worktree is merged (and never when a sub-agent finishes or while still in an unmerged worktree) **CHANGED** (11-19), rendered in a tab with an Edit source and Preview toggle.
  - **Browser tab**:
    - Streams the daemon machine's browser as JPEG frames under an address bar and sends pointer, wheel, keyboard, paste and touch input back (the agent's browser is section 17).
  - **Terminal tab** **NEW** (10-05):
    - Uses xterm.js 6.0.0 on a shell of the daemon's machine, or of the chat's machine, in the chat's folder.
    - Tabs are titled Terminal, Terminal 2 and so on, belong to the client that opened them and re-attach after a reconnect.
    - In bash and zsh, the Up key of a new tab recalls the commands entered in the other tabs (10-09; section 14).

### Settings

- A sheet titled with the product name, the version and the commit hash (11-07).
- **Daemon addresses** (listed above the controls; a daemon in Docker lists none, 10-16, section 14):
  - Addresses, each a link with a Copy URL button:
    - The machine's `ntfy.sh` topic page, where a tunnel's URL is posted (the active URL itself when no topic can be derived).
    - Local (this machine).
    - When other devices can connect, LAN (local network).
  - While the daemon serves a certificate from its own CA, followed by a line on trusting it (`kiss-web --trust-ca`, a `ca.crt` link).
- **Saving and remote-machine behaviour**:
  - Edits to the daemon's settings are saved when the sheet closes, by its close button, a click outside it or Escape: the fields edited since it opened are sent and no others **CHANGED** (10-28; before, the whole form was written back).
  - For a chat on another machine, it loads and saves that machine's settings **NEW** (10-17) and leaves the addresses out; when the chat header's machine changes while the sheet is open, the edits are first saved to the machine they were made for (10-30).
- **Removed settings**:
  - Auto commit and Use worktree left the sheet on 10-09.
  - Classify tasks and Use Jev left on 10-15.
  - Chat in the editor left with the extension on 10-14.
- **Controls**:
  - **Remote password**:
    - **Default**: 20 random letters and digits, drawn by the first `install.sh` run and kept by later ones; empty without that run.
    - **Notes**: Plain text behind an eye toggle, editable, with a button that fills in a new random 20-character password, stored like any edit when the sheet closes **CHANGED** (10-15 to 10-19); while it is empty only this computer can connect (section 14).
  - **Default working directory** **NEW** (11-21):
    - **Default**: None (`default_work_dir` empty; section 6).
    - **Notes**:
      - A drop-down like the chat header's working-directory drop-down ("None", the recent folders and "Choose ...").
      - The page, not the daemon, applies it when a new chat is opened, when a chat is opened from the Chats list, when a chat is put into a project, and when the field is saved while the chat on screen has not run a task yet (a chat that has run a task keeps the folder it is in); a chat in a project with a default folder of its own gets that one instead, and a folder picked in the chat header afterwards overrides it until another chat is opened.
      - Chats of this machine only (a chat bound to a registered remote machine gets none); a plain tab switch or a reload applies none, and a task started without the page (a scheduled job or a messaging channel) runs in the daemon's current working directory.
  - **Max budget per task ($)**: Default 100.
  - **Use web tools**: Default on.
  - **Use persistent memory**: Default on (section 13).
  - **Compaction** **NEW** (11-05): Default on (section 6).
  - **Drive my Chrome** **NEW** (10-28): Default off (section 17); the **?** button next to it pops up the steps that install the Chrome extension **CHANGED** (11-19, they were the checkbox's label before).
  - **Play a sound when a task finishes** **NEW** (11-04; the label was shortened on 11-19): Default on; kept per browser.
  - **Auto submit spoken task**: Default Yes; kept per browser (section 18).
  - **Wake phrase ("Hey Sorcar") sensitivity**: Default 80; slider from 0 to 100; kept per browser.
  - **Tips, Update, Reset Server, Update Models** (buttons):
    - Reset Server asks "Restart the server?" first when a task is running.
    - Update, Reset Server and Update Models act on this daemon and are hidden on another machine's sheet.
  - **API Keys** (folded) **CHANGED** (11-19): 5 fields: Anthropic key and workspace id, OpenAI, Gemini, OpenRouter.
  - **Custom Models** (folded):
    - Name, endpoint, API key and headers boxes with Add.
    - Under them, the models saved in `~/.kiss/MY_MODELS.json` (section 8), each with Edit (which loads it into the boxes, swaps Add for Save and Cancel and may rename it) and a two-step Delete.
    - Closing the sheet over an unsaved model edit asks Discard changes or Keep editing.
  - **Bedrock** **NEW** (10-17) (folded): Bearer token or access key pair, session token, region.
  - **Claude Code** **NEW** (10-17) (folded): CLI status, "Bill my Claude subscription" (on), subscription token (section 8).
  - **Codex** **NEW** (10-18) (folded): CLI status, "Bill my ChatGPT subscription" (on), Sign in with ChatGPT, access token (section 8).

### Sounds, phone layout, appearance, updates and sign-in

- **Sounds** **NEW** (11-04):
  - `victory.mp3` plays when a task of the page finishes, on this machine or another one, and not for a sub-agent; a play that the browser's autoplay rule refuses waits for the next click or key press.
  - A spoken task is acknowledged with `working-on-it.mp3` when it is submitted.
  - Speech from the `talk` tool plays in the page (section 18).
- **Phone layout**:
  - Below 900 px, the two sidebars are drawers.
  - **Four-screen swipe navigation** **NEW** (10-08):
    - On a phone or tablet, the page is four screens in a row:
      1. Chats.
      2. Chat.
      3. Open file.
      4. Task info.
    - A sideways swipe of at least 60 px moves one screen, skipping the file screen when nothing is open.
    - A swipe right on the file screen returns to the chat, on its machine, that the screen was reached from (11-12); over a streamed browser page it does so only when the machine's daemon says that nothing under the finger had anything left to scroll that way (`browserTouch`).
    - The two drawer buttons are hidden there and a sticky notice explains the swipe **NEW** (10-19); it returns on every page load until its "Do not show this again" box is ticked, which that browser remembers (`kiss-swipe-hint-dismissed`).
  - **Composer folding**:
    - The composer's text box folds away while a task runs, and **New chat** unfolds and focuses it (11-06).
- **Appearance**:
  - **Themes and modes**:
    - Light (VS Code Light Modern, the default) and dark (Dark Modern), plus High contrast and Auto mode **NEW** (10-17; Auto is dark from 19:00 to 07:00 on the device clock).
    - All are "..." entries remembered per browser, and a switch also recolours Monaco, code highlighting and terminals.
  - **CSS tokens and typography**:
    - `main.css` defines 90 root tokens (7 font sizes, 11 spacing steps, 9 radii, 5 shadows, 3 durations, a 17-layer z-index scale and the colours).
    - `remote-codex.css` sets 64 custom properties on the page body from the theme palette that the daemon inlines.
    - Text defaults to 13 px (`--theme-font-size`, adjustable from 10 px to 20 px via the Chats sidebar footer buttons across the UI, Monaco editors and widgets, terminals, preview context menus and rendered Markdown previews, **NEW** 11-19).
  - **Auto-hiding scrollbars** **NEW** (11-19):
    - Scrollbars in the sidebars, transcript and menus stay hidden at rest and show a slim pill thumb while the user scrolls (for 1.2 s after the last user scroll, or while hovered or dragged), keeping their gutter reserved so rows do not shift.
- **Updates and refresh**:
  - When the daemon finds a newer release (section 14):
    - A sticky toast offers Update, Update when idle (installed the first time no task is running; Cancel calls it off) and Remind me later (24 hours, in every window).
    - The Update button in Settings carries a badge.
  - **NEW** (10-23): The page reloads itself when it reconnects to a restarted daemon.
  - `sw.js` keeps the page's shell on screen while the connection is down.
- **Sign-in window** **NEW** (10-20):
  - While the brand's sign-in hook (`$KISS_HOME/signin`) reports that nobody is logged in, the daemon makes every window show a dialog with one button per provider.
  - A click opens the provider's page in a new tab; when the sign-in completes, the dialog closes in every window.
  - A paste row takes the address of the final page for a browser on another computer than the daemon.
  - Not now dismisses the dialog.
- **Welcome page**: A new chat shows the product logo in a light or dark variant, "Welcome to KISS Sorcar" and the tagline from `brand.json` ("Your SI assistant. Ask me anything!" in the stock build). The logo is 560px wide **CHANGED** (11-22; 280px before), and narrower in a narrow pane or a short window.

## 17. Browser, web research and sign-in hand-off

### Agent browser, profiles and stealth

- **Agent browser** (`src/kiss/agents/sorcar/web_use_tool.py`, 3,894 lines):
  - `WebUseTool` drives Chromium through Patchright (`patchright>=1.63.0`; stock Playwright is the import fallback) and prefers an installed Google Chrome to the bundled Chromium, which is downloaded on first use when missing (900 s deadline).
  - A page comes back as an accessibility tree: a `Page:`/`URL:` header and interactive elements numbered `[N]`, cut at 50,000 characters.
  - 15 tools (parameters: section 7): `go_to_url` (also `tab:list` and `tab:N`), `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `evaluate_script`, `list_console_messages`, `list_network_requests`, `wait_for`, `attach_to_chrome`, `use_my_chrome`, `show_browser`, `close_browser`.
  - Navigation and driver recovery:
    - A tab the page opens itself (`target="_blank"`) becomes the current one **CHANGED** (11-04).
    - A navigation Chromium aborts with `net::ERR_NETWORK_CHANGED` is issued up to 3 times **CHANGED** (09-30).
    - A call on a Playwright driver that has died raises at once instead of hanging, and the next call relaunches the browser **CHANGED** (10-25).
  - With **Use web tools** unticked in Settings no browser tool is built and `WEB_TOOLS_OFF_NOTE` tells the model that the research rules do not apply.
- **Profiles**:
  - The top-level agent uses the persistent profile `~/.kiss/browser_profile`, so cookies and logins survive tasks and restarts, under a file lock (`<profile>.lock`, 120 s); when a live Chromium of another process holds the profile, numbered siblings `browser_profile_1` to `_99` are tried.
  - Each sub-agent gets a throwaway profile (`kiss_web_profile_*` in the temp directory, deleted when the tool closes), so concurrent children do not contend for the shared one.
  - The tool's own Chromium aborts every request to `accounts.google.com`.
- **Display and fingerprint**:
  - "headless", the default, is a headed Chromium on a private Xvfb display (`1920x1080x24`, one per process) on Linux machines with Xvfb, and real headless elsewhere.
  - A real-headless launch passes Chrome's own `--user-agent` switch without the `HeadlessChrome` token **CHANGED** (10-27; before, an in-page rewrite of `navigator.userAgent` that Cloudflare detected).
  - `AutomationControlled` is disabled in every launch and WebAuthn in unattended ones **CHANGED** (10-01), because a passkey request there opens a native dialog that nobody sees and that swallows every click.
  - The viewport is 1280×900.
- **Human-like input**:
  - The pointer glides along a Bézier curve to a random point 35 to 65% into the element, pauses 40 to 160 ms and holds the button 45 to 140 ms.
  - Text is typed key by key (35 to 115 ms inside a word, 90 to 260 ms at a space, 150 to 450 ms before Enter).
  - Scrolling moves in wheel notches of about 300 px, at most 50 per call.

### Bot-protection pages

- **Bot-protection pages** (`web_stealth.py`, 1,192 lines):
  - 12 kinds of page are recognised from the title, the first 1,500 characters of body text and the script URLs.
  - After `go_to_url`, and since 11-04 after `click`, `type_text` with Enter, `press_key` and a tab switch **CHANGED**, the tool:
    - Waits for such a page to clear while it moves the pointer.
    - Presses the widget a person would press once it has shown for 1.2 s.
    - When the page stays, puts a `Note:` above the tree that names the vendor and the way out (`show_browser()` or another source).
  - Each press gives the page the table's full wait again, counted from the end of the press:

| Page | Wait | What the tool does |
| --- | --- | --- |
| Cloudflare challenge ("Just a moment...") | 12 s | ticks the Turnstile checkbox once |
| Anubis proof-of-work check | 45 s **CHANGED** (10-26; 12 s before) | waits |
| PerimeterX "Press & Hold" | 12 s | holds the button 8 to 10 s, up to twice **NEW** (10-26) |
| AWS WAF token challenge **NEW** (10-31) | 12 s | waits |
| AWS WAF CAPTCHA puzzle **NEW** (10-31) | 3 s | reports it |
| Stack Overflow and Stack Exchange "Human verification" **NEW** (11-04) | 12 s | ticks the reCAPTCHA box once; stops waiting when the picture puzzle opens |
| Reddit network-policy block **NEW** (11-04) | none | reports it; the note says to log in through `show_browser()` |
| Google "unusual traffic" | none | opens the same query on Bing |
| Cloudflare block, Akamai block, Imperva Incapsula block, generic CAPTCHA | 12 s | reports it |

- **Semantic Scholar fallback** **NEW** (11-01):
  - A `semanticscholar.org` page that stays blocked is looked up on Crossref instead when its URL is a search with a non-empty `q` or a paper link with a title slug before the 40-hex id, unless the user can see the browser.
  - `api.crossref.org/works`: 10 works for a search, 5 for a paper, rendered into the page.

### DevTools, user Chrome and streamed Browser tab

- **DevTools-style tools** **NEW** (10-28):
  - `evaluate_script` runs JavaScript in the page and returns the result, a string as it is and anything else as JSON (cut at 20,000 characters).
  - `list_console_messages` and `list_network_requests` return the page's last 200 console entries and 300 requests.
  - `wait_for` waits up to 120 s for visible text to appear or disappear.
  - The `/devtools` SEA (`src/kiss/agents/seas/devtools/devtools_sea.py`, 167 lines; section 9) appends an act-then-verify workflow built on them to the system prompt and switches the browser tools on.
- **The user's own Chrome** **NEW** (10-28), by two routes:
  - `attach_to_chrome(cdp_url)`:
    - Attaches to a Chrome started with `--remote-debugging-port=9222`.
    - The page tools (`go_to_url` to `wait_for` above) then act on the user's tabs until `close_browser()` detaches or `show_browser` moves the page out, both of which leave that Chrome running.
    - A lost attachment is an error rather than a silent return to the tool's own Chromium.
  - `use_my_chrome`:
    - Reaches the everyday Chrome (default profile, Gmail, SSO, passkeys) through the Chrome bridge (`chrome_bridge.py`, 756 lines): a loopback WebSocket server on an ephemeral port that the "KISS Sorcar Bridge" extension (Manifest V3) dials, every 3 s while Chrome is awake and from a 30 s alarm.
    - The extension's two files are installed into `~/.kiss/chrome_extension/`, the folder to pick once under chrome://extensions → Developer mode → Load unpacked, beside `bridge-config.json` (mode 600: the live port and a per-start token, re-read on every connect).
    - Ten actions: `list_tabs`, `open_tab`, `navigate`, `activate_tab`, `close_tab`, `snapshot` (the visible interactive elements, numbered), `click` and `type_text` by snapshot uid, `evaluate`, `screenshot`.
    - The handshake is a mutual SHA-256 challenge-response, so the token never crosses the wire and a peer that does not know it (a web page, a process that occupies the port) can neither drive the browser nor pose as the daemon; the mode of `bridge-config.json` keeps the token from other OS users, not from the user's own processes.
    - Execution and queueing:
      - Page JavaScript runs in the page's own world and goes through `chrome.debugger` only when the page's CSP refuses it.
      - Focus-stealing commands are serialised.
      - Each command has a deadline (30 s by default, 120 s at most), and one whose connection closed or whose deadline passed while it queued does not run.
    - Bridge lifecycle:
      - The daemon serves the bridge from start-up when the Settings checkbox "Drive my Chrome" (`chrome_bridge`, off by default) is ticked (ticking it starts the bridge on save; unticking it leaves a running bridge up until the daemon stops).
      - Otherwise the first `use_my_chrome` call starts it, waits up to 5 s for the extension, and defers to a bridge another KISS process already serves.
    - Tests: `test_chrome_bridge.py` (a scripted extension peer over a real socket), `test_chrome_extension_js.py` (the page-side JavaScript of `snapshot`, `click` and `type_text`, extracted from the shipped `background.js`), `test_web_devtools_tools.py`.
- **`show_browser`**:
  - Under the daemon the page moves into the streamed Browser tab (section 14) on every surface together with its URL, cookies and the origin's `localStorage`; the user clicks and types there while the agent keeps driving the same page over a second CDP connection.
  - Out of a Chrome reached with `attach_to_chrome` only the URL moves: that Chrome's cookies and `localStorage` stay in it.
  - `show_browser(visible=False)` carries the session back and closes the tab; a page still shown when the task ends stays open.
  - That browser has its own persistent profile, `~/.kiss/browser-tab-profile`.
  - Without a daemon the agent's profile reopens in a visible window.

### Web research, sign-in hand-off and recorded use

- **Web research protocol** (`src/kiss/SYSTEM.md`):
  - Google research is the default, skipped only when the agent is fully confident it can do the task without it.
  - At least 10 distinct sites per research session, each visited with `go_to_url` (curl and wget do not count).
  - Notes and a `Websites visited: N/10` counter are kept in `<scratch dir>/information.md` **CHANGED** (10-25; `./tmp/information-{id}.md` before).
  - A real-time question must be looked up with tools and may stop at one authoritative site.
  - When Google blocks the search, the agent opens it in the browser and asks the user to pass the bot check, then turns to other search engines.
- **Sign-in hand-off** (`src/kiss/core/browser_handoff.py`, 505 lines):
  - `open_for_user(url)` puts a sign-in page where the user can act on it, trying in fallback order and returning `"browser_tab"`, `"default_browser"` or `""`:
    1. The Browser tab on every surface first.
    2. Then the machine's default browser.
    3. Else nothing.
  - `browser_handoff_note` turns the answer into the instruction the agent reads.
  - Only `http`, `https` and `file` URLs are opened, a URL opened in the last 120 s is not opened again, and the default-browser step is skipped in a headless environment (`KISS_HEADLESS`, a container, Linux without a display).
  - Callers (section 19):
    - `portal_handoff`, which opens the page and returns that note, in 12 third-party agent SEAs.
    - The device-code and Nextcloud sign-ins of `_device_auth.py`.
    - The WhatsApp and Signal QR pages.
    - The Composio Google link.
    - MCP OAuth.
- **Recorded use**:
  - 13,092 `go_to_url`, 818 `screenshot`, 894 `click`, 322 `get_page_content` and 68 `show_browser` calls are recorded in the task database.
  - 211 `evaluate_script` calls are recorded since the DevTools tools arrived on 10-28.

## 18. Voice

### Wake word, capture and speaker routing

- **Wake word** (`src/kiss/webapp/media/voice.js`, 1,379 lines, with the vendored `vosk.js`):
  - The **Voice trigger** item of the composer's "More actions" menu makes the page listen locally for **"Hey Sorcar"** with the Vosk small English model (`vosk-model-small-en-us-0.15`, cached under `~/.kiss/models` and served to the page at `/voice-model.tar.gz`); the choice survives reloads.
  - Two recognizers hear every audio block:
    - One whose grammar holds only the phrase, spelled "hey sore car", "hey sir car" and "hey sar car" because the name is not in the model's vocabulary.
    - A grammar-free one that confirms on the word "hey" or on closing words within 3 edits of "heysorcar".
  - A wake needs a final result from both within 2.5 s **CHANGED** (10-20: final results only, where a partial followed by a pause used to suffice; 11-08: the second recognizer).
  - The phrase was "Abracadabra Sorcar" and then "Abracadabra" from 10-17 and is "Hey Sorcar" again since 11-08.
- **Capture and transcription**:
  1. After a wake the page waits up to 5 s for speech and gives up when nobody speaks.
  2. It records from the first loud block until 2 s of silence or 30 s of recording, puts the audio from before the wake in front (2 s, plus the time, up to 2.5 s, that the wake waited for its grammar-free confirmation, in whole audio blocks) and sends the lot to the daemon as `voiceTranscribe` (16 kHz mono PCM).
  3. On the daemon (`src/kiss/server/speech.py`, 1,019 lines), `gpt-audio` transcribes it and translates it into English, with a second attempt on 1 s more silence when the reply reads like a refusal **CHANGED** (10-01); each audio request is bounded at 60 s (`KISS_VOICE_AUDIO_TIMEOUT`).
  4. The daemon then looks for the name in the transcript:
     - A word among the first eight that is "sorcar", one of its spellings (`sorkar`, `sarkar`, `sarcar`, `soccer`, `sir car`, `sore car`, `sar car`) or one edit from it is cut together with everything before it.
     - A transcript without one is dropped, so a sound-alike such as "hey, so the car" lights the mic but never submits a task.
     - Only speech reported as non-English, whose translation can lose the wake word, passes on the page's two agreeing recognizers alone **CHANGED** (11-08).
- **Speakers**:
  - A Vosk x-vector model (`vosk-model-spk-0.4`) on the daemon numbers voices in order of first appearance (cosine distance of at most 0.6 from a known speaker's first sample reuses the number).
  - The agent reads `Speaker #N says in the language <tag> that: <text>`.
- **Where the words go**:
  - A transcript belongs to the conversation that was on screen when its capture began and is dropped when that conversation is no longer on screen.
  - It then takes the typed path: it starts a task, is added to the running task, or answers a pending question.
  - With **Auto submit spoken task** set to No it is inserted at the cursor as bare words instead, except an answer to a pending question, which is always sent.

### Settings, failures, spoken replies and diagnostics

- **Settings** (kept in the browser's `localStorage`, followed by its other tabs):
  - `Wake phrase ("Hey Sorcar") sensitivity`, 0 to 100 in steps of 5 with default 80 (higher accepts less confident words).
  - **Auto submit spoken task**, default Yes.
- **Failures**:
  - Loading the speech engine, loading the model and the microphone request are each given 30 s.
  - A microphone that goes away, a recognizer error or a dead speech worker turns the trigger off with the reason on its button.
  - One sticky toast states the remedy for a denied, missing or busy microphone.
- **Spoken replies**:
  - The system prompt tells the agent to answer a spoken message in the speaker's language with `talk(language, text, emotion)` (section 7; 12 named emotions).
  - The text is synthesised once with `gpt-audio-1.5` and the `cedar` voice (`src/kiss/core/speech_synthesis.py`), and the MP3 travels in the `talk` event to every device that has the task's tab open, where clips play one at a time; a client stays silent when synthesis failed, the Web Speech fallback being gone.
  - For a task that a local API client on the daemon's machine started (a `daemon_client` run such as the Python client API of section 5; the `sorcar` CLI runs its agent in its own process and never connects) the daemon plays the clip on that machine's speakers (`src/kiss/server/talk_player.py`: `afplay` on macOS, else `mpg123`, `ffplay` or `mpv`; `KISS_SORCAR_PLAY_CMD` overrides).
  - The synthesis spend is added to the task's.
- **Sounds**:
  - `working-on-it.mp3` plays when a spoken message is submitted.
  - `victory.mp3` **NEW** (11-04), a half-second synthesised flourish rendered by `kiss.scripts.make_victory_sound`, plays when a top-level task ends with `task_done`, whether it succeeded or not and whether it ran on this daemon or on a registered machine (section 15); sub-agents, errors, stops and replayed transcripts are silent.
  - The Settings checkbox "Play a sound when a task finishes", on by default and kept per browser, mutes it; a play the browser's autoplay policy refuses is kept for the next click or keypress.
- **Debug captures** **NEW** (11-07):
  - Every capture the daemon can decode is saved before transcription as a WAV plus a JSON note (the wake flags, then the transcript, its language and whether it was accepted) under `~/.kiss/voice-debug/`, newest 200 kept.
  - In the page, `localStorage.kissVoiceDebug = '1'` logs recognizer results and partials, the microphone track and RMS levels to the console.
- **Removed**: the host-side listener (`voice_wake.py`, `voiceWake.ts`) went with the VS Code extension on 10-14, after the daemon's `voiceWake*` commands on 10-08; voice now lives only in the web app's page.
- **Recorded use**:
  - 119 top-level tasks whose text starts with `Speaker #` (dictation inserted as bare words is not counted) and 272 `talk` calls are recorded, 111 and 208 of them in July.
  - 3 tasks and 7 calls are dated 10-01 or later.

## 19. Messaging and third-party agents

### Agent catalogue and layout

- **Package layout** (`src/kiss/agents/third_party_agents/`):
  - Holds **44 agents**, one folder each (`<name>/<name>_sea.py`; 33,726 lines in all), plus `muse_auth/` (8 files, 4,751 lines), `auth_status.py`, `govee.py` (Govee lights CLI, not one of the 44) and shared helpers (`_channel_agent_utils.py` 2,155 lines, `_channel_cli.py`, `_kiss_web_launcher.py`, `_backend_utils.py`, `_device_auth.py`, `_oauth_apps.py`, `_composio_google.py`, `_google_workspace_utils.py`, `_overleaf_realtime.py`).
  - The agents define 151 sign-in tools and 365 API tools between them; an agent always offers its sign-in tools and adds its API tools once it is authenticated.
  - Entry points and configuration:
    - Each has a console script (`kiss-<name>`; `kiss-gchat` for googlechat, `kiss-ha` for homeassistant).
    - Every one except the hidden `oai` is a slash command `/<name>` (section 9).
    - Per-agent configuration lives under `~/.kiss/third_party_agents/<name>/` (the directory is `brave_search` for brave, `openai_compat` for oai, and `google_calendar`, `google_docs`, `google_drive`, `google_sheets` for gcal, gdocs, gdrive, gsheets).
- **32 messaging and device channels** (233 API tools): bluebubbles, dingtalk, discord, email, feishu, gmail, googlechat, homeassistant, imessage, irc, line, matrix, mattermost, msteams, nextcloud, nostr, ntfy, phone, qq, signal, simplex, slack, sms, synology, telegram, tlon, twitch, webhook, wecom, weixin, whatsapp, zalo.
- **10 service APIs** (128 API tools):
  - brave, firecrawl, gcal, gdocs, gdrive, github, gsheets and notion.
  - **overleaf**: 48 tools over the user's own `overleaf_session2` browser cookie, CSRF token, socket.io file tree and optional Git bridge token.
  - postgres:
    - Read-only unless its configuration says otherwise, with a 60 s statement timeout on every tool except `pg_execute`.
    - `pg_query` returns one complete JSON document with text cells cut at 2,000 characters **CHANGED**, where output was sliced at 8,000 characters until 11-07.
- **2 infrastructure surfaces** (both derive from `BaseSea`, not `ChannelSea`, so channel dispatch leaves them out):
  - `kiss-a2a` (Agent-to-Agent protocol): an agent card at `/.well-known/agent-card.json` and JSON-RPC `message/send`, on `127.0.0.1:18091` by default, 20 messages per `contextId` per hour, every inbound POST request appended to `a2a_audit.jsonl`; tools `a2a_discover`, `a2a_call`, `a2a_get_task`.
  - `kiss-oai`: OpenAI-compatible HTTP server started with `kiss-oai --serve`: `GET /v1/models` and `POST /v1/chat/completions` with a bearer key.

### Dispatch, CLI and gateways

- **Dispatch** **CHANGED**:
  - A channel is a folder whose class derives from `ChannelSea` (`seas/base/base_sea.py`; base classes replaced the per-SEA `settings()` flag on 10-07), and `available_channels()` returns the 42 names.
  - `run_agent(agent="<channel>", task=...)` from any task runs the channel as a sub-agent with a nested tab (section 10); `/<channel> <task>` runs it in the chat itself.
  - Either way it works in `~/.kiss/channel_work` without the worktree tools, classifier, web tools or memory (`WORKER_DEFAULTS`; those keys and `work_dir` are locked against overrides).
  - A dispatched channel takes nothing from the calling task (chat, model, budget share, container, prompt suffixes; `inherit: true` is refused) and receives the task text verbatim.
  - Prompt and tool assembly:
    - The daemon appends `CHANNEL_PREAMBLE` to the system prompt, and the SEA's `system_prompt()` then appends its `channel_system_prompt` (29 agents define one).
    - The SEA's `tools()` adds the agent's tools.
  - Workspace isolation:
    - The `options` key `workspace` (default `default`) names the account: it is published in `KISS_CHANNEL_WORKSPACE` for the run, runs of one workspace overlap and a run of another waits up to 900 s (`channel_workspace.py`).
    - Slack is the only agent that reads it (`~/.kiss/third_party_agents/slack/<workspace>/token.json`, `kiss-slack --list-workspaces`, `--delete-workspace WS`).
- **Command line** **CHANGED**:
  - `kiss-<channel> -t "task"` (or `-f file`; `-m` model, `-b` budget, `--workspace`) runs the agent's SEA file through `kiss.server.sorcar.run(sea_path=...)` on a daemon the CLI starts inside its own process: a loopback WSS endpoint on an ephemeral port, with its endpoint file and tab registry in a private temporary directory.
  - The run therefore needs no `kiss-web` service and stays out of `~/.kiss/tabs.json`.
  - That private daemon listened on a Unix socket until 10-01, and the `tools=` file contract it used was removed on 10-03.
- **Gateways and scheduled delivery**:
  - A channel whose module passes a backend factory to `channel_main()` (25 messaging channels; not gmail, homeassistant, imessage, nostr, tlon, twitch, wecom) supports one-poll ticks `kiss-<channel> --channel=<chat> [--allow-users a,b] [--pairing] [--quiet]` and pairing admin `--approve CODE` / `--list-pending`.
  - Each tick (`ChannelRunner`) runs this pipeline:
    1. Takes a non-blocking lock so overlapping ticks exit.
    2. Polls up to 50 messages from a persisted cursor.
    3. Admits every sender unless `--allow-users` or `--pairing` is given and then only the allow list and the approved users (with `--pairing` an unknown sender is sent a one-time code).
    4. Runs one task per message with the daemon chat resumed per thread.
    5. Records each reply in a delivery ledger: sent with one retry, redelivered on the next tick if that fails.
  - 5 failed ticks in a row pause the channel for 900 s.
  - Slack alone continues a thread from later replies (`poll_thread_messages`, at most 20 threads per tick).
  - The config keys `channel_model_name` and `channel_max_budget` apply when `-m` / `-b` are omitted, and `[SILENT]` / `NO_REPLY` suppress delivery.
  - An always-on gateway is that tick scheduled as a cron command job, and the same channels are cron delivery targets such as `telegram:123`, `slack:eng`, `ntfy` (section 20).

### Sign-in, Muse auth, MCP connectors and recorded use

- **Sign-in**:
  - The Connectors subpanel (titled Apps until 11-20) lists the auth state of the 42 channels as connected, not connected or unknown (`auth_status.py` in a subprocess: 8 workers, 20 s for all probes together, 45 s for the subprocess, result cached 30 s), and a click on an unconnected app starts a "Connect my <app>" task; the same probe feeds `/recommendations` (section 9).
  - Every sign-in page is handed to the user in the Browser tab or the default browser (section 17).
  - Flows in `_device_auth.py` and QR pairing:
    - Device code: GitHub, Teams, Twitch, Matrix.
    - PKCE on `http://localhost:53682/callback`: Slack, Discord.
    - Nextcloud Login Flow v2.
    - Signal links by `signal-cli link` QR and WhatsApp pairs a locally built bridge by QR.
  - OAuth client IDs and bot tokens:
    - The four client IDs embedded in `_oauth_apps.py` are empty, so browser sign-in to GitHub, Teams, Slack and Discord needs `KISS_<PROVIDER>_CLIENT_ID`; Twitch takes the user's own client ID and Matrix registers itself with the homeserver.
    - Discord's browser sign-in reaches only `list_guilds` and `post_message` to the authorised webhook channel: its other 9 API tools and poll mode need a bot token, which `authenticate_discord(bot_token=...)` stores without a client ID.
  - Credential probing and enrolment:
    - Five agents probe a pasted credential with one direct request before storing it (`probe_credential`; Nextcloud keeps an older probe of its own).
    - `enroll_credential` writes the config file and the vault entry under one lock and restores the previous file when the store fails **NEW**.
  - Google agents and Gmail:
    - The six Google agents (gmail, gcal, gdocs, gdrive, gsheets, googlechat) go through Composio Connect Links and `tools.proxy`, so no Google token reaches the process (project keys `ak_…`; consumer `ck_…` keys are rejected).
    - Their auth checks and cron's pre-flight (section 20) ask Composio for the live connection state through `connection_status()` **NEW**.
    - Gmail exposes 13 tools with no permanent delete, and `reply_to_message(draft_only=True)` and `create_draft(thread_id=...)` leave reply drafts inside the thread **NEW**.
- **Muse auth**:
  - A credential-isolation daemon (`muse_auth/`, protocol v8) holding real tokens in a vault under `~/.kiss/muse_auth/` and handing surrogate tokens to 18 connectors (17 SEAs plus Govee) under a host allow-list and a read/write policy (reads allowed, writes need a grant of scope `once`, `session`, `perpetual` or `ttl`).
  - It is the default on Linux only, because it identifies its peers with `SO_PEERCRED`; `KISS_MUSE_AUTH=0` opts out and `KISS_MUSE_AUTH=1` forces it.
  - CLI verbs `status, import, grant, revoke, audit, clear, daemon, stop, export`; daemon spawn serialised behind `spawn.lock`; Microsoft Teams requires it.
  - **CHANGED** since 09-30:
    - Enrolment checks read the vault directory, so building a tool list or probing auth state never starts the daemon.
    - The daemon exits when its state directory is deleted.
    - A `generation` request tells `enroll_credential` whether an interrupted store reached the vault.
- **MCP connectors**:
  - Remote MCP servers sign in through `connect_mcp_server` / `finish_mcp_server_connect` or `python -m kiss.agents.sorcar.mcp_oauth <name> [url] [transport]`:
    - Known names: notion, linear, asana, zoom.
    - CIMD, dynamic client registration, or a pre-registered app given in `KISS_MCP_<NAME>_CLIENT_ID` with an optional `KISS_MCP_<NAME>_CLIENT_SECRET`, the path the module documents for Zoom.
    - Loopback redirect on port 53683; the wait for the redirect expires after 600 s.
    - Tokens in `~/.kiss/mcp_auth/<name>.<digest>.json`, one file per server name and URL since 11-01 **CHANGED**.
  - `connectors/` at the repository root is a catalogue of 16 MCP servers (`catalog.json`: 14 launched locally, 2 remote without sign-in; 7 flagged default, none enabled automatically):
    - `enable.py list|enable|disable` writes `~/.kiss/mcp.json` or, with `--scope project`, `<project>/.kiss/mcp.json`.
    - `verify.py` connects to every configured server.
    - Includes two organisation cost reports over the providers' admin APIs (`bin/anthropic_costs.sh`, `bin/openai_costs.sh`).
  - Section 7 covers the MCP tools themselves.
- **Recorded use**:
  - Runs in the `sea` column: `gmail_sea` 852 runs (822 of them inside cron-launched tasks), `slack_sea` 80, `github_sea` 5, `whatsapp_sea` 3, `gcal_sea` 3, `gdrive_sea` 3 and one each of `telegram_sea`, `overleaf_sea`, `ntfy_sea`, `gsheets_sea`, `gdocs_sea` and `dingtalk_sea`.
  - Tool calls: 1,517 `list_messages`, 789 `check_gmail_auth`, 544 `get_message`, 90 `get_profile`, 85 `post_message`, 84 `check_slack_auth`, 57 `get_thread`, 49 `read_messages`, 47 `reply_to_message` and 23 `send_email`.

## 20. Scheduled automations

### Agent, store and scheduler

- **Cron SEA** (`cron_agent.py`, 3,044 lines):
  - Dispatched with `run_agent(agent="cron", task=...)` or with the built-in `/cron` command **NEW** (10-04; section 9).
  - `CronAgentSea` is a `ChannelSea` whose `settings()` returns `settings | {"work_dir": cron_work_dir()}` (`~/.kiss/cron/work`), whose `system_prompt()` appends the cron dispatch preamble and whose `tools()` adds `cron_job` and `gateway_command`; only that session holds the `cron_job` tool, so the chat agent never edits schedules directly.
  - Bundled SEAs, the Schedule subpanel and the `kiss-cron` console script call the same module.
- **Store**:
  - `~/.kiss/cron/jobs.json`, written atomically with mode `0600` under an exclusive file lock on `~/.kiss/cron/jobs.lock` (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows); edits refuse a store that cannot be parsed instead of overwriting it.
  - Each run gets a scratch dir `~/.kiss/cron/runs/<id>-<rand>`, removed when the run ends unless its task may still be running.
  - Delivered results are appended to `~/.kiss/cron/output/<job_id>.md`.
- **Scheduler**:
  - A thread inside the blocking `kiss-web` lifecycle ticks every 60 s, reschedules due jobs before running them and runs them concurrently, one thread each.
  - `kiss-cron --tick` / `--daemon` run it standalone (command jobs then work alone, prompt jobs still need a reachable daemon).
  - In-process `start_async()` embedders, private daemons and a server built with `scheduler=False` **NEW** (the Terminal-Bench daemon, 10-27) run no scheduler.
  - Schedules: `every 30m`, 5-field cron, one-shot durations, ISO timestamps; cron expressions and offset-less ISO timestamps are evaluated in **America/Los_Angeles** (an explicit offset is kept).
- **No catch-up, no overlap**:
  - A repeating job found more than 10 minutes overdue (`MISSED_RUN_GRACE_SECONDS`) is rescheduled from now without running, so a restart after downtime does not fire every missed job at once; a one-shot still runs, however late **NEW** (10-04).
  - A job is not started while:
    - Its previous run is alive in this process;
    - Another live process runs it (`running_pid`) **NEW** (11-07); or
    - The daemon task of a run that timed out without a confirmed stop, or lost its connection, may still be running (`running_task_id`) **NEW** (10-25).
  - A job with an unreadable schedule or timestamp is disabled, with the reason in `last_summary`, instead of aborting the tick.

### Job kinds, gateways and unattended execution

- **Job kinds**:
  - *prompt* jobs **CHANGED**:
    - Launch the bundled `cron_prompt_sea.py` (81 lines; until 10-04 a SEA file was written per run) through the `run_agent` machinery as a top-level task in a fresh session, with the job's model, budget and timeout as call arguments and `work_dir` as an option only when the job names one.
    - `CronPromptSea` is a `BaseSea` whose `settings()` turns the classifier off and whose `system_prompt()` swaps the full system prompt for `SYSTEM_PROMPT_LITE` plus five scheduled-run rules (10-25).
    - The per-job `use_worktree` / `auto_commit` flags were removed 10-09: a run inside a repository uses its own worktree tools (section 12).
    - A prompt that is a slash command (`/recommendations now`) runs that SEA directly **NEW** (10-12).
    - An `in_chat` job **NEW** (10-12) submits its prompt verbatim as a message typed into a new chat that every surface opens and the user answers in; it works in `~/.kiss/cron/chats/<job_id>` unless it names a `work_dir`.
  - *Command* jobs:
    - Run a shell command with no LLM in the `Bash` tool's shell, in their own process group with `KISS_HOME` set, and deliver the command's stdout, stripped of surrounding whitespace and otherwise unchanged; a non-zero exit is an error reporting the exit code, stdout and stderr.
    - A timeout kills the command's processes, best effort: on POSIX the process group and the descendants that left it (those a `/proc` snapshot taken before the kill finds), on Windows the Job Object the command was enrolled in at start or, failing that, its `taskkill /F /T` tree.
  - Default timeouts are 3,600 s for a prompt job and 600 s for a command.
- **Gateways**:
  - A channel gateway is a command job whose command `gateway_command` builds (`kiss-telegram --channel=-100123 --pairing --quiet`).
  - `create` refuses a prompt that describes a gateway tick and stores a gateway command whose `deliver` names a channel target with `none` instead (the default `local` stays, and jobs already stored are normalised the same way when the store is loaded), so a tick never posts status into the chat it serves **NEW** (10-02; section 19).
- **Unattended runs**:
  - A prompt job's task text is assembled in order:
    1. A preamble (nobody can answer questions, no scheduling during the run, the final summary is delivered verbatim, exactly `[SILENT]` when there is nothing to report).
    2. Then a run-context block **NEW** (10-25) giving the time now in Pacific time and UTC, the schedule, the start of the window the run covers, the previous attempt with its status, and the last 1,500 characters of its stored result (`last_summary`, which keeps only a result's first 4,000 characters).
    3. Then the prompt.
  - Watermark:
    - The window starts at the watermark `covered_until`: the job's creation, then the start of the last run that ended `silent`, or `ok` with its result at every target (until 11-01 any `ok` run counted).
    - An `error` run never moves it, even when the error reached every target.
    - A job stored before the watermark existed gets a window start one schedule interval back until a run sets the watermark, or none when its run times give no interval.
  - `ask_user_question` returns an error in such a run and in its sub-tasks instead of blocking.
  - Before dispatch each Google connector the prompt names (Gmail, Calendar, Docs, Drive, Sheets, Chat) has its `connection_status()` checked **NEW** (10-25): a status other than `ACTIVE` or `UNKNOWN` fails the run with a fixed message; `in_chat` jobs and prompts that ask to authenticate or reconnect are exempt.
- **Results and delivery**:
  - A run ends `ok`, `silent` (a successful run whose result is empty or, for prompt jobs only, is `[SILENT]` or `NO_REPLY`; nothing is delivered or logged) or `error` (delivered like a result).
  - `deliver` accepts a comma list of `<channel>[:<chat>]` targets, `local` (the default) or `none`, both meaning the output log only; `until_delivered` disables the job once a successful, non-silent result has reached every target.
  - Repeated failure backoff:
    - A failure whose first non-blank line repeats the previous failed result's is not delivered again and pushes the next run at least 1 h out, doubling with each repeat up to 24 h **NEW** (10-25).
    - When the previous failure's delivery to a target failed, the repeat is delivered again and nothing backs off (11-07).

### Actions, seeded jobs, UI and recorded use

- **Tool actions**:
  - Actions: `create` (refuses a duplicate: the same prompt or command, work dir, schedule and delivery targets as a scheduled or paused job), `ensure` (idempotent by name; resumes a paused job), `list`, `remove`, `pause`, `resume`, `run_now` (refused while the job is running).
  - A job takes `name`, `schedule`, exactly one of `prompt` or `command`, and optionally `deliver`, `model_name`, `max_budget`, `until_delivered`, `work_dir`, `timeout` and `in_chat`.
  - `kiss-cron` offers `--list`, `--create`, `--remove`, `--pause`, `--resume` and `--run` from the shell.
- **Default jobs** **NEW** (10-12):
  - On start the daemon seeds `DEFAULT_JOBS` once per home (recorded in `~/.kiss/cron/default-jobs-seeded.json`):
    - `/recommendations now` every six hours (`0 */6 * * *`, $2, 900 s, `deliver` `none`).
    - The `in_chat` `/good_morning now` at 08:00 Pacific (`0 8 * * *`, $0.50, 300 s).
  - Section 9 describes both SEAs.
  - A default the user removed, paused or edited is never re-created or reset.
- **Home hand-over** **NEW** (10-26):
  - Each home has its own store and only the running daemon's store ticks.
  - Before seeding, `adopt_previous_home_jobs()` records the daemon's home in `~/.kiss-web-home` and, when the previous daemon ran from another home (a white-label install, section 21), copies the jobs whose names this store lacks:
    - A repeating job gets a fresh next run and a reset failure streak.
    - A one-shot keeps its deadline.
    - A finished job stays finished.
  - Only a home that is a direct child of the user's home directory takes part: a `KISS_HOME` anywhere else records no marker and adopts nothing.
- **Auto-scheduled jobs**:
  - Picking `autorouter` or `orautorouter` in the model picker ensures "Weekly rsi7d: autorouter evidence and prompt (Sat 1am PT)" (`0 1 * * 6`, `claude-fable-5-2-max` when runnable, since 11-09 **CHANGED**, $25 + $5 relay, 2 h nested run inside a 2 h 10 min job).
  - `/git_extract_knowledge` registers "git-knowledge daily update: <memory slug>" at 04:00 PT (`claude-fable-5-1`, $25 + $2 relay, 4 h) and replaces an outdated job for the same repository rather than adding a second.
- **Schedule subpanel and history**:
  - `getCronJobs` returns the rows of `cron_jobs_report()` (enabled jobs first, soonest next run first, each with a `running` flag).
  - `cronAction` **NEW** (10-11; `create` 10-12) applies `create`, `update`, `remove`, `pause`, `resume` or `run_now` through `apply_job_action()` and answers with the fresh list.
  - The subpanel shows next-run times in Pacific time, per-job edit, run-now, enable/disable, copy and delete buttons and a `+` form in its header (section 16); with a machine picked in the chat header both commands go to that machine's daemon (section 15).
  - Tasks the scheduler launched and their sub-tasks carry `is_cron` in `task_history` **NEW** (10-17; older rows are backfilled), and the history list hides them unless its Cron chip, off by default, is on.
- **Recorded use**:
  - On the development machine 13 jobs are configured (10 prompt, 3 command; 12 enabled): daily README update, test fix, cost-calculation audit, model-catalogue update, knowledge refresh and browser bot-protection re-check, two weekly rsi7d jobs, the two default jobs, a 4-hourly sync of the task database and memories to a shared machine, a nightly work index, and a finished one-shot audit.
  - The scheduler launched 1,506 of the 9,685 recorded top-level tasks (`is_cron = 1`; $4,398 of spend), 1,392 of them since 09-30.
  - 1,149 top-level tasks mention cron, 1,020 of them launched by the scheduler, whose prompt preamble names it; 131 `cron_job` calls are recorded.

## 21. Installation, deployment, Docker and release

### Local installer and standalone app

- **One-line install**:
  - `curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash` runs the bootstrap `scripts/install.sh` (306 lines):
    1. Takes the update lock.
    2. Installs git when missing.
    3. Clones `ksenxx/kiss_ai` into `~/.kiss/kiss_ai` (a ZIP of `main` when git is unusable) or fast-forwards the existing clone (when the pull fails, local edits are stashed and the clone is reset to its upstream).
    4. Runs that clone's `install.sh` from the directory the command was typed in.
  - `./install.sh` in a checkout installs that checkout as it stands.
- **`install.sh`** (110,026 bytes, 2,380 lines) **CHANGED**:
  - Since 10-14, the single installer of the app itself (the `kiss-web` daemon, its web app and the `sorcar` CLI) for macOS and Linux, and since 11-24 for Windows too (run in Git Bash), on `x86_64`, `aarch64` and `arm64`.
  - It asks nothing and needs no administrator password, compiler, Homebrew or git (11-24): what a machine lacks (uv, Python, Chromium, cloudflared) is downloaded into the home directory. The Homebrew step and its `[Y/n]` question were removed; on macOS cloudflared now comes from its GitHub release, as on Linux.
  - Its former steps (git, Node, VS Code CLI, extension build, extension install) went with the VS Code extension, removed 10-14.
  - A re-run updates the app in place.
  - **Five steps**:
    1. `[1/5]`:
       - Copy the tracked files of `src/kiss` (no tests; of the web app only `media/`) with `pyproject.toml`, `uv.lock`, `README.md` and `LICENSE` into `~/.kiss/kiss`.
       - Write the checkout's short commit into that copy's `core` package as `_commit.txt` (a file written at install time, not tracked in the repository; Settings shows it beside the version, 11-07).
       - Apply the `.brand/` overlay (up to 7 media files).
    2. `[2/5]`: Build the Python environment with `uv sync` (uv is fetched from astral.sh when missing, with no version pin).
    3. `[3/5]`: Byte-compile the library and delete its sources, keeping 8 patterns that are run as scripts, read as text or shipped as source to another machine (the bundled SEAs, `*_sea.py`, `cron_agent.py`, `_version.py`, the database-sync and memory-sync modules).
    4. `[4/5]`: Install the `~/.local/bin/sorcar` wrapper, the `rsorcar` and `sorcar-docker` launchers, Playwright's Chromium and cloudflared:
       - The installed `sorcar` sets `KISS_WORKDIR` to the current directory and runs the `sorcar` console script (`kiss.agents.sorcar.sorcar_agent:main`) from the app directory, whereas the checkout's own 6-line `sorcar` script runs `kiss.agents.sorcar.worktree_sorcar_agent` through `uv run` (section 5).
       - The two launchers are wrappers that run the checkout's `rsorcar` and `sorcar-docker`, so they stay bound to the checkout the installer was run from.
    5. `[5/5]`:
       - Refresh `~/.kiss/MODEL_INFO.json`.
       - Run the executables of `~/.kiss/post-install.d/` in name order with `KISS_HOME` and `KISS_APP_DIR` set (a failing hook is a warning) **NEW**.
       - Set or keep the remote-access password, install into other brand homes and start the daemon.
  - **Process safety**:
    - Started from inside the daemon's own systemd service (its Terminal tab, the Update button, a task's shell), the script first moves itself into a transient scope with `systemd-run --user --scope`, so stopping `kiss-web.service` cannot kill the installer **NEW**.
    - A non-interactive run re-executes in a new session (`setsid`).
    - An `flock` on `~/.kiss/.update.lock` makes a second installer exit at once; the kernel drops the lock when its holder dies, so a dead run never blocks a retry.
    - One stray `SIGINT` or `SIGTERM` is ignored and a second within 3 s aborts.
    - When the terminal closes (`SIGHUP`), the run carries on with its output in the log; the password is then not printed, and Settings shows it.
  - **Daemon service**:
    - `kiss-web` is registered as the user's service (the LaunchAgent `com.kiss.web-server` on macOS, the systemd user unit `kiss-web.service` with lingering on Linux, a detached process on a Linux without a systemd user manager, a logon entry under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` on Windows, written by `src/kiss/server/windows_install.py`) and serves the directory the installer was run from (the home directory when that is inside `~/.kiss`).
    - The restart is skipped when the running daemon already serves this exact runtime (a fingerprint of the launcher, the work dir and the newest modification time among the library's `.py` and `.pyc` files, the bytecode being written afresh by every full install), which keeps its tunnel URL.
    - The restart never happens while the daemon reports tasks in flight.
  - **Deferred restart** **NEW** (10-30):
    - Handoff sequence (`deferred_restart.py`):
      1. With tasks in flight, the installer writes `{installer, home, workdir}` to the file the busy daemon names, `~/.kiss/.kiss-web.restart-pending`.
      2. The daemon looks for it every 5 s and, once no task is in flight and no installer it started is still running, raises the run-admission barrier and runs the recorded `install.sh --restart-only --non-interactive`, which installs nothing and only restarts the daemon onto the app already on disk.
    - The record is taken away as it is claimed, so a broken one is not retried.
    - A task can therefore run `./install.sh`, and the daemon it runs under moves to the new build once the task is done.
    - From a checkout, `scripts/reinstall-kiss-web-when-idle.sh` re-runs the whole installer as a detached one-shot job once the daemon has been idle for 3 probes in a row (4 h deadline) **NEW**.
  - **Remote-access password** **NEW** (10-15):
    - The first install draws 20 random letters and digits and stores them as `remote_password` in `~/.kiss/config.json` (`python -m kiss.core.remote_password`); every later run keeps the stored one, so an update never invalidates the password other devices know.
    - It is stored as typed, not hashed: the file's forced `0600` mode protects it, and the `remote_password_hash` key of the hashed scheme of 10-15, replaced the same day, is purged on every write.
    - The installer prints the password in a box of `*` at the end of the install, on the terminal only, never in the log.
    - `--rotate` draws a new one and `--show` prints the stored one; Settings shows it in an editable field with a button that fills in a new random one (section 16).
    - On every check, the daemon stats the file and reads it again only when its inode, modification time or size has changed, so a change is in force without a restart (section 14).
  - **Brand homes** **NEW**:
    - The state directory is a brand property (`home_dir` in `brand.json`, 10-03; `KISS_HOME` overrides it), and the app directory is named after it (`~/.kiss/kiss` for the stock brand).
    - On a machine that also holds a white-label brand's home (a `~/.<name>` with an executable `post-install.d` hook, or the home the registered service runs from), the installer repeats itself into that home, each home keeping its own app copy, model catalogue, hooks and password, and then starts the brand's daemon rather than the stock one.
    - A hook that is an `exec <script>` stub inside a git checkout has that checkout rebased onto its upstream first; if the rebase cannot complete, the checkout is reset to the upstream and the local commits stay on a `brand-update-backup-<time>` branch.
    - Executables in `~/.kiss/services.d/` run for as long as the daemon does (`services.py`, 10-29): each in its own process group, output in `~/.kiss/logs/services/<name>.log`, restarted after 1 s doubling to 60 s.
  - **Failures and notices** **NEW** (10-16):
    - Every warning and failure of a run goes to `~/.kiss/.install-problems`, which the web app shows as a sticky error notification for up to 7 days or until a clean run (`install_problems.py`).
    - On a machine that still has the retired extension:
      - The installer uninstalls `ksenxx.kiss-sorcar` through the `code`, `code-insiders`, `cursor` and `code-server` CLIs (`KISS_CODE_CLI` names the one to use) and leaves `VSCODE-EXTENSION-REMOVED.txt`.
      - For 7 days, the web app shows a "VS Code extension removed" notification with a "Do not show this again" checkbox (`vscode_retired.py`).
  - **Opening the app**:
    1. The installer waits up to 120 s (`KISS_WEBAPP_WAIT_SECS`; 900 s before) for the daemon's `remote-url.json`.
    2. It opens `http://127.0.0.1:8787` with `kiss-app` and shows the launcher's folder in the file manager; without a usable browser it falls back to the default browser, and on a remote machine (an SSH session, or Linux without a display) it prints the tunnel URL instead.
  - **Controls**:
    - `--non-interactive` (or `KISS_NONINTERACTIVE`): for an automated caller; the run detaches from the terminal's signals. No run asks a question.
    - `--restart-only`.
    - `KISS_SKIP_LAUNCH` (no daemon start, no window), `KISS_NO_REVEAL` (the launcher's folder is not shown), `KISS_HOME`, `KISS_WEB_PORT` (8787) and `KISS_CODE_CLI`.
    - Timeouts: `KISS_WEB_START_TIMEOUT` (90 s), `KISS_WEB_DRAIN_TIMEOUT` (30 s) and `KISS_HEARTBEAT_INTERVAL` (15 s).
    - The log is `~/.kiss/install.log`.
- **Installers for a machine with nothing set up** **NEW** (11-24):
  - Both unpack the same payload (`scripts/installer-payload.sh`: the sources `install.sh` installs, `install.sh` itself and the scripts it needs) into `~/.kiss/kiss_pkg` and run `install.sh --non-interactive` there.
  - **Mac package** `installers/KISS-Sorcar.pkg` (`scripts/build-macos-pkg.sh`): installs for the current user without Apple's developer tools.
  - **Windows setup program** `installers/KISS-Sorcar-Setup.exe` (`scripts/build-windows-exe.sh`, NSIS; builds on macOS, Linux and Windows):
    - Installs for the user who runs it, without administrator rights, on 64-bit Windows 10 and later; `/S` installs without a window.
    - `scripts/windows-exe/setup.ps1` finds Git Bash or downloads the portable Git for Windows into the profile, then runs `install.sh`.
    - Registers an uninstaller (Settings > Apps) that removes the app, the Start Menu shortcut, the logon start and the CLI and keeps the settings and history in `~/.kiss`.
    - It is not signed: Windows SmartScreen asks before running a downloaded copy.
  - `scripts/release.sh` rebuilds both from the released tree and commits them with the version bump.
- **`kiss-app`** **NEW** (10-16; `src/kiss/server/standalone_app.py`, 1,486 lines):
  - Opens the web app as a standalone window, Chromium's app mode (`--app=<URL>`) under its own profile `~/.kiss/webapp-profile`, choosing the browser in priority order:
    1. The browser `$KISS_APP_BROWSER` names (once set it is the only candidate: when it does not resolve, no window opens and the command exits with status 1).
    2. Without it, the first of the system's Chrome, Chromium, Edge or Brave, and Playwright's Chromium.
  - `--install-launcher` registers a brand-named launcher whose icon the running window carries too:
    - Linux: a `.desktop` entry with `StartupWMClass` set to the window's Wayland app id.
    - macOS: a clone `.app` bundle in `~/Applications` holding APFS-cloned copies of the browser's binary and frameworks, the binary re-signed ad hoc under the brand's identifier, because Chromium's sandboxed helper processes read only the bundle they run from.
    - Windows (11-24): a Start Menu shortcut with the brand's icon that runs `kiss-app --home <state directory>` without a console window.
  - While the daemon is not answering yet, the window opens on a packaged waiting page (`appStart.html`) that polls it every half second.
  - On macOS, a launch while the app is open brings its window to the front.
  - The other flags are `--url`, `--wait` (120 s), `--foreground`, `--print-command`, `--uninstall-launcher` and `--reveal-launcher`.
- **Updates**:
  - The daemon asks PyPI for a newer release at start and every hour, and the toast offers:
    - **Update**: runs the bootstrap of `~/.kiss/kiss_ai` (it syncs the clone, then installs) detached and non-interactive, output in `~/.kiss/update.log` (the curl one-liner when the clone is gone; `$KISS_UPDATE_BOOTSTRAP_URL` overrides its URL), one run at a time and with the run-admission barrier up, so no new task starts under the installer.
    - **Update when idle**: re-checks every 5 s and starts the update in the same atomic step that finds no task running.
    - **Remind me later**: a 24 h snooze that a still newer release breaks.
  - An open page reloads itself once it reconnects to a daemon whose per-process token differs (`GET /api/daemon`, 10-23) **NEW**.

### Remote deployment and Docker

- **`rsorcar user@host`** (60,557 bytes, 1,019 lines):
  - Deploys the checkout to a remote Linux machine and publishes its web app in nine numbered steps plus six lettered sub-steps (1a to 1c, 4b to 4d), using 12 helper scripts under `scripts/` and 11 Python modules that it checks are present before it starts.
  - Nothing is installed on the local machine any more (the closing `install.sh` run there was removed 10-20), but the two-way syncs of steps 3 and 4b to 4d change both machines.
  - **Deployment pipeline**:
    1. **Steps 1 and 1a to 1c**:
       - Runs an ssh check.
       - Installs the missing ones of `git`, `curl`, `tar`, `python3` and the ssh client through the host's package manager.
       - Renames a `sorcar.db` from before 2026.10.2 to `history.db`, leaving the old name as a symlink **NEW**.
       - Checks room under the remote home for the local task database plus `SORCAR_DISK_HEADROOM_GB` (3) GiB, with `scripts/move-home-to-disk.sh` to move the home onto a larger disk.
    2. **Step 2** **CHANGED** (10-12):
       - Private ssh keys stay home by default and the ssh agent is forwarded (`ssh -A`) for the sync.
       - `SORCAR_SKIP_SSH_COPY=0` copies `~/.ssh` (never `authorized_keys`) and moves the files it replaces to `~/.kiss/ssh-replaced-<time>/`.
    3. **Step 3**: Two-way repository sync through `origin`:
       - Uncommitted work is committed first, a diverged branch is merged, and nothing is force-pushed or deleted.
       - The branch deployed is the main checkout's (`SORCAR_GIT_BRANCH` overrides it), never a `kiss/wt-*` branch (`main` instead).
    4. **Steps 4 and 4b to 4d**:
       - API keys go only to a remote that has none, with a probe that fails closed.
       - Runs two-way `history.db` merge, memory sync with tombstones and clock-skew tolerance, and `AUTOROUTER.md` and `MODEL_DECISIONS.md` merged both ways.
       - A failure of 4b, 4c or 4d is a warning, not a failed deploy.
       - **The `history.db` merge** **CHANGED**:
         - Sends only the rows the other side lacks while both web apps keep running.
         - Since 10-20, a task still in flight is held back until a later sync, and deleted tasks and chats travel as deletions (`deleted_tasks`, `deleted_chats`) instead of coming back from the other machine: the receiving database drops the deleted tasks' events and rows, except those of a task still running there, which wait for a later sync, and of a task row newer than its tombstone (section 15).
         - Since 10-17, the two machines may be on different releases, where the merge used to refuse: a `task_history` or `events` column one database lacks is added with the other's declaration, and `is_cron` is worked out for rows written without it, by the same `history_columns.py` the web app migrates with.
         - `chat_summaries` is recomputed on the receiving side, never copied.
         - Projects never travel (11-21): the `projects` table and the chats' `project_id` are taken out of a whole-database snapshot before it leaves the machine (`scripts/sync-task-db.sh`), because a project can hold API keys.
    5. **Steps 5 and 6**:
       - The remote runs `install.sh` (`KISS_SKIP_LAUNCH=1`, `--non-interactive`), then `uv sync` in the checkout, and `kiss-web` becomes a lingering systemd user service run from the checkout's `.venv` with `--workdir` set to the checkout.
       - A running task no longer stops the deploy **CHANGED** (10-20): while one runs on the remote (its newest event within `SORCAR_LIVE_TASK_WINDOW`, 300 s) only the restart is skipped, and `SORCAR_FORCE_RESTART=1` restarts anyway.
    6. **Steps 7 to 9**:
       - Configures the web app password (this machine's by default, `SORCAR_PASSWORD` overrides it), a Cloudflare tunnel and a machine-unique ntfy topic.
       - Sends GitHub credentials over stdin (`gh auth login --with-token`), with the remote checkout's `git@github.com:` URLs rewritten to `https` when the keys stayed home.
       - Verifies the public URL from this machine and opens it in the browser.
  - Fourteen `SORCAR_*` environment overrides (thirteen on 09-30; `SORCAR_SKIP_SSH_COPY` is the new one).
  - **`/install` command** **NEW** (11-04; section 15):
    - Runs `./install.sh --non-interactive` over ssh, concurrently, on every machine registered with an ssh login, or on the machines it names, in that machine's clone of the private `kiss.git` (`~/kiss`, else another direct subdirectory of the home, matched by its `origin` URL).
    - A machine registered by URL or without such a clone (a public `kiss_ai` clone does not count) is reported as failed.
- **Docker** **CHANGED** (10-14):
  - `sorcar-docker [PORT] [--rebuild]` builds a plain `debian:bookworm-slim` image (`Dockerfile`, uv **0.12.19** baked in) in place of the `codercom/code-server` image that served the extension on port 8080.
  - The entrypoint `scripts/docker-startup.sh`:
    1. Clones the repository (the private `kiss.git` with `GH_TOKEN`, else the public `kiss_ai.git`).
    2. Runs `install.sh`.
    3. Sets the web app password (`KISS_WEB_PASSWORD`, else the stored one, else a generated one printed in `docker logs`).
    4. Runs `kiss-web` in the foreground on port 8787 (host port `PORT`, 8787 by default).
  - State persists in the `kiss-sorcar-web-home` volume and 13 variables are forwarded (`GH_TOKEN`, `KISS_WEB_PASSWORD`, the Anthropic key and workspace id, the OpenAI, Gemini and OpenRouter keys, five AWS variables, `CLAUDE_CODE_OAUTH_TOKEN`).
  - The launcher probes `https://localhost:PORT`, whose self-signed certificate is accepted once or installed from `/ca.crt`, in a loop of up to 900 rounds that sleeps 1 s after each failed probe, then once more after the loop whether it succeeded or ran out (901 requests at most); a probe has no time limit of its own, so 900 s is not a hard deadline.
  - Inside tasks, `docker_image=` runs the shell and file tools in a container with the work dir mounted at its host path (section 7).

### Release, packaging, mobile and platform support

- **Release** (`scripts/release.sh`, 1,030 lines, 11 steps; 13 on 09-30) **CHANGED**:
  - **Release sequence**:
    1. Stash.
    2. Purge the 18 paths of `scripts/exclude.json` from every branch and tag of the public history.
    3. Bump the version (`YEAR.MONTH.N`) in `_version.py` and `README.md` (`SYSTEM.md` is no longer edited: since 10-18 the prompt renders the version through the `VERSION` placeholder).
    4. Commit and push to `origin`.
    5. Push one `Release X.Y.Z` commit (origin's tree minus the excluded paths, parented on the previous release commit) and its tag to `ksenxx/kiss_ai`.
    6. GitHub release.
    7. PyPI publish (`uv build`, a 100 MiB per-file check, `uv publish`).
    8. Local `install.sh`.
    9. Restore the stash.
  - Since 10-17, the public `main` holds one release commit per release and no development commits (before, it received the full development history with the excluded paths filtered out, plus a release tip commit); a `main` of any other shape is replaced in one atomic push under a lease per ref, and every earlier release tag is re-pointed to its release commit.
  - The VSIX build, the `package.json` bump and the Marketplace publish were removed 10-14.
- **Package** (`pyproject.toml`):
  - `kiss-agent-framework` on PyPI, `requires-python >=3.13`, with version read from `src/kiss/core/_version.py`.
  - 29 dependencies with one exact pin (`vosk==0.3.44`) and `cryptography` held below 49 on Intel macOS, plus 14 `constraint-dependencies` floors.
  - 53 console scripts (among them `sorcar`, `kiss-web`, `kiss-app` and `check`).
  - The wheel's only packages are `src/kiss` and `projects/swedefend`, and the sdist ships those two trees plus `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore` and a generated `PKG-INFO`.
- **iPhone Home Screen icon** **NEW** (11-03):
  - `python -m kiss.scripts.webclip` (361 lines, 8 options) writes an iOS Web Clip profile (`.mobileconfig`, payload `com.apple.webClip.managed` with `FullScreen`) that opens the web app without browser controls.
  - Its options: `--url` is the address to open (default: the daemon's tunnel URL, else its first LAN URL, from `remote-url.json`); `--label`, `--icon` and `--icon-background` replace the brand's name and icon (a PNG, JPEG or GIF is centre-cropped and flattened onto that colour as an opaque 400x400 PNG); `--organization` names the profile's organization; `--out` is the file written (default: the label plus `.mobileconfig`); `--tls-dir` is the signing CA's directory (default: `$KISS_HOME/tls`); `--unsigned` writes the plain XML profile.
  - The profile is signed with a certificate issued by the daemon's local CA (`profile-signer.pem` and `profile-signer-key.pem`, kept beside the CA), so a phone that installed `/ca.crt` shows it as Verified.
  - The daemon serves the same profile for its tunnel URL, else its LAN URL, at `GET /webclip.mobileconfig` (the "..." menu's "iPhone Home Screen" item).
  - On an iPhone with rounded display corners that the page knows (the Face ID models from the iPhone X on, told apart by screen size and pixel ratio), held upright in the Home Screen web app, the composer keeps one 8px gap to the left, right and bottom edge of the screen and rounds its bottom corners by the display's corner radius less that gap so the two curves run parallel, keeping what is in the composer above the home indicator; nothing changes in a Safari tab, with the phone held sideways, or on a phone not in the list (`media/displayCorners.js`, inlined in the page head, and the `.display-corners` rules of `media/remote-codex.css`) **NEW** (11-21).
- **Windows**:
  - Windows is installed by the setup program `installers/KISS-Sorcar-Setup.exe` or by `install.sh` in Git Bash (11-24, section 21 "Local installer and standalone app"); from 10-14, when the extension that installed MinGit and uv there was removed, until then there was no Windows installer.
  - `src/kiss/server/windows_install.py` holds what the installer needs there: the logon start, starting the daemon hidden in a fresh logon environment, stopping the daemon of one app directory, the user `PATH` entry and the uninstall.
  - Fixes from the cross-platform test runs landed through 11-24 (section 22).

## 22. Developer tooling and tests

### Verification, impacted-test selection and reporting

- **`uv run check`** (`src/kiss/scripts/check.py`):
  - **Stages**:
    1. Starts by deleting `dist`, `build`, `*.egg-info`, `.claude`, `artifacts` and `uv_cache` at the repository root; `--no-clean` skips that step and `--clean-only` stops after it.
    2. Runs `uv sync`, `generate-api-docs`, `sea lint` and `sea docs` **NEW** (10-05), `python -m compileall -q src/`, `ruff check src/` and `mypy src/`.
    3. `--full` adds `pyright src/` and, when `node_modules` exists under `src/kiss/webapp` and `npm` is on `PATH`, the web app's `npm run lint` (eslint over `media/**/*.js`, stylelint over the CSS, htmlhint over the HTML) **CHANGED** (10-14: the VS Code extension's `npm run typecheck`, its lint and its version sync were removed with the extension).
  - Every stage runs even after an earlier one failed, except that a failed `uv sync` ends the run, and the error lines of each failed stage are repeated at the end of the output, at most 60 per stage.
  - `--read-only` **NEW** (11-11): leaves out the cleaning, `uv sync` and the two generated docs and runs every stage with `uv run --no-sync`, so that it can run beside the tests.
  - `--fix` **NEW** (10-25):
    - First runs `ruff check --fix src/` (lint fixes only, no formatter), prints one status line per stage and only the residual errors, and stores the verdict in `<OS temp dir>/kiss-tmp/check-cache-<12-hex-of-root>.json` (skipped when `kiss-tmp` is not owned by the current user) **CHANGED** (11-18; `tmp/check-cache.json` before), keyed by `--full` and by a SHA-256 fingerprint of the path, size and modification time of every source file under `src/` and of `pyproject.toml`, `uv.lock` and `pyrightconfig.json`.
    - A repeat run on an unchanged tree still does the clean-up (unless `--no-clean`) and the auto-fix, then prints the stored verdict and runs none of the stages.
  - Inside a task's worktree **NEW** (11-02), the stages use the main checkout's `.venv` with the worktree's `src/` and `projects/` first on `PYTHONPATH`:
    - Dependencies are synced with `uv sync --inexact --no-install-project`.
    - The other stages run under `uv run --no-sync`, and the project's console scripts run as `python -m <module>`, so the shared editable install is never re-pointed at a worktree.
  - **Rules**:
    - ruff selects `E`, `F`, `W`, `I`, `N` and `UP` at 100 columns for Python 3.13.
    - mypy sets `disallow_untyped_defs` everywhere but `kiss.tests` and `kiss.viz_trajectory`.
    - pyright reads `pyrightconfig.json`.
- **`uv run findings-report`** **NEW** (11-11; `src/kiss/scripts/findings_report.py`):
  - Builds the findings table of a final report from the files fixers and reviewers append to, one JSON object per finding on one line (`file`, `line`, `kind`, `status`, `summary`, `test`).
  - Prints the counts by status and kind and one row per finding, open ones first, as an HTML fragment (`-o` writes it to a file).
  - Exits 1 naming every malformed line without writing a report.
- **`uv run impacted-tests`** **NEW** (10-25; `src/kiss/scripts/impacted_tests.py`):
  - Prints the test files within two import hops (`--depth N`; `0` removes the bound) of the changed Python files under `src/`:
    1. Parses the `kiss` imports of every source file with `ast`.
    2. Walks the reverse import graph from the changed modules (`git diff --name-only HEAD` plus untracked files, or the paths given) and keeps the `test_*.py` files it reaches; a changed test file selects itself, and the importers of a deleted module are still found.
  - `--splits N` prints instead a JSON list of at most N commands of the form `uv run pytest -q -p no:cacheprovider --timeout=600 <files>` for `run_commands_parallel`; `--timeout S` changes the per-test limit and `0` omits it.
  - `--owned <paths>` (11-09) selects only the given test files and the tests named for the given modules (`test_<module>.py`, `test_<module>_*.py`), with no import hops, for a sub-agent whose tree also holds its siblings' edits; a module with no such test is named on stderr.
  - Splits are balanced by measured durations (11-09):
    - The suite's `conftest.py` sums the setup, call and teardown seconds of each test file and merges them at session end, under a file lock, into `<OS temp dir>/kiss-tmp/test-durations.json` **CHANGED** (11-18; `tmp/test-durations.json` of the main checkout before), which the main checkout, every worktree and the commands of sub-agents on the machine share since its last reboot (`$KISS_TEST_DURATIONS` names another file; a sub-agent's commands have its private folder in `kiss-tmp` as their temp directory, and `kiss_tmp_dir()` then names the directory that folder is in).
    - Files go longest first into the lightest split, and a file with no record weighs its size times the median seconds per byte of the recorded files.

### Test suites, isolation and policy

- **Python tests**:
  - 2,073 `test_*.py` files with 14,406 test functions (558,090 lines across the 2,140 Python files under `src/kiss/tests/`): 742 files under `server/`, 711 under `agents/`, 377 under `core/`, 142 under `webapp/`, 74 under `scripts/` and 27 elsewhere; 67 of them import Playwright.
  - `uv run pytest` collects `test_*.py` only, deselects the `slow` and `process_killer` markers (`addopts = "-m 'not slow and not process_killer'"`), applies no per-test timeout (`timeout = 0`; the `impacted-tests` splits pass their own) and measures coverage only when `--cov=src/kiss --cov-branch` is given.
  - `live_api` marks tests that call a provider's API and are skipped without its key; `live_cli` marks tests that spawn the real Claude Code or Codex CLI.
  - `pytest>=9.0.3,<9.1` stays pinned because 9.1.0 and 9.1.1 lose a package's `conftest.py` fixtures in split runs.
- **JavaScript tests**:
  - 407 `*.test.js` files under `src/kiss/webapp/test/`, plain Node scripts that use `assert`, with 21 helper modules.
  - `npm test` (`node test/run-all.js`) **CHANGED** (10-18) runs every suite in its own `node` process, as many at a time as the machine has CPUs up to 32 (`--serial` runs one at a time), and kills a suite's process group after 10 minutes; before, the suites ran one after another.
  - 23 `*.coverage.js` gates run jsdom suites under `NODE_V8_COVERAGE` (a suite that a gate names runs only through that gate):
    - 22 run one suite each and fail unless every code line fenced by `// <mark>-coverage:start` and `// <mark>-coverage:end` comments in the `media/*.js` files was executed.
    - `remotePasswordBypass.coverage.js` runs five suites and requires every code line of the whole of `transport.js` and `assetLoadGuard.js` and every branch except the empty `catch` after a `location.reload()`.
  - `npm run check` is the lint plus the tests.
  - `scripts/` also holds 11 shell tests (`test_*.sh`) of the deployment and release scripts (section 21).
- **Suite isolation**:
  - The root `conftest.py` registers the reaper plugin, sets `KISS_DISABLE_TASK_CLASSIFIER=1` and raises the open-file soft limit to 4,096 on POSIX.
  - `src/kiss/tests/conftest.py` (1,401 lines) gives the session its own `KISS_HOME` (a `kiss_test_*` directory removed at exit), sets `BROWSER=true` and `KISS_MUSE_AUTH=0`, turns the tips and the update toast off, and:
    - Routes every temporary file of the session and of its child processes into one `kiss-pt-*` root, deleted at exit or by the next session when its owner died **NEW** (10-30, after leaked test directories had used 960,000 of the 1,048,576 inodes of `/tmp` and 2,500 tests failed at setup; POSIX only).
    - Sets `GIT_CEILING_DIRECTORIES` to the temp roots and drops the repository-scoping `GIT_*` variables, so no `git` call of a test can reach the developer's checkout **NEW** (10-30, after tests had made seven commits on work in progress).
    - Applies autouse fixtures that give each test its own `KISS_WORKDIR`, undo changes to `config.json`, empty `tabs.json`, hide ambient AWS credentials **NEW** (10-18) and point `CODEX_HOME` at a missing directory unless the test is marked `live_cli` **NEW** (10-18).
    - Offers ephemeral OAuth redirect ports **NEW** (10-12).
    - Offers helper modules **NEW** that start:
      - A second `kiss-web` daemon in its own process (`remote_daemon.py`, 10-10).
      - A private unprivileged `sshd` on a loopback port (`ssh_server.py`, 10-11).
      - A client of the daemon's local WSS endpoint (`local_ws.py`, 10-01).
      - A re-branded copy of the package (`branded_package.py`, 10-16).
    - A fifth, `lock_injection.py` (10-13), offers `hammer`, which loops a real code path on a worker thread, injects up to 300 `KeyboardInterrupt`s the way a forced Stop does, optionally while another thread contends for a lock, and asserts that the worker ended and that `STATE_LOCK` and the named locks are free.
- **Subprocess reaper** (`kiss.tests.subprocess_reaper`):
  - Wraps `Popen.__init__`, so every child, including those of `subprocess.run` and asyncio, is attributed to the running test, to the class-, module-, package- or session-scoped fixture being set up, or to the pytest process, and is swept when that scope ends:
    1. SIGTERM.
    2. Up to 5 s of grace.
    3. SIGKILL.
  - At a test's teardown, only the processes that nothing references any more are swept; one that a shared object still holds waits for the end of the run.
  - A child that leads a process group is signalled as a group; on Windows its Job Object or process tree is terminated.
  - Reporting:
    - A test or fixture sweep that had to kill something emits a `LeakedSubprocessWarning` naming the test or fixture and the commands.
    - The sweep at the end of the session prints its report on the terminal instead, and a last-resort `atexit` sweep prints on `stderr`.
  - Processes created without `Popen` (`os.fork`, `os.posix_spawn`, `multiprocessing`) are not tracked.
  - Written after a run left 529 detached `muse_auth.daemon` processes.
- **Testing policy** (`src/kiss/SYSTEM.md`) **CHANGED**:
  - **Standing rules**:
    - End-to-end tests only, with no unit tests, mocks, patches, fakes or test doubles and no structural tests that assert on source code.
    - 100% branch coverage of new and modified code wherever a branch is reachable without a test double, an unreachable branch being documented in the test file.
    - An issue is reproduced by a test before it is fixed.
    - Lint and type checks run once per task, at the end, and only when the task created or modified code files.
    - Only the impacted tests run after a change, and the full suite runs only on request or when changes cross module boundaries.
    - A verification that passed is not repeated unless a later change could invalidate it.
    - A load-dependent flake is named in the summary and not chased with a third full run.
  - **Changed since 09-30**:
    - The end-of-task check is `uv run check --full --fix` (10-25).
    - The impacted tests come from `uv run impacted-tests`, never from a directory glob (10-25); a sub-agent in a shared tree runs `--owned` on its own files and the parent runs the impacted closure once per round (11-09).
    - Test volume **NEW** (11-02): a regression test goes into the existing test module of the code it exercises, one scenario per behaviour, and the test lines a task adds stay near the number of product lines it changes, with duplicated scenarios merged once they pass twice that.
    - The full suite is budgeted (11-02): one run after the last product change, as a background job, plus at most one re-run of each failing split.
    - A suspected flaky test is repeated at most three times, in one `run_commands_parallel` call (10-25).
    - The split rule (10-12): tests run in parallel through `run_commands_parallel` with as many splits as the cores and their load allow; it was an equal split by test methods over `cores - 2` workers.

### SEA tooling, reference viewers and benchmarks

- **SEA tooling** **NEW** (for Sorcar Extension Agents; section 9 has the contract, the rule codes and the tables):
  - `uv run sea lint [paths] [--fix] [--registered]` (10-05) checks every bundled SEA script and built-in command without calling a model, prints one line per finding and exits 1 when there is any; `--fix` rewrites the four mechanical kinds of finding (a stage of `check`).
  - `uv run sea docs [--check]` (10-05) regenerates the six vocabulary tables in `src/kiss/server/README.md` and two website pages from the code; `--check` only reports stale tables and exits 1 (also a stage of `check`).
  - `uv run python scripts/run_sea.py <sea> "<task>"` (10-09) runs one SEA in the calling process, with no daemon and no UI, and exits 0 on success and 1 otherwise; `--json` prints the whole outcome with the trajectory file paths.
- **Generated reference and viewers**:
  - `generate-api-docs`, a stage of `check`, rewrites `API.md` (669 lines) from the syntax trees of six modules, listing their public classes and functions (`core/kiss_agent.py`, `agents/sorcar/relentless_agent.py`, `sorcar_agent.py`, `chat_sorcar_agent.py`, `daemon_client.py`, `server/sorcar.py`).
  - The trajectory visualiser also runs standalone, `python -m kiss.viz_trajectory.server <artifact_dir> [--host 127.0.0.1] [--port 5050]`: a Flask page over the `trajectory_*.yaml` files of the `job_*` directories that merges the jobs of every `.kiss.artifacts/jobs` under the project root, worktrees included, and opens the browser (the daemon serves the same page at `/trajectories`, section 14).
  - `scripts/ui_theme_screenshots.py` captures six scenes of the web app's chat page at desktop and phone width and the shared-chat page, each in the dark and light themes, and one screenshot of the trajectory visualiser.
  - `scripts/ui_style_probe.py` writes one line of computed style per distinct control of the same chat scenes and of the shared-chat page, each in its default theme only, so a control can be compared across surfaces and across a CSS change.
- **Benchmark harnesses**:
  - **Terminal-Bench under Harbor** **NEW** (10-25; `src/kiss/benchmarks/terminal_bench/`, 7 Python files, 1,182 lines; `harbor>=0.24.0`):
    - `python -m kiss.benchmarks.terminal_bench.daemon` starts a private `kiss-web` daemon with its own home (`~/.kiss-tb`, port 8797) in which the web tools, memory, task classification and the cron scheduler are switched off (the `is_worktree` and `auto_commit_mode` keys of its config template are read by nothing, section 24) and a trial cannot switch models (section 10).
    - `python -m kiss.benchmarks.terminal_bench.run --model M --job-name N` builds the `harbor run` command for the dataset `terminal-bench/terminal-bench`, named without a version so that Harbor takes its latest release (`--dataset org/name@version` pins one):
      - 5 attempts per task, 40 concurrent trials and a 100 USD cap per trial by default.
      - `-x` and `-i` exclude and include tasks.
      - `--env modal` runs the trials on Modal instead of local Docker.
      - `--effort`, `--reviewer`, `--reading-check` and `--blind-verifier` set the reasoning effort, the reviewer of the coding SEA's finish gate, the reader of the agent's interpretation ledger and the writer of the gate's independent check (section 9).
      - `--agent oracle` runs the reference solutions instead, the sandbox check Terminal-Bench asks for before an agent run.
    - The Harbor agent `SeaHarborAgent`:
      - Writes a coding SEA per trial and drives it on the trial's container through the daemon.
      - Lists the task's Agent Skills and MCP servers in the prompt.
      - Reaches a sandbox that is not a local Docker container through a loopback exec bridge (10-26).
      - Ends a trial whose sandbox is gone with `EnvironmentLostError` (10-30).
    - `python -m kiss.benchmarks.terminal_bench.summarize <job_dir> --attempts K`:
      - Writes `summary.json` and `summary.md` with the mean reward, the passed attempts divided by the number of tasks times `K`.
      - `K` must be the attempts per task of the launch: `--attempts` defaults to 1, not to the launcher's 5, and an attempt that never ran counts as failed.
      - With `--dataset-dir`, dataset tasks that have no trial count as failed; a further job directory replaces the trials of the tasks it re-ran.
    - Section 3 reports the runs, which the ICML paper and the reports call Terminal-Bench 4.0.
  - `scripts/transport_bench/` **NEW** (09-30) drives one real daemon with a stand-in model over the local WebSocket (WSS) transport and records latency percentiles and the daemon's CPU time and memory for five workloads (`connect`, `ping`, `models`, `burst`, `task`); `report.py` renders the results as HTML.
  - `scripts/web_stealth_bench.py` **NEW** (10-26) loads one page per bot-protected host through the real browser tool, prints `OK`, `FALLBACK`, `BLOCKED` or `ERROR` per URL and exits 1 when a page was blocked or failed (section 17).
  - `kiss.scripts.memory_search_eval` scores memory search against a question set (section 13).
  - `kiss.scripts.cost_levers_experiment` runs A/B cost experiments with real model calls.
  - `kiss.scripts.redundancy_analyzer` lists the test methods whose coverage (coverage.py branch arcs per test, LCOV records) other tests already produce.
  - The `benchmarkings` tree of the previous inventory (the HarnessTax, memoryfield and task-classifier drivers and their job data) has since been moved out of the repository.

### Maintenance scripts, CI and repository hygiene

- **Maintenance scripts** (`src/kiss/scripts/`, 24 Python files, 12,178 lines; run as `uv run python -m kiss.scripts.<name>`):
  - `update_models` and `update_responses_api_support` maintain the model catalogue, and `cost_report` prints the token-cost KPIs of a `history.db` window (section 8).
  - `backfill_subtask_costs` re-adds sub-task spend that was never folded into its parent (it only reports unless `--apply` is given), and `backfill_task_metadata` fills missing tags, SEA names and chat summaries (section 11).
  - `sync_db`, `carry_over_tables`, `db_fingerprint`, `relocate_work_dir`, `running_tasks`, `remote_config` and `legacy_task_db` serve the installer and `rsorcar` (section 21).
  - `sync_kiss_notes` and `merge_memory_pages` serve the memory sync (section 13).
  - `webclip` builds the iOS Web Clip profile (section 21), and `make_victory_sound` renders `media/victory.mp3` (section 18).
- **Continuous integration**:
  - One GitHub workflow remains, `.github/workflows/check.yml` ("Code Quality", on pushes and pull requests to `main`, Python 3.13 with uv).
  - Its only command is `uv run check --clean`, which `argparse` resolves to `--clean-only`, so the job deletes build artefacts and runs no stage (section 24).
  - The three Windows workflows were removed 10-14 with the VS Code extension; no workflow runs a test.
- **Repository hygiene**:
  - A worktree holds tracked files only, so the main checkout's ignored `node_modules` directories are symlinked into each new one (section 12).
  - Inside a worktree, the Bash tool:
    - Rewrites an unambiguous `uv run <program>` to the main checkout's `.venv/bin/<program>` with the worktree's sources on `PYTHONPATH`, and refuses `uv sync`, `uv add`, `uv remove` and any `uv run` it cannot rewrite, naming the command to run instead **NEW** (11-02; the rewrite 11-03).
    - Refuses a command that names the main checkout's path outside its `.venv` (section 7).
  - Task output paths:
    - Scratch files go to `<OS temp dir>/kiss-tmp/<task-id>/` (`<OS temp dir>/kiss-tmp/` with no task id, `<OS temp dir>/kiss-tmp/kiss-sub-<random>/` for a sub-agent, and `/tmp/kiss-tmp/…` in a container; `kiss-tmp-<uid>` when `kiss-tmp` is taken in a shared `/tmp`, private to the user and emptied on the first use after a reboot) **CHANGED** (11-18; `./tmp/<task-id>/` from 10-25, `./tmp/` before).
    - The progress log goes to `<scratch dir>/PROGRESS.md` **CHANGED** (11-18; `./tmp/PROGRESS.md` before).
    - Reports longer than about 800 words are written as HTML under `./reports`.
  - Agent notes and rules:
    - `AGENTS.md` at the repository root (32 lines; `SORCAR.md` until 10-03) and `CLAUDE.md` (30 lines) **NEW** (10-15) carry notes for coding agents: one home-automation pointer in `AGENTS.md` and the same block of AWS rules in both.
    - The build and test rules for agents are in the "Sorcar repo specific" block of `src/kiss/SYSTEM.md`.

## 23. What the trajectories show

### Scope and summary figures

- **Source and window**: Figures from the development machine's task database (`history.db`), read on 2026-11-10 and covering the rows and events dated before 2026-11-10 00:00 UTC:
  - First row: 2026-04-22.
  - Two more rows carry the timestamp `1234567890` (2009-02-13) and a NULL id.
- **Multi-machine history**: The file holds more than one machine's history:
  - The history sync (sections 15 and 21) merges task databases.
  - Since 11-08 a copied row names the machine it ran on in `machine`.
- **Task rows**: 41,127 (9,685 top-level, 31,442 sub-agents, 4,477 side channels).
- **Rows naming another machine**:
  - 13,165 from 8 hosts (9,377, 3,046 and 726 from the three largest), 3,019 of them top-level with $25,070 of spend.
  - Of the 27,962 unstamped rows, 692 have macOS working directories although the development machine runs Linux.
- **Chats**:
  - 8,614 chat ids (4,927 with a top-level task, 3,687 used by sub-agents only; the two rows dated 2009 have none).
  - 4,927 stored summaries, 4,764 of them non-empty.
- **Persisted events**: 23,152,517:
  - 16,389,163 `thinking_delta`
  - 385,054 `llm_call` since 09-27
- **Tool calls**: 968,429 under 177 names.
- **Spend on top-level tasks**: $77,565 · 69.8 billion tokens · 674,671 steps.
- **Sum over every row** (double counts folded sub-agent spend): $137,184 · 119.1 billion tokens · 1,081,998 steps.
- **Per-call check**:
  - The `llm_call` events sum to $38,819.
  - The top-level tasks launched since the first of them (09-27) record $38,769.
- **Tasks over one hour / over 1,000 steps / over $100**:
  - 564 / 89 / 117 top-level (1,027 / 99 / 146 across all rows).
  - Duration runs from launch to finish, with `timestamp` standing in where a legacy row has `start_ts` 0 (the 09-30 figure of 2,300 subtracted that 0).
- **Task with the most steps**:
  - 22,104 steps, $1,479, 12.4 hours ("Can you precisely and thoroughly find and fix all race conditions, hangs, deadlocks, dead code, and redundancies …").
  - The costliest took $1,853 over 14,514 steps, and 5 tasks cost more than $1,000.
- **Distinct top-level working directories / named models**:
  - **Working directories**: 1,332, of which 1,205 are per-run cron directories (2,534 across all rows).
  - **Named models**: 87 (64 rows record neither a directory nor a model).
- **Cron-launched tasks**: 1,506 top-level tasks with `is_cron = 1` ($4,398) and 1,640 sub-task rows.
- **Finish verdicts**: 1,742 top-level tasks with `success = 1`, 399 with `success = 0` and 150 with none since the first recorded verdict (10-10).
- **Worktree rows / auto-commit top-level tasks / parallel top-level tasks**:
  - 3,532 / 4,585 / 7,644.
  - The first two columns are legacy, last written on 10-09 and 10-10 (section 12).
- **Worktree tool calls**:
  - 580 `create_worktree` in 525 tasks
  - 675 `commit_and_merge` in 502
- **Voice tasks / slash-command tasks**:
  - **Voice tasks**: 119 top-level tasks whose text starts with `Speaker #` (section 18; dictation inserted as bare words carries no prefix and is not counted).
  - **Slash-command tasks**: 591:
    - `/sync`: 254
    - `/good_morning`: 96
    - `/sync_history_and_memory`: 53
    - `/review_paper`: 33
    - `/revise_and_review_paper`: 28
    - `/sh`: 27
    - `/write_paper`: 26
    - `/install`: 20
    - `/machine`: 16
    - `/remember`: 9
- **Since 2026-09-30**:
  - 13,467 rows, 3,243 top-level (1,392 of them cron-launched), 10,224 sub-agents.
  - $36,951 of top-level spend, 12,470,709 events, 395,051 tool calls.

### Tool calls by name

Figure 2. Tool calls by name:

- **Recorded calls shown**: 965,940 of the 968,429 recorded calls; the remaining 2,489 are spread over 126 smaller tools such as `search_knowledge_pages` (114), `run_transcript` (104), `stored_recommendations` (96) and `sea_findings` (91).
- **Group column**: Group follows the tool groups of section 7; the other rows name the SEA or agent that owns the tool.

| Tool | Calls | Group |
| --- | --- | --- |
| `Bash` | 481,927 | shell |
| `Read` | 211,124 | files |
| `Edit` | 85,055 | files |
| `summary` | 53,838 | interaction |
| `finish` | 33,814 | interaction |
| `Write` | 18,117 | files |
| `go_to_url` | 13,092 | browser |
| `run_commands_parallel` | 11,094 | shell |
| `memory_search` | 5,945 | memory |
| `run_parallel` | 5,609 | dispatch |
| `bash_job` | 4,890 | shell |
| `set_model` | 4,267 | interaction |
| `memory_pull` | 4,079 | memory |
| `memory_write` | 4,016 | memory |
| `memory_read` | 3,579 | memory |
| `run_agent` | 3,066 | dispatch |
| `task_transcript` | 2,571 | SEA (`/task_update`) |
| `task_context` | 2,362 | SEA (`/ask`) |
| `code_graph` | 1,524 | retired tool (removed 08-04) |
| `list_messages` | 1,517 | channel SEA (Gmail) |
| `decide` | 1,079 | interaction |
| `git_sync` | 993 | SEA (`/sync`) |
| `read_paper` / `build_paper` / `check_review` / `check_paper` | 916 / 738 / 373 / 337 | paper SEAs |
| `click` | 894 | browser |
| `agent_job` | 872 | dispatch |
| `screenshot` | 818 | browser |
| `number_of_cores` | 807 | dispatch |
| `check_gmail_auth` | 789 | channel SEA (Gmail) |
| `commit_and_merge` | 675 | worktree |
| `ask_user_question` | 608 | interaction |
| `create_worktree` | 580 | worktree |
| `get_message` | 544 | channel SEA (Gmail) |
| `recent_tasks` / `save_recommendations` / `connected_apps` | 383 / 359 / 305 | SEA (`/recommendations`) |
| `get_page_content` | 322 | browser |
| `close_browser` | 300 | browser |
| `talk` | 272 | interaction |
| `write_knowledge_page` | 221 | SEA (`/git_extract_knowledge`) |
| `evaluate_script` | 211 | browser |
| `log_decision` | 185 | SEA (`/autorouter`) |
| `type_text` / `press_key` / `scroll` | 179 / 159 / 140 | browser |
| `patch_sea_prompt` | 144 | SEA (`/rsi7d`) |
| `cron_job` | 131 | cron agent |
| `show_browser` | 68 | browser |
| `checkpoint` | 52 | checkpoints |

### Monthly top-level tasks and spend

Figure 3. Top-level tasks and spend per month:

- **Time zone and exclusions**: UTC; the two rows dated 2009 are left out.
- **Cron-launched column**: Cron-launched counts the tasks with `is_cron = 1` among them.

| Month | Tasks | Cron-launched | Spend |
| --- | --- | --- | --- |
| 2026-04 | 513 | 0 | $414 |
| 2026-05 | 1,807 | 0 | $1,506 |
| 2026-06 | 888 | 0 | $2,451 |
| 2026-07 | 883 | 0 | $6,885 |
| 2026-08 | 843 | 5 | $10,382 |
| 2026-09 | 1,629 | 114 | $19,952 |
| 2026-10 | 2,596 | 1,171 | $24,354 |
| 2026-11 (to the 9th) | 524 | 216 | $11,622 |

### Model usage and spend

- **Models on top-level tasks**:
  - `claude-fable-5-1`: 2,781
  - `claude-fable-5`: 1,978
  - `claude-opus-4-7`: 1,830
  - `claude-opus-4-6`: 1,044
  - `claude-opus-5-5`: 298
  - `claude-opus-4-8`: 231
  - `claude-opus-5-5-medium`: 181
  - `claude-fable-5-1-max`: 165
  - `gpt-6-astra`: 160
  - `openrouter/anthropic/claude-opus-5.5-medium`: 147
  - `openrouter/anthropic/claude-fable-5.1-max`: 138
  - `gpt-5.6-sol`: 136
  - `gpt-5.5`: 105
- **Models across all rows**:
  - `claude-fable-5-1` leads with 9,372 (3,159 of them side channels)
  - then `gpt-5.6-sol`: 8,422 (plus 2,972 `-xhigh`)
  - `claude-fable-5`: 7,371
  - `claude-opus-4-7`: 2,665
  - and `gpt-6-astra`: 2,224 (plus 35 `-high`)
- **Sub-agents and review work on `gpt-5.6-sol` and `gpt-6-astra`**:
  - 11,234 of the 11,394 rows of the two `gpt-5.6-sol` variants and 2,064 of the 2,224 `gpt-6-astra` rows are sub-agents:
    - "review" occurs in the task text of 3,941 and 1,088 of them.
    - The `review` tag survives the three-tag cap (section 11) on 2,674 and 694 (the 09-30 inventory counted it on 8,878 of the `gpt-5.6-sol` sub-agents).
- **Spend by model since 10-24** (when rows began to carry `model_costs`, the cost of a row's own model calls; 5,300 rows, $21,157 in all):
  - `claude-fable-5-1-max`: $8,318
  - `claude-fable-5-1`: $4,867
  - `gpt-6-astra`: $3,133
  - `claude-opus-5-5-max`: $1,659
  - `openrouter/anthropic/claude-fable-5.1-max`: $1,367
  - `openrouter/anthropic/claude-opus-5.5-max`: $423
  - `openrouter/anthropic/claude-fable-5.1`: $410
  - `gpt-5.6-sol`: $333
  - `claude-opus-5-5-medium`: $209
  - `openrouter/openai/gpt-6-astra`: $194

### Keywords and tags

- **Kinds of work by keyword in top-level task text** (case-insensitive substring):
  - test: 1,817
  - fix: 1,764
  - report: 1,516
  - bug: 1,498
  - review: 1,431
  - cron: 1,149
  - git: 1,001
  - slack: 842
  - gmail: 788
  - paper: 603
  - implement: 332
  - benchmark: 177
  - audit: 158
  - optimiz*: 131
  - research: 121
  - docker: 96
  - rsorcar: 82
  - refactor: 80
  - blog: 70
  - deploy: 61
  - latex: 43
  - slides: 32
  - whatsapp: 24
  - telegram: 20
- **Cron-launched share of keyword counts**: The 1,506 cron-launched tasks repeat a few prompts and supply most of four of these; without them:
  - report: 496
  - cron: 129
  - slack: 145
  - gmail: 100
- **Tags**:
  - **Coverage and cap**:
    - 41,123 of the 41,127 rows carry them.
    - At most three per row since 10-12, and the rows written before the cap were re-tagged under it, leaving 376 rows of 10-10 to 10-12 with four to six.
  - **Most frequent combinations**:
    - `work,coding,testing` (6,579)
    - `work,chore,question` (4,227)
    - `work,chore,failed` (3,711)
    - `work` alone (2,658)
    - `work,testing,debugging` (2,101)
  - **By single tag**:
    - `coding`: 13,952
    - `testing`: 12,428
    - `chore`: 11,851
    - `failed`: 8,884
    - `question`: 5,969
    - `review`: 5,073
    - `subagent`: 4,983 of the 31,442 sub-agent rows
    - `debugging`: 4,349
    - `secret`: 1,012

## 24. Caveats, corrections and stale documentation

### Gated, off by default or platform-bound

- **CI checks nothing**:
  - The only workflow, `.github/workflows/check.yml`, runs `uv run check --clean`; `argparse` reads the flag as `--clean-only`, so the job only deletes build artefacts.
  - No workflow runs `pytest` or the JavaScript tests (section 22).
- **Sub-agents of the caller's own model family**: Section 10's same-family rule refuses a plain `run_agent` on the caller's own family unless the call is a review or the caller's own task asks for sub-agents.
- **Off by default** (ship disabled):
  - Starting the Chrome bridge at daemon boot (`chrome_bridge`; while it is off, the first `use_my_chrome` call starts the bridge, and unticking it takes effect at the next daemon start).
  - `KISS_REASONING_KEEP_TURNS` (0).
  - `KISS_ROUTINE_TURN_EFFORT` (empty).
  - The Terminal-Bench launcher's `--blind-verifier` (`none`).
- **Settings without a control**:
  - `classify_tasks`, `classify_with_decisions` and `memory_dir` have no control in the app and are set in `~/.kiss/config.json`.
  - The web client always sends `isParallel: true` and never sends `useMemory`:
    - Only the Python API starts a sequential run.
    - Memory is switched per run through the Python API, a SEA setting or `run_agent` options.
- **Needs something first**:
  - **Remote access**: Needs a stored remote password (section 14).
  - **Browser sign-in to GitHub, Microsoft Teams, Slack and Discord**:
    - Needs `KISS_<PROVIDER>_CLIENT_ID`, the four embedded OAuth client IDs being empty.
    - Microsoft Teams needs Muse auth, on by default on Linux only (section 19).
  - **Google sign-in**: Needs the Browser tab or the user's Chrome; the agent's own Chromium aborts requests to `accounts.google.com` (section 17).
- **SSH machines only**:
  - `Bash(machine=)`, `/sync_history_and_memory` and `/install` work only for machines registered by SSH login.
  - `/install` needs a clone of the private development repository there (section 15).
- **Terminal tab**:
  - Has no Windows implementation.
  - Is not mirrored to other clients.

### Behaviour a name or a message hides

- **A SEA that names `bestrouter`**:
  - A SEA whose `settings()` set `model` to `bestrouter` runs on the router's simple model, `claude-opus-5-5-medium`.
  - The routing protocol is layered only when `bestrouter` is the tab's pick (section 9).
  - No bundled SEA does so since 11-16, when `/discover`, `/write`, `/review_paper`, `/review_patent` and `/revise_and_review_paper` went back to the caller's model.
- **Update Models**:
  - The completion notice says new tasks pick up the refreshed catalogue.
  - `MODEL_INFO` is loaded once at import, so the running daemon keeps the old one until it restarts (section 8).
- **Worktree guard**:
  - Covers `Write`, `Edit` and `rollback`, including a shell rewrite that `Bash` applies as `Write` or `Edit` (section 7).
  - Any other `Bash` write run before `create_worktree`, such as a here-document that creates a file, goes straight into the checkout.
  - After a daemon restart an unmerged worktree is not offered for merging again, although its branch and directory remain (section 12).
- **Cron**:
  - A repeating job's missed occurrences are skipped.
  - A repeated identical failure is not delivered again.
  - Scheduled runs stay out of the history list until its Cron chip is on (section 20).
- **Literal `~/.kiss` paths**:
  - The update lock, the install log and the update clone ignore `$KISS_HOME` and a brand's `home_dir`.
  - `scripts/sync-memory.sh` (the memory step of `rsorcar`) and the Terminal-Bench daemon know no brand home either.

### Limits of the figures

- **The task database is a merge**:
  - The file behind section 23 holds several machines' history:
    - 13,165 of its 41,127 rows name one of 8 other machines (3,019 top-level tasks with $25,070 of the $77,565).
    - 692 unstamped rows have macOS working directories although the development machine runs Linux.
  - A later sync can still add rows dated before the cutoff.
- **Spend**:
  - The top-level sum, $77,565, is the spend; the sum over every row ($137,184) counts folded sub-agent spend twice.
  - Since 09-27 the `llm_call` events give an independent total: $38,819, against $38,769 on the top-level tasks launched since then.
- **Unbilled turns**:
  - Subscription billing, the default for `cc/*` and `codex/*` models, records a turn at $0 (89 of the 130 rows on those models carry no cost).
  - Nine other catalogue entries have a $0 input price.
  - Recorded spend understates what such runs consumed.
- **Cron runs dominate several counts**:
  - The scheduler launched 1,392 of the 3,243 top-level tasks since 09-30, each in its own run directory.
  - They account for 1,205 of the 1,332 working directories and for most keyword hits on report, cron, slack and gmail.
- **Tags and legacy columns**:
  - Rows were re-tagged under the three-tag cap of 10-12, so tag counts are not comparable with 09-30: `subagent` survives on 4,983 of 31,442 sub-agent rows.
  - `is_worktree` and `auto_commit_mode` were last written on 10-09 and 10-10 and exist only in databases created before then.
- **Section 1 counts**:
  - The capability counts repeat the catalogue's flags, which mark `gemini-embedding-2` for generation and function calling instead of embedding and give 7 OpenRouter entries no capability at all.
  - The 14,406 test functions are `test_*` functions at module level and `test_*` methods of module-level classes in `test_*.py` files.
- **Dates and hashes**:
  - Section 2's range and peak are in UTC (90 commits on 10-08; 88 by author-local date), while month-day dates are author-local, so the removal of `benchmarkings/` dated 09-29 (`b9ed868023`) is inside the window.
  - Hashes are development-repository commits:
    - Since 10-17 the public repository holds one commit per release.
    - None of the 31 release tags since 09-30 is an ancestor of the development HEAD.

### Documents and docstrings behind the code

- **Website** (under `website/kisssorcar.github.io/`):
  - `docs/cli.md`, `docs/overview.md`, `docs/tips.md`, `docs/api.md` and `docs/sea-commands.md` still describe pre-warmed worktrees, a `use_worktree` option, a classifier that decides whether a task needs a worktree and a `classify_tasks` toggle in Settings; the agent now calls `create_worktree` and `commit_and_merge` itself (sections 11 and 12).
  - `index.html` says 695 models and `docs/models.md` 761, against 1,077 entries.
  - `docs/cli.md` still turns memory off for `cc/*` and `codex/*` models, a gate removed 10-18.
- **`README.md`**:
  - Line 173 calls the unmerged-worktree bar "Merge / Discard" (it has three buttons: "Auto-commit and merge", "Do nothing", "Discard").
  - Line 178 says a remote task row has no delete button (since 11-08 delete, favourite and title edits go to the row's machine).
  - Line 316 says `--splits` balances by file size (by recorded duration, with file size as the fallback).
- **`API.md` and `src/kiss/server/README.md`**:
  - `API.md:644` still describes the "Use Jev" and "Classify tasks before running" checkboxes, and `src/kiss/server/README.md:1820-1823` the second; both left Settings on 10-15.
  - The `tool_profile` list at `API.md:191` omits `none`.
- **`src/kiss/SYSTEM.md`**:
  - Line 201 says `Write` and `Edit` of a checkout path are refused once a worktree exists (the work dir's own paths are remapped).
  - Line 33 caps `run_parallel` nesting at two, which no code enforces.
- **`summary` cadence**:
  - The prompts at `bestrouter_sea.py:75`, `merge_sea.py:78`, `rsi7d_sea.py:331` and `git_extract_knowledge_sea.py:222` say the 10th, 20th, ... call "is refused".
  - The root SEA layer lets the call run and appends a reminder (section 9).
- **Tool docstrings the model reads**:
  - `talk` says each client speaks the text in the language's voice (clients play only the synthesised MP3).
  - `run_agent` says a cron sub-task runs in `channel_work` (`/cron` runs in the home's `cron/work`).
- **Transports**:
  - `_kiss_web_launcher.py:293` still submits "over its Unix-domain socket", removed 10-01.
  - `talk_player.py:11` and `web_server.py:3551` list a `sorcar` CLI among the daemon's local clients, and `_channel_cli.py:9` says the `sorcar` command no longer exists; the console script exists and runs its agent in its own process.
- **History sync**:
  - `peer_sync.py:46-47` and `rsorcar:106` say no row is ever deleted.
  - `_apply_deletions` in `sync_db.py` deletes the events and the row of a task recorded as deleted on either side, except:
    - A task still running there (it waits for a later sync).
    - A row younger than its deletion (section 21).
- **Strings in the product**:
  - The Save chat tooltip names `reports/chat-<id>.html` (a titled chat is saved as `chat-<title-slug>-<chat-id>.html`).
- **Install and recipes**:
  - `pyproject.toml:383-384` calls uv 0.12.19 "the version the installers pin" (only the `Dockerfile` pins it).
  - `RECIPES.md:14` points at `templates/write_paper_prompt.md`, which is not in the tree.

### Corrections to the 2026-09-30 inventory

- **Tasks over one hour**:
  - The 2,300 of 09-30 subtracted a `start_ts` of 0 on 1,984 legacy rows.
  - Measured from the launch instant, the count was 352 then and is 564 now.
- **Rows dated 2009**: Not a clock artefact; both carry the timestamp `1234567890` and a NULL id.
- **Working directories**:
  - The 09-30 counts of 125 and 430 included the empty `work_dir`.
  - Without it they were 124 and 429.
- **`summary` and Docker**:
  - The `summary` cadence is no longer left to the prompt alone.
  - `bash_job` works in a container through `ContainerJobs`.
  - A Docker run still lacks the checkpoint, worktree and memory tools (section 7).
- **Retired caveats**:
  - The `knowledge/` residue was deleted on 09-30 (`f4c197420e`).
  - `website/kisssorcar.github.io/docs/sea-commands.md` now carries a generated table of all 72 commands.
  - `benchmarkings/` left the tree on 09-29.

### Leftovers

- **Dead launcher**:
  - The root `sorcar` script runs `kiss.agents.sorcar.worktree_sorcar_agent`, a module deleted on 10-10, and fails.
  - The `sorcar` console script works (section 5).
- **Unused files**:
  - `quoted_evidence.py` (194 lines) in `src/kiss/agents/seas/coding/` is imported by nothing.
  - `ROUTING.md` at the root is an empty tracked file.
  - `.pylintrc` is referenced by no tool.
- **Unused settings**:
  - `auto_commit_mode` and `is_worktree` survive only as two unread keys of the Terminal-Bench daemon's config template.
  - The `redundancy_check` pytest marker is used by no test.
- **Names that outlived a removal**: `git_worktree.py` keeps `load_spare_marker` and `spare_has_content` only to discard spares of the worktree pool removed 10-09.
- **Stale model names**:
  - `DEFAULT_MODEL_NAME` in `relentless_agent.py` is still `claude-opus-4-6` (it reaches only a bare `RelentlessAgent`; `API.md:115` repeats it).
  - `DAILY_UPDATE_MODEL` of `/git_extract_knowledge` still pins `claude-fable-5-1`.
