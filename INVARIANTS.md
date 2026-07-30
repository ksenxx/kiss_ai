# INVARIANTS

Durable invariants (standing rules, constraints, and preferences) collected from the full KISS
Sorcar task history (2026-04-22 through 2026-07-28). Each invariant is tagged with the date it was
(last) stated. Where invariants conflicted, the newer invariant takes precedence; superseded rules
are noted inline. One-off task instructions are excluded.

## 1. Model usage policy

- [2026-07-28] Use the `claude-fable-5` model for all tasks, including software development. Use
  `gpt-5.6-sol` (not codex) for a thorough read-only review and debugging of the other model's
  work, thoroughly checking whether the other model has missed any code or wiring or introduced any
  bugs. Use at most 20% of the task budget in gpt-5.6-sol for reviewing and debugging. Use the
  model names literally without hallucinating new model names; no need to check whether the models
  exist. (Evolution, newest wins: claude-opus-4-7 + gpt-5.5 [2026-06-19] → reviewer gpt-5.5-xhigh
  [2026-06-29] → coder claude-fable-5 [2026-07-04] → reviewer gpt-5.6-sol-xhigh [2026-07-09] →
  reviewer gpt-5.6-sol with ≤20% budget cap [2026-07-16].)
- [2026-07-24] Task-specific override: for the receipt-artifact reproduction project
  (./projects/receipts), use `openrouter/moonshotai/kimi-k3` for all tasks including software
  development, with `gpt-5.6-sol` for read-only review at ≤20% budget.
- [2026-06-27] Iterative review-fix loop: have the reviewer model review the work, write end-to-end
  tests reproducing the reported bugs, fix and test with the coding model, run all tests in
  parallel, and repeat until the review fails to produce reproducible bugs.
- [2026-06-22] Standing model notes: claude-opus-4-7 was best for SWE work, gpt-5.5 for reviewing,
  and openrouter/z-ai/glm-5.2 for low-budget SWE; pick the best model per sub-task from
  ~/.kiss/MODEL_INFO.json. (Historical; the 2026-07-28 policy above takes precedence.)
- [2026-07-09] Models must not be called directly anywhere in the project outside
  ./src/kiss/core/models/; use KISSAgent instead of direct model calls.
- [2026-06-28] The `set_model` tool must NOT change the model shown on the model picker (which is
  the user's default choice).

## 2. Testing and bug-fixing process

- [2026-07-14] Bug fixes: FIRST reproduce the issue by writing a real end-to-end/integration test
  (jsdom-based for webview/UI issues), THEN fix the issue, then verify the test passes. (Restated
  continuously since 2026-04-23.)
- [2026-07-14] New features: write end-to-end 100%-coverage (real) tests for the feature FIRST,
  then implement the feature. (Restated continuously since 2026-06-27.)
- [2026-07-14] When running all tests: split the set of tests by the number of test methods into
  (number of cores − 2) splits and run all splits in parallel using the `run_parallel` tool;
  determine whether each failure is a project bug or a test bug and fix accordingly. (Supersedes
  the fixed 12-group split of 2026-05-19.)
- [2026-06-16] Do not write structural tests that assert on the source code; remove any such tests.
- [2026-05-16] `uv run pytest` must not collect coverage.
- [2026-05-17] Slow tests must be marked as slow and excluded from default pytest runs.
- [2026-07-22] Tests that kill the pytest process mid-run must be skipped.
- [2026-05-18] Remove tests for legacy features no longer in the codebase; keep the test suite
  minimal by deleting/merging redundant tests while preserving covered features and regressions.
- [2026-06-26] Code that is used only by tests counts as dead code and should be removed along with
  other redundant/dead code.
- [2026-06-30] CLI/terminal features must be tested end-to-end on a real terminal (e.g., driven via
  tmux), not only with simulated tests.
- [2026-04-28] Integration tests for features like budget limit, custom endpoint/key, use web
  browser, and API key setup/deletion must call an actual model end-to-end.
- [2026-04-28] Tests must live in the src/kiss/tests/ folder.
- [2026-07-14] During AI discovery, ensure there is no reward hacking or cheating; review the
  implementation for it.

## 3. Architecture and code organization

- [2026-07-16] Dependency layering: no code in ./src/kiss/core/ may depend on code outside that
  folder; no code in ./src/kiss/agents/sorcar/ may depend on code outside that directory except
  code in ./src/kiss/core/. Enforce this even if code must be moved.
- [2026-07-26] All user interfaces (VS Code extension and remote webapp in ./src/kiss/agents/vscode/)
  must interact with the server ONLY via the API in ./src/kiss/server/sorcar.py instead of sending
  direct messages/commands to ./src/kiss/server/web_server.py. No raw `{"type": ...}` protocol
  message construction is allowed outside sorcar.py/SorcarAPI; a lint/CI check must fail the build
  if new raw protocol messages are introduced elsewhere. (The 2026-07-25 version also covered
  ./src/kiss/ui/cli/, with an exception only for code that installs or starts the server.)
- [2026-07-17] kiss.server.sorcar.run API: minimal and simple, runs synchronously; TaskResult
  includes result, cost, tokens, steps, chat id, and task id; accepts an optional chat id and an
  optional list of tools. Tools are passed as a file path to a Python file whose top-level public
  functions become tools (clients never serialize Python functions).
- [2026-07-18] Agents in ./src/kiss/agents/third_party_agents/ must be implemented using the
  kiss.server.sorcar.run API; they launch via \_CommandsMixin.\_cmd_run() as kiss-web registered
  agents instead of calling SorcarAgent.run directly [2026-07-12].
- [2026-07-16] No backward compatibility shims: do not create alias/shim modules (static re-exports
  plus `sys.modules[...]` at old import paths) when moving or refactoring code; remove existing
  shims when droppable without breaking functionality or tests.
- [2026-05-16] Single-server architecture: one server/daemon serves both the VS Code extension and
  remote web clients (no per-tab/per-chat server processes).
- [2026-05-17] Frontend events are sent directly to clients over sockets (UDS/WSS) with no
  stdout/stdio hop; legacy stdio-based printer paths must not be reintroduced.
- [2026-05-20] Backend/frontend separation: sorcar_agent.py and chat_sorcar_agent.py must contain
  NO references to tab_id or parent_tab_id (frontend concepts); the backend emits a "new_tab"
  message carrying a task_id and the frontend calls resumeSession — including for subagent tabs.
- [2026-05-24] parent_tab_id must NOT be stored in the database.
- [2026-05-17] Agent lifecycle: do not maintain an agent/thread per tab or per chat_id; create the
  agent and its thread when a task executes and dispose of them when the task completes.
  `running_agent_states` maps task_id → RunningAgentState (no special treatment of subagents); the
  agent registers itself as soon as the task's row is added to the DB and is removed when the task
  finishes.
- [2026-06-15] Every ChatSorcarAgent must be registered with the server, and events of agents run
  outside a chat webview (e.g., from the sorcar CLI) must be recorded in the database so they can
  be loaded and live-streamed in a chat webview tab.
- [2026-05-18] There is no need to maintain the tab_id == chat_id invariant in the source tab
  (supersedes the earlier tab-id = chat-id scheme of 2026-05-16).
- [2026-06-19] All asyncio.open_unix_connection and asyncio.open_connection call sites must pass
  limit=16*1024*1024 to prevent buffer-overflow issues.
- [2026-06-08] All subprocess pipes opened with text=True must also pass encoding="utf-8" so
  non-ASCII text is handled consistently across platforms.
- [2026-07-13] New code (e.g., the code_graph tool) must be minimally coupled with the existing
  source code; new features are implemented in new/separate files with minimal modifications to
  existing files [2026-04-23].
- [2026-06-19] Reuse/share code as much as possible when porting features across the extension,
  remote webapp, and CLI clients; the remote webapp must be as similar as possible to the
  extension and share as much code as possible [2026-06-09].
- [2026-04-23] src/kiss/channels/cron_manager_daemon.py must be a thin client to the `cron`
  command: it must not parse/manage cron jobs or run them with daemon threads; clients send the
  exact crontab entry; the SOCKS interface is kept for inter-process communication.
- [2026-07-21] Cron jobs created by the agent must have names prefixed with "kiss-".

## 4. Coding style and source hygiene

- [2026-07-26] Files under ./src/kiss/ must contain no comments except within the first 4 lines of
  each file (enforced via AST); docstrings of public methods are kept. (First seen 2026-05-03.)
- [2026-06-03] Every program file (.py, .ts, .js, .html, .css — using each language's comment
  syntax) must start with the 4-line author header: "Author: Koushik Sen (ksen@berkeley.edu) /
  Contributors: / Koushik Sen (ksen@berkeley.edu) / add your name here".
- [2026-04-26] src/kiss/SYSTEM.md must be formatted with at most 100 characters per line.
- [2026-05-15] Temporary/scratch/intermediate files must be created under PWD/tmp/ (never directly
  in PWD) and removed at task completion.
- [2026-07-17] Dependency groups in ./pyproject.toml must be organized by folders in ./src/kiss/:
  'core' installs dependencies of ./src/kiss/core/; 'sorcar' those of core + agents/sorcar;
  'server' those of server + agents/sorcar + core.

## 5. Models registry and configuration

- [2026-06-20] MODEL_INFO lives as JSON in src/kiss/core/models/MODEL_INFO.json; update_models.py
  updates MODEL_INFO.json and model_info.py loads it. (Per 2026-06-28: MODEL_INFO.json is NOT
  copied to ~/.kiss/ at install; instead ~/.kiss/MY_MODELS.json is auto-created with short
  documentation and an example entry, and is loaded before the bundled MODEL_INFO.json. This
  supersedes the 2026-06-20 copy-to-~/.kiss/ rule.)
- [2026-05-03] The default model and fast model must NOT be hardcoded anywhere in
  src/kiss/agents/vscode/; they must be obtained from src/kiss/core/models/model_info.py.
- [2026-06-29] In update_models.py, every model that supports xhigh thinking gets two entries: the
  plain model name with thinking=high, and the name suffixed `-xhigh` with thinking=xhigh.
  (Supersedes the 2026-06-19 rule that plain GPT model names default to thinking=xhigh.)
- [2026-06-19] update_models.py and model_info.py must exclude gpt pro and gpt codex models.
- [2026-07-25] Models with context length 1,000,000 or above must have their context length set to
  500,000 in update_models.py / MODEL_INFO.json.
- [2026-07-14] The context length of fable-5 is pinned to 400000 in MODEL_INFO.json — keep it that
  way (tests updated accordingly).
- [2026-07-14] Thinking tokens must be revealed for claude-fable-5 and every other thinking-capable
  model; any adaptive-thinking model in MODEL_INFO.json must have `display: summarized` and must
  not use forced `tool_choice: any` (a regression test must fail if a new adaptive-thinking model
  violates this).
- [2026-06-29] Cost computation must be accurate for all models, including xhigh thinking models;
  each task's reported cost must reflect actual cost [2026-07-20], and cost calculation from LLM
  responses through cost collection must be accurate [2026-07-16].
- [2026-05-07] For `cc/*` models (e.g. cc/opus), the Claude Code CLI is used purely as a model — a
  single call returning a single response, never invoking the Claude Code agent internally; Sorcar
  itself extracts and executes the tool calls.
- [2026-05-02] When codex is not installed as a CLI binary but the Codex UI is installed, the
  binary from the UI installation must be used for the codex model.
- [2026-07-02] Switching models mid-conversation must work in both directions among all model
  formats (e.g., Anthropic ↔ OpenAI), with message histories converted correctly.
- [2026-06-27] MiniMax API key support is removed; Z-ai and Moonshot.ai API keys are supported
  (config.py and the settings panel).

## 6. Git, worktree, and auto-commit semantics

- [2026-05-18] Command-line git is configured with user.name "KISS Sorcar" and user.email
  "kisssorcar@gmail.com".
- [2026-05-23] Worktrees are NOT associated with a chat_id: create a fresh worktree for each
  worktree task; no auto-commit/squash-merge of the prior task before starting a new one.
- [2026-05-18] In worktree mode the agent's PWD is the git worktree directory (not the original
  repo dir), and the agent reports it as such.
- [2026-05-20] When an agent is stopped in worktree mode, do NOT discard the worktree.
- [2026-05-18] "Auto commit" semantics: when ON, skip the merge/diff workflow and auto-commit the
  agent's changes; if worktree mode is also ON, additionally auto-merge with the original branch.
- [2026-05-20] Auto-commit + worktree, task succeeds with no file changes: do NOT commit/merge;
  discard the worktree. Auto-commit + worktree, task FAILS with modified files: behave as if auto
  commit is off — show the diff/merge workflow followed by the worktree merge workflow.
- [2026-05-18] The worktree merge commit message must NOT be "Squashed commit ..."; use the last
  commit message of the worktree branch if the agent modified it.
- [2026-07-14] In auto-commit, the task description and results must be appended to the
  auto-generated commit message before committing; auto-generated commit messages include the user
  prompt of the task [2026-05-18].
- [2026-05-22] The agent must NOT commit changes before the task finishes (commits happen only at
  task completion).
- [2026-06-03] A working directory must be provided for every git operation (git_worktree.py:\_git()
  takes cwd as a required parameter, no None default); VSCodeServer git operations use a
  per-command work_dir.
- [2026-06-03] When a task runs in a git repo and the agent modified no files, never report "Not a
  git repository."; in auto-commit mode, print nothing at all if no file changed.
- [2026-07-09] When a worktree branch is discarded, do not show a notification or print anything in
  the chat webview.
- [2026-05-02] Every functionality of the VS Code extension must work correctly without error when
  the working directory is not in a git repo.
- [2026-04-23] All git popups in VS Code must be suppressed.
- [2026-06-28] When saving the working directory for each task in the task history table, strip the
  worktree part from the directory name.
- [2026-04-28] USER_PREFS.md must be in .gitignore and untracked; it is copied to the worktree when
  a task runs in worktree mode and copied back to the original branch when the worktree is merged.
- [2026-06-22] kiss-sorcar.vsix must not be committed to the git repo, and install.sh must not
  commit it.
- [2026-04-24] \*.aux, \*.out, \*.blg, \*.bbl LaTeX build artifacts must never be added to git.
- [2026-04-30] No one may be able to submit pull requests to https://github.com/ksenxx/kiss_ai.

## 7. work_dir rules

- [2026-06-10] work_dir is set to the open workspace/folder when VS Code launches; each VS Code
  window and each remote webapp instance maintains its own work_dir (the directory of the window's
  open workspace). This invariant must never be violated.
- [2026-06-28] Newest work_dir configurability rule: if the working directory is changed (e.g.,
  via the settings page), that change must NOT alter the directory of any tab bound to a real
  task's chat id. (History: 2026-05-04 made work_dir customizable in Settings; 2026-05-18/
  2026-06-03 removed it from vscode_config.py/config.json, the settings UI, and
  update_settings-related code; the 2026-06-28 rule is the newest and governs any surviving
  work-dir change path.)
- [2026-05-18] When VS Code starts or opens a (different) folder, the workspace root becomes the
  agent's PWD / self.work_dir.
- [2026-06-08] In non-interactive contexts, when the sorcar CLI is launched, work_dir is the
  directory from which sorcar is invoked.
- [2026-06-08] The file picker in the input textbox of both the extension and webapp chat webviews
  must resolve paths with respect to the work_dir.

## 8. Tabs, chat ids, and isolation

- [2026-06-27] A tab in a chat webview must not interfere (functionally or in UI) with another tab
  or chat webview — across VS Code windows and remote webapp/browser instances — unless they have
  the same chat id.
- [2026-06-28] No two tabs on a client may have the same chat id; opening a task from history whose
  chat id is already open simply switches to that tab.
- [2026-06-10] When a task is running, any tab in any browser or VS Code window that has opened the
  task's chat id must see the events streaming live from the running task; all clients viewing the
  same running task see the streaming events [2026-05-16].
- [2026-06-16] A tab where a task started must behave exactly the same (UI and user interactions)
  as a tab that loads the task: identical event rendering, buttons, and labels; user input is
  never ignored during or after the task, and new tasks are runnable after the task ends.
- [2026-05-16] A running agent is NOT stopped when its chat tab is closed; the agent finishes its
  task, and reopening the task from history loads persisted events and streams new ones live.
- [2026-05-16] A server-side tab state is disposed once the frontend tab is closed AND the task
  lifecycle has completed AND there are no subscribers (deferred tab-disposal), in both the
  extension and the web server.
- [2026-05-25] Loading a completed (non-running) task from the task history must NOT create a new
  agent or RunningAgentState entry — only replay the task's events in the tab.
- [2026-05-23] There is NO enforcement of a maximum number of open tabs (the 5-tab limit of
  2026-05-02 was removed entirely on 2026-05-23).
- [2026-04-30] When the last tab is closed (extension and remote), a new chat is created instead of
  closing the sidebar.
- [2026-06-28] When a task completes in a tab, or a running task asks the user a question, the UI
  must switch to that tab.
- [2026-06-28] Ask-user-question dialogs appear on all tabs of all clients with the same chat id;
  responding in one tab closes the dialog everywhere.
- [2026-06-21] If the user changes global settings, new tasks run in a tab use the updated global
  settings, not the settings of the last task in that chat.
- [2026-04-27] The input chatbox is NEVER disabled while a task runs in a tab; the user can queue
  unlimited tasks, which run when the running task ends. Merge/diff and auto-commit are delayed
  until ALL queued tasks have run (no merge_dir snapshot for queued tasks) [2026-04-27].
- [2026-05-07] While a task is running, the buttons below the input textbox are not disabled and
  the model picker remains functional.
- [2026-07-01] Chat/session history must survive restarts: after relaunching VS Code, submitting a
  task to the same tab gives the agent the previous tasks of the same chat id; daemon restart or
  session resumption must never orphan agent state.

## 9. Subagents (run_parallel)

- [2026-05-15] When a run_parallel argument is a list of strings, each string runs as a separate
  task by its own agent in parallel (never the whole list as one task, never per-character splits).
  Parallel sub-agents run with the same chat_id as the parent and accumulate responses in the
  shared chat history.
- [2026-05-18] Subagent marking: extra_payload["subagent"] = {"parent_task_id": \<parent's
  task_history.id>}; is_subagent is implied by the key's presence. A subagent task is a regular
  task except in tab display: its tab does not load other tasks of the chat id and has a different
  header color/icon.
- [2026-07-15] Subagent tab lifecycle (extension and remote webapp): whenever an agent or subagent
  calls run_parallel, tabs open for the spawned subagents (regardless of how many run_parallel
  calls are made); as soon as a subagent finishes, only its tab closes; collapsing a run_parallel
  tool-call panel closes all tabs of subagents it spawned; uncollapsing reopens them.
- [2026-07-15] While a subagent task is running, its chat webview shows the input textbox and
  buttons (removed when it completes) so the user can inject prompts; a message sent there shows
  up as a prompt in that subagent's trajectory; the subagent's stop button stops only that
  subagent. (Supersedes the 2026-05-16 rule that subagent tabs never show the input textbox; a
  loaded finished subagent task still shows no input textbox/buttons.)
- [2026-05-24] After the UI auto-switches to a newly created subagent tab, it switches back to the
  parent non-subagent tab.
- [2026-05-22] Closing a regular agent tab closes all of its subagent tabs.
- [2026-05-18] The stop event for a task stops the whole tree: the main agent and all transitively
  nested subagents.
- [2026-05-19] Budget/metrics aggregation: the parent agent's cost/tokens/steps include the sum of
  all subagents' plus its own, chaining correctly through nested run_parallel levels; the cost and
  tokens shown at the top of the chat webview and in the sorcar CLI reflect agent+subagents cost at
  every turn [2026-07-16].
- [2026-07-15] Agents must enforce the budget from the settings panel and stop when over budget;
  budget is distributed meaningfully across subagents so no subagent can spend the entire budget.
- [2026-05-20] Subagent tasks are NOT shown in the task history panel, and adjacent-task
  navigation/scrolling excludes subagent tasks [2026-05-24].
- [2026-05-22] Subagent result panels are NEVER shown in the parent agent's chat webview.
- [2026-06-11] When an agent with subagents is restored after a VS Code restart, it loads its own
  chat events in a tab and each subagent's events in separate tabs to the right of the parent tab.
- [2026-05-22] \_resolve_parent_tab_id_for_sub must never produce a self-referential parent_tab_id
  and returns "" when no real parent state exists.
- [2026-06-28] While run_parallel is running, a spinner shows to the left of the send button.
- [2026-05-16] Sub-agent tabs are not counted as regular tabs for tab-management decisions.

## 10. Chat webview UI (extension and remote webapp)

- [2026-07-31] The chat webview has no auto-scroll: streamed chat and nested-panel updates must
  never move the chat or inner-panel scroll positions; only explicit user navigation may scroll.
- [2026-07-19] The fixed task panel and the input textbox+buttons panel are drawer-style
  collapsible widgets; collapsed space is used for events; on mobile both open collapsed.
- [2026-07-28] The input drawer button must not show the "Collapse input panel" (or "Expand input
  panel") tooltip; its aria-label is kept for screen readers.
- [2026-07-21] The fixed task panel has no "Collapse/Uncollapse Chats" button; the "expand task
  panel" button increases the panel height to show the entire task text within the webview.
- [2026-07-19] Hovering over the task text in the fixed task panel shows a tooltip with the entire
  task text, in the same font size as the panel's task text.
- [2026-07-19] Each event panel title shows a human-readable compact timestamp to the left of the
  copy button.
- [2026-06-27] Each panel shows its elapsed time at the bottom, updating every second while active.
- [2026-06-27] The thought panel is shown (with live time) even before any thought tokens arrive;
  all model thinking text renders inside the collapsible Thoughts panel, never outside it
  [2026-05-12].
- [2026-06-09] Copy buttons on chat panels copy the raw text (e.g., markdown source), not formatted
  text; copy buttons have no tooltips [2026-05-23].
- [2026-05-16] The return values of all function/tool calls made by the agent are shown in the chat
  webview and the console.
- [2026-07-18] When a chat webview (extension and remote webapp) sees a 'summary' tool call, it
  nests the grouped event panels as sub-panels of the collapsed 'summary' panel with the
  description fully visible; grouping includes the steps after the last `record` call (or from the
  beginning), per the latest SYSTEM.md; "(click to expand)" appears next to the 'summary' label
  [2026-07-20]. (The 5-step summary cadence itself is in section 22.)
- [2026-07-13] Desktop remote webapp layout: codex-style colorless chat — fixed task panels as
  chat bubbles aligned right, all other panels aligned left; no colors in the main chat webview;
  the agent history panel opens from the burger menu on the left; existing remote-webview controls
  remain unchanged when changing layout; the fixed task panels fit the color/style aesthetics of
  the rest of the UI [2026-07-15].
- [2026-07-14] Desktop remote webapp: the history panel occupies 1/4 of the browser screen (and is
  horizontally resizable); chat panels and the fixed task panel occupy 90% of the webview width;
  the input textbox+buttons panel is as wide as the chat webview [2026-07-19].
- [2026-07-15] Font size and style are uniform across chat panel contents (matching the fixed task
  panel), including subpanels of event panels.
- [2026-07-14] Side scrolling across all tasks of a chat id must always work, including while a
  task is running and when previous tasks have very short trajectories.
- [2026-06-27] The "Bash" panel title header is shown in cyan.
- [2026-06-27] Notifications use KISS Sorcar's own notification system, appear at the top right of
  the chat webview with contrasting boundaries, and stay active half as long as before.
- [2026-06-27] "Generating commit message ..." notification persists until generation finishes;
  after commit, show "Committed {a line of the commit message}".
- [2026-05-02] When is_continue is true, the result panel shows "Status: Continue", not
  "Status: Failed"; multi-session failures still show a Result panel [2026-05-23]; the finish
  summary of a multi-session task summarizes ALL sessions [2026-05-03].
- [2026-04-26] When a user stops an agent or the agent is killed, the error message is part of the
  result panel; if cost exceeds max budget the task stops and shows "Budget exceeded" [2026-04-28].
- [2026-04-26] If a task runs with no model available, the result panel must show exactly
  "No model available. Set at least one API key in the environment.".
- [2026-06-27] The "Previous sessions" panel appears before the "Results" panel for multi-session
  tasks.
- [2026-06-28] Clicking a filepath in the chat webview opens it in the VS Code editor or a native
  viewer when the file exists.
- [2026-06-27] The chat webview has no "Frequent tasks" button; "Inject instruction" uses a
  syringe icon (needle 80% of body length) with tooltip "Inject promptlet" [2026-06-28]; injected
  instruction strings are included in fast-complete suggestions at sentence starts.
- [2026-05-02] Chevron collapse state is per task in the chat session; clicking a tab's chevron
  collapses/uncollapses only that task's panels.
- [2026-05-12] Tooltips disappear when the element that triggered them scrolls out of view.
- [2026-07-14] Input-row button internal padding is 2px / 2px 3px (halved from 4px / 4px 6px); the
  gap between input-row buttons is a quarter of the original.
- [2026-06-22] Welcome-page suggested prompts show at most 3 lines with a full-text tooltip and are
  labeled "suggested prompt".
- [2026-05-05] Showing suggestions must not scroll the chat webview to the end; clicking a task in
  the history panel scrolls the chat to that task, not the end [2026-05-04].

## 11. Input textbox and keyboard

- [2026-06-30] Shift+Enter, Alt/Option+Enter, Command+Enter, and Ctrl+J insert a newline while
  typing a prompt — never submit the task or trigger autocomplete — in the chat webview and the
  sorcar CLI interactive (Shift+Enter newline everywhere, first seen 2026-06-10).
- [2026-06-30] The input textbox height adjusts dynamically to show every line of a multi-line
  prompt; there is no cap on vertical growth [2026-05-03]; the CLI input area starts at 3 lines.
- [2026-06-30] A multi-line prompt is a single task: lines are never sent as separate tasks or
  queued steers.
- [2026-07-02] Accepting a fast-autocomplete completion preserves the existing typed text (webview
  and CLI); autocomplete never adds the 'head' to completions [2026-07-01].
- [2026-05-18] Fast complete includes the contents of previous tasks with the same chat id; the
  loaded chat context is cached, not re-parsed per keystroke.
- [2026-04-30] Up/down-arrow task-history navigation grows the textbox when the text does not fit;
  it works on mobile too, along with fast autocomplete.
- [2026-07-01] A steering message sent to a running agent is not merely appended to the trajectory:
  it is prefixed "User says: " with an instruction to take it into account and finish the task.

## 12. Task history panel

- [2026-06-28] When a task starts, its DB entry is created first and the task appears immediately
  in the task history panel (also for tasks started from remote/mobile clients [2026-07-14];
  running rows always pass the workspace filter, with path normalization).
- [2026-06-28] Status circles: a running task shows a pulsing green circle that becomes solid green
  on completion; failed tasks — including user-cancelled ones [2026-06-27] — show a red circle;
  no solid green circle is shown otherwise. Indicators sit at the middle left of the panel. When a
  task is run and the burger menu is opened, its panel shows at the TOP of the history list.
- [2026-06-15] Tab titles carry the same status: a pulsing/blinking green circle while running (no
  spinner, no lightning bolt for running subagents), a solid green circle for successfully
  finished tasks and a solid red circle for failed tasks in any created or loaded tab; no green
  tick / red cross for a loaded non-running subagent [2026-05-19].
- [2026-07-14] Task panels have no background colors; the status color goes on the left
  margin/border; all task metadata (steps, tok, cost, duration, time, work dir, model, wt,
  parallel, auto-commit — the 2026-07-14 request literally wrote "auto-complete", the 2026-06-20
  one "auto commit/manual commit" — chat id, task id) is shown on a single line with wrapping.
  (Refines the 2026-06-20 dot-separated metadata line and the 2026-05-22 4-line task text rule.)
- [2026-07-19] The filter buttons and dates live under a collapsible "Filters" panel (visible when
  uncollapsed): Running / Errored / Succeeded checkboxes (2026-07-14 names; originally
  Running/Errors/Completed per 2026-05-22), From/To date pickers (all selected initially)
  [2026-05-22]; a "Workspace" checkbox (default checked) shows only tasks whose
  work_dir equals the client's work_dir, before the "Favorites" checkbox [2026-06-16]; the filter
  buttons are smaller [2026-07-14]; "From" label, textbox, and picker never split across lines
  [2026-06-20].
- [2026-05-18] Task deletion: an always-visible compact black trash-can button with NO tooltip
  (the original "Delete task" tooltip of 2026-05-02 is superseded) shows white-background "Delete"
  (dark red) / "Cancel" (dark green) confirmation; confirming deletes the task and its events from
  the DB, re-parents children, and removes the task's chat from all open tabs on all clients
  [2026-07-14].
- [2026-05-23] Task panels (history and frequent tasks) have a copy button left of the delete
  button copying the whole task text; these copy buttons have no tooltips.
- [2026-05-22] Clicking a task in the history panel also copies the task text into the input
  textbox.
- [2026-06-11] Every tab shows the running time as elapsed-since-start while running and
  (end − start) when ended; task cards show time spent in hh:mm:ss after the cost [2026-06-20].
- [2026-05-02] There is NO 10000-task cap in the database; \_MAX_FILE_USAGE_ENTRIES is 10000.
- [2026-05-02] frequent_tasks table (task, count, timestamp) capped at 100 entries; every sent task
  increments count and refreshes timestamp; eviction removes the lowest-count oldest entry; the
  frequent-tasks panel shows up to 50 entries with a scrollbar [2026-05-07], opens bottom-to-top
  at 90%×75% of the webview, shows two lines of task text per card with delete-and-confirm like
  the task history [2026-05-22]. (Note: the "Frequent tasks" button was removed from the chat
  webview on 2026-06-27, so the panel has no launcher there; the database invariant stands.)
- [2026-05-20] The "Advanced options" tooltip is removed; no tooltip on the history delete button
  [2026-05-18].

## 13. Settings panel and configuration

- [2026-05-04] The Settings panel is scrollable, has NO "Save configuration" button, and saves
  automatically when it closes; it opens right-to-left as a standalone panel (no Settings tab in
  the menu panel, no "Running" tab either) from a settings button right of the + button; + has 2px
  right padding, 8px between + and the settings button, and 8px to its right [2026-05-22].
- [2026-05-15] User settings have a single source of truth (no duplication across config.json and
  other files); all settings are updatable via the update_settings tool (extension, webapp, and
  slack_channel_sorcar_poller.py) — but update_settings must NOT allow updating API keys
  [2026-05-18].
- [2026-05-15] Tasks can read/modify runtime options: is_parallel, is_worktree, model, max budget,
  use-web-browser, remote_password, demo mode, and auto_commit (worktree/demo/auto-commit actually
  effected via WorktreeAgent).
- [2026-06-11] API keys in the settings panel are secret (masked) by default with an eye-icon
  toggle, like the remote password field. (Supersedes 2026-05-02 "API keys shown in plain text".)
- [2026-05-20] The Remote password field sits at top right after the web app URL, is secret with an
  eye toggle; Custom endpoint/key fields are at the bottom; "Use parallel agents" and "Auto
  commit" are settings checkboxes with the "git commit" button next to the "Auto commit" label;
  the Auto commit button outside the menu has tooltip "git commit" [2026-05-18]; is_parallel and
  auto commit default to True.
- [2026-05-01] The Web/mobile app URL appears at the top of the settings panel (label "Web/mobile
  app", URL on its own line, clickable and copyable); the panel displays the ntfy.sh URL rather
  than the raw Cloudflare URL [2026-05-12].
- [2026-05-12] remote_password, once set in config.json, is never overwritten by code until the
  user sets it again in the settings panel.
- [2026-06-05] The "Update" button appears on a new line after the "Demo mode" select with the
  "Git Commit" button to its left (same style); pressing Update runs install.sh and notifies the
  user that an update is being installed.
- [2026-07-07] The Tips launcher is a light-bulb button labeled "Tips" in the settings page, left
  of "Git Commit". (Supersedes the 2026-07-05 placement next to the mic button.)
- [2026-06-27] Clicking "Server reset" while an agent runs asks for confirmation via a floating
  dialog inside the settings panel (not a system dialog) with OK/Cancel.
- [2026-04-26] The max budget comes from max_budget in src/kiss/core/config.py (default $100), not
  hardcoded; all config values are accessible via the command line through config_builder.py.
- [2026-04-28] A custom endpoint (+optional key) becomes a selectable model in the model picker; if
  use-web-browser is false, the Chromium browser is not used; options (except API keys) persist to
  ~/.kiss/config.json. The panel offers API-key setters for all LLM providers (defaults ""); keys
  set there are saved to the user's shell rc (.zshrc/.bashrc/config.fish) and refreshed in sorcar;
  defaults: max budget $100, custom endpoint/key "", use web browser true, remote password "".
- [2026-06-27] The model picker persists the user's last selected model across VS Code sessions;
  when a new chat opens, the picker shows the last model picked by the user, read from the DB
  (never another tab's task model) [2026-05-02]; last_model_used and usage counts update ONLY on
  user model-picker selection [2026-04-26].
- [2026-05-02] The menu (burger) button — 3 horizontal lines, no background — holds "Use
  worktree", "Use parallelism", "Auto commit", and "Toggle demo mode" (tooltips = item names; menu
  closes on click). The Auto commit and Do nothing buttons appear only when git status shows
  changes.

## 14. Installer, updates, and daemon/server

- [2026-07-15] ./install.sh must be totally simple and non-interactive: it installs necessary
  software, builds the extension, and installs it WITHOUT touching kiss-web; kiss-web restarts
  only as part of extension installation (no double restart); perl-free where possible.
- [2026-07-06] install.sh must never hang or get stuck (e.g., when stopping the old kiss-web
  daemon), and the extension must not get stuck at "KISS Sorcar Server is restarting ...".
- [2026-06-29] Never send Ctrl-C/SIGINT to install.sh; the update flow completes without terminal
  abort or early window reloads; the installer's git pull must not fail (reconcile divergent
  branches automatically).
- [2026-07-07] install.sh must not refer to claude skills at all. (Supersedes 2026-05-03/2026-05-13
  claude_skills install/keep rules.)
- [2026-05-13] install.sh installs git if missing before cloning, handles missing xcode-select
  (macOS: install command-line tools non-interactively [2026-05-01]), and pulls (not re-clones) if
  kiss_ai already exists; if the repo is dirty it stashes, pulls, installs, and unstashes
  [2026-06-03].
- [2026-04-28] README.md documents full installation via curl of install.sh; install.sh clones the
  repo when git exists, else downloads and unzips main.zip to ~/kiss_ai.
- [2026-06-03] Daemon health requires BOTH the TCP listener and ~/.kiss/sorcar.sock; the macOS
  LaunchAgent is explicitly kickstarted after bootstrap/load; the daemon auto-restarts if killed
  and survives reboots [2026-04-28]; it keeps responding after lid close/system sleep and restarts
  itself if the host IP changes [2026-04-29].
- [2026-04-30] VS Code extension activation RESTARTS the remote web server (its code may have
  changed); ensureDependencies() returns immediately if uv + .venv + Chromium + daemon are all
  present; findKissProject in DependencyInstaller.ts performs only checks 1 and 2 and skips the
  rest; the daemon runs in the kiss_ai directory [2026-04-28].
- [2026-07-01] A "Server restart complete" notification fires once the server restarts and accepts
  connections; after an update, the update notification for the now-installed version stops
  appearing; update checks happen on VS Code launch with a permanent notification and an SVG
  update button [2026-06-27].
- [2026-06-24] Until kiss-web has (re)started, the KISS Sorcar view shows "KISS Sorcar Server is
  starting ..."; a reconnecting client (e.g., iPhone Safari returning from background) shows
  "Reconnecting to KISS Sorcar Server ..." and reconnects fast [2026-06-30].
- [2026-06-28] On installation: do not copy SAMPLE_TASKS.md, INJECTIONS.md, or MODEL_INFO.json to
  ~/.kiss/; auto-create ~/.kiss/MY_TASK_TEMPLATES.md ("Hi!"), ~/.kiss/MY_INJECTION.md (the
  100%-coverage-tests-first promptlet), and ~/.kiss/MY_MODELS.json (docs + example entry) when
  missing; load the MY\_\* files before the bundled ones. (Supersedes 2026-06-22 copy-and-overwrite
  rules for SAMPLE_TASKS.md/INJECTIONS.md.)
- [2026-05-06] After install + restart, VS Code opens the KISS Sorcar tab in the secondary sidebar,
  focuses the input textbox, and sets the sidebar width to 1/3 of the window.
- [2026-05-05] The "Restart VS Code" notification stays until clicked; the "Installing
  Dependencies" notification remains until the restart notification appears.
- [2026-04-28] The user's email (comma-separated allowed) is collected only if absent from
  config.json, stored there, exposed near the top of the settings panel, and the web-app URL is
  emailed via resend from "KISS Sorcar<kisssorcar@gmail.com>". (The original install.sh email and
  remote_password prompts [2026-05-12] are superseded by the 2026-07-15 non-interactive install.sh
  rule; email/password acquisition happens via the extension/settings UI, not installer prompts.)
- [2026-05-13] scripts/build-extension.sh creates its output directory if missing.
- [2026-07-20] sorcar-linux is idempotent: re-running it against a provisioned host detects the
  existing code-server/kiss_ai install and skips reinstalling.
- [2026-04-23] scripts/release.sh versions use the format yyyy.mm.minor_number and the version is
  propagated to README.md, src/kiss/SYSTEM.md ("Your version is ..."), and every other file using
  it [2026-05-14].

## 15. Remote web server and security

- [2026-04-30] src/kiss/agents/vscode/web_server.py is HTTPS/TLS-only (no plain-HTTP code) with
  tunnelling always on; the CLI flags --host, --port, --tunnel-token, --certfile, --keyfile, and
  --tunnel/--no-tunnel do not exist; --tls and --tunnel are the defaults for `uv run kiss-web`
  [2026-04-28]; it runs a single 30s-interval timer and has 100%-coverage end-to-end tests.
- [2026-07-16] The remote web app never bypasses the remote-password check: launching it asks for
  the remote password; the session persists so the password is not re-requested until a match
  fails [2026-05-13].
- [2026-05-13] The Cloudflare tunnel URL stays stable as long as possible; cloudflared is not
  needlessly restarted, and a tunnel restart never restarts the kiss web server [2026-06-15].
- [2026-05-12] Whenever kiss-web starts or the Cloudflare URL changes, the URL is posted privately
  to a stable ntfy.sh topic (same topic across restarts; existing post updated on change).
- [2026-04-28] The remote web server handles ALL webview commands, reusing the extension code with
  minimal modification; no extension feature may be missing from the remote server.
- [2026-05-02] Blocking of accounts.google.com is implemented via a Playwright context.route() call
  in \_launch_browser right after launch_persistent_context returns (NOT in web_use_tool.py).
- [2026-06-03] RemoteAccessServer.start() calls logging.basicConfig(level=logging.INFO).
- [2026-07-17] When a remote webapp opens, it loads all running tasks in separate tabs and focuses
  the tab running the latest task.
- [2026-04-30] The remote webview fits mobile screens horizontally, uses the browser's regular font
  on all devices [2026-05-03], submits on Enter, and has right-side padding on the + button for
  edge presses [2026-05-12]; the welcome page centers the input textbox and buttons (no
  SAMPLE_TASKS suggestions) until a task starts [2026-05-02].
- [2026-06-27] A refreshed remote webapp tab never shows a stale merge/diff UI; the merge/diff view
  works in the remote app exactly as in the extension (hunk navigation, accept/reject with visible
  feedback) [2026-05-13].

## 16. Merge/diff workflow

- [2026-04-23] The merge/diff UI is shown in ALL tabs whose tasks have finished (no deferral while
  another tab is active); when shown, the tab is made active; per-tab editor-file snapshots are
  restored after all diffs/merges resolve.
- [2026-05-18] Rejecting changes to a file newly created by the agent DELETES the file; binary
  files with no hunk are opened and accepted/rejected the normal way (delete on reject).
- [2026-04-23] VS Code commands like kissSorcar.acceptChange / rejectChange / nextChange are
  non-keyboard-accessible.

## 17. Sorcar CLI

- [2026-06-05] Sorcar CLI interactive mode behaves like the Claude Code CLI: a Claude-CLI-style
  single merged input panel (input + steer dialogs merged) with complete borders, a right chevron
  and blinking cursor, horizontal lines above and below, an always-visible input bar
  [2026-06-20], screen cleared at launch, no raw YAML output, nothing printed after the Result
  panel, and ConsolePrinter panels content-matching the chat webview.
- [2026-07-07] The CLI REPL input header reads "TAB for autocomplete . Alt+Enter/Shift+Enter for
  newline"; left/right arrows move the cursor for editing; up/down arrows navigate within the
  input, traversing task history when nothing is typed — identical to the chat webview.
- [2026-06-12] Pressing @ shows a file/folder completion list navigable with arrows and selectable
  with tab/enter; matching fast completions appear as a list; pressing / shows all commands;
  in-place menus work even when the input box moves up from the screen bottom [2026-06-20].
- [2026-06-20] Messages entered while a task runs are added to the agent's context, as in the
  extension and remote webapp.
- [2026-06-15] The CLI defaults to worktree mode ON, parallel mode ON, and auto commit ON.
- [2026-06-21] The non-interactive CLI uses only SorcarAgent; the options -c/--chat-id,
  -l/--list-chat-id, --use-chat, --cleanup, and --use-worktree are removed from the project.
- [2026-06-11] The CLI supports Claude-compatible custom commands (.claude/commands .md files) and
  Agent Skills (on-demand SKILL.md loading via a skill tool, discovery from .claude/skills/ and
  ~/.claude/skills/, pattern-based permissions).
- [2026-07-08] All CLI notifications are shown in yellow.

## 18. Voice interface (mic, wake word, talk tool)

- [2026-07-05] The `talk` tool exists with language as the first argument and the audio text as the
  second; it plays the text in that language on the default speakers of every device with a tab
  open for the running task; it is a SorcarAgent default tool; SYSTEM.md documents that users can
  speak to a running task and the agent must respond via talk. The talk tool never speaks anything
  twice.
- [2026-07-07] Speech synthesis for the talk tool uses a suitable GPT model (from MODEL_INFO.json)
  in ALL interfaces; the voice must be highly natural and human-like with emotion — never
  robotic — without breaks or overlaps. Web Speech system voice is never used as a fallback
  anywhere in the project [2026-07-11].
- [2026-07-07] Wake-word speech submitted to the agent is prefixed "Speaker #N says in the language
  X that: " (N = unique per-speaker number from voice recognition, X = language detected during
  transcription); the spoken task is translated to English, placed in the input textbox, and
  submitted to the highlighted tab or as a steering instruction. (Supersedes the 2026-07-05 prefix
  "Speaker #N says that: ".)
- [2026-07-07] Voice transcription is done by a KISSAgent, which also returns the language of the
  user's speech.
- [2026-07-18] The default sorcar wake-word sensitivity is 80%. (Supersedes 85 from 2026-07-07.)
  The sensitivity is adjustable via a settings-panel slider that actually changes the sensitivity
  [2026-07-05].
- [2026-07-17] On first install, the mic is closed by default so it does not respond to the wake
  word. (Supersedes the 2026-07-05 mic-on-at-launch rule.) When the mic is listening, it turns OFF
  when the secondary VS Code bar closes and ON when it opens [2026-07-11].
- [2026-07-07] When the wake word is heard and the mic flashes red, "Listening ..." blinks in large
  fonts in the input textbox, then the assistant says "Working on it"; in the CLI, blinking
  "Listening ..." / "Transcribing ..." appear at the start of the input header while text input
  remains allowed [2026-07-08].
- [2026-07-08] The CLI voice interface works functionally the same as the chat webview voice
  interface (including the Speaker #N prefix) and they share code.
- [2026-07-07] The ask-user-question panel includes a mic button and also listens for the spoken
  reply after the wake word, sending it as the response to the agent; the ask window fits in the
  chat webview with reply textbox and buttons visible and long questions scrollable [2026-05-18].

## 19. Demo mode

- [2026-07-09] Demo mode demos only the clicked task (not the whole chain); prompts are narrated
  with the "User says " prefix through the shared talk queue (playTalkEvent) with the replay
  awaiting the promise so visuals never run ahead of audio; talk and run_parallel tool calls are
  actually executed so subagent tab creation is simulated; a finished demo leaves no state that
  interferes with the next demo.
- [2026-07-09] While a demo plays: hide the input textbox, burger menu, model picker, attach,
  inject-promptlet, mic, and send buttons; show a stop button in place of send (stopping animation
  and speech) with a pause/play toggle to its left; when the demo ends or is stopped, show only
  the play button, which restarts the demo [2026-07-10].
- [2026-04-23] Demo replay loads each panel in 0.5s and collapses it; the task text loads into the
  input textbox with a 2s pause; the result panel streams slowly; between tasks the replay stays
  in the same chat (never showing the welcome page) [2026-04-26].

## 20. Tips window

- [2026-07-06] The tips window (shown after a fresh install, and from the settings "Tips" button)
  renders the markdown after each "# Tip" line of ./src/kiss/TIPS.md with previous/next/close
  buttons; it has fixed height and width, centered both ways, scrollable overflow, a contrasting
  border, copy buttons on every \`\`\` code block, aesthetically pleasing (non-Claude/GPT-default)
  colors, and larger fonts.

## 21. Papers, website, and content rules

- [2026-07-15] Papers must be indistinguishable from human-written text: checked for consistency,
  hallucinations, citation correctness, duplication, and AI slop; less verbose with repeated text
  removed; follow the style of ./papers/swedefend/ and ./papers/kisssorcar/ [2026-07-19];
  first-person "we", Strunk & White style [2026-04-24]; no minted — prompts verbatim
  [2026-04-26]; each paragraph on a single line [2026-04-26]; marketing tone toned down
  [2026-05-14].
- [2026-07-14] Cite with LaTeX \\cite{...} (never "(arxiv:xxx)", listed titles, or obscure
  abbreviations); no made-up attributions (e.g., "Tramèr Standard") — cite specific papers, not
  people; citations must be well known, actually read, validated via internet search, with related
  work post-August-2025 and highly cited [2026-04-23/2026-06-06].
- [2026-05-03] Papers include the NeurIPS 2026 checklist and meet NeurIPS 2026 requirements; for
  papers/kisssorcar/ks_assistant.tex all content before References fits in 9 pages, verified by
  building the PDF and taking screenshots [2026-05-07]; always build any paper you modify
  [2026-05-04] and surface auto-commit/do-nothing buttons when a tracked artifact changes; VS Code
  settings wrap lines when a .tex file is shown [2026-04-26].
- [2026-06-26] Code-size/line-count claims ignore empty lines, comment-only lines, and docstrings
  of private functions/methods (also for the README Core Agents LoC count [2026-07-12]).
- [2026-07-21] KV store paper: do not describe bugs found/fixed; engine improvement/hardening only
  in Section 6 plus one intro paragraph; Sections 3–5 and the Conclusion never reference "the
  agent", KISS Sorcar, or the intermediate engine.
- [2026-07-21] The HydraKV engine score must not fall below 5.5 Mops/s (and never below the current
  best score) when fixing bugs or making changes.
- [2026-07-12] Do not mention "62.2% on Terminal Bench 2.0" on the website or in README.md.
- [2026-07-12] The kisssorcar.github.io site is white by default with the top logo blending into
  the page; it lists no authors (supersedes earlier author rules); Features precede the 5-layer
  architecture (both after setup); the Einstein quote follows the logo; it must not look
  AI-generated and must not contain "Built by hand, in Berkeley.", "Meet the Author", or badge
  rows; it includes "Time-Tested Engineering Principles" and general-purpose-assistant /
  third-party-agent features [2026-04-28]; the KISS-Sorcar.png logo sits at the top spanning the
  same width as the UI screenshot [2026-04-28]; the cleverest+ paper is removed from the website
  [2026-07-12].
- [2026-04-28] README and the website emphasize time-tested robust software engineering principles
  but never contain the fragment "—read before writing, test before fixing, plan before
  executing, verify before finishing—".
- [2026-06-26] Generated artifacts (slides, reports, reviews, animations) must not look/read
  AI-generated; slides are professional and elegant with large fonts and Python-syntax-highlighted
  code without indented imports [2026-04-29]; paper reviews suggest improvements, examine related
  work, judge novelty, and produce a ~2-page text/html report in ./reports, opened for the user.
- [2026-07-21] Social posts (LinkedIn/HackerNews) are modest in tone, free of AI slop, not
  AI-sounding; HackerNews post text stays within 4000 characters.
- [2026-07-26] Recommendation letters and similar prose must contain no AI slop and read as written
  only by a human, following ~/work/letters/sample.txt; the Mukul Prasad letter is at most 2000
  words.
- [2026-06-12] Fact-check all content of generated artifacts; verify claims independently.
- [2026-07-21] When replying to Slack, format the final summary with Slack mrkdwn — *bold*,
  _italic_, ~strike~, `code`, fenced code blocks, \<url|label> links, "- item" bullets — never
  markdown **bold**, __italic__, or [label](url). (First seen 2026-05-15.)

## 22. Agent behavior and process rules

- [2026-05-16] Internet research: visit at least 10 distinct websites using go_to_url() (not
  curl/wget), tracked in PWD/tmp/information-\*.md with a "Websites visited: N/10" counter, ≤10
  bullets per site; prefer small targeted pages; do not call get_page_content() right after
  go_to_url(). (Supersedes the 2026-05-04 "at least 30 websites" rule.)
- [2026-06-26] Use internet search extensively for research, feature design, reviews, and
  fact-finding.
- [2026-05-04] User-visible summaries must contain the full detailed answer, never a
  meta-description like "Answered the user's question…".
- [2026-07-18] The chat sorcar agent calls the summary tool after every 5 steps, summarizing the
  last 6 steps.
- [2026-07-28] When collecting invariants from task history into ./INVARIANTS.md, newer invariants
  take precedence over older conflicting invariants.
- [2026-07-11] Remember: KISS Sorcar is the most reliable SWE agent (see the paper).
- [2026-07-11] Pending user request: in a couple of weeks the user may ask to implement the full
  "Vision & Physical-World Control" feature set (camera capture, Home Assistant integration,
  scheduled camera monitors, closed-loop skills, ROS 2 bridge, safety guardrails); reuse govee.py
  for the light-actuation layer.
- [2026-05-14] Lines in USER_PREFS.md are cross-checked against the latest code and stale text
  removed.
- [2026-07-20] Project layout: the ./kv_adversarial/ and ./papers/swedefend/ artifacts were moved
  under ./projects/ (e.g., ./projects/kv_adversarial/ including KV_TASK.md; also
  ./projects/receipts/); paper references must point to the moved locations.
- [2026-04-23] Terminal Bench evaluation (src/kiss/benchmarks/terminal_bench/) builds the Python
  package before evaluation and uses that package.
- [2026-05-01] The "Run current file as prompt" button and its functionality do not exist anywhere
  in the project.
- [2026-05-18] Available values of a task's `extra` column are written when the task-history row is
  added; remaining values may be added after the task completes.
- [2026-05-20] Steps/tokens/cost shown in task panels come from RelentlessAgent's budget_used,
  total_tokens_used, and total_steps (kiss_agent.py is not modified for this).
- [2026-05-22] The tab bar in the menu panel is horizontally scrollable; sidebar panels are 90% of
  the chat webview width.
- [2026-04-23] The file picker's directory scan depth is 10.
- [2026-06-27] The Read tool matches Claude Code's Read tool in both signature (offset/limit) and
  implementation [2026-06-24], supports a start_line parameter, and supports binary files for ALL
  models and whatever MIME types each model supports [2026-05-22].
