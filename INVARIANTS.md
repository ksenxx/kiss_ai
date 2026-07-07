# KISS Sorcar — Invariants

This document collects the **invariants** (properties that MUST always hold)
that the user described across tasks recorded in `~/.kiss/sorcar.db`
(`task_history` table). Each invariant is phrased as a canonical "MUST / MUST
NOT" statement, grouped by subsystem. Numbers in parentheses are the source
`task_history.id`s where the invariant was described (duplicates from retries
are merged).

> Note: pure research questions, one-off paper/slide-formatting requests, and
> machine-generated bug-hunt/audit prompts were excluded — only durable
> behavioral invariants of the system are listed.

______________________________________________________________________

## Cloudflare tunnel & remote web server

- When a Cloudflare tunnel is restarted by
  `src/kiss/agents/vscode/web_server.py`, it MUST NOT restart the WSS/kiss-web
  server. (3745, 3746)
- The Cloudflare tunnel URL MUST remain stable and stay the same as long as
  possible (it MUST NOT change every few minutes). (1049, 1052, 1057, 1086)
- When `kiss_web` starts or the Cloudflare URL changes, the cloudflared URL MUST
  be posted as a private message to a widely-used public message board that
  requires no login; that board MUST NOT change across machines. (1036)
- The `remote_password`, once set in `config.json`, MUST NOT be overwritten
  until set again in the Settings panel. (1044)
- The remote web server MUST remain responsive even after the MacBook lid is
  closed. (412)
- The remote web server MUST run as a daemon (started by `install.sh`), MUST
  auto-restart if killed, MUST survive machine reboots, and MUST run in the
  `kiss_ai` directory. (379)
- `--tls` and `--tunnel` MUST be the default when `uv run kiss-web` is run. (377)
- A single server MUST serve both the extension and remote web clients. (1749)
- The remote webapp MUST be made as similar to the extension as possible and
  share code as much as possible. (349, 353, 1741, 3271)
- The remote webapp MUST show only the chat webview (tabs, input textbox,
  features) and no other KISS Sorcar / VS Code UI components. (344)
- A tab that runs a task and a tab that loads a running task MUST behave
  similarly in both the extension and the web server. (1741)

## Sorcar CLI

- The sorcar CLI MUST set worktree mode on, parallel mode on, and auto-commit on
  by default. (3747)
- When the sorcar CLI is launched, `work_dir` MUST be set to the directory from
  which sorcar is invoked. (3162)
- If neither `-t` nor `-f` is provided to `sorcar`, `--use-chat` MUST be turned
  on and sorcar MUST behave like the Claude Code CLI (e.g. wait for user input
  after a task finishes); all fast completes MUST work exactly like the
  extension input textbox. (3055, 3056)
- In the sorcar CLI, pressing `@` MUST show a list of files/folders to complete;
  the list MUST support up/down arrows and tab/enter to select. (3670)
- Fast completions MUST be shown as a list when a match exists; the user MUST be
  able to use up/down arrows and tab to accept an option. (3673)
- The sorcar CLI MUST show only one input panel (the sorcar input panel), not
  both the steer and sorcar input panels. (3083)
- In the steer input dialog, the cursor MUST blink after the chevron the same
  way as in the sorcar input dialog. (3087)

## ChatSorcarAgent & agent lifecycle

- Any `ChatSorcarAgent` MUST be registered with the server. (3744)
- `ChatSorcarAgent` MUST maintain a class attribute `running_agents` mapping task
  id → agent; an entry MUST be created as soon as the task is persisted in the DB
  and removed as soon as the task finishes. (1799)
- Closing a tab MUST NOT stop the running agent — the agent MUST finish its task
  even though the tab on the chat webview closes. (1662)
- When a `ChatSorcarAgent` is executed on a task outside the UI:
  - the task MUST show a pulsing/blinking green circle in the task history panel
    as long as the task is running; (3742)
  - the user MUST be able to stream the agent's events into a chat-webview tab by
    clicking the task in the task history panel, and that tab's title MUST show
    the blinking green circle while the agent is running. (3740)
- Every row in the History panel MUST carry a middle-left status circle: a
  pulsing green circle while running, a solid green circle once finished
  cleanly, and a solid red circle when the task failed. A freshly-started task
  MUST appear at the top of the History list when the burger menu is opened.

## Tabs & chat webview

- Every tab MUST show the running time as the actual elapsed time since the
  task's start while running, and as (end − start) once the task has ended; this
  status MUST NOT be overridden by newer runs in another window. (3640)
- When a running task is loaded into a tab, its events MUST be printed exactly as
  when the tab itself runs the task, and the buttons/labels below the input
  textbox MUST also be shown. (1762, 1763)
- When an agent with sub-agents is restored after a VS Code restart, it MUST load
  its own chat events in a tab and then load each sub-agent's events in separate
  tabs to the right of the parent tab. (3538)
- When a regular agent tab is closed, it MUST close all of its sub-agent
  tabs. (2289, 2301)
- Clicking the chevron of a task in the fixed panel MUST collapse/expand only the
  panels of that task, and the chevron state MUST be specific to each task in the
  chat session. (583)
- When a task is clicked in the task history panel, the tab MUST scroll to the
  clicked task, not to the end. (758)
- Showing "Suggested next" MUST NOT scroll the chat webview to the end. (811)
- Copying from any chat-webview panel MUST copy the raw text (e.g. markdown), not
  the formatted text. (773, 3195)
- The tab id MUST be set to the chat id when an agent starts running and when a
  task is loaded from the task history; maintaining a separate `tab_id == chat_id`
  translation is not required. (1708, 1709, 1824)
- The file picker in the input textbox of both the extension and webapp chat
  webviews MUST be relative to `work_dir`. (3163)

## Sub-agents & run_parallel

- Sub-agent result panels MUST NOT be shown in the parent agent's chat
  webview. (2468, 2477, 2593, 2594, 2606)
- Each parallel sub-agent MUST get its own tab that shows that sub-agent's
  events; such tabs MUST NOT be counted as regular tabs when deciding to close
  old tabs. (1278, 1414, 1651)
- In the chat webview, a `run_parallel` tool-call panel and the tabs of its
  sub-agents MUST stay in sync: when the panel is uncollapsed the sub-agent
  tabs are open, and when the panel is collapsed those tabs are closed.
- Once the `run_parallel` tool finishes (its tool_result arrives), the
  automatic collapse passes MUST collapse its panel like any other tool panel
  — closing the fan-out's sub-agent tabs; while the fan-out is still running,
  the automatic passes MUST NOT collapse the panel (its live sub-agent tabs
  stay open).
- Replaying or re-rendering the parent tab MUST NOT break the `run_parallel`
  panel ⇔ sub-agent tab association: the replayed panel must adopt already-open
  sub-agent tabs before deciding whether to collapse or remain open.
- A sub-agent task MUST be treated like a regular task in `RunningAgentState`,
  but its tab MUST NOT show/load other tasks of the same chat id and MUST use a
  different color/icon for its header. (1905, 1907)
- `parent_tab_id` MUST NOT be stored in the database. (2764)

## Cost accounting

- The final cost of an agent MUST be greater than or equal to the sum of its
  sub-agents' costs plus the agent's own cost. (2016, 2019)
- If a task's cost exceeds the user-set max budget, the result panel MUST show
  "Budget exceeded". (314)

## Diff/merge & auto-commit

- In auto-commit mode, nothing MUST be printed if the agent did not change any
  file. (2901)
- When a task runs in a git repo and modifies no file, the error
  "Not a git repository" MUST NOT be reported. (2898)
- When auto-commit mode and worktree mode are both on, no files were modified,
  and the task completes successfully, the agent MUST NOT auto-commit/merge — it
  MUST discard the worktree instead. (2171)
- In the diff/merge workflow, when the user rejects changes to a new file that
  was created by the agent, the file MUST be deleted. (1916, 1926)
- If "Auto commit" is toggled on, the merge/diff workflow MUST be skipped and the
  agent's changes MUST be auto-committed; if auto-commit is on and worktree mode
  is on, the merge/diff workflow MUST be skipped, changes auto-committed, and
  auto-merged with the original branch. (1843, 1844)
- When the user queues tasks while a task is running, merge/diff and auto-commit
  MUST be delayed until all queued tasks have run; once the queue becomes empty,
  diff/merge (if there were filesystem changes) followed by auto-commit MUST
  start. (276, 286, 287)
- A queued task that runs MUST NOT snapshot the file state in `merge_dir`
  (because the previous merge/diff was skipped). (277)
- The diff/merge UI MUST be shown in all tabs that have finished their tasks (no
  deferral). (67)

## Task history & frequent tasks panels

- Deleting a task MUST remove it from the database and from the task list panel;
  any child of the deleted task MUST be re-parented to the deleted task's
  parent. (571, 573)
- Clicking the delete button MUST show tick/X confirmation buttons in its place;
  on tick the task MUST be deleted, otherwise the delete button returns. (574)
- The copy button on a task panel MUST copy the entire task text to the
  clipboard. (2554)
- The `frequent_tasks` table MUST be restricted to a maximum of 100 entries; on
  each task submission the task's count MUST be incremented and its timestamp set
  to the current time; when evicting, the lowest-count and oldest entry MUST be
  removed; the panel MUST show the top 20 tasks by decreasing count; clicking an
  item MUST copy the task to the input textbox. (649, 650)
- When using up/down arrows over history in the input textbox, the textbox MUST
  grow taller if the task text does not fit. (450)

## Settings / configuration

- Configurations MUST be saved whenever the settings panel closes (no separate
  "Save configuration" button). (759)
- The folder / working-directory picker MUST open showing the current working
  directory. (305, 752)
- `last_model_used` and model usage counts MUST be updated only when the user
  selects a model using the model picker. (217)
- The default model and fast model MUST NOT be hardcoded anywhere in
  `src/kiss/agents/vscode/`; they MUST be obtained from
  `src/kiss/core/models/model_info.py`. (717)
- The menu button MUST have horizontal bars and no background, matching the style
  of the other buttons. (586, 587)
- If a task is run without any model, the result panel MUST show "No model
  available. Set at least one API key in the environment." (232)

## Installation / release / update

- `install.sh` MUST be located at `~/kiss_ai/`. (3180)
- The Settings "Update" button MUST run `install.sh` and notify the user that an
  update is being installed. (3035)
- `install.sh` MUST ask for the `remote_password` if it is not set, before
  running `kiss_web`. (1034, 1035)
- `install.sh` MUST NOT refer to the Claude skills; downloading and bundling
  the skills is owned by `release.sh` and `scripts/build-extension.sh`. (676)
- `release.sh` MUST update the version number in `src/kiss/SYSTEM.md`,
  `README.md`, and every other file that uses the version number; the version
  number MUST have the format `yyyy.mm.minor`. (90, 91, 1176)
- During VS Code extension activation, `DependencyInstaller.ts` MUST restart the
  web server (its code may have changed) instead of calling
  `ensureKissWebDaemon`. (448)

## Tools

- `Read` MUST support binary files. (2507)
- `update_models.py` MUST include all GPT models that can be used with
  codex. (660)
- When codex is installed only via the UI (no CLI), the binary from the UI
  installation MUST be used when codex is used as a model. (659)

## Models

- The `cc/opus` model MUST behave like an ordinary model (e.g.
  `claude-opus`): the Claude Code CLI MUST be used as a model making a single
  call and returning a response, without invoking the Claude Code agent
  internally; Sorcar MUST extract the tool calls and execute them
  explicitly. (929, 931, 934, 938, 940, 942)

## Demo mode

- In demo mode the red stop button and the spinner MUST be shown; pressing the
  stop button MUST stop the demo and bring back the send button. (220)
- When the demo switches from one task to the next, the animation MUST be a
  continuation of the same chat (the welcome page MUST NOT be shown). (219)

## Other components

- Papers that have been modified MUST always be rebuilt. (768)
- In the code-size claim, empty lines and lines with comments or private
  function/method docstrings MUST be ignored. (3125)
