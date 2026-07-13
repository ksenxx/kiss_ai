# KISS Sorcar Python API Reference

> The core Python classes of the KISS Agent Framework (`kiss-agent-framework` on PyPI): KISSAgent, RelentlessAgent, SorcarAgent, ChatSorcarAgent, WorktreeSorcarAgent, and GitWorktreeOps. Auto-generated from source; the canonical version lives at <https://github.com/ksenxx/kiss_ai/blob/main/API.md>.

## `kiss.core.kiss_agent` — Core KISS agent with native function calling

### `class KISSAgent(Base)`

A KISS agent using native function calling.

**Constructor:** `KISSAgent(name: str) -> None`

- **run** — Runs the agent's main ReAct loop to solve the task.

  ```python
  run(model_name: str, prompt_template: str, arguments: dict[str, str] | None = None,
      system_prompt: str = '', tools: list[Callable[..., Any]] | None = None,
      is_agentic: bool = True, max_steps: int | None = None, max_budget: float | None = None,
      model_config: dict[str, Any] | None = None, printer: Printer | None = None,
      verbose: bool | None = None, attachments: list[Attachment] | None = None,
      print_prompts: bool = True) -> str
  ```

  Key parameters: `model_name` (LLM to use), `prompt_template` + `arguments` (task prompt with substitutions), `tools` (callables exposed to the model; a built-in `finish` tool is always added), `max_steps` (default 100), `max_budget` (default $10), `attachments` (images/PDFs for the initial prompt). Returns the result string of the agent's task.

- **finish** — `finish(result: str) -> str`. The agent must call this with the final answer.

## `kiss.core.relentless_agent` — Auto-continuation for long tasks

### `class RelentlessAgent(Base)`

Base agent with auto-continuation across multiple sub-sessions for long-horizon tasks.

- **perform_task** — `perform_task(tools, attachments=None) -> str`. Executes the task with auto-continuation across sub-sessions; returns a YAML string with `success` and `summary` keys.
- **run** — Full signature adds: `model_name`, `prompt_template`, `arguments`, `system_prompt`, `max_steps` (per sub-session), `max_budget` (USD), `model_config`, `work_dir`, `printer`, `max_sub_sessions`, `docker_image` (run tools inside a container), `verbose`, `tools`, `attachments`. Returns YAML with `success` and `summary`.
- **finish** — `finish(success: bool, is_continue: bool = False, summary: str = '') -> str`. `is_continue=True` pauses an incomplete task so it resumes in a new sub-session.

## `kiss.agents.sorcar.sorcar_agent` — Coding + browser automation

### `class SorcarAgent(RelentlessAgent)`

Agent with both coding tools and browser automation for web + code tasks.

**Constructor:** `SorcarAgent(name: str) -> None`

- **run** — Adds on top of `RelentlessAgent.run`: `web_tools: bool = True` (set False for terminal-only), `is_parallel: bool = False` (enables the `run_parallel` tool for spawning parallel sub-agents), `current_editor_file` (path appended to the prompt), `ask_user_question_callback` (collects a text response from the user). Returns YAML with `success` and `summary`.

### Module helpers

- **`auto_commit_changes(commit_dir, user_prompt, message_fn, notify_fn=None) -> bool`** — Stage all changes, generate a commit message (typically via an LLM), and commit. Re-stages just before committing so late-arriving files are included. Falls back to a generic message if `message_fn` raises. Returns True if a commit was created.
- **`run_tasks_parallel(tasks, max_workers=None, model_name=None, work_dir=None, printer=None, totals_out=None) -> list[str]`** — Execute multiple SorcarAgent tasks concurrently with a thread pool; each task gets its own `ChatSorcarAgent`. Returns YAML result strings in the same order as `tasks`.

## `kiss.agents.sorcar.chat_sorcar_agent` — Chat-session persistence

### `class ChatSorcarAgent(SorcarAgent)`

SorcarAgent with chat-session state management.

- **chat_id** *(property)* — Current chat session ID ("" means new session).
- **new_chat()** — Reset to a new chat session.
- **resume_chat_by_id(chat_id)** — Resume a chat session by stable identifier.
- **resume_from_task_id(task_id)** — One-shot seed of the next prompt's context from a task's parent chain.
- **build_chat_prompt(prompt) -> str** — Load chat context and prepend previous tasks/results to the prompt.
- **`run(prompt_template='', **kwargs) -> str`** — Run with chat-session context: loads prior context, persists the new task, runs the underlying agent, and saves the result to history.

## `kiss.agents.sorcar.git_worktree` — Git worktree operations

### `class GitWorktree` — Immutable snapshot of a pending worktree task.

### `class MergeResult(enum.Enum)` — Outcome of a merge operation (`SUCCESS` / `CONFLICT`).

### `class GitWorktreeOps` — Stateless helper with all git worktree operations

Highlights (all methods take explicit `repo`/`wt_dir` paths):

- `discover_repo(path)`, `current_branch(repo)`, `head_sha(wt_dir)`, `branch_exists(repo, branch)`
- `create(repo, branch, wt_dir)`, `remove(repo, wt_dir)`, `prune(repo)`, `cleanup_partial(repo, branch, wt_dir)`
- `stage_all(wt_dir)`, `commit_all(wt_dir, message)`, `commit_staged(wt_dir, message, no_verify=False)`, `staged_diff(wt_dir)`
- `has_uncommitted_changes(wt_dir)`, `status_porcelain(wt_dir)`, `unstaged_files(repo)`, `staged_files(repo)`
- `checkout(repo, branch)`, `stash_if_dirty(repo)`, `stash_pop(repo)`, `delete_branch(repo, branch)`
- `squash_merge_branch(repo, branch)`, `squash_merge_from_baseline(repo, branch, baseline)` — squash-merge a task branch (or only the agent's changes after a baseline commit) into HEAD, returning `MergeResult`
- `copy_dirty_state(repo, wt_dir)`, `save_baseline_commit` / `load_baseline_commit`, `save_original_branch` / `load_original_branch`
- `ensure_excluded(repo)` — adds `.kiss-worktrees/` to the repo-local git exclude
- `ensure_scratch_merge_driver(repo)` — installs a merge driver that auto-resolves agent scratch files (e.g. `PROGRESS.md`)

### Module helpers

- **`repo_lock(repo) -> threading.RLock`** — Per-repo re-entrant lock serializing multi-step git operations across concurrent tabs.
- **`strip_worktree_suffix(path) -> str`** — Strip the ephemeral `.kiss-worktrees/kiss_wt-<slug>` suffix so persisted paths always point at the parent repo.

## `kiss.agents.sorcar.worktree_sorcar_agent` — Isolated task branches

### `class WorktreeSorcarAgent(ChatSorcarAgent)`

SorcarAgent that isolates every task in a git worktree.

- **`run(prompt_template='', **kwargs) -> str`** — Creates a new worktree and branch, redirects `work_dir` into the worktree, and delegates to `ChatSorcarAgent.run()`. Any previously pending branch is auto-committed and squash-merged first. Falls back to direct execution when `use_worktree=False`, when `work_dir` is not in a git repo, when the repo has no commits, or when HEAD is detached.
- **merge() -> str** — Merge the task branch into the original branch. Idempotent; auto-commits uncommitted worktree changes and stashes/restores user edits on main.
- **discard() -> str** — Throw away the task branch and worktree, checkout the original branch. Idempotent.
- **new_chat()** — Reset to a new chat session, auto-merging any pending worktree first.
