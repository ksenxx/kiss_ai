# KISS Framework API Reference

> **Auto-generated** — run `uv run generate-api-docs` to regenerate.

<details><summary><b>Table of Contents</b></summary>

    - [`kiss.core.kiss_agent`](#kisscorekiss_agent)
      - [`kiss.agents.sorcar.relentless_agent`](#kissagentssorcarrelentless_agent)
      - [`kiss.agents.sorcar.sorcar_agent`](#kissagentssorcarsorcar_agent)
      - [`kiss.agents.sorcar.chat_sorcar_agent`](#kissagentssorcarchat_sorcar_agent)
    - [`kiss.server.sorcar`](#kissserversorcar)
      - [`kiss.agents.sorcar.daemon_client`](#kissagentssorcardaemon_client)

</details>

---

#### `kiss.core.kiss_agent` — *Core KISS agent implementation with native function calling support.*

##### `class KISSAgent(Base)` — A KISS agent using native function calling.

**Constructor:** `KISSAgent(name: str) -> None`

- **budget_used** — Cumulative USD spend, derived from the atomic usage snapshot.<br/>`budget_used() -> float` *(property)*
- **budget_used**<br/>`budget_used(value: float) -> None`
- **total_tokens_used** — Cumulative tokens, derived from the atomic usage snapshot.<br/>`total_tokens_used() -> int` *(property)*
- **total_tokens_used**<br/>`total_tokens_used(value: int) -> None`
- **step_count** — Completed agentic steps, derived from the atomic usage snapshot.<br/>`step_count() -> int` *(property)*
- **step_count**<br/>`step_count(value: int) -> None`
- **usage_snapshot** — Return one coherent ``(budget_used, total_tokens_used, step_count)``. Reads the immutable snapshot attribute ONCE, so the triple can never mix dimensions from two different accounting states — unlike three separate property reads, between which a concurrent response-accounting publish could land.  Concurrent readers (``sorcar_agent._executor_usage`` polled by the live usage monitor and by abandoned-child reclaims, the parent's session bank) rely on this.<br/>`usage_snapshot() -> tuple[float, int, int]`
- **reset_usage** — Zero this agent's usage counters before a run. The ``(budget_used, total_tokens_used, step_count)`` triple is replaced with ONE store, so a concurrent :meth:`usage_snapshot` reads either the previous totals or all zeros, never a mix. Called by :meth:`_reset` and by the task runner before a run (``task_runner._zero_usage_counters``).<br/>`reset_usage() -> None`
- **run** — Runs the agent's main ReAct loop to solve the task.<br/>`run(model_name: str, prompt_template: str, arguments: dict[str, str] | None = None, system_prompt: str = '', tools: list[Callable[..., Any]] | None = None, is_agentic: bool = True, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, printer: Printer | None = None, verbose: bool | None = None, attachments: list[Attachment] | None = None, print_prompts: bool = True, llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None, tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None) -> str`
  - `model_name`: The name of the model to use for the agent.
  - `prompt_template`: The prompt template for the agent.
  - `arguments`: The arguments to be substituted into the prompt template. Default is None.
  - `system_prompt`: Optional system prompt to provide to the model. Default is empty string (no system prompt).
  - `tools`: The tools to use for the agent. If None, no tools are provided (only the built-in finish tool is added).
  - `is_agentic`: Whether the agent is agentic. Default is True.
  - `max_steps`: The maximum number of steps to take. Default is 10000.
  - `max_budget`: The maximum budget to spend. Default is 10.0.
  - `model_config`: The model configuration to use for the agent. Default is None.
  - `printer`: Optional printer for streaming output. Default is None.
  - `verbose`: Whether to print output to console. Default is None (verbose enabled).
  - `attachments`: Optional file attachments (images, PDFs) to include in the initial prompt. Default is None.
  - `print_prompts`: Whether to print the system prompt and task prompt to the printer. Internal helper agents (e.g. the summarizer in RelentlessAgent) pass False so their internal prompts never surface as user-visible "prompt" events in a shared printer's event stream. Default is True.
  - `llm_call_hook`: Optional hook called before every ``generate_and_process_with_tools`` LLM call with the list of new messages (those added to the conversation since the previous LLM call) about to be sent to the LLM. Its return value — a possibly modified list of messages — replaces those new messages in the conversation before the call is made. After a turn in which the model called tools natively (Codex over MCP) the new messages include those calls and their results, and the turn's answer. Default is None (no hook).
  - `tool_call_hook`: Optional hook called before every tool call with the tool's name and its arguments dict; it returns a :class:`~kiss.core.tool_verdict.Verdict`: ``ALLOW`` lets the tool execute as usual, ``refuse(text)`` suppresses the execution and the model reads :func:`~kiss.core.tool_verdict.refusal_result` of *text* ("REFUSED: the X call did NOT run and had no effect. ..."), then a reminder to re-issue the call on the next allowed call of another tool. The hook runs before (and its rejection takes precedence over) the framework's :attr:`tool_call_guard`; allowing does not override a guard block. An implicit finish (text-only turns) also consults the hook (with ``("finish", {})``) and is suppressed when the hook refuses. Default is None (no hook).
  - **Returns:** str: The result of the agent's task.

- **finish** — The agent must call this function with the final answer to the task.<br/>`finish(result: str) -> str`
  - `result`: The result generated by the agent.
  - **Returns:** Returns the result of the agent's task.

**`is_routine_turn`** — Return whether the model turn after this tool call is a routine one. Routine turns (P14 of the 2026-10-24 trajectory audit) are the one after a refused call, a :data:`ROUTINE_TOOLS` call, a ``bash_job`` / ``agent_job`` ``wait`` that ended with the job still running (the model only waits again), or a ``Read`` whose window was unchanged and already in the context.  Planning, debugging and review turns are never routine: a wait that returned a finished job's result (a test run, a reviewer's findings) is one of those.<br/>`def is_routine_turn(name: str, args: dict[str, Any], result: str, refused: bool) -> bool`

- `name`: The tool that was called.
- `args`: Its arguments.
- `result`: Its result text (or the refusal text).
- `refused`: Whether a hook or guard refused the call.

**`unknown_argument_hint`** — Return a hint for the arguments of a tool call that its signature does not take. A tool that knows its own vocabulary says so through an ``unknown_arguments_hint`` attribute, a function from the unknown ``{name: value}`` pairs to the hint text (``run_agent`` and ``run_parallel`` tell an options key from a renamed, removed or misspelled one).  Otherwise a tool with an ``options`` parameter takes its run settings as one JSON object, so a setting passed as a keyword (``use_memory=False``) is pointed there; the agent cannot know the accepted keys, so that hint is conditional: the tool itself rejects a key that is not a setting.<br/>`def unknown_argument_hint(sig: inspect.Signature | None, args: dict[str, Any], tool: Any = None) -> str`

- `sig`: The tool's signature, ``None`` when unknown.
- `args`: The arguments of the call.
- `tool`: The tool callable, for its ``unknown_arguments_hint``.
- **Returns:** The hint (starting with a newline), or ``""`` when every argument is a parameter or the tool takes no ``options``.

---

#### `kiss.agents.sorcar.relentless_agent` — *Base relentless agent with smart continuation for long tasks.*

##### `class SummarizerFinish` — The failed-session summarizer's ``finish`` tool.

- **finish** — Finish the summary with the work done so far.<br/>`finish(result: str) -> str`
  - `result`: A precise chronologically-ordered account of the work done so far, formatted as HTML (never Markdown).
  - **Returns:** The structured continuation result (YAML with ``success``, ``is_continue`` and ``summary`` keys).

##### `class RelentlessAgent(Base)` — Base agent with auto-continuation for long tasks.

**Constructor:** `RelentlessAgent(name: str) -> None`

- `name`: The name identifier for the agent.

- **budget_used** — Cumulative USD spend banked so far (see :class:`_UsageEvent`).<br/>`budget_used() -> float` *(property)*
- **budget_used**<br/>`budget_used(value: float) -> None`
- **total_tokens_used** — Cumulative tokens banked so far (see :class:`_UsageEvent`).<br/>`total_tokens_used() -> int` *(property)*
- **total_tokens_used**<br/>`total_tokens_used(value: int) -> None`
- **total_steps** — Cumulative steps banked so far (see :class:`_UsageEvent`).<br/>`total_steps() -> int` *(property)*
- **total_steps**<br/>`total_steps(value: int) -> None`
- **reset_usage** — Reset the whole accounting state to zero in ONE atomic store. Swaps in a fresh ledger epoch with a single ``STORE_ATTR`` — atomic under both thread interleaving and asynchronously injected exceptions.  A writer that already loaded the OLD epoch appends its record there: the transaction linearizes BEFORE the reset and is discarded with the old epoch.  A writer that loads the ledger after the swap lands in the new epoch with its whole triple.  Either way every observable state is coherent — never a mix of pre- and post-reset dimensions, and never a torn triple.  A session bank retried across the swap counts exactly once: its duplicate record carries the same session key, and the old epoch's record is no longer summed. The swap is also the EPOCH BOUNDARY for adjustment sources that outlive a run: a daemon sub-task's spend is tagged with the epoch token (:meth:`_usage_epoch`) when the job starts and its fold commits into that exact object, so a prior epoch's late spend settles in the discarded ledger and is never banked into the new epoch (an explicit, documented undercount versus real provider spend — see ``kiss.server.task_update``). Also the coherent replacement for zeroing the three counter properties one by one (the server's ``_zero_usage_counters``), which could otherwise interleave with a racing attribution.<br/>`reset_usage() -> None`
- **usage_snapshot** — Return one coherent ``(budget_used, total_tokens_used, total_steps)``. Sums ONE ledger epoch (see :func:`_ledger_totals`), so the triple always describes a single prefix of the append-only history.  Reading the three properties separately instead can TEAR: each property read sums the ledger afresh, and a concurrent append or reset between two of those reads yields an impossible mix (e.g. the old budget with the new tokens/steps) — a final abandoned-child reclaim that reads such a mix permanently loses the unseen dimension from the parent's accounting.<br/>`usage_snapshot() -> tuple[float, int, int]`
- **live_usage_snapshot** — Return :meth:`usage_snapshot` plus the in-flight session's spend, counted once. A session's spend is folded into the ledger only when the session ends, so mid-session it is visible only on ``_current_executor``.  ``perform_task`` banks the session BEFORE dropping it from ``_current_executor``, so a reader here can observe both; the executor's triple is read first and then added only when the ledger prefix that produced the totals does not already contain the session (:func:`_ledger_totals_and_banked`), so every interleaving counts the session exactly once.  A session handed off (or an epoch reset by :meth:`reset_usage`) between the executor read and the ledger read is detected by re-reading ``_current_executor`` afterwards, and the read starts over, so the live triple is never added to a ledger it does not belong to.<br/>`live_usage_snapshot() -> tuple[float, int, int]`
- **perform_task** — Execute the task with auto-continuation across multiple sub-sessions. Each sub-session is a fresh :class:`KISSAgent`; one that returns ``finish(is_continue=True, ...)`` hands its summary to the next. The ``IMPORTANT_INSTRUCTIONS`` suffix names the work dir when the tools can reach it: on the host, or in a container that bind-mounts it (every container kiss starts from an image does; see :data:`WORK_DIR_LINE`).<br/>`perform_task(tools: list[Callable[..., Any]], attachments: list[Attachment] | None = None) -> str`
  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys on successful completion.

- **run** — Run the agent with the provided tools.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str = '', max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, verbose: bool | None = None, tools: list[Callable[..., Any]] | None = None, attachments: list[Attachment] | None = None, llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None, tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None) -> str`
  - `model_name`: LLM model to use. Defaults to "claude-opus-4-6".
  - `prompt_template`: Task prompt template with format placeholders.
  - `arguments`: Dictionary of values to fill prompt_template placeholders.
  - `system_prompt`: System-level instructions passed to the underlying LLM via model_config. Defaults to empty string (no system instructions).
  - `max_steps`: Maximum steps per sub-session. Defaults to 10000.
  - `max_budget`: Maximum budget in USD. Defaults to 200.0.
  - `model_config`: Optional dictionary of additional model configuration parameters (e.g. temperature, top_p). Defaults to None.
  - `work_dir`: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
  - `printer`: Printer instance for output display.
  - `max_sub_sessions`: Maximum continuation sub-sessions. Defaults to 10000.
  - `docker_image`: Docker image name to run tools inside a container.
  - `verbose`: Whether to print output to console. Defaults to True.
  - `tools`: List of callable tools available to the agent during execution.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - `llm_call_hook`: Optional hook installed on every per-session executor :class:`KISSAgent` (see :meth:`kiss.core.kiss_agent.KISSAgent.run`): called before every LLM call with the new messages about to be sent, and its return value replaces them.  Defaults to None (no hook).
  - `tool_call_hook`: Optional hook installed on every per-session executor :class:`KISSAgent` (see :meth:`kiss.core.kiss_agent.KISSAgent.run`): called before every tool call with the tool's name and arguments and returns a :class:`~kiss.core.tool_verdict.Verdict`: ``refuse(text)`` suppresses the call and the model reads *text* as the tool's result, ``ALLOW`` lets it run. Defaults to None (no hook).
  - **Returns:** YAML string with 'success' and 'summary' keys.

**`publish_usage_offsets`** — Publish *agent*'s banked totals as *printer*'s usage offsets. The printer adds its per-task offsets to every session-relative event, so the UI shows the task's cumulative usage.  The two writers — the session-start rebase on the agent thread (:meth:`RelentlessAgent._set_printer_offsets`) and a finished child's fold on its own thread (``sorcar_agent._attribute_sub_usage``) — both call this under :data:`_OFFSET_PUBLISH_LOCK`, so publishes land in snapshot order and the offsets never decrease. The offsets belong to the agent's own task: a fold runs on whatever thread finished the child (a dispatch job thread, the ``/update`` worker, a server thread), whose thread-keyed setters would file them under that thread's task, so the agent's persisted task is named when it has one (``last_task_id``) and the printer can take a task id (``set_usage_offsets``); otherwise the plain setters serve (a console printer, an agent without a task row).<br/>`def publish_usage_offsets(agent: RelentlessAgent, printer: Any) -> None`

- `agent`: The agent whose ledger totals are published.
- `printer`: Its printer.

**`resolve_work_dir`** — The absolute working directory a run with this ``work_dir`` argument uses. ``None`` or ``""`` means ``artifact_dir/kiss_workdir``.  The directory is not created here; :meth:`RelentlessAgent._reset` does that when the run starts. A *work_dir* inside a ``.kiss-worktrees/kiss_wt-*`` worktree that has already been torn down (a finished worktree task's ``work_dir`` handed to a follow-up run) is remapped to the parent repository (:func:`~kiss.agents.sorcar.useful_tools.remap_vanished_worktree`), so ``_reset``'s ``mkdir`` never resurrects the removed worktree directory.<br/>`def resolve_work_dir(work_dir: str | None) -> str`

- `work_dir`: The ``work_dir`` argument of :meth:`RelentlessAgent.run`.

---

#### `kiss.agents.sorcar.sorcar_agent` — *Sorcar agent with both coding tools and browser automation.*

##### `class SorcarAgent(RelentlessAgent)` — Agent with coding tools, browser automation and git-worktree isolation.

**Constructor:** `SorcarAgent(name: str) -> None`

- **perform_task** — Execute the task, building docker-aware tools after docker_manager is set.<br/>`perform_task(tools: list, attachments: list | None = None) -> str`
  - `tools`: Extra tools passed by the caller (from run(tools=...)).
  - `attachments`: Optional file attachments for the initial prompt.
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **classify_task_for_run** — Classify *task* now and pre-seed the verdict for coming runs. For external drivers that classify BEFORE calling :meth:`run` (the server's task runner shows a "Classifying task…" phase). The pre-seeded verdict is then reused by the run itself (system prompt selection) and by every later subtask of the same submission, so the driver and the agent can never disagree. The seed lasts until the next ``classify_task_for_run`` call on this agent.<br/>`classify_task_for_run(model_name: str | None, task: str, model_config: dict[str, Any] | None = None, enabled: bool | None = None) -> TaskClassification | None`
  - `model_name`: The model the run will use, possibly None.
  - `task`: The (already substituted) task prompt about to run.
  - `model_config`: The model configuration the run will use.
  - `enabled`: Per-run override of the persisted ``classify_tasks`` setting — the ``classifyTasks`` wire field of the ``run`` command (the *auto_classify* parameter of :func:`kiss.server.sorcar.run`).  ``True`` forces classification on, ``False`` skips it (the run then behaves exactly as it would without a classifier), and ``None`` (the default) follows the config.  The ``KISS_DISABLE_TASK_CLASSIFIER`` environment kill switch wins over any override.
  - **Returns:** The task's classification, or ``None`` when classification is disabled or failed.

- **run** — Run the assistant agent with coding tools and browser automation. A worktree still pending from an earlier run of this agent is retired first (:meth:`_retire_previous_worktree`: committed to its own branch, never merged), so the task starts in *work_dir* and its ``create_worktree`` tool makes a fresh worktree.  The warnings left by the retirement are flushed to *printer*.  A run that ends with a worktree unmerged leaves its branch marked preserved (:meth:`GitWorktreeOps.save_preserve_marker`), so no orphan reclaim in another process merges it behind the user's back.  An unexpected exception becomes a ``success: false`` result; a :class:`KISSError` or :class:`SeaError` (a setup failure the caller must see as a diagnostic) propagates.<br/>`run(model_name: str | None = None, prompt_template: str = '', arguments: dict[str, str] | None = None, system_prompt: str | None = None, tools: list[Callable[..., Any]] | None = None, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, web_tools: bool = True, is_parallel: bool = True, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list[Attachment] | None = None, ask_user_question_callback: Callable[[str], str] | None = None, base_system_prompt: str = '', append_basic_tools: bool = True, inherited_tools: list[Callable[..., Any]] | None = None, llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None, tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None, use_memory: bool | None = None, tool_profile: str = '', live_browser: Any = None, prompt_suffix: str = '', system_prompt_hook: Callable[[str], str] | None = None, tools_hook: Callable[[list[Callable[..., Any]]], list[Callable[..., Any]]] | None = None, task_begun: bool = False) -> str`
  - `model_name`: LLM model to use. Defaults to config value.
  - `prompt_template`: Task prompt template with format placeholders.
  - `arguments`: Dictionary of values to fill prompt_template placeholders.
  - `system_prompt`: system prompt to be appended to the actual system prompt.  Also forwarded to every sub-agent spawned via ``run_parallel``, appended to each sub-agent's own base system prompt — like *base_system_prompt*, so the extra instructions constrain the whole task tree.
  - `tools`: List of tools to be added in addition to bash and web tools.
  - `max_steps`: Maximum steps per sub-session. Defaults to 10000.
  - `max_budget`: Maximum budget in USD. Defaults to config value.
  - `work_dir`: Working directory for the agent. Defaults to the current directory.
  - `printer`: Printer instance for output display.
  - `max_sub_sessions`: Maximum continuation sub-sessions. Defaults to config value.
  - `docker_image`: Docker image name to run tools inside a container.
  - `web_tools`: Whether to include browser/web tools. Defaults to True.
  - `live_browser`: The daemon's ``BrowserTabService``; when given, ``show_browser()`` opens the page in the Browser tab on every surface (forwarded to every sub-agent).  None (no daemon) shows a local window instead. Set to False for terminal-only environments.
  - `prompt_suffix`: The caller-supplied text (the daemon's ``appendToPrompt`` wire field, e.g. a ``run_agent`` call's ``add_to_prompt`` option) that the caller has ALREADY appended to *prompt_template*; it is not added again here.  Recorded as ``_prompt_suffix`` so a ``run_agent`` sub-task dispatched during the run inherits it as its own ``add_to_prompt`` option (see ``agent_dispatch.inherit_from_parent``).  Defaults to "" (the run has no suffix).  Last in the signature so every earlier argument keeps its position.
  - `is_parallel`: Whether to include the run_parallel tool. Defaults to True. When True, the agent can spawn parallel sub-agents for independent tasks.
  - `verbose`: Whether to print output to console. Defaults to config verbose setting.
  - `current_editor_file`: Path to the currently active editor file, appended to prompt.
  - `attachments`: Optional file attachments (images, PDFs) for the initial prompt.
  - `ask_user_question_callback`: Optional callback used by the ask_user_question tool to collect a text response from the user.
  - `base_system_prompt`: Custom base system prompt.  When non-blank it REPLACES the default ``SYSTEM.md`` system prompt (:data:`kiss.core.base.SYSTEM_PROMPT`) for this agent and for every sub-agent it spawns via ``run_parallel``.  The *system_prompt* suffix, the active-editor-file line, and the per-run operational instructions (work dir, PID, ``$KISS_HOME/AGENTS.md``) are still appended.  Blank (default) keeps the default system prompt.  Appended after the historical positional arguments so every one of them keeps its position.
  - `append_basic_tools`: Whether :meth:`perform_task` prepends the built-in basic toolset (:meth:`_get_tools`: Bash, Read, Edit, Write, browser tools, run_agent, ask_user_question, talk, set_model, decide, run_parallel, ...) to the caller's *tools*.  Defaults to True.  When False the agent runs with ONLY the ``finish`` tool (added by ``RelentlessAgent.perform_task``) and the caller's *tools* — *web_tools* and *is_parallel* then have no effect, since the tools they toggle are never built.
  - `inherited_tools`: Extra tools taken over from the task that dispatched this run (a ``run_agent`` sub-task gets the caller's SEA ``tools()``, see ``task_runner``).  Added after the built-in toolset and *tools*; one whose name this run already has (a built-in the caller lacked, or one of *tools*) is skipped rather than registered twice.  ``None`` (default) adds nothing.
  - `llm_call_hook`: Optional hook forwarded to the underlying :meth:`kiss.core.kiss_agent.KISSAgent.run` of every sub-session this agent runs (see that docstring): called before every LLM call with the new messages about to be sent, and its return value replaces them.  Applies to this agent only, not to ``run_parallel`` sub-agents. Defaults to None (no hook).
  - `tool_call_hook`: Optional hook forwarded to the underlying :meth:`kiss.core.kiss_agent.KISSAgent.run` of every sub-session this agent runs (see that docstring): called before every tool call with the tool's name and arguments and returns a :class:`~kiss.core.tool_verdict.Verdict` (``ALLOW``, or ``refuse(text)``: the call is suppressed and the model reads *text* as its result).  Applies to this agent only, not to ``run_parallel`` sub-agents.  Defaults to None (no hook).
  - `use_memory`: Per-run persistent-memory toggle (:mod:`kiss.core.memoryfield`).  ``True`` gives the run the ``memory_*`` tools and the ``MEMORY_PROTOCOL`` prompt block, ``False`` withholds them, and ``None`` (the default) falls back to the ``KISS_USE_MEMORY`` environment variable / the stored ``use_memory`` setting (see :func:`_memory_settings`).  A boolean never bypasses the hard gates of :func:`_memory_root_for_run` (stripped basic tools, Docker runs, a caller ``model_config["system_instruction"]``).  Forwarded to every ``run_parallel`` sub-agent, so one override governs the whole task tree.
  - `tool_profile`: Name of the tool profile this run's built-in toolset is cut down to — a key of :data:`TOOL_PROFILES` (the composites ``"full"``, ``"review"``, ``"assistant"``, ``"bash"`` or the groups ``"shell"``, ``"edit"``, ``"browser"``, ``"memory"``, ``"agents"``, ``"mcp"``, ``"skills"``, ``"user"``, ``"decide"``, ``"control"``), several keys joined with ``+`` (``"shell+edit+browser"`` keeps the union of their tools; see :func:`resolve_tool_profile`), or ``""`` (the default) to let :meth:`_tool_profile` decide (``full`` for a top-level task, ``review`` for a reviewer sub-agent). Applies to this agent only: ``run_parallel`` children pick their own profile.
  - `system_prompt_hook`: The SEA's ``system_prompt`` method (:func:`kiss.agents.sorcar.sea_commands.base_system_prompt`): called once with the assembled system prompt (base or *base_system_prompt*, plus *system_prompt*) and its return value is the run's system prompt, verbatim.  Not forwarded to sub-agents (their own SEA layers apply theirs; they inherit *base_system_prompt* and *system_prompt* as given).  ``None`` (default) changes nothing.
  - `tools_hook`: The SEA's ``tools`` method (:func:`kiss.agents.sorcar.sea_commands.base_tools`): called once by :meth:`perform_task` with the built-in toolset plus *tools* and *inherited_tools*, and its return value is the run's toolset.  ``None`` (default) changes nothing.
  - `task_begun`: True when the caller already ran :meth:`_begin_task` for this task (``ChatSorcarAgent.run`` does, before it replaces the chat state the retirement's commit message is generated from).
  - **Returns:** YAML string with 'success' and 'summary' keys.

- **worktrees** — Every worktree pending merge/discard, in creation order.<br/>`worktrees() -> list[GitWorktree]` *(property)*
- **create_worktree** — Create a git worktree for the repository of *path* (default: the work dir). Call this once per repository, before the first change (create, edit, write, or make a folder) to a Git-tracked file or folder of that repository.  The task's edits to the repository then land on a private ``kiss/wt-*`` branch checked out under ``<repo>/.kiss-worktrees/`` instead of the user's checkout.  For the repository containing the work dir the work dir moves into the worktree, so relative paths and Bash commands use it from then on; for any other repository, edit its files under the returned worktree directory (``<repo>/x.py`` becomes ``<worktree>/x.py``).  Call ``commit_and_merge`` when the work is done: it merges every worktree.  A no-op when the repository already has a worktree or *path* is not inside a git repository.<br/>`create_worktree(path: str = '') -> str`
  - `path`: A file or directory inside the repository to isolate; relative to the work dir.  Empty: the work dir itself.
  - **Returns:** The worktree directory and branch (and the new work dir when it moved), or the reason no worktree was created (edit the files in place then).

- **commit_and_merge** — Commit every worktree's changes and merge them into their original branches. For each worktree created by ``create_worktree``, commits every change on its task branch (with a generated commit message), squash-merges the branch into the branch the task started from, deletes the branch and removes the worktree; the work dir returns to the user's checkout.  A merge conflict is resolved by the merge SEA running as a sub-agent; when it cannot, that branch is kept and the manual steps are returned.  Call this after the work is complete and verified, before ``finish``.<br/>`commit_and_merge() -> str`
  - **Returns:** What was merged, or why a merge did not complete.

- **inherit_worktrees** — Adopt the calling tasks' worktrees before this agent runs as a sub-agent. A sub-agent never creates a worktree for a repository a calling task (any ancestor) has already isolated: ``create_worktree`` points at the ancestor's worktree instead, and the write guard maps the repository's checkout paths into it.  The ancestors commit and merge those worktrees.<br/>`inherit_worktrees(worktrees: dict[Path, Path]) -> None`
  - `worktrees`: Repository root -> worktree directory, as :meth:`isolated_repos` of the calling agent returns it.

- **isolated_repos** — Repository root -> worktree directory for every repo this task isolates. Its own worktrees and the ones it inherited from calling tasks, so a sub-agent of a sub-agent still sees the top task's worktrees. Runs on a sub-task's thread (``task_runner`` reads the parent's map at the child's run start) while this agent's own thread keeps storing into and popping from ``_wts``: ``list(...)`` is one C-level copy under the GIL, so the Python-level loop never sees the dict change size.  ``_inherited_worktrees`` is only ever replaced wholesale, never mutated.<br/>`isolated_repos() -> dict[Path, Path]`
- **merge** — Merge every pending task branch into its original branch. Every step is idempotent — safe to re-run after a crash. Commits any uncommitted changes in each worktree before merging it.  If a main working tree has uncommitted changes, they are stashed before the merge and restored afterward.  A worktree whose merge fails stays pending; the others are still merged.<br/>`merge(conflict_resolver: MergeConflictResolver | None = None) -> str`
  - `conflict_resolver`: Resolves a merge conflict and completes the merge (see :meth:`_do_merge`) instead of this method returning the manual-resolution steps.
  - **Returns:** One success or error message per worktree, blank-line separated.

- **leave_as_is** — Detach from the pending worktrees, leaving them untouched on disk. Serves the "Do nothing" button of the post-task worktree bar: each branch, its worktree directory, and any uncommitted changes inside it are left exactly as the task ended.  A preserve-for-review marker is persisted on every branch so :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` in a future process also refuses to silently publish it.<br/>`leave_as_is() -> str`
  - **Returns:** Messages telling the user where the untouched worktrees are.

- **discard** — Throw away every pending task branch and worktree, checkout original. Every step is idempotent — safe to call multiple times.<br/>`discard(*, rescue_ignored: bool = False) -> str`
  - `rescue_ignored`: When True, git-ignored files the task created in a worktree are copied into its main repository (never overwriting existing files) before the directory is removed.  The automatic discard of a worktree without changes passes True, because the changed-files probe cannot see ignored files.
  - **Returns:** One confirmation message per worktree (including a warning if checkout to the original branch failed), or a "Discard deferred" message when ignored output could not be rescued.

- **retire_for_disposal** — Retire the pending worktree because its tab is going away.<br/>`retire_for_disposal() -> bool`
  - **Returns:** True when this agent no longer holds a worktree claim and the caller may drop it; False when the claim is retained (the keep decision could not be made durable) and the caller must keep this agent reachable so a later disposal attempt can retry.

- **new_chat** — Start afresh (the chat page's 'Clear'), preserving any pending worktree. The warning the preserve sets is flushed to the attached printer now: if the user never runs another task on this agent, no later :meth:`run` would flush it.<br/>`new_chat() -> None`
- **add_warning** — Combine *text* with the pending warning atomically.<br/>`add_warning(text: str, *, prepend: bool = False) -> None`
  - `text`: The warning to add.  Joined to the pending warning with a newline; becomes the whole warning when none is pending.
  - `prepend`: Put *text* before the pending warning instead of after it.

**`canonical_tool_profile`** — Return *name* with every part a :data:`TOOL_PROFILES` key, or raise.<br/>`def canonical_tool_profile(name: str) -> str`

- `name`: A profile name: one key or alias, or several joined with ``+``.  Whitespace around a part is ignored; the empty name (no profile chosen) is returned unchanged.
- **Returns:** The parts, aliases replaced by their key (:data:`~kiss.agents.sorcar.sea_settings.PROFILE_ALIASES`), joined with ``+``.

**`classification_chooses_prompt`** — Return whether a run's task classification can affect its system prompt. A run that cannot edit files — no built-in toolset (*append_basic_tools* off: a script-only SEA such as a channel agent) or a profile without the edit tools (``review``, ``shell``, ...) — has no use for the coding, testing and research rules of SYSTEM.md: it runs on the reduced SYSTEM_LITE.md prompt whatever the task, so classifying the task would only cost a model call (P12 of the 2026-10-24 trajectory audit: a three-tool channel child paid most of its budget for a 30k-character prompt).  Every run that does classify shares this gate so none pays for a verdict it would ignore.<br/>`def classification_chooses_prompt(allowed: frozenset[str] | None, append_basic_tools: bool) -> bool`

- `allowed`: The run's tool names as :func:`resolve_tool_profile` returns them (``None`` for every tool).
- `append_basic_tools`: Whether the run offers the built-in toolset.
- **Returns:** ``True`` when the verdict chooses between SYSTEM.md and SYSTEM_LITE.md, ``False`` when the run gets SYSTEM_LITE.md anyway.

**`resolve_tool_profile`** — Return the tool names a (possibly composite) profile *name* allows. *name* is one :data:`TOOL_PROFILES` key (or an alias of one) or several joined with ``+``; the result is the union of their tool sets.  ``None`` means every tool the agent can build: the ``full`` profile, alone or as a part of a composite, and the empty name (no profile chosen).<br/>`def resolve_tool_profile(name: str) -> frozenset[str] | None`

- `name`: The profile name, e.g. ``"review"`` or ``"shell+edit+browser"``.
- **Returns:** The allowed tool names, or ``None`` for everything.

**`summary`** — Record a progress summary at a natural checkpoint (a plan made, a milestone reached). The framework writes a recap of the tool calls every 10 steps on its own; call this only when you have something the recap cannot say.<br/>`def summary(description: str) -> str`

- `description`: Natural language summary in 5-10 sentences of what the agent did since the last call to `summary`, written in Markdown format (use bullet lists for the steps, and ``**bold**`` / backtick code spans).
- **Returns:** A short confirmation string.

**`auto_commit_changes`** — Stage all changes, generate a commit message, and commit. Stages once so *message_fn* can compute the diff, runs *message_fn* (typically a slow LLM call) to generate the commit subject/body, then re-stages immediately before the commit so any file that appeared in the worktree during the LLM call (e.g. ``PROGRESS.md`` rewrites, macOS ``.DS_Store`` materializing after an ``open`` of the report, an editor side-channel saving swap files) is included in the same commit.  Without the second ``stage_all`` those late-arriving files would be left uncommitted, ``_finalize_worktree`` would see them via ``has_uncommitted_changes`` and abort the auto-merge with the misleading "pre-commit hook may have rejected" warning (observed in production on 2026-06-26 07:23:14 for worktree ``kiss_wt-1782483430-cb03445c`` even though the repo had no custom pre-commit hooks installed). Falls back to a generic commit message when *message_fn* raises (e.g. the LLM-based generator is unavailable).<br/>`def auto_commit_changes(commit_dir: Path, user_prompt: str | None, message_fn: Callable[[Path, str | None, str | None], str], notify_fn: Callable[[str, str], None] | None = None, task_result: str | None = None) -> bool`

- `commit_dir`: Directory whose changes are staged and committed.
- `user_prompt`: The user's task prompt, woven into the commit message (or its fallback), or ``None`` when unavailable.
- `message_fn`: Callable producing a commit message from ``(commit_dir, user_prompt, task_result)``.
- `task_result`: The task's result summary, appended to the commit message (or its fallback) under a ``Result:`` heading, or ``None`` when unavailable.
- `notify_fn`: Optional UI callback invoked at two life-cycle points so the chat page can render toasts: - ``notify_fn("generating", "")`` immediately before *message_fn* runs (typically a slow LLM call) so the user sees "Generating commit message" while the LLM works. - ``notify_fn("committed", subject)`` immediately after a successful commit, where *subject* is the first non-empty line of the committed message. Both hooks are SKIPPED when there is nothing to commit (no staged diff after the initial ``stage_all``), so the chat page never sees a misleading "Generating commit message" toast without a follow-up.  When ``commit_staged`` returns ``False`` after *message_fn* (e.g. a pre-commit hook rejected the commit), ``notify_fn("failed", "")`` is invoked instead so the sticky "generating" toast always gets a terminal update. All ``notify_fn`` exceptions are swallowed so a broken UI hook can never block the commit itself.
- **Returns:** True if a commit was created, False if nothing to commit.

**`subagent_parent_tab_id_of`** — Return the frontend tab id children of *parent_agent* hang off. The daemon's helper sub-agents (task update, merge-conflict resolver, ``run_agent`` dispatches) stamp this as their ``parent_tab_id``; it is :meth:`SorcarAgent._subagent_parent_tab_id` when the parent has it, else the parent's raw ``_tab_id`` (a duck-typed parent).  A parent that is itself a sub-agent carries a synthetic ``_tab_id`` no chat page has, so stamping that raw id would make every client drop the child's ``new_tab``.<br/>`def subagent_parent_tab_id_of(parent_agent: Any) -> str`

- `parent_agent`: The agent spawning the child.
- **Returns:** The tab id, or ``""`` when the parent runs headless.

---

#### `kiss.agents.sorcar.chat_sorcar_agent` — *Stateful Sorcar agent with chat-session persistence.*

##### `class ChatSorcarAgent(SorcarAgent)` — SorcarAgent with chat-session state management.

**Constructor:** `ChatSorcarAgent(name: str) -> None`

- **chat_id** — Return the current chat session ID ("" means new session).<br/>`chat_id() -> str` *(property)*
- **last_task_id** — Return the ``task_history`` row id this agent last allocated. The readers live on other threads — the WebSocket command handler stamping a queued user message, the merge/discard flow, the printer's broadcast fan-out — so the read takes ``_task_id_lock``, the same lock the publishing assignment in :meth:`run` takes.  That pairing is what makes the lock mean anything: a lock only the writer holds excludes nobody.<br/>`last_task_id() -> str` *(property)*
  - **Returns:** The row id, or ``""`` before this agent's first ``run``.

- **last_user_prompt** — Return the task text of this agent's current (or last) :meth:`run`. The text as submitted — what the user typed, or the ``task`` a ``run_agent`` dispatch passed — before the chat history, the bare-path directive or ``AGENTS.md`` are added to the prompt the model sees.  A SEA's tool reads it through :func:`kiss.server.agent_state.current_agent` to parse options out of its own task (rsi7d's scope).<br/>`last_user_prompt() -> str` *(property)*
  - **Returns:** The text, or ``""`` before this agent's first ``run``.

- **new_chat** — Reset to a new chat session (the chat page's 'Clear'). Retires any pending worktree first (``SorcarAgent.new_chat``), and drops any pending one-shot :meth:`resume_from_task_id` seed: a brand-new chat must never have its first prompt augmented with the previous task's parent-chain context.<br/>`new_chat() -> None`
- **resume_chat_by_id** — Resume a chat session using a stable chat identifier.<br/>`resume_chat_by_id(chat_id: str) -> None`
  - `chat_id`: String chat session identifier to resume.

- **resume_from_task_id** — Seed the next prompt's context from a task's parent chain. Called when the tab that owns this agent was opened by a specific task id (history click / ``resumeSession`` with ``taskId``) and no task has been run in the tab since: the first :meth:`build_chat_prompt` after this call traverses the ``parent_task_id`` links starting at *task_id* (via :func:`_load_task_chain_context`) instead of loading the whole chat.  The seed is one-shot — subsequent prompts fall back to the normal chat-context path.<br/>`resume_from_task_id(task_id: str) -> None`
  - `task_id`: The ``task_history.id`` the tab was opened with. Empty strings are ignored.

- **build_chat_prompt** — Load chat context and augment prompt with previous tasks/results.<br/>`build_chat_prompt(prompt: str) -> str`
  - `prompt`: The original task prompt.
  - **Returns:** The augmented prompt with chat history prepended, or the original prompt if no prior context exists.

- **run** — Run the agent with chat-session context management. Loads prior chat context, persists the new task, augments the prompt with previous tasks/results, runs the underlying agent, and saves the result back to history. Only the result summary is persisted here.  Callers that record chat events (e.g. the daemon) persist events incrementally via :func:`~kiss.agents.sorcar.persistence._append_chat_event`.<br/>`run(prompt_template: str = '', model_name: str | None = None, arguments: dict[str, str] | None = None, system_prompt: str | None = None, tools: list[Callable[..., Any]] | None = None, max_steps: int | None = None, max_budget: float | None = None, model_config: dict[str, Any] | None = None, work_dir: str | None = None, printer: Printer | None = None, max_sub_sessions: int | None = None, docker_image: str | None = None, web_tools: bool = True, is_parallel: bool = True, verbose: bool | None = None, current_editor_file: str | None = None, attachments: list[Attachment] | None = None, ask_user_question_callback: Callable[[str], str] | None = None, base_system_prompt: str = '', append_basic_tools: bool = True, inherited_tools: list[Callable[..., Any]] | None = None, llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None, tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None, use_memory: bool | None = None, tool_profile: str = '', live_browser: Any = None, prompt_suffix: str = '', system_prompt_hook: Callable[[str], str] | None = None, tools_hook: Callable[[list[Callable[..., Any]]], list[Callable[..., Any]]] | None = None, _skip_persistence: bool = False, _on_task_id_allocated: Callable[[str, str], None] | None = None, _history_prompt: str | None = None, _open_bare_path: bool = True, _start_ms: int = 0) -> str`
  - `prompt_template`: The task prompt.  First so the historical ``run("task", ...)`` call keeps working; every other parameter up to *tools_hook* is forwarded verbatim to :meth:`SorcarAgent.run`, in that method's order and with its defaults — see its docstring for what each one does.  *model_name* is resolved here once (:meth:`_resolve_model_name`) and the resolved name is what the base run receives.
  - `model_name`: LLM model to use.  Defaults to the config value.
  - `arguments`: Values filling the placeholders of *prompt_template*.
  - `system_prompt`: Suffix appended to the actual system prompt.
  - `tools`: Tools added on top of the built-in toolset.
  - `max_steps`: Maximum steps per sub-session.
  - `max_budget`: Maximum budget in USD; also recorded on the task-history row (``DEFAULT_MAX_BUDGET`` when omitted).
  - `model_config`: Extra model configuration.
  - `work_dir`: Working directory; its resolved absolute path is recorded on the task-history row.
  - `printer`: Printer for output display; its ``broadcast`` (if any) carries the ``task_settings`` / ``new_tab`` / ``tasks_updated`` events of this run.
  - `max_sub_sessions`: Maximum continuation sub-sessions.
  - `docker_image`: Docker image to run tools inside.
  - `web_tools`: Whether to include browser/web tools.
  - `is_parallel`: Whether to include the ``run_parallel`` tool.
  - `verbose`: Whether to print output to the console.
  - `current_editor_file`: Active editor file appended to the prompt.
  - `attachments`: File attachments for the initial prompt.
  - `ask_user_question_callback`: Callback of the ``ask_user_question`` tool.
  - `base_system_prompt`: Replacement for the default base system prompt.  A non-blank value also disables the bare-path "open it" directive (see *_open_bare_path*).
  - `append_basic_tools`: Whether the built-in toolset is prepended.
  - `inherited_tools`: Tools taken over from the dispatching task.
  - `llm_call_hook`: Hook called before every LLM call.
  - `tool_call_hook`: Hook called before every tool call.
  - `use_memory`: Per-run persistent-memory toggle.
  - `tool_profile`: Tool profile name the toolset is cut down to; the EFFECTIVE profile (``review`` for a reviewer sub-agent asked for none) is what the ``task_settings`` event reports.
  - `live_browser`: The daemon's ``BrowserTabService``.
  - `prompt_suffix`: Caller-supplied suffix already appended to *prompt_template*, recorded so a ``run_agent`` sub-task inherits it.
  - `system_prompt_hook`: The SEA's ``system_prompt`` method.
  - `tools_hook`: The SEA's ``tools`` method.
  - `_skip_persistence`: When True the final result / usage / replay-event save is skipped (the daemon persists them itself); the task-history row is still allocated.
  - `_on_task_id_allocated`: Called with ``(task_id, chat_id)`` as soon as the history row exists, before any event of the run is broadcast.
  - `_history_prompt`: The raw user text to persist and display (history row, frequent-tasks table, last-user-prompt cache) in place of *prompt_template*, which may carry an internal directive the LLM must see (a slash-command run).  ``None`` or blank: *prompt_template* is used.
  - `_open_bare_path`: Whether a *prompt_template* that is only a filesystem path gets the "open it" directive.  The task runner passes False for a run it starts on behalf of a SEA or of another agent.
  - `_start_ms`: The run's start as a ms epoch, when the caller stamped it before calling (the daemon's ``<task>`` block); 0 (default) dates the run at row allocation.
  - **Returns:** YAML string with 'success' and 'summary' keys.

---

#### `kiss.server.sorcar` — *The Sorcar server API and a minimal synchronous client for it.*

##### `class ApiCommand` — One command of the Sorcar server API.

##### `class Auth` — The outcome of a successful ``auth`` handshake.

##### `class ApiContext` — Transport context of one in-flight server API call.

##### `class ServerBackend(Protocol)` — Daemon capabilities the server API dispatches onto.

- **local_token**<br/>`local_token() -> str` *(property)*
- **local_only**<br/>`local_only() -> bool` *(property)*
##### `class ServerApi` — The Sorcar server's code-level API.

**Constructor:** `ServerApi(backend: ServerBackend) -> None`

- `backend`: The daemon object providing the transports and command implementations (in production the ``RemoteAccessServer``).

- **dispatch** — Route one client command to its API method. The single entry point of the code API.  Applies, in order: 1. Canonicalises a string ``tabId`` (surrounding whitespace stripped) and writes it back into *cmd*, so the registry, the agent-state registry, the printer and the cleanup tail all key the tab identically. 2. Validates *cmd* against the catalog (:func:`validate_command`) and answers an invalid command with a direct ``error`` event to the sender only. 3. Records the command's ``tabId`` in the connection's bookkeeping (:meth:`_record_tab`). 4. Stamps the connection's ``conn_id`` as ``connId`` — overwriting any client-supplied value so it cannot be spoofed — which keys the backend's per-connection autocomplete state. 5. Blanks a ``workDir`` naming a filesystem root (``/``, ``C:\`` — see :func:`kiss.core.utils.is_root_dir`) so it is treated exactly like an absent one: a client whose cwd degenerated to the root can never make the daemon's single global working directory the whole disk, nor run a task there.  A command without a ``workDir`` resolves to that global directory (``config.json`` ``work_dir``, set from any surface's chat-header work-dir drop-down via ``setWorkDir``). 6. Sends a task command of a tab bound to a remote machine to that machine (:class:`~kiss.server.remote_tabs.RemoteTabs`), and a workspace command stamped ``machine`` (the Explorer, Source Control view, folder picker and file editor of such a tab: :data:`kiss.server.sidebar_panels.MACHINE_WORKSPACE_REPLIES`), a settings command stamped ``machine`` (the settings sheet opened for the machine: :data:`kiss.server.sidebar_panels.MACHINE_SETTINGS_REPLIES`) a subpanel command stamped ``machine`` (the Schedule, Apps, Spend and Recommendations subpanels shown for it: :data:`kiss.server.sidebar_panels.MACHINE_PANEL_REPLIES`) or the task-info panel's ``getTaskUpdate`` stamped ``machine`` (the Task update of the task such a tab runs there: :func:`kiss.server.sidebar_panels.machine_task_update_reply`) to the machine's daemon, answering the sender with its reply. 7. Invokes the :class:`ServerApi` method named by the command's catalog entry.<br/>`async dispatch(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The parsed JSON command dictionary (the transport guarantees a dict).
  - `ctx`: The transport context of this call.

- **cancel_background_replies** — Cancel and await every background reply and detached command (at shutdown). The threads they wait on (``asyncio.to_thread``) run to their own bounds; only the tasks, which would otherwise hold the loop open for a slow machine's answer, end here.  A command dispatched from here on (one received while the cancelled tasks are awaited) spawns no reply, so none outlives this. The brand's running sign-in hook, a child process, is ended too: it must not finish a login after the daemon is gone.<br/>`async cancel_background_replies() -> None`
- **authenticate** — Authenticate a WSS client with the ``auth`` handshake. Every client's entry point into the API: before a connection may issue any catalog command, its very first frames must complete this handshake.  The remote webapp's transport sends ``{"type": "auth", "password": ...}`` as soon as the socket opens; local Python clients send ``{"type": "auth", "token": ...}`` with the daemon's per-start local token read from the endpoint file (:mod:`kiss.agents.sorcar.local_endpoint`), whose 0600 mode gates the token to the owning user. Protocol serviced here, in order: -1. A peer that presented a client certificate issued by an enrolled CA (``tls/trusted-clients/*.pem``, see :meth:`~kiss.server.web_server.RemoteAccessServer._peer_client_ca`) is another daemon's machine link: its first frame must be an ``auth`` frame (its content is irrelevant), answered with ``auth_ok`` (``"local": true``); it returns ``"local"`` whatever the password setting or the lockout. 0. An ``auth`` frame carrying a ``token`` equal to the backend's ``local_token`` (constant-time compare) from a LOOPBACK peer is answered with ``auth_ok`` (``"local": true``) and returns ``"local"``.  A token from a non-loopback peer is never honoured, whatever the password setting; a wrong token counts as a wrong guess.  A token client that looks like the retired VS Code extension (no ``User-Agent`` and no ``Origin`` upgrade header, see :func:`_is_legacy_vscode_client`) is then also sent a ``notice`` event pointing it at the web app. 0b. On a daemon that is not ``local_only``, an ``auth`` frame without a token from a LOOPBACK BROWSER (:meth:`_is_loopback_browser`: the raw TCP peer is loopback, :meth:`_client_ip` is loopback too, so the upgrade request carried no forwarded client IP, and its ``Origin`` names a loopback host) is answered with ``auth_ok`` (``"local": false``) whatever password it carries and returns ``"loopback"``: the web app opened at ``http://127.0.0.1`` on the daemon's own machine needs no password, and no later frame of it is password-checked.  It gets no local privileges.  Visitors relayed by cloudflared also arrive from loopback but carry ``Cf-Connecting-Ip`` / ``X-Forwarded-For``; programs on this machine (machine links, the Python client) send no ``Origin``; both still need their password or token.  Any OTHER reverse proxy on loopback that neither sets a forwarded-IP header nor changes the page's origin would expose the daemon to everyone it relays. 1. A source IP that is still rate-limited after too many failed logins is answered with ``auth_locked`` (carrying ``retry_after`` seconds) and closed — telling the client WHY instead of leaving its loading overlay spinning.  The local token (0a) and the loopback browser (0b) are still admitted: the lockout guards the password, which neither is guessing. 2. When no ``remote_password`` is configured, a non-loopback TCP peer is refused (``error`` + close) before any credential is examined: with no password set, only localhost may connect.  The check runs before the handshake AND again for every attempt (against a freshly re-loaded config), so it mirrors the transport's 403 gate (``RemoteAccessServer._process_request``) and also covers a password cleared at any point mid-handshake. 3. Otherwise up to two ``auth`` attempts are read: a correct password (a constant-time compare against the configured ``remote_password`` as re-loaded for that attempt, so a password change applies immediately) is answered with ``auth_ok``; the first wrong password elicits an ``auth_required`` retry prompt; the second failure is answered with an ``error`` event and the socket is closed. A first message that is not an ``auth`` at all closes the socket without counting a failed login. 4. Only NON-EMPTY wrong guesses count toward the brute-force lockout: every fresh page load probes with the (possibly empty) password stored in ``localStorage``, and behind the shared cloudflared tunnel penalising that benign empty probe would let a handful of normal page loads lock the password prompt away from every visitor.<br/>`async authenticate(websocket: Any) -> Auth | None`
  - `websocket`: The client's WebSocket connection.
  - **Returns:** The :class:`Auth` of the admitted client (``kind`` ``"local"`` for a token-authenticated loopback client, ``"loopback"`` for a direct loopback peer admitted without a password, ``"remote"`` for a password-authenticated one; ``link`` when its ``auth`` frame declared it a daemon link), ``None`` when authentication failed (the socket is then already closed).

- **forward** — Run *cmd* on the backend agent server. The default handler: commands with no daemon-side special casing (``run``, ``stop``, ``getModels``, ``getConfig``, ``getHistory``, ``complete``, …) are executed by the backend ``AgentServer`` in the thread-pool executor.<br/>`async forward(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The validated, connection-stamped command.
  - `ctx`: The transport context of the current call (unused).

- **resume_session** — Resume a chat session in the issuing tab. Translates the wire field ``id`` to the backend's ``chatId`` (:func:`translate_client_command`), then forwards.<br/>`async resume_session(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``resumeSession`` command.
  - `ctx`: The transport context of the current call.

- **worktree_action** — Merge, discard or detach a tab's pending worktree in the background. The backend's ``worktreeAction`` handler reports through a tab-stamped ``worktree_result`` broadcast and claims the tab's ``is_merging`` flag atomically before it starts, so a second click is refused and a ``run`` on the tab is turned down while it works.  A merge generates its commit message with an LLM call and may run the merge-conflict SEA for minutes, and each connection's commands are dispatched one at a time, so the command runs as a background task: a ``stop`` of another tab's task, a ``submit`` or the transport's ``ping`` sent right after the click is not held up behind the merge.<br/>`async worktree_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``worktreeAction`` command.
  - `ctx`: The transport context of the current call (unused).

- **autocommit_action** — Run the Git Commit button's command on the backend. The backend reports through a tab-stamped ``autocommit_done`` broadcast, which a daemon-link peer is never sent.  Another daemon forwards the Git Commit of a tab picked onto this machine over such a link and waits for that event as the reply, so a link's command is stamped ``replyTo`` with its connection id and the backend sends that connection its own copy of the outcome (``_commit_main_tree`` of :mod:`kiss.server.merge_flow`).<br/>`async autocommit_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``autocommitAction`` command.
  - `ctx`: The transport context of the current call.

- **ready** — Initialize a (re)loaded chat page. Sanitizes the command's ``restoredTabs`` (warnings included) and writes the cleaned list back for the backend's ready handler, then fans the command out through it (models / input history / config / session replay).<br/>`async ready(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``ready`` command.
  - `ctx`: The transport context of the current call.

- **submit** — Start a task from a chat page's ``submit``. The backend translates the ``submit`` into a ``run`` (path resolution, follow-up routing) including the path-only shortcut: a prompt that is just the path of an existing file is answered on the submitting connection with the file's ``fileContent`` and starts no task.<br/>`async submit(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``submit`` command.
  - `ctx`: The transport context of the current call; its endpoint receives a path-only prompt's reply.

- **open_file** — Serve a clicked file link. A client clicked a file or directory link in a chat page.  The daemon resolves the path (``~``, the tab's work dir, its pending worktree) and answers with the file's content (or a plain-text directory listing) for an in-page content tab.<br/>`async open_file(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``openFile`` command.
  - `ctx`: The transport context of the current call.

- **save_file** — Write a remote-web client's edits back to a file on disk. The remote webapp opens a file (``openFile``) in an editable Monaco editor inside a content tab; Ctrl/Cmd+S or the tab's Save button sends the editor's full text here.  The file must already exist (the editor never creates files) and is replaced atomically; the ``version`` stamp taken from the ``fileContent`` reply lets the daemon refuse to overwrite a file that changed on disk since it was opened unless ``force`` is set.  The reply is a ``fileSaved`` event sent to the requester only.<br/>`async save_file(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``saveFile`` command (``path``, ``content``, optional ``workDir``, ``tabId``, ``token``, ``version``, ``force``).
  - `ctx`: The transport context of the current call.

- **check_paths** — Report which file paths exist to the requesting client. The chat page linkifies file-path-looking strings in event panel contents lazily: a path only becomes a clickable link after this check confirms that clicking it (``openFile``) would actually open something — a file or a directory — with the same resolution ``openFile`` applies.<br/>`async check_paths(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``checkPaths`` command.
  - `ctx`: The transport context of the current call.

- **get_task_update** — Report the ``/ask`` agent's task update to a task-info panel. The remote webapp's task-info panel (docked on desktop, a drawer on mobile) polls this command while the visible tab's task runs: the info subpanel shows the :mod:`~kiss.agents.seas.ask.ask_sea` agent's short answer to what that task has done so far and its partial results.  The first poll once the task is a minute old runs the agent (as a sub-agent of the task, in the task's chat; its cost counts towards the task), later polls re-run it every 10 minutes, and a poll with ``refresh: true`` (the panel's refresh button) re-runs it at once.  The direct ``taskUpdate`` reply goes back to the asking endpoint, local or remote.  A poll stamped ``machine`` (the tab is bound to a registered machine, where its task runs) never reaches this handler: :meth:`dispatch` forwards it to the machine's daemon and relays its reply (:func:`~kiss.server.sidebar_panels.machine_task_update_reply`).<br/>`async get_task_update(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getTaskUpdate`` command (``tabId``, optional ``knownSig``, ``token``, ``refresh``, ``machine``).
  - `ctx`: The transport context of the current call.

- **get_cron_jobs** — Send a client the scheduled cron jobs for its Schedule subpanel. The right sidebar's "Schedule" subpanel polls this command. The direct reply, to whichever endpoint (local or remote) asked, is ``{"type": "cronJobs", "jobs": [...]}`` with the rows of :func:`kiss.server.sidebar_panels.cron_jobs_report`.<br/>`async get_cron_jobs(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getCronJobs`` command (no fields).
  - `ctx`: The transport context of the current call.

- **get_apps_status** — Send a client every third-party agent's authentication status. Feeds the right sidebar's "Apps" subpanel.  The status comes from :func:`kiss.server.sidebar_panels.apps_status` (a cached probe subprocess; ``refresh: true`` — the subpanel's refresh button, or an app waiting for its connect task — probes again). The direct reply is ``{"type": "appsStatus", "apps": [...], "checkedAt": <epoch ms>}``. A probe takes seconds, and each connection's commands are dispatched one at a time, so the reply is produced by a background task: a ``submit`` or ``stop`` sent right after the poll is not held up behind the probe.<br/>`async get_apps_status(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getAppsStatus`` command (optional ``refresh``).
  - `ctx`: The transport context of the current call.

- **cron_action** — Apply a Schedule subpanel button to a cron job and send the fresh list. ``action`` is ``create`` (the header's ``+`` form: ``name``, ``schedule``, ``what`` and ``kind``, ``prompt`` or ``command``; no ``jobId``), or ``remove``, ``pause``, ``resume``, ``run_now`` or ``update`` (with ``name``, ``schedule`` and ``what``) on the job ``jobId``, applied by :func:`kiss.agents.sorcar.cron_agent.apply_job_action`.  The direct reply is a ``cronJobs`` message (as for ``getCronJobs``) carrying the ``action`` and an ``error`` text when it was refused, so the subpanel re-renders either way (and closes its in-place form only when its own create or update went through).<br/>`async cron_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``cronAction`` command (``action``, optional ``jobId``, ``name``, ``schedule``, ``what``, ``kind``).
  - `ctx`: The transport context of the current call.

- **get_machines** — Send a client the machines for its MACHINES subpanel. The rows come from :func:`kiss.server.sidebar_panels.machines_report` (the local machine first, then every registered one with its reachability; ``refresh: true`` pings again).  The direct reply is ``{"type": "machines", "machines": [...], "checkedAt": <epoch ms>, "action": "", "error": ""}``, produced by a background task like ``appsStatus`` so the pings never hold up the connection's other commands.<br/>`async get_machines(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getMachines`` command (optional ``refresh``).
  - `ctx`: The transport context of the current call.

- **broadcast_machines** — Send a fresh ``machines`` reply (pinged anew) to every connection. Called by the backend when ``machines.json`` changed under the daemon (a ``/machine add`` task or the CLI registered or removed a machine), so every surface's MACHINES subpanel refreshes at once -- and unfolds for a new machine -- instead of at its next poll.  The reply's ``action`` is ``""`` as for a poll.<br/>`async broadcast_machines() -> None`
- **machine_action** — Add or remove a machine for the MACHINES subpanel and send the fresh list. Applied by :func:`kiss.server.sidebar_panels.machine_action`; the direct reply is a ``machines`` message (as for ``getMachines``, pinged anew) carrying the ``action`` and an ``error`` text when it was refused.<br/>`async machine_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``machineAction`` command (``action``: ``add`` or ``remove``, ``name``, optional ``address`` and ``machinePassword`` for ``add``; the machine's own password travels under that name because the frame's ``password`` is this daemon's credential).
  - `ctx`: The transport context of the current call.

- **get_machine_work_dirs** — Send a client the working directories of a registered machine. The chat header's WORKDIR drop-down of a tab bound to a machine lists that machine's directories: its daemon's working directory and the directories selected there so far; the task-info panel shows the directory and the machine's default task budget for a chat headed there whose task has no settings yet.  The direct reply is ``{"type": "machineWorkDirs", "machine": <name>, "workDir": <dir>, "recentWorkDirs": [{"path", "ts"}, ...], "maxBudget": <number or null>, "error": ""}`` from :func:`kiss.server.sidebar_panels.machine_work_dirs` (an unreachable or unknown machine answers with ``error``).<br/>`async get_machine_work_dirs(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getMachineWorkDirs`` command (``machine``).
  - `ctx`: The transport context of the current call.

- **set_work_dir** — Make ``workDir`` the global working directory -- here or on a machine. Without ``machine`` the backend adopts the directory (``_cmd_set_work_dir``: persisted, ``workDirChanged`` broadcast).  With ``machine`` (the chat header's pick for a tab bound to that machine) the directory is one of the machine's: its daemon adopts it, and the direct reply is the machine's fresh ``machineWorkDirs`` (as for ``getMachineWorkDirs``, with ``set: true``; ``error`` names a machine that could not be reached) -- this daemon's own working directory is untouched.<br/>`async set_work_dir(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``setWorkDir`` command (``workDir``; optional ``machine``).
  - `ctx`: The transport context of the current call.

- **get_spend_report** — Send a client the task history's spend for its Spend subpanel. The right sidebar's "Spend" subpanel polls this command for its all-time totals, daily cost heatmap and cost-by-model bars.  The direct reply is ``{"type": "spendReport", ...}`` carrying the ``total``, ``days``, ``totalByModel`` and ``daysByModel`` fields of :func:`kiss.server.sidebar_panels.spend_report`.<br/>`async get_spend_report(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getSpendReport`` command (no fields).
  - `ctx`: The transport context of the current call.

- **get_recommendations** — Send a client the suggested tasks for its Recommendations subpanel. The right sidebar's "Recommendations" subpanel polls this command for the five tasks the last ``/recommendations`` run suggested.  The direct reply is ``{"type": "recommendations", "generatedAt": ms, "tasks": [...]}`` from :func:`kiss.server.sidebar_panels.recommendations_report`.<br/>`async get_recommendations(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getRecommendations`` command (no fields).
  - `ctx`: The transport context of the current call.

- **list_dir** — List a directory for the remote webapp's Explorer view. The remote webapp's task-history panel carries an activity bar whose Explorer view browses the workspace: opening the view lists the work dir, expanding a folder lists that folder, and clicking a file goes through ``openFile``.  The reply is a ``dirListing`` event sent to the requester only. With ``machine`` (the Explorer, folder picker and the pick check of the chat header's WORKDIR drop-down of a tab bound to that machine) the listing is the machine's, from its daemon: :meth:`dispatch` forwards it before this handler runs (:data:`kiss.server.sidebar_panels.MACHINE_WORKSPACE_REPLIES`).<br/>`async list_dir(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``listDir`` command (optional ``path``, ``workDir``, ``tabId``, ``token``, ``machine``).
  - `ctx`: The transport context of the current call.

- **terminal_open** — Start (or re-attach) the shell of a terminal tab.<br/>`async terminal_open(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``terminalOpen`` command (``tab_id``; optional ``cols``, ``rows``, ``workDir``).
  - `ctx`: The transport context of the current call.

- **terminal_input** — Write the keystrokes of a terminal tab to its shell.<br/>`async terminal_input(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``terminalInput`` command (``tab_id``, ``data``).
  - `ctx`: The transport context of the current call.

- **terminal_resize** — Resize the pty of a terminal tab.<br/>`async terminal_resize(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``terminalResize`` command (``tab_id``, ``cols``, ``rows``).
  - `ctx`: The transport context of the current call.

- **terminal_close** — Hang up the shell of a closed terminal tab.<br/>`async terminal_close(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``terminalClose`` command (``tab_id``).
  - `ctx`: The transport context of the current call.

- **git_status** — Report working-tree changes for the remote Source Control view. The activity bar's Source Control view lists the repository's staged, unstaged and untracked changes (VS Code's "Changes" section) from this command's ``gitStatus`` reply, sent to the requester only.<br/>`async git_status(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``gitStatus`` command (optional ``workDir``, ``tabId``, ``token``).
  - `ctx`: The transport context of the current call.

- **git_log** — Report recent commits for the remote Source Control graph. The activity bar's Source Control view draws a commit graph (VS Code's "Graph" section) with each commit's modified files from this command's ``gitLog`` reply, sent to the requester only.<br/>`async git_log(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``gitLog`` command (optional ``workDir``, ``tabId``, ``token``, ``limit``).
  - `ctx`: The transport context of the current call.

- **git_show** — Serve a commit's patch / a file at a commit / a revision diff. The remote Source Control graph's commit context menu ("Open Changes", "Open File", "Compare with...") reads its text from this command's ``gitShow`` reply, sent to the requester only.<br/>`async git_show(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``gitShow`` command (``sha``, optional ``path``, ``base``, ``mode``, ``workDir``, ``tabId``, ``token``).
  - `ctx`: The transport context of the current call.

- **git_action** — Run a commit context-menu git action for the remote graph. "Checkout (Detached)", "Create Branch...", "Create Tag..." and "Cherry Pick" of the remote Source Control graph's commit menu each send one ``gitAction``; the outcome comes back as a ``gitActionResult`` to the requester only.<br/>`async git_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``gitAction`` command (``action``, ``sha``, optional ``name``, ``message``, ``workDir``, ``tabId``, ``token``).
  - `ctx`: The transport context of the current call.

- **fs_action** — Run an Explorer context-menu file action for the remote webapp. New File..., New Folder..., Rename..., Delete, Paste, Find in Folder... and Compare Selected of the remote Explorer's context menu each send one ``fsAction``; the outcome comes back as an ``fsResult`` to the requester only.<br/>`async fs_action(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``fsAction`` command (``action``, ``path``, optional ``dest``, ``name``, ``query``, ``overwrite``, ``workDir``, ``tabId``, ``token``).
  - `ctx`: The transport context of the current call.

- **share_chat** — Write a chat page's transcript as a standalone HTML page. The chat page serialized the highlighted tab's static task panel and event panels (its ``shareChat`` command carries the markup) and asks the daemon to save them as ``reports/chat-<title-slug>-<chatId>.html`` under the tab's work dir.  The reply is a direct ``share_done`` event to the requester.<br/>`async share_chat(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``shareChat`` command (``chatId``, ``html``, optional ``title``, ``workDir``, ``tabId``).
  - `ctx`: The transport context of the current call.

- **share_chat_tasks** — Send the requester every task of a chat for a share export. The chat page's share button exports the whole chat, but after a reload its DOM holds only one task's transcript — the daemon's session replay repaints a single task.  This command returns the persisted transcripts of ALL of the chat's tasks (oldest first) as a direct ``share_tasks`` reply; the page replays them into detached containers, splices in the live DOM for the on-screen task, and sends the assembled page back via ``shareChat``.<br/>`async share_chat_tasks(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``shareChatTasks`` command (``chatId``, optional ``tabId``, optional ``taskId`` — a sub-agent tab's share narrows the reply to that one task and its sub-agents).
  - `ctx`: The transport context of the current call.

- **get_history** — Answer a CHATS panel page in the background, in the connection's order. The page merges every registered machine's history (:func:`kiss.agents.sorcar.machines.merged_history`): a second or more over the machines' links, up to their timeout when one is down.  Each connection's commands are dispatched one at a time, so producing it inline would hold up a ``submit`` or ``stop`` sent right after it, and the page would itself wait behind the slow forwards in front of it (a transcript's ``checkPaths`` on a remote machine, a panel poll).  The connection's pages are answered in the order they were asked for: a later page of a view extends the earlier one.<br/>`async get_history(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getHistory`` command.
  - `ctx`: The transport context of the current call.

- **voice_transcribe** — Transcribe a browser's post-wake utterance. The chat page heard the "Abracadabra" wake word and captured the utterance that followed.  The audio is translated by gpt-audio (:func:`kiss.server.speech.transcribe_pcm`) and answered with the ``voiceSpeech`` message ``voice.js`` handles.  The transcription is a network round-trip (and, on first use, the speaker model's download), so it runs in the background: a ``stop`` or ``submit`` the page sends meanwhile is not held up behind it.  The page's dictations are answered in the order they were sent, though: ``voice.js`` pairs each ``voiceSpeech`` with the oldest of its outstanding rounds (the conversation the words were spoken into), so a short dictation answered before a longer earlier one would put each transcript into the other's conversation.<br/>`async voice_transcribe(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``voiceTranscribe`` command carrying the audio.
  - `ctx`: The transport context of the current call.

- **active_tasks_query** — Report in-flight agent tasks back to the requesting client.<br/>`async active_tasks_query(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``activeTasksQuery`` command (unused).
  - `ctx`: The transport context of the current call.

- **get_welcome_info** — Send the welcome-screen info (the active remote URL) to the asking client.<br/>`async get_welcome_info(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``getWelcomeInfo`` command (unused).
  - `ctx`: The transport context of the current call (its connection id addresses the reply).

- **run_update** — Run the KISS Sorcar installer to update the checkout.<br/>`async run_update(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``runUpdate`` command (unused).
  - `ctx`: The transport context of the current call; supplies the requesting ``conn_id`` so acknowledgement notifications reach only the requesting window.

- **update_models** — Refresh the user-local model catalog (``~/.kiss/MODEL_INFO.json``). Services the settings panel's "Update Models" button: the daemon runs ``kiss.scripts.update_models --model-info`` against the user-local catalog copy as a detached subprocess.<br/>`async update_models(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``updateModels`` command (unused).
  - `ctx`: The transport context of the current call; supplies the requesting ``conn_id`` so acknowledgement notifications reach only the requesting window.

- **snooze_update** — Snooze the update notification for 24 hours. Services the "Remind me later" action of the update toast: records the snooze in the update-check cache and rebroadcasts the ``update_available`` state so every client's toast disappears.<br/>`async snooze_update(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``snoozeUpdate`` command; its optional ``latest`` field names the release being snoozed.
  - `ctx`: The transport context of the current call (unused — the resulting rebroadcast must reach every window).

- **tips_opt_out** — Persist, or forget, the "Don't show tips again" choice. Services the tips window's checkbox.  The choice is the marker file ``$KISS_HOME/TIPS_DISABLED`` that ``tips_data`` reads, so a choice made in one browser holds in every browser.  ``optOut`` ``false`` (checkbox unticked again) removes the marker; absent or any other value opts out.<br/>`async tips_opt_out(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``tipsOptOut`` command with an optional boolean ``optOut``.
  - `ctx`: The transport context of the current call (unused).

- **dismiss_notice** — Service a toast's "Do not show this again" checkbox. Only the "VS Code extension removed" notice offers the box today; its dismissal is a marker file under ``$KISS_HOME``, so the choice holds in every browser of this daemon.  Every window then gets a ``close`` for the toast.  An unknown id is ignored.<br/>`async dismiss_notice(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``dismissNotice`` command with the notification ``id``.
  - `ctx`: The transport context of the current call (unused — the close must reach every window).

- **signin_login** — Service a click on a provider button of the sign-in window. Starts the brand's ``$KISS_HOME/signin login <provider>`` (:class:`kiss.server.signin.SigninLogin`); the window gets ``signinUrl`` as soon as the hook prints the page to open, every window ``signinDone`` when the sign-in succeeds, the clicking window ``signinError`` when it fails.<br/>`async signin_login(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``signinLogin`` command with the ``provider`` id.
  - `ctx`: The transport context of the current call; supplies the ``conn_id`` the URL and errors are stamped with.

- **signin_paste** — Forward a pasted redirect URL or code to the running sign-in hook. For a browser on another machine, whose redirect to the hook's loopback port cannot land.  Without a running login the window gets a ``signinError`` saying to start over.<br/>`async signin_paste(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``signinPaste`` command with the pasted ``text``.
  - `ctx`: The transport context of the current call.

- **signin_cancel** — End the running sign-in hook: the window was closed with "Not now".<br/>`async signin_cancel(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``signinCancel`` command (unused).
  - `ctx`: The transport context of the current call (unused).

- **update_when_idle** — Arm (or cancel) an update that runs once no task is running. Services the "Update when idle" action of the update toast: the daemon polls its agent registry and launches ``install.sh`` the first time no task is in flight.  The ``update_available`` state is rebroadcast with ``pendingIdle`` so every chat window's toast reflects the armed state.<br/>`async update_when_idle(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``updateWhenIdle`` command; ``cancel: true`` disarms a pending idle update instead of arming one.
  - `ctx`: The transport context of the current call (unused — the resulting rebroadcast must reach every window).

- **ping** — Answer a client's ordering probe with a direct ``pong``. A connection's commands are dispatched one after another, so the ``pong`` reaches the sender only once every command it sent before the ``ping`` has been taken.  The chat page's transport (``media/transport.js``) relies on that: after a (re)connect it flushes the commands queued while the socket was down, sends ``ping`` and keeps the batch until ``pong`` arrives, re-sending it on the next connection if this one dies first.  Another daemon's :class:`~kiss.agents.sorcar.machines.MachineLink` pings this one the same way; the ``pong`` names this machine (its short host name, ``machine``) so that daemon can label this machine by its name rather than by the address it was registered at.<br/>`async ping(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``ping`` command (no fields are used).
  - `ctx`: The transport context of the current call; the reply goes to this sender only.

- **auth** — Discard an ``auth`` frame on an already-authenticated connection. The real handshake is :meth:`authenticate`, run before the dispatch loop starts; a repeated frame (a client re-sending its queued batch after a reconnect) carries nothing to act on.<br/>`async auth(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``auth`` command (unused).
  - `ctx`: The transport context of the current call (unused).

- **server_reset** — Restart the kiss-web daemon at the user's request.<br/>`async server_reset(cmd: dict[str, Any], ctx: ApiContext) -> None`
  - `cmd`: The ``serverReset`` command (unused).
  - `ctx`: The transport context of the current call; supplies the requesting ``conn_id`` so acknowledgement notifications reach only the requesting window.

- **trajectory_jobs** — List all trajectory jobs (the ``/api/jobs`` endpoint). Mirrors the ``/api/jobs`` endpoint of the standalone trajectory visualizer (:mod:`kiss.viz_trajectory.server`, imported lazily so this client-importable module stays light).<br/>`trajectory_jobs() -> tuple[int, str, bytes]`
  - **Returns:** ``(200, "application/json", body)`` with the JSON job list.

- **job_trajectories** — Serve one job's trajectory list (``/api/jobs/<job>/trajectories``). Mirrors the ``/api/jobs/<job_name>/trajectories`` endpoint of the standalone trajectory visualizer.<br/>`job_trajectories(path: str) -> tuple[int, str, bytes]`
  - `path`: Request path of the form ``/api/jobs/<job_name>/trajectories``.  The transport has already URL-decoded it exactly once; the job segment must NOT be unquoted again or names containing literal percent-escapes would spuriously 404.
  - **Returns:** ``(200, "application/json", body)`` with the trajectory list, a 400 reply for an invalid job name, or a 404 reply when the job directory does not exist.

**`validate_command`** — Validate one client command against the server API catalog.<br/>`def validate_command(cmd: Any) -> str | None`

- `cmd`: The parsed JSON value received from a client.
- **Returns:** ``None`` when *cmd* is a valid API command, otherwise a human-readable error string (unknown command name or missing required field).

**`translate_client_command`** — Translate a chat-page wire command into a backend command. The chat page (``media/main.js``) speaks the wire dialect of this API; the backend agent server expects slightly different field names for one command.  Translation applied: * ``resumeSession`` → renames the ``id`` field to ``chatId`` (``media/main.js`` posts ``userAnswer`` directly, so no ``userActionDone`` rewrite is needed here.)<br/>`def translate_client_command(cmd: dict[str, Any]) -> dict[str, Any]`

- `cmd`: Raw command dictionary received from a client.
- **Returns:** The (possibly copied and modified) command dictionary ready for the backend agent server (``AgentServer._handle_command``).

**`passwords_equal`** — Compare two secrets (the local token) in constant time.<br/>`def passwords_equal(a: str, b: str) -> bool`

**`is_loopback_ip`** — Return True when *ip* is an IPv4/IPv6 loopback address. A loopback TCP peer on the public port is the local ``cloudflared`` tunnel relaying a remote visitor, or a process (a browser, a Python client) on this host.  IPv6-mapped IPv4 loopback (``::ffff:127.0.0.1``) counts too.  A malformed / empty string is not loopback.<br/>`def is_loopback_ip(ip: str) -> bool`

- `ip`: A textual IP address (no port).
- **Returns:** True when *ip* parses to a loopback address.

---

#### `kiss.agents.sorcar.daemon_client` — *Synchronous client for running tasks on the ``kiss-web`` daemon.*

##### `class StopUnconfirmedTimeoutError(TimeoutError)` — Timeout whose ``stop_on_timeout`` stop was sent but never confirmed.

##### `class CancelledError(Exception)` — The caller's *cancel* event was set while :func:`run` waited.

**Constructor:** `CancelledError(message: str, result: TaskResult, confirmed: bool) -> None`

##### `class StoppedOnTimeoutError(TimeoutError)` — Timeout whose ``stop_on_timeout`` stop the daemon confirmed.

**Constructor:** `StoppedOnTimeoutError(message: str, result: TaskResult) -> None`

- `message`: The timeout message.
- `result`: The stopped task's :class:`TaskResult`.

##### `class TaskResult` — Final outcome of one synchronous daemon task run.

**`resolve_sea_path`** — Validate a client-supplied SEA path and resolve it. Client-side counterpart of the daemon's ``kiss.agents.sorcar.sea_apply.apply_sea``: the path is resolved against the CLIENT's working directory (the daemon may run with a different one) and validated eagerly so a bad value fails fast, before any daemon connection is made.<br/>`def resolve_sea_path(sea_path: str | None) -> str`

- `sea_path`: Path string of a SEA file (a Python file defining a ``BaseSea`` subclass), or ``None``/empty for no SEA.
- **Returns:** The absolute path as a string, or ``""`` when *sea_path* is ``None`` or empty.

**`run_command`** — Encode a run's arguments as the ``run`` command :func:`run` sends the daemon. One wire field per argument; the setting arguments land in the fields :data:`~kiss.agents.sorcar.sea_settings.SETTING_FIELDS` names, and *provenance* marks which of them the caller passed explicitly.  Pure: :func:`run` sends what this returns.  The arguments not listed below are those of :func:`run`.<br/>`def run_command(prompt: str, *, tab_id: str, run_token: str, work_dir: str = '', scope_work_dir: str = '', parent_task_id: str = '', parent_tab_id: str = '', parent_reviewer: bool = False, side_channel: bool = False, cron_job: bool = False, model: str = '', chat_id: str = '', system_prompt: str = '', sea_path: str = '', max_budget: float | None = None, model_config: dict[str, Any] | None = None, use_web_tools: bool | None = None, auto_classify: bool | None = None, use_memory: bool | None = None, is_parallel: bool = True, add_to_system_prompt: str = '', add_to_prompt: str = '', tool_profile: str = '', docker_image: str = '', inherit_tools: bool = False, isolated_repos: dict[str, str] | None = None, workspace: str = '', provenance: dict[str, str] | None = None, timeout: float | None = None, machine: str = '', sea_command: str = '', attachments: list[dict[str, str]] | None = None) -> dict[str, Any]`

- `prompt`: The task text.
- `tab_id`: The tab the run is shown under.
- `run_token`: The client-minted per-submission run token.
- `sea_path`: The resolved SEA file (``""`` for none).
- `timeout`: The seconds recorded as the run's ``timeout`` setting (``None`` for the daemon default).
- `machine`: The registered remote machine the run executes on (wire field ``machine``, set only when non-empty).
- `sea_command`: The registered command whose SEA the executing daemon runs as its own copy (wire field ``seaCommand``, set only when non-empty).
- `attachments`: The prompt's files, each ``{"mimeType", "data"}`` with base64 *data* (wire field ``attachments``, set only when non-empty).
- **Returns:** The command dict.

**`run`** — Run *prompt* as a task on the local Sorcar daemon and block until done. Connects to the ``kiss-web`` daemon's local WSS endpoint, sends the same ``run`` command a chat page would, streams the task's events, and returns once the daemon reports the task finished.<br/>`def run(prompt: str, *, work_dir: str = '', scope_work_dir: str = '', parent_task_id: str = '', parent_tab_id: str = '', parent_reviewer: bool = False, side_channel: bool = False, cron_job: bool = False, model: str = '', chat_id: str = '', system_prompt: str = '', sea_path: str = '', max_budget: float | None = None, model_config: dict[str, Any] | None = None, use_web_tools: bool | None = None, auto_classify: bool | None = None, use_memory: bool | None = None, is_parallel: bool = True, add_to_system_prompt: str = '', add_to_prompt: str = '', tool_profile: str = '', docker_image: str = '', inherit_tools: bool = False, isolated_repos: dict[str, str] | None = None, workspace: str = '', provenance: dict[str, str] | None = None, timeout: float | None = 3600.0, stop_on_timeout: bool = False, record_timeout: float | None = None, endpoint_file: str | Path | None = None, cancel: threading.Event | None = None, running: threading.Event | None = None, machine: str = '', sea_command: str = '', attachments: list[dict[str, str]] | None = None) -> TaskResult`

- `prompt`: The task instruction to run.
- `work_dir`: Working directory for the task; the daemon's current default is used when empty.
- `scope_work_dir`: The CALLING workspace recorded on the task's tab in the daemon's shared tab registry (``scopeWorkDir``), kept separate from *work_dir* (the channel/cron scratch directory a standalone dispatch executes in). Informational only: every client shows every registry tab whatever folder it runs in.  Empty (the default) records nothing.  Irrelevant for a sub-agent run (non-empty *parent_task_id*), which gets no registry tab at all.
- `parent_task_id`: The persisted ``task_history`` row id of the CALLING task, when this run is dispatched on behalf of one (the ``run_agent`` tool).  Non-empty marks the run as a SUB-AGENT of that task, giving its tab the same frontend behavior as a ``run_parallel`` sub-task: no top-level tab of its own — instead every client viewing the parent gets a nested sub-agent tab (via the run's ``new_tab`` broadcast), the run's history row nests under the parent task, and a ``subagentDone`` broadcast stops the tab's running indicator when the run ends.  Empty (the default) runs as an ordinary top-level task.  It is a client/UI-transport parameter with no SEA setting: a dispatched script must not be able to re-parent itself under an unrelated task.
- `parent_tab_id`: Frontend tab id of the calling task's tab, forwarded on the sub-agent's ``new_tab`` broadcast so the chat page knows which tab spawned it (nested placement and cascade-close).  Only meaningful with *parent_task_id*; empty spawns a parentless sub-agent tab, exactly like a headless ``run_parallel`` fan-out.  No SEA setting.
- `parent_reviewer`: Whether the dispatched run belongs to a reviewer's sub-tree — the caller is a reviewer sub-agent, or *prompt* itself is a review task (see :mod:`kiss.agents.sorcar.fanout_guard`).  Stamped on the child's ``_subagent_info`` so it and its helpers keep the read-only ``review`` tool profile.  Only meaningful with *parent_task_id*; no SEA setting, for the same reason as *parent_task_id*.
- `side_channel`: Whether the run is a side channel of the parent — a sub-agent whose result is delivered into the PARENT's transcript (the ``/ask`` answer panel), so its own nested tab is scaffolding that is closed when the run ends and never re-opened by a replay.  Persisted on the child's history row; only meaningful with *parent_task_id*; no SEA setting.
- `cron_job`: Whether the cron scheduler launched the run (an in-chat job; prompt jobs are recognised by their preamble). Persisted as the history row's ``is_cron`` flag, which the CHATS list hides unless its Cron chip is on; no SEA setting.
- `model`: Model name; the daemon's selected default when empty.
- `chat_id`: Optional existing chat session id to continue.  Pass the ``chat_id`` of a previous :class:`TaskResult` to run this task in the same chat — the agent then sees the prior tasks and results of that chat as context.  A new chat is started when empty.
- `system_prompt`: Optional custom system prompt for the run. When non-empty it is used as the system prompt of the agent AND of every sub-agent it spawns (``run_parallel``), replacing the default system prompt shipped in ``src/kiss/SYSTEM.md``.  The daemon still appends its per-run operational instructions (work directory, process id, ``$KISS_HOME/AGENTS.md``) so the agent's tool contract keeps working.  Empty (default) runs with the default system prompt as usual.
- `sea_path`: Optional path — a string — to a Python file defining a Sorcar Extension Agent (SEA) that configures this run **on the daemon**.  When non-empty, the daemon loads the file's one subclass of :class:`kiss.agents.seas.base.base_sea.BaseSea` and applies its methods (:mod:`kiss.agents.sorcar.sea_commands`, :func:`kiss.agents.sorcar.sea_apply.apply_sea`) on top of the values passed to this call: a setting the SEA declares replaces the parameter of the same name; one it does not declare keeps the value passed here. File format — one class deriving from ``BaseSea`` that overrides any of its methods (every one has an identity default):: from kiss.agents.seas.base.base_sea import BaseSea class MySea(BaseSea): def description(self) -> str: ...            # /xxx help text def settings(self, settings: dict) -> dict:  # settings | {kind, run() keywords} ... def prompt(self, task: str) -> str: ...      # the task text -> the prompt def system_prompt(self, system_prompt: str) -> str:  # assembled -> the run's ... def tools(self, tools: list) -> list: ...    # toolset -> the run's toolset def llm_call_hook(self, new_messages: list) -> list: ... def tool_call_hook(self, name: str, args: dict) -> Verdict: ...  # ALLOW/refuse def register_as_model(self) -> bool: ...     # model-picker entry def on_picked_as_model(self, work_dir: str) -> str: ... A SEA extends another by deriving from its class (the launcher runs every class of the chain, base first; do not call ``super()``).  ``settings`` returns its argument with any of the keyword parameters of this function except the transport, identity and prompt ones (``prompt`` and ``system_prompt`` are the methods above, not settings) laid over it: ``work_dir``, ``model``, ``chat_id``, ``max_budget`` (finite), ``model_config``, ``use_web_tools``, ``auto_classify``, ``use_memory``, ``tool_profile``, ``docker_image``; plus two dispatcher keys: ``timeout`` (seconds a ``run_agent`` call waits for this SEA's sub-task) and ``locked`` (keys an explicit caller argument may not change).  ``prompt(task)`` receives the task text and returns the prompt body; every ``{task_id}`` of the result is replaced by *parent_task_id*.  A ``None`` value, or ``""`` for a string key, means "no override".  The base class says what the SEA is: ``BaseSea`` changes nothing; ``WorkerSea`` lays the worker defaults under the explicit keys (``auto_classify``, ``use_web_tools``, ``use_memory`` all off) and drops the worktree tools; ``ChannelSea`` (a worker) marks a messaging channel: it defaults ``work_dir`` to ``$KISS_HOME/channel_work`` and locks it with the worker keys, gives the run the channel preamble and a workspace held for the run, makes a ``run_agent`` sub-task of it inherit nothing from the caller, and lists it as a channel. ``system_prompt(system_prompt)`` receives the run's assembled system prompt (the base prompt plus *add_to_system_prompt*) and returns the run's: the same text with additions, or a replacement.  ``tools(tools)`` receives the built-in toolset and returns the run's: a list of tool callables (never a file path); with ``"tool_profile": "none"`` it starts empty, so what it returns and ``finish`` are the run's whole tool set (the default ``SYSTEM.md`` prompt, whose workflow rules name ``Read``, ``Edit``, ``Bash`` and the browser tools, should then usually be replaced by a ``system_prompt`` written for the tools the run has).  Each tool's name, docstring (Google-style ``Args:`` section) and annotated keyword-bindable parameters define the tool schema the agent sees, exactly like a native tool. The two hook methods are passed to the underlying :meth:`kiss.core.kiss_agent.KISSAgent.run` of every task-executor sub-session of the task's agent (internal helper sessions, e.g. the failed-session trajectory summarizer, are not hooked).  Per that method's contract, ``llm_call_hook(new_messages)`` is called before every LLM call and its return value replaces the new messages about to be sent, and ``tool_call_hook(name, args)`` is called before every tool call and returns a ``Verdict``: the tool executes on ``ALLOW``; on ``refuse(text)`` the model is given *text* as the tool's result instead.  The hooks apply to the task's own agent, not to sub-agents it spawns via ``run_parallel``. A SEA whose ``register_as_model()`` returns ``True`` is listed in the model picker under its command name (the bundled ``autorouter`` and ``bestrouter``); picking it runs every task of the tab through the SEA on the model its ``settings()["model"]`` names (else the default model), with the protocol its ``system_prompt`` method adds to the system prompt. Everything the script defines runs **in the daemon process**; nothing is serialized by the client. ``timeout``, *stop_on_timeout*, *endpoint_file*, *scope_work_dir*, *parent_task_id*, and *parent_tab_id* have no settings key by design: the first three are client-transport parameters — the script only runs on the daemon that *endpoint_file* selects, *timeout* bounds this client's local wait, and *stop_on_timeout* picks this client's timeout behavior — and *scope_work_dir* / *parent_task_id* / *parent_tab_id* are the CALLING task's identity, which the script must not be able to forge.  The *sea_path* itself is resolved against this process's working directory and validated eagerly.  A broken SEA (deleted before the daemon reads it, raising at import time, a non-callable getter, a raising ``settings()`` or getter, an unknown settings key or kind, or a wrong-typed value) stops the task: the daemon fails the run and the returned :class:`TaskResult` carries the diagnostic error in its ``text`` with ``success=False``.
- `max_budget`: Per-task budget override in USD; ``None`` uses the daemon's configured default.
- `model_config`: Per-task model configuration override (custom endpoint / headers); ``None`` uses the daemon's configured model endpoint.  Must be JSON-serializable.
- `use_web_tools`: Per-task browser-tool enablement override, mapped to the agent's ``web_tools`` toggle (:meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`); ``None`` uses the daemon's configured default (the settings panel's "Use web tools" checkbox, persisted as ``use_web_browser``).
- `auto_classify`: Per-task override of pre-run task classification (``kiss.agents.sorcar.task_classifier``), which runs one lightweight model call before the task — a typed question to a decisions model when an OpenRouter key is configured and the settings panel's "Use Jev" checkbox (``classify_with_decisions``) is on, else a non-agentic call on the run's own LLM — to pick the system prompt (lite vs. full).  ``True`` forces classification on, ``False`` skips it — the run then uses the full system prompt — and ``None`` (the default) uses the daemon's configured default (the settings panel's "Classify tasks before running" checkbox, persisted as ``classify_tasks``).
- `use_memory`: Per-task persistent-memory override (``kiss.core.memoryfield``), mapped to the agent's ``use_memory`` toggle (:meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`). ``True`` gives the run (and its ``run_parallel`` sub-agents) the ``memory_*`` tools plus the ``MEMORY_PROTOCOL`` prompt block, ``False`` withholds them, and ``None`` (the default) uses the daemon's configured default (the settings panel's "Use persistent memory" checkbox, persisted as ``use_memory``, or the daemon process's ``KISS_USE_MEMORY`` environment variable).  A boolean override never bypasses the memory safety gates: a run without the basic toolset (the ``none`` tool profile), a Docker run, or a caller-supplied ``model_config["system_instruction"]`` stays memory-free even with ``True``.
- `is_parallel`: Whether the agent gets the ``run_parallel`` tool (the user's parallel-mode toggle).  Defaults to True.
- `add_to_system_prompt`: Extra text appended to the run's system prompt when the agent is executed — after the default ``SYSTEM.md`` prompt (or the *system_prompt* replacement) and before the daemon's per-run operational instructions.  ``run_parallel`` sub-agents inherit the suffix on their own system prompts, like a *system_prompt* replacement, so the extra instructions constrain the whole task tree.  Empty (default) appends nothing.
- `add_to_prompt`: Extra text appended to the executed task prompt.  A multi-``<task>`` *prompt* runs the agent once per subtask, and the text is appended to EACH subtask's prompt.  The appended text is part of the prompt the agent actually runs with, so it is also what the chat history records and what follow-up tasks of the same chat see as context.  Empty (default) appends nothing.
- `tool_profile`: Name of the tool profile the task's built-in toolset is cut down to — a key of :data:`kiss.agents.sorcar.sorcar_agent.TOOL_PROFILES` (the composites ``"full"``, ``"review"``, ``"assistant"``, ``"bash"`` or the groups ``"shell"``, ``"edit"``, ``"browser"``, ``"memory"``, ``"agents"``, ``"mcp"``, ``"skills"``, ``"user"``, ``"decide"``, ``"control"``), or several keys joined with ``+`` for the union of their tools (``"shell+edit+memory"``); ``bash`` is the single-command runner of the bundled ``/sh`` agent: ``Bash`` and ``finish`` only; ``none`` keeps no built-in tool at all (``finish`` plus what the SEA's ``tools()`` returns, the bundled ``/ask`` agent).  Empty (the default) keeps the daemon's usual choice (the full toolset). An unknown name stops the task with a diagnostic error.
- `docker_image`: Run the task's file and shell tools (``Bash``, ``run_commands_parallel``, ``Read``, ``Edit``, ``Write``) inside a Docker container instead of on the daemon's host. An image name (``"python:3.12"``) starts a fresh container that is removed when the task ends; ``container:<name-or-id>`` attaches to a container the caller already runs and leaves it running.  ``run_parallel`` sub-agents share the task's container.  ``bash_job`` and persistent memory are unavailable in a Docker run.  Empty (default) runs the tools on the host.
- `isolated_repos`: Repository root -> worktree directory of the calling agent's git worktrees (its ``isolated_repos()``). The task never creates a worktree for one of these repositories: it edits the caller's worktree, which the caller merges.  Carried on the wire because a caller run outside the daemon (the CLI) has no registered state for the daemon to read it from.
- `inherit_tools`: Whether the task also gets the extra tools of the task *parent_task_id* names — the tool callables that parent's SEA added through ``tools()`` (and those the parent inherited itself), resolved on the daemon from the running parent (a callable cannot travel the wire).  They are added to the task's built-in toolset before the task's own SEA's ``tools()`` sees it, skipping names the task already has; a SEA on the ``none`` tool profile keeps exactly its own set. ``run_agent`` sets this for the sub-tasks it dispatches in path mode, so a sub-task that inherits the caller's system prompt also has the tools that prompt refers to.  ``False`` (default) adds nothing; ignored without *parent_task_id*.
- `workspace`: Workspace/account identifier for multi-account channels.  A channel SEA's run (``channel: True``) holds it (``KISS_CHANNEL_WORKSPACE``) for its whole lifetime, so its channel tools load that account's credentials; empty means ``"default"``.  Ignored by every other run.
- `provenance`: Where the command's values come from, ``{setting key: "explicit" | "inherited"}`` (see :mod:`kiss.agents.sorcar.run_config`): a field the caller was given explicitly, or one it filled from the agent calling it.  Recorded in the task's ``task_settings`` event (``inherited``), so the result's ``ran:`` line and the persisted history show which values the sub-task took over. ``None`` (default): every value is a persisted setting or the daemon's default.
- `timeout`: Maximum seconds to wait for the task to finish; ``None`` waits indefinitely.  With *stop_on_timeout* it is also sent to the daemon (wire field ``timeout``), which records it in the task's ``task_settings`` event.
- `stop_on_timeout`: Whether a *timeout* expiry also STOPS the task.  ``False`` (the default) keeps the documented timeout contract — the caller stops waiting, the task keeps running.  With ``True`` the client sends the daemon a ``stop`` for the task and keeps reading until the task's terminal status confirms it is dead (waiting up to :data:`_STOP_CONFIRM_GRACE_SECONDS`; on a wedged daemon :class:`StopUnconfirmedTimeoutError` is raised instead, the stop then staying best-effort) before raising the ``TimeoutError``. ``True`` is for callers that must not let the task outlive the wait (the ``/ask`` side channel, ``commands.py``).  The ``run_agent`` dispatch does not use it: its ``timeout`` bounds the call only, and its sub-task is stopped through *cancel* (an ``agent_job`` kill, or the calling run's end). When the task finished ON ITS OWN — a SUCCESSFUL terminal ``result`` AND the terminal status raced the stop onto the wire — the completed :class:`TaskResult` is returned instead of a ``TimeoutError`` that would discard the finished work (only natural completions carry ``success: true``; every daemon stop/cancel/failure path broadcasts ``success: false``).  A successful result WITHOUT the terminal status never settles the wait: the daemon's persistence / worktree cleanup still run after the result is emitted, so only the terminal status proves the task is dead, and the grace expiring with just the result in hand raises :class:`StopUnconfirmedTimeoutError` all the same.
- `record_timeout`: A bound to send as the wire field ``timeout`` (recorded in the task's ``task_settings`` and checked against a SEA's locked ``timeout``) when this wait itself has none: ``run_agent`` waits here without a deadline on a job thread and bounds the call by joining that thread, so it passes the call's bound through this.  ``None`` sends *timeout* when *stop_on_timeout*, else nothing.
- `endpoint_file`: Daemon endpoint file override (defaults to ``$KISS_SORCAR_LOCAL`` or ``$KISS_HOME/sorcar-local.json``).
- `cancel`: An event the caller may set from another thread to stop the task (``agent_job(..., "kill")``): the daemon is sent a ``stop`` and the wait continues until the task's terminal status confirms it is dead (or the confirmation grace expires), then :class:`CancelledError` is raised with ``confirmed`` set accordingly.  ``None`` (the default) waits for the task or the timeout.
- `running`: An event set when the task's initial ``status running=true`` arrives — the moment its tab exists on every client.  Not set by a wait that ends without one: the ``run_agent`` job thread sets it after recording the dispatch's outcome, so a ``wait="false"`` call woken by it either has a tab to report or the finished result.
- `machine`: The name of a registered remote machine (:func:`kiss.agents.sorcar.machines.load_machines`) the task executes on: the local daemon forwards the ``run`` to that machine's daemon and relays its frames back (:mod:`kiss.server.remote_tabs`), so this wait is the same. *work_dir* and *sea_path* are then paths on that machine (*sea_path* is still checked to exist here).  Empty (the default) runs the task on the local daemon.
- `sea_command`: The name of a registered command: the executing daemon runs its own copy of that command's SEA (the wire field ``seaCommand``, resolved by the daemon's SEA pipeline; an unknown name fails the run).  How a run on a *machine* names a SEA that machine has under its own paths; it replaces *sea_path*.  Empty (the default) sends *sea_path*.
- `attachments`: Files sent with the prompt (an image or PDF the user attached), each ``{"mimeType", "data"}`` with base64 *data* — the webapp's ``submit`` wire shape.
- **Returns:** A :class:`TaskResult` with the result text, success flag, cost (USD), total tokens, step count, chat id, and task id of the task.  ``chat_id`` is the daemon chat session id and ``task_id`` the persisted ``task_history`` row id — both usable later to look up or resume the run in the daemon's history.

---
