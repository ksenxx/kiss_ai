# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

# CLI plumbing lives in cli_helpers; re-exported here for backwards
# compatibility (tests and callers import these from this module).
from kiss.agents.sorcar.cli_helpers import (
    _DEFAULT_TASK as _DEFAULT_TASK,
)
from kiss.agents.sorcar.cli_helpers import (
    _resolve_task as _resolve_task,
)
from kiss.agents.sorcar.cli_helpers import (
    cli_ask_user_question as cli_ask_user_question,
)
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_default_model
from kiss.core.models.model_info import model as _model_factory
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent

logger = logging.getLogger(__name__)


def _save_setting_to_config(key: str, value: Any) -> None:
    """Persist a single setting to ~/.kiss/config.json.

    Loads the existing config, updates the specified key, and saves
    atomically. Also applies the change to the running environment.

    Args:
        key: The config key (e.g. "max_budget", "work_dir").
        value: The new value for the key.
    """
    try:
        from kiss.agents.vscode.vscode_config import (
            apply_config_to_env,
            load_config,
            save_config,
        )

        cfg = load_config()
        cfg[key] = value
        save_config(cfg)
        apply_config_to_env(cfg)
    except Exception:
        pass


def _apply_setting(
    updated: list[str],
    broadcast: Callable[[dict[str, Any]], None] | None,
    key: str,
    value: Any,
    label: str | None = None,
    broadcast_value: Any = None,
) -> None:
    """Record, persist, and broadcast a single settings update.

    Appends a human-readable entry to *updated*, persists *value* under
    *key* via :func:`_save_setting_to_config`, and broadcasts an
    ``updateSetting`` event to the frontend when *broadcast* is set.

    Args:
        updated: Accumulator of "key=value" summary strings.
        broadcast: Frontend broadcast callable, or ``None``.
        key: The config / broadcast key.
        value: The new value to persist.
        label: Override for the summary entry (default ``"{key}={value}"``);
            used to mask secrets (e.g. ``"remote_password=<updated>"``).
        broadcast_value: Override for the broadcast payload value
            (default *value*); used to avoid leaking secrets to the UI.
    """
    updated.append(label if label is not None else f"{key}={value}")
    _save_setting_to_config(key, value)
    if broadcast:
        broadcast({
            "type": "updateSetting",
            "key": key,
            "value": value if broadcast_value is None else broadcast_value,
        })


def _generate_commit_message(
    commit_dir: Path, user_prompt: str | None = None,
) -> str:
    """Generate a commit message for staged changes using an LLM.

    Gets the staged diff and delegates to
    :func:`~kiss.agents.vscode.helpers.generate_commit_message_from_diff`.
    When *user_prompt* is provided, it is forwarded so the user's
    task prompt is incorporated into the commit message.

    Args:
        commit_dir: The directory containing staged changes.
        user_prompt: The user's task prompt that produced these
            staged changes, or ``None`` when not available.

    Returns:
        A commit message string.
    """
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps
    from kiss.agents.vscode.helpers import generate_commit_message_from_diff

    diff_text = GitWorktreeOps.staged_diff(commit_dir)
    return generate_commit_message_from_diff(diff_text, user_prompt=user_prompt)


def auto_commit_changes(
    commit_dir: Path,
    user_prompt: str | None,
    message_fn: Callable[[Path, str | None], str],
) -> bool:
    """Stage all changes, generate a commit message, and commit.

    Stages all changes once, generates a commit message from the
    staged diff via *message_fn*, then commits the already-staged
    changes (without re-staging).  Falls back to a generic commit
    message when *message_fn* raises (e.g. the LLM-based generator
    is unavailable).

    Args:
        commit_dir: Directory whose changes are staged and committed.
        user_prompt: The user's task prompt, woven into the commit
            message (or its fallback), or ``None`` when unavailable.
        message_fn: Callable producing a commit message from
            ``(commit_dir, user_prompt)``.

    Returns:
        True if a commit was created, False if nothing to commit.
    """
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    GitWorktreeOps.stage_all(commit_dir)
    try:
        msg = message_fn(commit_dir, user_prompt)
    except Exception:
        logger.debug(
            "LLM commit message generation failed; using fallback", exc_info=True,
        )
        msg = "kiss: auto-commit agent changes"
        if user_prompt:
            from kiss.agents.vscode.helpers import _append_user_prompt

            msg = _append_user_prompt(msg, user_prompt)
    return GitWorktreeOps.commit_staged(commit_dir, msg)


def _yaml_failure(exc: BaseException) -> str:
    """Return a YAML result string for an unhandled sub-agent exception."""
    failure: str = yaml.dump(
        {"success": False, "summary": f"Unhandled exception: {exc}"},
        sort_keys=False,
    )
    return failure


def _agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return ``(budget_used, total_tokens_used, total_steps)`` for *agent*."""
    return (
        float(getattr(agent, "budget_used", 0.0) or 0.0),
        int(getattr(agent, "total_tokens_used", 0) or 0),
        int(getattr(agent, "total_steps", 0) or 0),
    )


def _broadcast_subagent_done(printer: Any, tab_ids: list[str]) -> None:
    """Broadcast ``subagentDone`` for each tab id so the frontend can
    stop the running indicator on the sub-agent tab.  Errors are
    swallowed (the broadcast is best-effort UI signalling)."""
    broadcast = getattr(printer, "broadcast", None)
    if broadcast is None:
        return
    for vid in tab_ids:
        try:
            broadcast({"type": "subagentDone", "tab_id": vid, "tabId": ""})
        except Exception:
            pass


def _attribute_sub_usage(agent: Any, budget: float, tokens: int, steps: int) -> None:
    """Attribute sub-agents' cost, tokens, and steps to the parent *agent*.

    Without this, sub-agent budgets would be invisible to the parent
    agent's global accounting and UI.  Also updates the printer offsets
    so the live status line in the current sub-session reflects the
    additional spend immediately (the offsets are otherwise
    snapshotted only at session start).
    """
    agent.budget_used = float(getattr(agent, "budget_used", 0.0) or 0.0) + budget
    agent.total_tokens_used = (
        int(getattr(agent, "total_tokens_used", 0) or 0) + tokens
    )
    agent.total_steps = int(getattr(agent, "total_steps", 0) or 0) + steps
    if agent.printer is not None:
        try:
            agent.printer.budget_offset = agent.budget_used
            agent.printer.tokens_offset = agent.total_tokens_used
            agent.printer.steps_offset = agent.total_steps
        except Exception:
            pass


# Attachment MIME-type prefixes and the human-readable labels used when
# describing attachments in the initial prompt, in display order.
_ATTACHMENT_KINDS: tuple[tuple[str, str], ...] = (
    ("image/", "image(s)"),
    ("application/pdf", "PDF(s)"),
    ("audio/", "audio file(s)"),
    ("video/", "video file(s)"),
)


def _attachment_parts(attachments: list[Attachment]) -> list[str]:
    """Return human-readable per-kind attachment counts (e.g. ``"2 image(s)"``)."""
    parts: list[str] = []
    for prefix, label in _ATTACHMENT_KINDS:
        count = sum(1 for a in attachments if a.mime_type.startswith(prefix))
        if count:
            parts.append(f"{count} {label}")
    return parts


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    # True only on subclasses that isolate every task in a git worktree
    # (see :class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`).
    uses_worktree: bool = False

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        # NOT redundant: the base class only sets ``docker_manager`` in
        # ``_reset`` (called from ``run``); tools can be built earlier.
        self.docker_manager: Any = None
        self._use_web_tools: bool = True
        self._is_parallel: bool = False

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute multiple independent tasks concurrently using parallel agents.

        Each task gets its own ``ChatSorcarAgent`` instance.  Subclasses can
        override this method to change the agent type or pass extra context
        (e.g. ``ChatSorcarAgent`` propagates ``chat_id``).

        This method is a pure parallel executor.  It has no knowledge of
        backend task ids or any frontend concepts (tabs, ``new_tab``
        broadcasts, etc.).  Any sub-agent-specific frontend behaviour is
        owned by the sub-agent itself — see
        :meth:`ChatSorcarAgent.run`, which self-broadcasts a ``new_tab``
        message whenever it detects ``self._subagent_info`` is set.

        Args:
            tasks: List of self-contained task description strings.
            max_workers: Maximum concurrent threads (``None`` = auto).

        Returns:
            List of YAML result strings in the same order as *tasks*.
        """
        totals: dict[str, float] = {}
        results = run_tasks_parallel(
            tasks,
            max_workers=max_workers,
            model_name=self.model_name,
            work_dir=self.work_dir,
            printer=self.printer,
            totals_out=totals,
        )
        _attribute_sub_usage(
            self,
            float(totals.get("budget_used", 0.0)),
            int(totals.get("total_tokens_used", 0)),
            int(totals.get("total_steps", 0)),
        )
        return results

    def _get_tools(self) -> list:
        """Build tool list, using DockerTools when docker_manager is active.

        Must be called after docker_manager is set up (i.e., from perform_task,
        not from run() before super().run()).
        """
        def _stream(text: str) -> None:
            if self.printer:
                self.printer.print(text, type="bash_stream")

        def ask_user_question(question: str) -> str:
            """Ask the user a question and wait for their typed response.

            Use when the agent needs clarification, confirmation, or additional
            information from the user in the middle of a task. The user sees
            the question in the chat window, types their answer, and clicks
            "I'm Done". The agent blocks until the answer is provided.

            Args:
                question: The question to display to the user.

            Returns:
                The user's typed response text.
            """
            ask_callback = getattr(self, "_ask_user_question_callback", None)
            if ask_callback:
                return str(ask_callback(question))
            return "(ask_user_question not available in this environment)"

        if self.docker_manager:
            from kiss.docker.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)
            tools: list = [
                self._docker_bash, docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            useful_tools = UsefulTools(
                stream_callback=_stream,
                stop_event=getattr(self, "_stop_event", None),
                work_dir=self.work_dir,
            )
            tools = [useful_tools.Bash, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self._use_web_tools and self.web_use_tool is None:
            self.web_use_tool = WebUseTool()
            tools.extend(self.web_use_tool.get_tools())
        def run_parallel(tasks: str, max_workers: str = "") -> str:
            """Run multiple independent tasks concurrently using parallel agents.

            Spawns a separate ChatSorcarAgent for each task string and executes
            them in parallel threads.

            **When to call run_parallel:**
            - Multi-source / multi-topic research ("research these 5
              companies", "summarize each of these N PDFs").
            - Codebase exploration across unrelated modules ("look at the
              frontend, backend, db layer, and auth in parallel").
            - Multi-perspective review of one artifact (correctness
              reviewer + security reviewer + style reviewer +
              architecture reviewer, each looking at the same diff with
              a different lens).
            - Generating N alternative candidates for the same problem
              so the orchestrator can pick the best.
            - Independent test suites or validations on disjoint targets.
            - Bulk file generation when each file is independent and the
              API contract between them is already pinned down in a
              spec.


            Args:
                tasks: A JSON-encoded list of task description strings.
                    Example::

                        '["Read src/foo.py and summarize its purpose", '
                        '"Read src/bar.py and summarize its purpose", '
                        '"Find the current weather in San Francisco"]'
                max_workers: Maximum number of concurrent threads, as a
                    string containing an integer (e.g. ``"4"``).  An empty
                    string (default) lets Python choose automatically.
                    Set to a lower number to limit concurrency.

            Returns:
                A YAML-formatted string containing a list of result
                objects, one per task, in the same order as the input.
                Each result object has ``success`` and ``summary`` keys.
            """
            task_list = _coerce_tasks(tasks)
            workers: int | None = int(max_workers) if max_workers else None
            results = self._run_tasks_parallel(task_list, max_workers=workers)
            result_str: str = yaml.dump(results, sort_keys=False)
            return result_str

        def update_settings(
            is_parallel: bool | None = None,
            is_worktree: bool | None = None,
            model_name: str | None = None,
            max_budget: float | None = None,
            use_web_browser: bool | None = None,
            remote_password: str | None = None,
            demo_mode: bool | None = None,
            auto_commit: bool | None = None,
            custom_endpoint: str | None = None,
            custom_headers: str | None = None,
        ) -> str:
            """Update task configuration settings during execution.

            Modifies runtime agent settings, persists config-level changes,
            and broadcasts UI updates to the frontend. Only provided
            (non-None) keys are updated; omitted keys are left unchanged.

            API keys (e.g. ``custom_api_key``, ``openai_api_key``,
            ``anthropic_api_key``) cannot be updated through this tool for
            security reasons.  Users must set them through the settings UI
            or environment variables.

            Args:
                is_parallel: Enable/disable parallel sub-agent spawning.
                is_worktree: Enable/disable git worktree isolation.
                model_name: Switch the LLM model for subsequent sub-sessions.
                max_budget: Set the maximum budget in USD.
                use_web_browser: Enable/disable browser/web tools.
                remote_password: Set the remote access password.
                demo_mode: Enable/disable demo replay mode in the UI.
                auto_commit: When True, trigger auto-commit of pending changes.
                custom_endpoint: Set a custom LLM endpoint URL (e.g. local model).
                custom_headers: Set custom HTTP headers (Key:Value, one per line).

            Returns:
                A summary of which settings were updated.
            """
            updated: list[str] = []
            broadcast = (
                getattr(self.printer, "broadcast", None)
                if self.printer
                else None
            )

            if is_parallel is not None:
                self._is_parallel = bool(is_parallel)
                _apply_setting(updated, broadcast, "is_parallel", self._is_parallel)

            if is_worktree is not None:
                _apply_setting(updated, broadcast, "is_worktree", bool(is_worktree))

            if model_name is not None:
                # Special-cased: the model is persisted via the last-model
                # store (not the config file) and applied to the live agent.
                self.model_name = model_name
                updated.append(f"model={model_name}")
                from kiss.agents.sorcar.persistence import _save_last_model
                _save_last_model(model_name)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "model",
                        "value": model_name,
                    })

            if max_budget is not None:
                self.max_budget = float(max_budget)
                _apply_setting(updated, broadcast, "max_budget", self.max_budget)

            if use_web_browser is not None:
                self._use_web_tools = bool(use_web_browser)
                _apply_setting(
                    updated, broadcast, "use_web_browser", self._use_web_tools,
                )

            if remote_password is not None:
                # Mask the secret in both the summary and the broadcast.
                _apply_setting(
                    updated, broadcast, "remote_password", remote_password,
                    label="remote_password=<updated>", broadcast_value=True,
                )

            if demo_mode is not None:
                _apply_setting(updated, broadcast, "demo_mode", bool(demo_mode))

            if auto_commit is not None and not auto_commit:
                # ``auto_commit`` is a one-shot action: ``False`` requests no
                # commit, but it is still a provided argument and must be
                # reported accurately (not as "all arguments were None").
                updated.append("auto_commit=not triggered (False)")
            elif auto_commit is not None and bool(auto_commit):
                # Special-cased: a one-shot action (commit pending changes),
                # not a persisted config value.
                updated.append("auto_commit=triggered")
                try:
                    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

                    wd = Path(self.work_dir).resolve()
                    if GitWorktreeOps.discover_repo(wd):
                        auto_commit_changes(
                            wd,
                            getattr(self, "_last_user_prompt", "") or None,
                            _generate_commit_message,
                        )
                except Exception:
                    pass
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "auto_commit",
                        "value": True,
                    })

            if custom_endpoint is not None:
                _apply_setting(updated, broadcast, "custom_endpoint", custom_endpoint)

            if custom_headers is not None:
                # Mask the header values in both the summary and broadcast.
                _apply_setting(
                    updated, broadcast, "custom_headers", custom_headers,
                    label="custom_headers=<updated>", broadcast_value=True,
                )

            if not updated:
                return "No settings were changed (all arguments were None)."
            return "Updated: " + ", ".join(updated)

        def number_of_cores() -> int:
            """Return the number of CPU cores available on the current machine.

            Useful for choosing a reasonable ``max_workers`` value when
            calling :func:`run_parallel`.

            Returns:
                The number of CPU cores available to the process,
                falling back to ``1`` when it cannot be determined.
            """
            return os.process_cpu_count() or 1

        def set_model(model_name: str) -> str:
            """Change the agent's LLM model dynamically.

            Args:
                model_name: New LLM model name (for example
                    ``"gpt-5.5"``, ``"claude-sonnet-4-8"``,
                    ``"gemini-3.5-flash"``).

            Returns:
                A human-readable confirmation string describing the
                change (or a "no change" message when the requested
                model is already active).
            """
            from kiss.agents.sorcar.persistence import _save_last_model

            old_model = getattr(self, "model", None)
            if old_model is None:
                self.model_name = model_name
                _save_last_model(model_name)
                return (
                    f"Model deferred-changed to {model_name} "
                    "(no live model yet)."
                )
            if old_model.model_name == model_name:
                return f"Model is already {model_name}; no change."

            # Reconstruct ``model_config`` for the factory.  The
            # ``OpenAICompatibleModel`` factory path strips ``base_url``
            # and ``api_key`` from ``model_config`` before storing it,
            # so we restore them from the live model's attributes so the
            # new model lands on the same endpoint.
            new_config: dict[str, Any] = dict(old_model.model_config or {})
            old_base_url = getattr(old_model, "base_url", None)
            old_api_key = getattr(old_model, "api_key", None)
            if old_base_url and "base_url" not in new_config:
                new_config["base_url"] = old_base_url
                if old_api_key is not None:
                    new_config["api_key"] = old_api_key
            new_model = _model_factory(
                model_name,
                model_config=new_config or None,
                token_callback=old_model.token_callback,
                thinking_callback=old_model.thinking_callback,
            )
            # Carry over the live conversation state so the next LLM
            # call resumes from the same point.  The kiss-agent
            # conversation list shape is OpenAI-style (``role`` /
            # ``content`` dicts plus tool_call entries) for every
            # provider in this codebase, so a direct hand-off is safe.
            new_model.conversation = old_model.conversation
            new_model.usage_info_for_messages = old_model.usage_info_for_messages

            previous_name = old_model.model_name
            self.model = new_model  # type: ignore[attr-defined]
            self.model_name = model_name
            # Rebuild the cached tools schema against the new model
            # (different providers can produce slightly different
            # schemas — e.g. Anthropic vs OpenAI).
            if getattr(self, "function_map", None):
                self._cached_tools_schema = new_model._build_openai_tools_schema(  # type: ignore[attr-defined]
                    self.function_map,
                )
            _save_last_model(model_name)
            return f"Model changed from {previous_name} to {model_name}."

        tools.append(ask_user_question)
        tools.append(update_settings)
        tools.append(set_model)
        if self._is_parallel:
            tools.append(run_parallel)
            tools.append(number_of_cores)
        return tools

    def perform_task(
        self,
        tools: list,
        attachments: list | None = None,
    ) -> str:
        """Execute the task, building docker-aware tools after docker_manager is set.

        Args:
            tools: Extra tools passed by the caller (from run(tools=...)).
            attachments: Optional file attachments for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        all_tools = self._get_tools() + tools
        # Wire up the pre-step hook so user prompts queued via the VS
        # Code frontend's ``appendUserMessage`` command while this task
        # is running get injected into the live model's conversation as
        # additional ``user`` messages immediately before the next model
        # call.  This MUST happen here (after ``RelentlessAgent.run`` has
        # already called ``_reset``, which clears ``pre_step_hook``) and
        # before ``super().perform_task`` runs the per-session executor
        # loop — that loop copies ``self.pre_step_hook`` onto each inner
        # executor.  Only meaningful when this agent has been bound to a
        # frontend tab (``_tab_id`` is set by
        # :meth:`_TaskRunnerMixin._run_task_inner`).
        if getattr(self, "_tab_id", None):
            self.pre_step_hook = self._drain_pending_user_messages
        else:
            self.pre_step_hook = None
        return super().perform_task(all_tools, attachments=attachments)

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        docker_image: str | None,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        resolved_model = model_name or _load_last_model() or get_default_model()
        super()._reset(
            model_name=resolved_model,
            max_sub_sessions=max_sub_sessions,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir or ".",
            docker_image=docker_image,
            printer=printer,
            verbose=verbose if verbose is not None else False,
        )

    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        web_tools: bool = True,
        is_parallel: bool = False,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: system prompt to be appended to the actual system prompt
            tools: List of tools to be added in addition to bash and web tools.
            max_steps: Maximum steps per sub-session. Defaults to config value.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            web_tools: Whether to include browser/web tools. Defaults to True.
                Set to False for terminal-only environments.
            is_parallel: Whether to include the run_parallel tool. Defaults to False.
                When True, the agent can spawn parallel sub-agents for independent tasks.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._is_parallel = is_parallel
        self.web_use_tool = None
        tl = getattr(printer, "_thread_local", None) if printer else None
        self._stop_event = getattr(tl, "stop_event", None) if tl else None
        # NOTE: the pending-user-messages pre-step hook is wired up in
        # :meth:`perform_task` (which runs *after* ``RelentlessAgent.run``
        # calls ``_reset``).  Installing it here would be useless because
        # ``RelentlessAgent._reset`` — invoked by ``super().run`` below —
        # resets ``self.pre_step_hook`` back to ``None`` before the
        # per-session executor loop reads it.
        try:
            system_instructions = (
                SYSTEM_PROMPT
                + (system_prompt if system_prompt else "")
            )
            prompt = prompt_template
            if attachments:
                parts = _attachment_parts(attachments)
                if parts:
                    prompt += (
                        f"\n\n# Important\n - User attached {', '.join(parts)}. "
                        f"The files are included in this message as inline content "
                        f"that you can see directly. "
                        f"Do NOT launch a browser, call screenshot(), go_to_url(), "
                        f"or any other browser tool to view these attachments — "
                        f"you already have them."
                    )
            if current_editor_file:
                system_instructions += (
                    "\n\n- The path of the file open in the editor is "
                    f"{current_editor_file}"
                )
            return super().run(
                model_name=model_name,
                system_prompt=system_instructions,
                prompt_template=prompt,
                arguments=arguments,
                max_steps=max_steps,
                max_budget=max_budget,
                model_config=model_config,
                work_dir=work_dir,
                printer=printer,
                max_sub_sessions=max_sub_sessions,
                docker_image=docker_image,
                verbose=verbose,
                tools=tools or [],
                attachments=attachments,
            )
        finally:
            if self.web_use_tool:
                self.web_use_tool.close()
            self.web_use_tool = None
            self._ask_user_question_callback = None
            self.pre_step_hook = None

    def _drain_pending_user_messages(self, model: Any) -> None:
        """Append any queued follow-up prompts to *model*'s conversation.

        Called once at the top of every model step (wired in via
        :attr:`kiss.core.kiss_agent.KISSAgent.pre_step_hook`).  Drains
        the owning :class:`_RunningAgentState`'s
        ``pending_user_messages`` list under
        :attr:`_RunningAgentState._registry_lock` (to keep the drain
        atomic against concurrent ``appendUserMessage`` commands from
        the frontend) and pushes each entry into *model*'s
        conversation as a ``user`` role message.  The list is emptied
        on every drain so the same queued message is never injected
        twice.

        Args:
            model: The live model whose conversation receives the
                queued user messages.
        """
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        tab_id = getattr(self, "_tab_id", "") or ""
        if not tab_id:
            return
        with _RunningAgentState._registry_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None or not tab.pending_user_messages:
                return
            queued = list(tab.pending_user_messages)
            tab.pending_user_messages.clear()
        for msg in queued:
            model.add_message_to_conversation("user", msg)


def _coerce_tasks(tasks: Any) -> list[str]:
    """Normalize the ``tasks`` argument to a ``list[str]``.

    LLM tool calls sometimes pass ``tasks`` in two malformed shapes that
    we recover from here:

    1. A JSON-encoded list string such as ``'["task A", "task B"]'``.
       Without recovery, the entire JSON string would be treated as one
       task and dispatched to a single sub-agent.  We parse it back into
       a proper ``list[str]``.
    2. A bare task string such as ``"hello"``.  Without this guard,
       ``enumerate(tasks)`` would iterate the string character-by-
       character and create one sub-agent (and one ``openSubagentTab``
       event) per character.  We wrap it into ``["hello"]``.

    Args:
        tasks: Either a ``list[str]``, a JSON-encoded ``list[str]`` string,
            or a single task ``str``.

    Returns:
        A ``list[str]``.  JSON-encoded list strings are parsed; other
        ``str`` inputs are wrapped in a one-element list.

    Raises:
        TypeError: If *tasks* is neither a ``str`` nor a ``list[str]``.
    """
    if isinstance(tasks, str):
        stripped = tasks.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                # A JSON empty list means zero tasks (NOT one task whose
                # text is "[]"); non-string elements (e.g. '[1, 2]') are
                # coerced to one task string per element.
                return [t if isinstance(t, str) else str(t) for t in parsed]
        return [tasks]
    if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
        return tasks
    raise TypeError(
        f"tasks must be list[str], got {type(tasks).__name__}: {tasks!r}"
    )


def run_tasks_parallel(
    tasks: list[str],
    max_workers: int | None = None,
    model_name: str | None = None,
    work_dir: str | None = None,
    printer: Printer | None = None,
    totals_out: dict[str, float] | None = None,
) -> list[str]:
    """Execute multiple SorcarAgent tasks concurrently using threads.

    Each task gets its own ``ChatSorcarAgent`` instance and runs in a
    separate thread via :class:`~concurrent.futures.ThreadPoolExecutor`.
    This is ideal for I/O-bound workloads (LLM API calls, network
    requests) where the GIL is released during I/O waits.

    This helper is a pure parallel executor: it has no knowledge of
    backend task ids or any frontend concepts.  It simply marks each
    spawned agent as a sub-agent (via ``_subagent_info``) and the
    sub-agent itself owns any sub-agent-specific behaviour (such as
    broadcasting ``new_tab`` to a browser-based frontend) inside its
    own ``run()`` method.

    Args:
        tasks: List of task description strings.  Each string is passed as
            the ``prompt_template`` argument to :meth:`SorcarAgent.run`.
            Example::

                [
                    "Summarize file A",
                    "Summarize file B",
                ]
        max_workers: Maximum number of threads.  ``None`` lets
            :class:`~concurrent.futures.ThreadPoolExecutor` pick a default
            (typically ``min(32, cpu_count + 4)``).
        model_name: LLM model name for all parallel agents.  ``None`` uses the
            default from persistence (same as :meth:`SorcarAgent.run`).
        work_dir: Working directory for all parallel agents.  ``None`` uses
            the default (``artifact_dir/kiss_workdir``).
        printer: Optional printer from the parent agent.  Forwarded
            verbatim to each sub-agent's ``run`` so live events
            continue to flow through the same channel.  The executor
            itself does not call any printer methods.

    Returns:
        List of YAML result strings in the **same order** as *tasks*.
        Each string contains ``success`` and ``summary`` keys.  If a task
        raises an unhandled exception the corresponding entry is a YAML
        string with ``success: false`` and the traceback in ``summary``.

    Raises:
        TypeError: If *tasks* is not a list of strings.  As a convenience
            for LLM tool callers that mistakenly pass a bare string,
            ``str`` is coerced to a one-element list.
    """
    tasks = _coerce_tasks(tasks)

    # Local import: ``chat_sorcar_agent`` imports from this module, so a
    # top-level import would be circular.
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    # Per-sub-agent usage so the caller can aggregate it back into the
    # parent agent's accounting.  Each entry is a tuple of
    # ``(budget_used, total_tokens_used, total_steps)``.
    sub_usage: list[tuple[float, int, int]] = [(0.0, 0, 0)] * len(tasks)

    # Capture the parent's thread-local ``task_id`` HERE, in the calling
    # thread.  ``printer._thread_local`` is a real ``threading.local``,
    # so reading it inside a worker thread would never see the parent
    # thread's value (and ``ChatSorcarAgent.run`` clears the worker's
    # own id before ``_run_single``'s ``finally`` runs) — the
    # ``subagentDone`` broadcast would then never fire.
    parent_tl = getattr(printer, "_thread_local", None) if printer else None
    parent_key = getattr(parent_tl, "task_id", "") if parent_tl else ""
    # Capture the parent's stop_event HERE too (same thread-local
    # reasoning as ``task_id`` above): ``SorcarAgent.run`` resolves
    # ``self._stop_event`` from the *worker* thread's
    # ``printer._thread_local``, so unless the event is copied into
    # each worker thread-local below, sub-agents never see the parent
    # stop request and Stop cannot kill their Bash process groups.
    parent_stop_event = getattr(parent_tl, "stop_event", None) if parent_tl else None

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.stop_event = parent_stop_event
        agent = ChatSorcarAgent(f"Parallel-{task[:40]}")
        # Mark the spawned agent as a sub-agent.  ``ChatSorcarAgent.run``
        # reads this marker to drive its own sub-agent-specific
        # behaviour (e.g. broadcasting ``new_tab`` to a browser-based
        # frontend, persisting the ``subagent`` extra field).  This
        # keeps the parallel executor itself free of any task-id or
        # frontend knowledge.  The base ``SorcarAgent`` path has no
        # parent ``task_id`` to record, so ``parent_task_id`` is
        # ``None`` here; the chat-aware override sets the real parent
        # id.
        agent._subagent_info = {"parent_task_id": None}
        try:
            # ``is_parallel=True`` propagates the parallel capability so
            # sub-agents themselves get the ``run_parallel`` tool and
            # can invoke nested parallel execution.  Without this, nested
            # parallel (sub-agent calls run_parallel) is impossible.
            result: str = agent.run(
                prompt_template=task,
                model_name=model_name,
                work_dir=work_dir,
                printer=printer,
                is_parallel=True,
            )
            return result
        except Exception as exc:
            return _yaml_failure(exc)
        finally:
            sub_usage[idx] = _agent_usage(agent)
            if printer is not None and parent_key:
                _broadcast_subagent_done(
                    printer, [f"{parent_key}__sub_{idx}"],
                )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_run_single, enumerate(tasks)))

    if totals_out is not None:
        totals_out["budget_used"] = sum(u[0] for u in sub_usage)
        totals_out["total_tokens_used"] = sum(u[1] for u in sub_usage)
        totals_out["total_steps"] = sum(u[2] for u in sub_usage)
    return results





