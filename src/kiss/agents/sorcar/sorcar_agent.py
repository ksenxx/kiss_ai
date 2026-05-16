"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_default_model
from kiss.core.printer import Printer
from kiss.core.relentless_agent import RelentlessAgent


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


class SorcarAgent(RelentlessAgent):
    """Agent with both coding tools and browser automation for web + code tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        self.docker_manager: Any = None
        self._use_web_tools: bool = True
        self._is_parallel: bool = False

    def _run_tasks_parallel(
        self,
        tasks: list[str],
        max_workers: int | None = None,
    ) -> list[str]:
        """Execute multiple independent tasks concurrently using parallel agents.

        Each task gets its own ``SorcarAgent`` instance.  Subclasses can
        override this method to change the agent type or pass extra context
        (e.g. ``ChatSorcarAgent`` propagates ``chat_id``).

        When the parent agent has a browser-based printer, each sub-agent
        gets its own tab in the UI so users can monitor progress.  The
        ``openSubagentTab`` event creates a read-only tab and each
        sub-agent's events are routed there via its thread-local
        ``tab_id``.  A ``subagentDone`` event marks the tab complete.

        Args:
            tasks: List of self-contained task description strings.
            max_workers: Maximum concurrent threads (``None`` = auto).

        Returns:
            List of YAML result strings in the same order as *tasks*.
        """
        return run_tasks_parallel(
            tasks,
            max_workers=max_workers,
            model=getattr(self, "model_name", None),
            work_dir=getattr(self, "work_dir", None),
            printer=self.printer,
        )

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

        stop_event = getattr(self, "_stop_event", None)
        useful_tools = UsefulTools(stream_callback=_stream, stop_event=stop_event)
        if self.docker_manager:
            from kiss.docker.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)
            tools: list = [
                self._docker_bash, docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            tools = [useful_tools.Bash, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
        if self._use_web_tools and self.web_use_tool is None:
            self.web_use_tool = WebUseTool()
            tools.extend(self.web_use_tool.get_tools())
        def run_parallel(tasks: list[str], max_workers: int | None = None) -> str:
            """Run multiple independent tasks concurrently using parallel agents.

            Spawns a separate SorcarAgent for each task string and executes
            them in parallel threads.  Use this tool when you have two or
            more **independent** sub-tasks that do not depend on each
            other's results (e.g. summarising several files, researching
            separate topics, running independent code changes, searching
            multiple solutions).

            Each parallel agent inherits the current model and working
            directory.  Results are returned in the same order as the
            input tasks.

            **When NOT to use**: Do not use this for tasks that must run
            sequentially or that depend on each other's output.

            Args:
                tasks: List of task description strings.  Each string is a
                    complete, self-contained instruction that a fresh
                    SorcarAgent can execute on its own.  Example::

                        [
                            "Read src/foo.py and summarize its purpose",
                            "Read src/bar.py and summarize its purpose",
                            "Find the current weather in San Francisco",
                        ]
                max_workers: Maximum number of concurrent threads.
                    ``None`` (default) lets Python choose automatically.
                    Set to a lower number to limit concurrency.

            Returns:
                A YAML-formatted string containing a list of result
                objects, one per task, in the same order as the input.
                Each result object has ``success`` and ``summary`` keys.
            """
            results = self._run_tasks_parallel(tasks, max_workers=max_workers)
            result_str: str = yaml.dump(results, sort_keys=False)
            return result_str

        def update_settings(
            is_parallel: bool | None = None,
            is_worktree: bool | None = None,
            model: str | None = None,
            max_budget: float | None = None,
            working_directory: str | None = None,
            use_web_browser: bool | None = None,
            remote_password: str | None = None,
            demo_mode: bool | None = None,
            auto_commit: bool | None = None,
            custom_endpoint: str | None = None,
            custom_api_key: str | None = None,
            custom_headers: str | None = None,
            gemini_api_key: str | None = None,
            openai_api_key: str | None = None,
            anthropic_api_key: str | None = None,
            together_api_key: str | None = None,
            openrouter_api_key: str | None = None,
            minimax_api_key: str | None = None,
        ) -> str:
            """Update task configuration settings during execution.

            Modifies runtime agent settings, persists config-level changes,
            and broadcasts UI updates to the frontend. Only provided
            (non-None) keys are updated; omitted keys are left unchanged.

            Args:
                is_parallel: Enable/disable parallel sub-agent spawning.
                is_worktree: Enable/disable git worktree isolation.
                model: Switch the LLM model for subsequent sub-sessions.
                max_budget: Set the maximum budget in USD.
                working_directory: Change the agent working directory.
                use_web_browser: Enable/disable browser/web tools.
                remote_password: Set the remote access password.
                demo_mode: Enable/disable demo replay mode in the UI.
                auto_commit: When True, trigger auto-commit of pending changes.
                custom_endpoint: Set a custom LLM endpoint URL (e.g. local model).
                custom_api_key: Set the API key for the custom endpoint.
                custom_headers: Set custom HTTP headers (Key:Value, one per line).
                gemini_api_key: Set the Gemini API key.
                openai_api_key: Set the OpenAI API key.
                anthropic_api_key: Set the Anthropic API key.
                together_api_key: Set the Together API key.
                openrouter_api_key: Set the OpenRouter API key.
                minimax_api_key: Set the MiniMax API key.

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
                updated.append(f"is_parallel={self._is_parallel}")
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "is_parallel",
                        "value": self._is_parallel,
                    })

            if is_worktree is not None:
                updated.append(f"is_worktree={bool(is_worktree)}")
                _save_setting_to_config("is_worktree", bool(is_worktree))
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "is_worktree",
                        "value": bool(is_worktree),
                    })

            if model is not None:
                self.model_name = model
                updated.append(f"model={model}")
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "model",
                        "value": model,
                    })

            if max_budget is not None:
                self.max_budget = float(max_budget)
                updated.append(f"max_budget={self.max_budget}")
                _save_setting_to_config("max_budget", self.max_budget)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "max_budget",
                        "value": self.max_budget,
                    })

            if working_directory is not None:
                resolved = str(Path(working_directory).resolve())
                Path(resolved).mkdir(parents=True, exist_ok=True)
                self.work_dir = resolved
                updated.append(f"working_directory={resolved}")
                _save_setting_to_config("work_dir", resolved)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "working_directory",
                        "value": resolved,
                    })

            if use_web_browser is not None:
                self._use_web_tools = bool(use_web_browser)
                updated.append(f"use_web_browser={self._use_web_tools}")
                _save_setting_to_config(
                    "use_web_browser", self._use_web_tools,
                )
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "use_web_browser",
                        "value": self._use_web_tools,
                    })

            if remote_password is not None:
                updated.append("remote_password=<updated>")
                _save_setting_to_config("remote_password", remote_password)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "remote_password",
                        "value": True,
                    })

            if demo_mode is not None:
                updated.append(f"demo_mode={bool(demo_mode)}")
                _save_setting_to_config("demo_mode", bool(demo_mode))
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "demo_mode",
                        "value": bool(demo_mode),
                    })

            if auto_commit is not None and bool(auto_commit):
                updated.append("auto_commit=triggered")
                try:
                    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

                    wd = Path(self.work_dir).resolve()
                    repo = GitWorktreeOps.discover_repo(wd)
                    if repo:
                        commit_dir = wd if wd != repo else repo
                        GitWorktreeOps.stage_all(commit_dir)
                        try:
                            from kiss.agents.vscode.helpers import (
                                generate_commit_message_from_diff,
                            )

                            diff = GitWorktreeOps.staged_diff(commit_dir)
                            msg = generate_commit_message_from_diff(diff)
                        except Exception:
                            msg = "kiss: auto-commit agent changes"
                        GitWorktreeOps.commit_staged(commit_dir, msg)
                except Exception:
                    pass
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "auto_commit",
                        "value": True,
                    })

            if custom_endpoint is not None:
                updated.append(f"custom_endpoint={custom_endpoint}")
                _save_setting_to_config("custom_endpoint", custom_endpoint)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "custom_endpoint",
                        "value": custom_endpoint,
                    })

            if custom_api_key is not None:
                updated.append("custom_api_key=<updated>")
                _save_setting_to_config("custom_api_key", custom_api_key)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "custom_api_key",
                        "value": True,
                    })

            if custom_headers is not None:
                updated.append("custom_headers=<updated>")
                _save_setting_to_config("custom_headers", custom_headers)
                if broadcast:
                    broadcast({
                        "type": "updateSetting",
                        "key": "custom_headers",
                        "value": True,
                    })

            api_key_params = {
                "gemini_api_key": ("GEMINI_API_KEY", gemini_api_key),
                "openai_api_key": ("OPENAI_API_KEY", openai_api_key),
                "anthropic_api_key": ("ANTHROPIC_API_KEY", anthropic_api_key),
                "together_api_key": ("TOGETHER_API_KEY", together_api_key),
                "openrouter_api_key": ("OPENROUTER_API_KEY", openrouter_api_key),
                "minimax_api_key": ("MINIMAX_API_KEY", minimax_api_key),
            }
            for param_name, (env_var, key_value) in api_key_params.items():
                if key_value is not None:
                    try:
                        from kiss.agents.vscode.vscode_config import (
                            save_api_key_to_shell,
                        )
                        save_api_key_to_shell(env_var, key_value)
                    except Exception:
                        import os
                        os.environ[env_var] = key_value
                    updated.append(f"{param_name}=<updated>")
                    if broadcast:
                        broadcast({
                            "type": "updateSetting",
                            "key": param_name,
                            "value": True,
                        })

            if not updated:
                return "No settings were changed (all arguments were None)."
            return "Updated: " + ", ".join(updated)

        tools.append(ask_user_question)
        tools.append(update_settings)
        if self._is_parallel:
            tools.append(run_parallel)
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

        try:
            system_instructions = (
                SYSTEM_PROMPT
                + (system_prompt if system_prompt else "")
            )
            prompt = prompt_template
            if attachments:
                pdf_count = sum(1 for a in attachments if a.mime_type == "application/pdf")
                img_count = sum(1 for a in attachments if a.mime_type.startswith("image/"))
                audio_count = sum(1 for a in attachments if a.mime_type.startswith("audio/"))
                video_count = sum(1 for a in attachments if a.mime_type.startswith("video/"))
                parts = []
                if img_count:
                    parts.append(f"{img_count} image(s)")
                if pdf_count:
                    parts.append(f"{pdf_count} PDF(s)")
                if audio_count:
                    parts.append(f"{audio_count} audio file(s)")
                if video_count:
                    parts.append(f"{video_count} video file(s)")
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
            if (
                isinstance(parsed, list)
                and parsed
                and all(isinstance(t, str) for t in parsed)
            ):
                return parsed
        return [tasks]
    if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
        return tasks
    raise TypeError(
        f"tasks must be list[str], got {type(tasks).__name__}: {tasks!r}"
    )


def run_tasks_parallel(
    tasks: list[str],
    max_workers: int | None = None,
    model: str | None = None,
    work_dir: str | None = None,
    printer: Printer | None = None,
) -> list[str]:
    """Execute multiple SorcarAgent tasks concurrently using threads.

    Each task gets its own ``SorcarAgent`` instance and runs in a separate
    thread via :class:`~concurrent.futures.ThreadPoolExecutor`.  This is
    ideal for I/O-bound workloads (LLM API calls, network requests) where
    the GIL is released during I/O waits.

    When *printer* is a browser-based printer (has ``broadcast`` and
    ``_thread_local``), each sub-agent gets a dedicated UI tab.  An
    ``openSubagentTab`` event creates the read-only tab, the sub-agent's
    streaming events are routed there via its thread-local ``tab_id``,
    and a ``subagentDone`` event marks completion.

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
        model: LLM model name for all parallel agents.  ``None`` uses the
            default from persistence (same as :meth:`SorcarAgent.run`).
        work_dir: Working directory for all parallel agents.  ``None`` uses
            the default (``artifact_dir/kiss_workdir``).
        printer: Optional printer from the parent agent.  When a
            browser-based printer is supplied, sub-agent tabs are created
            in the UI for live monitoring.

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

    broadcast = getattr(printer, "broadcast", None) if printer else None
    thread_local = getattr(printer, "_thread_local", None) if printer else None
    parent_tab_id = getattr(thread_local, "tab_id", None) if thread_local else None

    sub_tab_ids: list[str] = []
    for i, task in enumerate(tasks):
        if parent_tab_id:
            sub_tab_id = f"{parent_tab_id}__sub_{i}"
        else:
            sub_tab_id = f"sub-{uuid.uuid4().hex[:12]}"
        sub_tab_ids.append(sub_tab_id)
        if broadcast:
            broadcast({
                "type": "openSubagentTab",
                "tab_id": sub_tab_id,
                "parent_tab_id": parent_tab_id,
                "description": task[:200],
                "taskIndex": i,
                "isSubagentTab": True,
            })

    def _run_single(args: tuple[int, str]) -> str:
        idx, task = args
        sub_tab_id = sub_tab_ids[idx]
        tl = getattr(printer, "_thread_local", None) if printer else None
        if tl is not None:
            tl.tab_id = sub_tab_id
        agent = SorcarAgent(f"Parallel-{task[:40]}")
        success = True
        try:
            result: str = agent.run(
                prompt_template=task,
                model_name=model,
                work_dir=work_dir,
                printer=printer,
            )
            return result
        except Exception as exc:
            success = False
            error_result: str = yaml.dump(
                {"success": False, "summary": f"Unhandled exception: {exc}"},
                sort_keys=False,
            )
            return error_result
        finally:
            if tl is not None:
                tl.tab_id = None
            if broadcast:
                broadcast({
                    "type": "subagentDone",
                    "tab_id": sub_tab_id,
                    "success": success,
                })

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_run_single, enumerate(tasks)))
    return results


_DEFAULT_TASK = """
can you find what the current weather is in San Francisco and summarize it?
"""


def _resolve_task(args: argparse.Namespace) -> str:
    """Determine the task description from parsed arguments.

    Priority: -f file > --task string > default task.

    Args:
        args: Parsed argparse namespace with 'f' and 'task' attributes.

    Returns:
        The task description string.

    Raises:
        FileNotFoundError: If -f path does not exist.
    """
    if args.file is not None:
        return Path(args.file).read_text()
    if args.task is not None:
        task: str = args.task
        return task
    return _DEFAULT_TASK


def cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


