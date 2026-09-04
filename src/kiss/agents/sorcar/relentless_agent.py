# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

import getpass
import logging
import os
import platform
import socket
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import (
    BudgetExceededError,
    ContextWindowExceededError,
    KISSError,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import model_runs_task_to_completion
from kiss.core.printer import Printer
from kiss.core.utils import _coerce_bool as _str_to_bool
from kiss.core.utils import finish, substitute_prompt_args

logger = logging.getLogger(__name__)

TASK_PROMPT = """
{task_description}

{previous_progress}
"""

IMPORTANT_INSTRUCTIONS = """
# MOST IMPORTANT INSTRUCTIONS
- **If the task is not complete and you are at risk of running out of context \
length, you MUST call finish(success=False, is_continue=True, \
summary_in_html="precise chronologically-ordered list of things the agent did \
with the reason for doing that along with relevant code snippets, formatted \
as HTML (e.g. <ol>, <p>, <pre><code>), never Markdown")**
- The summary_in_html argument of finish MUST always be formatted as HTML.
- Work dir: {work_dir}
- Current process PID: {current_pid} — NEVER kill this process.
"""

TASK_SETTINGS_HEADER = "\n# Task Settings\n"

#: Budget cap (USD) a run falls back to when the caller states none.
DEFAULT_MAX_BUDGET = 200.0

CONTINUATION_PROMPT = """
# Task Progress (Continuation {continuation_number})

{progress_text}

# Continue
- Complete the rest of the task.
- **DON'T** redo completed work.
- If you have been retrying the same approach without progress, step back \
and rethink the strategy from scratch.
"""

SUMMARIZER_PROMPT = """
# Summarizer

The executor's trajectory is saved at: {trajectory_path}

Read relevant portions of the file using your tools:
- Read the first ~50 lines to understand the task and system instructions.
- Read the last ~200 lines to see the most recent steps and outcomes.
- Do NOT read the entire file; it may be very large.

# Instructions
- Analyze the trajectory file.
- Return a precise chronologically-ordered list of things the agent did
  with the reason for doing that along with relevant code snippets.
- Format the summary as HTML (e.g. <ol>, <p>, <pre><code>), never Markdown.
- Call finish(result="detailed summary of work done so far, in HTML").
"""

MAX_PROGRESS_CHARS = 60_000


def _local_ip_address() -> str:
    """Best-effort primary IP address of this machine.

    Opens a UDP socket "connected" to a public address, which selects
    the outbound interface without sending any packets (the numeric
    destination also avoids DNS), and reads the socket's own address.
    Returns ``"unknown"`` when the host has no route — deliberately
    with no hostname-resolution fallback, which could stall on a
    broken resolver or report loopback.

    Returns:
        The machine's primary IPv4 address, or ``"unknown"``.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "unknown"


def _nonempty(value: str) -> str:
    """*value* stripped, or ``"unknown"`` when nothing remains.

    ``platform.uname()`` reports fields it cannot determine as ``""``
    (per its documented contract); this keeps such fields readable.

    Args:
        value: A possibly empty host-identification field.

    Returns:
        The stripped value, or ``"unknown"`` if it is empty.
    """
    return value.strip() or "unknown"


def _host_settings() -> dict[str, str]:
    """User and host identification for the "# Task Settings" section.

    Returns:
        Label → value pairs for the unix user name, the machine's
        primary IP address, the OS name and release, and the machine's
        hostname and hardware architecture.
    """
    try:
        user = getpass.getuser()
    except OSError:
        user = "unknown"
    uname = platform.uname()
    return {
        "User id": _nonempty(user),
        "IP address": _local_ip_address(),
        "OS": f"{_nonempty(uname.system)} {_nonempty(uname.release)}",
        "Machine info": f"{_nonempty(uname.node)} ({_nonempty(uname.machine)})",
    }


def _capped_progress_text(summaries: list[str]) -> str:
    """Join attempt summaries newest-last, keeping the total within ``MAX_PROGRESS_CHARS``.

    The most recent summaries are the most relevant for continuing the
    task, so older ones are dropped first.  When any are dropped, a note
    stating how many were omitted is prepended.

    Args:
        summaries: All prior session summaries, oldest first.

    Returns:
        Markdown text of "### Attempt N" sections separated by
        ``\\n\\n---\\n\\n``, at most ``MAX_PROGRESS_CHARS`` characters of
        summary content, possibly preceded by an omission note.
    """
    separator = "\n\n---\n\n"
    budget = MAX_PROGRESS_CHARS - 200
    sections = [f"### Attempt {i + 1}\n{s}" for i, s in enumerate(summaries)]
    kept: list[str] = []
    total = 0
    for section in reversed(sections):
        if len(section) > budget:
            section = section[:budget] + "\n(...summary truncated.)"
        cost = len(section) + len(separator)
        if kept and total + cost > budget:
            break
        kept.append(section)
        total += cost
    kept.reverse()
    omitted = len(sections) - len(kept)
    if omitted > 0:
        kept.insert(0, f"({omitted} earlier attempt summaries omitted.)")
    return separator.join(kept)


def _prior_sessions_section(summaries: list[str]) -> str:
    """Join prior session summaries into "<h3>Previous Session N</h3>" HTML sections."""
    return "\n\n---\n\n".join(
        f"<h3>Previous Session {i + 1}</h3>\n{s}" for i, s in enumerate(summaries)
    )


def _build_exhaustion_summary(summaries: list[str], banner: str) -> str:
    """Compose the merged failure summary emitted on sub-session exhaustion.

    The exhaustion banner (``"Task failed after N sub-sessions"``) is
    appended AFTER a "<h3>Previous Session N</h3>" section when any prior
    session summaries exist. This layout matches the front-end
    (``splitMultiSessionSummary`` in ``main.js``): it splits on the
    trailing ``\\n\\n---\\n\\n`` separator so the banner renders as the
    terminal ``Result`` panel while the prior sessions become the
    ``Previous Sessions`` panel.

    Args:
        summaries: Prior session summaries (from ``is_continue=True``
            returns), in chronological order. May be empty when the
            very first session was already exhausted (single-session
            exhaustion → banner-only).
        banner: The short exhaustion message.

    Returns:
        The full summary string suitable for the ``summary`` field of a
        ``type="result"`` event.
    """
    if not summaries:
        return banner
    return f"{_prior_sessions_section(summaries)}\n\n---\n\n{banner}"


class RelentlessAgent(Base):
    """Base agent with auto-continuation for long tasks."""

    work_dir: str = ""

    def __init__(self, name: str) -> None:
        """Initialize the agent and its usage-counter lock.

        Args:
            name: The name identifier for the agent.
        """
        super().__init__(name)
        # Serializes every read-modify-write of the cumulative usage
        # counters (``budget_used``, ``total_tokens_used``,
        # ``total_steps``).  The writers run on different threads of
        # the SAME agent: the agent thread (:meth:`_accumulate_usage`
        # at session end, ``_attribute_sub_usage`` when a fan-out or a
        # ``talk`` synthesis banks its spend) and server threads
        # (``reclaim_abandoned_subagents`` from worktree cleanup /
        # teardown / discard).  Without one lock over all of them, two
        # concurrent read-modify-writes interleave and one side's
        # increment silently vanishes from the task's accounting.
        self._usage_lock: threading.Lock = threading.Lock()

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
        default_work_dir = str(Path(config_module.artifact_dir).resolve() / "kiss_workdir")

        self.work_dir = str(Path(work_dir or default_work_dir).resolve())
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

        self.max_sub_sessions = max_sub_sessions if max_sub_sessions is not None else 10000
        self.max_steps = max_steps if max_steps is not None else 10000
        self.max_budget = (
            max_budget if max_budget is not None else DEFAULT_MAX_BUDGET
        )
        self.model_name = model_name if model_name is not None else "claude-opus-4-6"
        self.verbose = verbose
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.total_steps: int = 0
        self._current_executor: KISSAgent | None = None
        self.docker_image = docker_image
        self.docker_manager: Any = None
        self.task_description: str = ""
        self.system_prompt: str = ""
        self.model_config: dict[str, Any] | None = None
        self.pre_step_hook: Callable[..., None] | None = None
        self.tool_call_guard: Callable[[str, dict[str, Any]], str | None] | None = None
        self.llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None
        self.tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None
        self.set_printer(printer, verbose=verbose)

    def _accumulate_usage(self, agent: Base) -> None:
        """Fold a sub-agent's budget, tokens and steps into the running totals.

        Held under ``_usage_lock``: a server-thread reclaim
        (``reclaim_abandoned_subagents``) can bank an abandoned child's
        spend into the same counters while a session ends on the agent
        thread, and an unserialized read-modify-write would lose one
        side's increment.
        """
        with self._usage_lock:
            self.budget_used += agent.budget_used
            self.total_tokens_used += agent.total_tokens_used
            self.total_steps += agent.step_count

    def _check_total_budget(self) -> None:
        """Raise :class:`KISSError` when the task's cumulative spend exceeds max_budget.

        Installed as :attr:`KISSAgent.budget_check_hook` on every
        per-session executor, so the executor's ``_check_limits`` also
        enforces the PARENT task's total budget.  ``self.budget_used``
        holds the spend of prior sub-sessions plus any spend attributed
        mid-session by parallel sub-agents (``_attribute_sub_usage``);
        the live executor's own spend is added on top because it is only
        folded into ``self.budget_used`` when its session ends.

        Raises:
            KISSError: If the cumulative spend exceeds ``self.max_budget``.
        """
        executor = self._current_executor
        live = executor.budget_used if executor is not None else 0.0
        total = self.budget_used + live
        if total >= self.max_budget:
            raise BudgetExceededError(
                f"Agent {self.name} budget exceeded "
                f"(${total:.4f} / ${self.max_budget:.2f})."
            )

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return str(self.docker_manager.Bash(command, description))

    def _system_prompt_task_settings(self) -> dict[str, str]:
        """Label → value pairs appended to the system prompt as "# Task Settings".

        Called once per task by :meth:`perform_task`, after ``_reset``
        resolved the run's model and budget, so the values describe the
        settings the task actually runs with, plus the host environment
        (unix user, IP address, OS, machine).  Subclasses extend the
        dict with the settings they know about (parallel mode, worktree
        mode, chat / task / parent ids, ...).

        Returns:
            Ordered mapping of setting labels to display values.
        """
        return {
            "Model name": self.model_name,
            "Max budget (USD)": f"${self.max_budget:.2f}",
            "Starting time": datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            ),
            **_host_settings(),
        }

    def _task_settings_section(self) -> str:
        """The "# Task Settings" system-prompt section for this run.

        Every value is collapsed to a single whitespace-normalized
        line, so host-derived strings (user name, hostname, ...)
        containing newlines cannot inject extra lines or headings into
        the system prompt.

        Returns:
            The formatted section, or ``""`` when
            :meth:`_system_prompt_task_settings` yields nothing.
        """
        settings = self._system_prompt_task_settings()
        if not settings:  # pragma: no cover — base hook never empty
            return ""
        lines = "".join(
            f"- {label}: {' '.join(str(value).split())}\n"
            for label, value in settings.items()
        )
        return TASK_SETTINGS_HEADER + lines

    def _executor_model_config(self) -> dict[str, Any]:
        """Return the model config for a sub-agent, carrying the work dir.

        A copy of :attr:`model_config` with ``work_dir`` defaulted to this
        agent's work directory.  ``work_dir`` is a framework-only config
        key: CLI-backed run-to-completion models (``cc/*``, ``codex/*``)
        launch their subprocess with it as the cwd — otherwise the CLI's
        native tools would act on the daemon's cwd instead of the task's
        (possibly worktree-redirected) work tree — and API adapters ignore
        it.  Sorcar's ``set_model`` copies the live model's config on a
        switch, so the work dir survives mid-run model changes.

        Returns:
            dict: The per-executor model config.
        """
        config: dict[str, Any] = dict(self.model_config or {})
        config.setdefault("work_dir", self.work_dir)
        return config

    def perform_task(
        self,
        tools: list[Callable[..., Any]],
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Execute the task with auto-continuation across multiple sub-sessions.

        Args:
            tools: List of callable tools available to the agent during execution.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys on successful completion.

        Raises:
            KISSError: If the task fails after exhausting all sub-sessions.
        """
        logger.info(
            "Executing task: agent=%s model=%s max_steps=%d "
            "max_budget=$%.2f pid=%d task=%r",
            self.name,
            self.model_name,
            self.max_steps,
            self.max_budget,
            os.getpid(),
            self.task_description[:200],
        )
        all_tools: list[Callable[..., Any]] = [finish, *tools]

        progress_section = ""
        summaries: list[str] = []
        current_pid = str(os.getpid())
        important_instructions = IMPORTANT_INSTRUCTIONS.format(
            work_dir=self.work_dir,
            current_pid=current_pid,
        )
        important_instructions += self._task_settings_section()
        sorcar_md = config_module.kiss_home() / "SORCAR.md"
        if sorcar_md.is_file():
            # User-authored: a cp1252 byte from a Windows editor must
            # not abort every task before its first model call (the
            # same tolerance ``skills.parse_frontmatter`` gives SKILL.md).
            important_instructions += "\n" + sorcar_md.read_text(
                encoding="utf-8", errors="replace",
            )
        system_prompt = self.system_prompt + important_instructions
        for session in range(self.max_sub_sessions):
            remaining_budget = self.max_budget - self.budget_used
            if remaining_budget <= 0:
                raise BudgetExceededError(
                    f"Agent {self.name} budget exhausted "
                    f"(${self.budget_used:.4f} / ${self.max_budget:.2f})."
                )
            if self.printer:
                self.printer.tokens_offset = self.total_tokens_used  # type: ignore[attr-defined]
                self.printer.budget_offset = self.budget_used  # type: ignore[attr-defined]
                self.printer.steps_offset = self.total_steps  # type: ignore[attr-defined]
            logger.info(
                "Session %d start: agent=%s budget_remaining=$%.4f "
                "total_tokens=%d total_steps=%d",
                session,
                self.name,
                remaining_budget,
                self.total_tokens_used,
                self.total_steps,
            )
            executor = KISSAgent(f"{self.name} Session-{session}")
            executor.pre_step_hook = getattr(self, "pre_step_hook", None)
            executor.tool_call_guard = getattr(self, "tool_call_guard", None)
            llm_call_hook = getattr(self, "llm_call_hook", None)
            tool_call_hook = getattr(self, "tool_call_hook", None)
            executor.budget_check_hook = self._check_total_budget
            self._current_executor = executor
            try:
                result = executor.run(
                    model_name=self.model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "previous_progress": progress_section,
                    },
                    system_prompt=system_prompt,
                    tools=all_tools,
                    max_steps=self.max_steps,
                    max_budget=remaining_budget,
                    model_config=self._executor_model_config(),
                    printer=self.printer,
                    verbose=self.verbose,
                    attachments=attachments if session == 0 else None,
                    llm_call_hook=llm_call_hook,
                    tool_call_hook=tool_call_hook,
                )
            except BudgetExceededError:
                self._current_executor = None
                self._accumulate_usage(executor)
                raise
            except Exception as exc:
                logger.debug("Exception caught", exc_info=True)
                is_context_overflow = isinstance(exc, ContextWindowExceededError)
                if (
                    (
                        not is_context_overflow
                        and (exc.__cause__ is not None or not isinstance(exc, KISSError))
                    )
                    or executor.step_count <= 1
                ):
                    self._current_executor = None
                    self._accumulate_usage(executor)
                    error_result = finish(False, False, f"{type(exc).__name__}: {exc}")
                    if self.printer:
                        self.printer.print(
                            error_result,
                            type="result",
                            step_count=executor.step_count,
                            total_tokens=executor.total_tokens_used,
                            cost=f"${executor.budget_used:.4f}",
                        )
                    return error_result
                if not getattr(self, "_append_basic_tools", True):
                    # Restricted runs (``append_basic_tools=False``)
                    # promise that NO LLM session of the task gets
                    # tools beyond ``finish`` and the caller's own —
                    # the trajectory summarizer's Read/Bash included —
                    # so skip the summarizer and continue with the
                    # plain failure text.
                    result = finish(False, True, f"Agent failed: {exc}")
                else:
                    result = finish(
                        False,
                        True,
                        self._summarize_failed_session(executor, session, exc),
                    )

            self._current_executor = None
            self._accumulate_usage(executor)

            try:
                payload = yaml.safe_load(result)
            except Exception:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)
                payload = {}
            if not isinstance(payload, dict):  # pragma: no cover
                payload = {}

            success = _str_to_bool(payload.get("success", False))
            is_continue = _str_to_bool(payload.get("is_continue", False))

            if not is_continue or success:
                if summaries:
                    final_summary = payload.get("summary", "")
                    prior_section = _prior_sessions_section(summaries)
                    if final_summary:
                        payload["summary"] = (
                            f"{prior_section}\n\n---\n\n<h3>Final Session</h3>\n"
                            f"{final_summary}"
                        )
                    else:
                        payload["summary"] = (
                            f"{prior_section}\n\n---\n\n<h3>Final Session</h3>\n"
                            "(no summary)"
                        )
                    result = yaml.dump(payload, sort_keys=False)
                    self._emit_merged_result_event(payload)
                return result

            summary = payload.get("summary", "")
            if summary:  # pragma: no branch
                summaries.append(summary)

                progress_section = CONTINUATION_PROMPT.format(
                    progress_text=_capped_progress_text(summaries),
                    continuation_number=session + 1,
                )
        banner = f"Task failed after {self.max_sub_sessions} sub-sessions"
        self._emit_merged_result_event(
            {
                "success": False,
                "is_continue": False,
                "summary": _build_exhaustion_summary(summaries, banner),
            }
        )
        err = KISSError(banner)
        err.terminal_result_broadcast = True  # type: ignore[attr-defined]
        raise err

    def _summarize_failed_session(
        self,
        executor: KISSAgent,
        session: int,
        exc: Exception,
    ) -> str:
        """Summarize a failed sub-session's trajectory with a helper LLM.

        Dumps *executor*'s trajectory to a temp file and asks a
        Read/Bash-equipped summarizer :class:`KISSAgent` to condense it
        into the progress text the next sub-session continues from.
        The summarizer's spend is folded into this agent's totals.
        Never called for restricted runs (``append_basic_tools=False``)
        — they must not hand ANY of the task's LLM sessions tools
        beyond ``finish`` and the caller's own, so
        :meth:`perform_task` uses the plain failure text instead.

        Args:
            executor: The failed sub-session's executor agent.
            session: Index of the failed sub-session.
            exc: The exception that ended the sub-session.

        Returns:
            The summary text, or the plain ``"Agent failed: ..."``
            fallback when summarization itself fails.
        """
        trajectory_path: Path | None = None
        try:
            tmp_dir = Path(self.work_dir) / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = tmp_dir / f"trajectory_{session}.json"
            trajectory_path.write_text(executor.get_trajectory(), encoding="utf-8")
            # The stop event lives on the printer's THREAD-LOCAL
            # (``_PrinterThreadLocal.stop_event``), not on the
            # printer: reading it off the printer always yields
            # None, which leaves the summarizer's shell command
            # unkillable by Stop.
            _tl = getattr(self.printer, "_thread_local", None) if self.printer else None
            _stop_ev = getattr(_tl, "stop_event", None) if _tl else None
            from kiss.agents.sorcar.useful_tools import UsefulTools

            shell_tools = UsefulTools(stop_event=_stop_ev)
            summarizer_budget = max(
                0.01, self.max_budget - self.budget_used - executor.budget_used
            )
            summarizer_agent = KISSAgent(f"{self.name} Summarizer")
            try:
                summarizer_result = summarizer_agent.run(
                    model_name=self.model_name,
                    prompt_template=SUMMARIZER_PROMPT,
                    tools=[shell_tools.Read, shell_tools.Bash],
                    arguments={
                        "trajectory_path": str(trajectory_path),
                    },
                    max_steps=self.max_steps,
                    max_budget=summarizer_budget,
                    model_config=self._executor_model_config(),
                    printer=self.printer,
                    verbose=self.verbose,
                    print_prompts=False,
                )
            finally:
                self._accumulate_usage(summarizer_agent)
            try:
                parsed = yaml.safe_load(summarizer_result)
                summary_text = (
                    parsed.get("result", summarizer_result)
                    if isinstance(parsed, dict)
                    else summarizer_result
                )
            except Exception:  # pragma: no cover
                logger.debug("Exception caught", exc_info=True)
                summary_text = summarizer_result
        except Exception:  # pragma: no cover – requires summarizer LLM failure
            logger.debug("Exception caught", exc_info=True)
            summary_text = f"Agent failed: {exc}"
        finally:
            if trajectory_path and trajectory_path.exists():  # pragma: no branch
                trajectory_path.unlink()
        return str(summary_text)

    def _emit_merged_result_event(self, payload: dict[str, Any]) -> None:
        """Emit a ``type="result"`` event with merged multi-session totals.

        Complements — never replaces — the per-session Result events emitted
        by the inner :class:`KISSAgent`.  Called from :meth:`perform_task`
        only when the terminal outcome depends on information the inner
        emit could not carry:

        * prior session summaries must be preserved (multi-session merge), or
        * all sub-sessions were exhausted (no inner session ever returned a
          terminal ``is_continue=False``).

        For single-session terminations, the inner Result event is already
        authoritative and this helper is not called.

        Args:
            payload: Dict with ``success``, ``is_continue`` and ``summary``
                keys.  Serialized to YAML as the event ``content``.
        """
        if self.printer is None:
            return
        offset_attrs = (
            ("tokens_offset", 0),
            ("budget_offset", 0.0),
            ("steps_offset", 0),
        )
        saved: dict[str, Any] = {}
        for attr, zero in offset_attrs:
            if hasattr(self.printer, attr):
                saved[attr] = getattr(self.printer, attr)
                try:
                    setattr(self.printer, attr, zero)
                except AttributeError:  # pragma: no cover
                    saved.pop(attr, None)
        try:
            self.printer.print(
                yaml.dump(payload, sort_keys=False),
                type="result",
                step_count=self.total_steps,
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
            )
        finally:
            for attr, value in saved.items():
                try:
                    setattr(self.printer, attr, value)
                except AttributeError:  # pragma: no cover
                    pass

    def run(
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        system_prompt: str = "",
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        work_dir: str | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        verbose: bool | None = None,
        tools: list[Callable[..., Any]] | None = None,
        attachments: list[Attachment] | None = None,
        llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> str:
        """Run the agent with the provided tools.

        Args:
            model_name: LLM model to use. Defaults to "claude-opus-4-6".
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: System-level instructions passed to the underlying LLM
                via model_config. Defaults to empty string (no system instructions).
            max_steps: Maximum steps per sub-session. Defaults to 10000.
            max_budget: Maximum budget in USD. Defaults to 200.0.
            model_config: Optional dictionary of additional model configuration
                parameters (e.g. temperature, top_p). Defaults to None.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to 10000.
            docker_image: Docker image name to run tools inside a container.
            verbose: Whether to print output to console. Defaults to True.
            tools: List of callable tools available to the agent during execution.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            llm_call_hook: Optional hook installed on every per-session
                executor :class:`KISSAgent` (see
                :meth:`kiss.core.kiss_agent.KISSAgent.run`): called before
                every LLM call with the new messages about to be sent, and
                its return value replaces them.  Defaults to None (no hook).
            tool_call_hook: Optional hook installed on every per-session
                executor :class:`KISSAgent` (see
                :meth:`kiss.core.kiss_agent.KISSAgent.run`): called before
                every tool call with the tool's name and arguments; any
                verdict other than ``"OK"`` suppresses the call and is
                returned to the model as the tool's result.  Defaults to
                None (no hook).

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        self._reset(
            model_name,
            max_sub_sessions,
            max_steps,
            max_budget,
            work_dir,
            docker_image,
            printer,
            verbose,
        )
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.llm_call_hook = llm_call_hook
        self.tool_call_hook = tool_call_hook
        args = arguments or {}
        self.task_description = substitute_prompt_args(prompt_template, args)

        if self.docker_image and model_runs_task_to_completion(self.model_name):
            # A run-to-completion CLI agent executes its native tools
            # directly on the host, so the container the caller asked for
            # would be silently bypassed — refuse rather than break the
            # isolation contract.
            raise KISSError(
                f"Model {self.model_name} is a CLI agent that runs natively on "
                f"the host and cannot honor docker_image="
                f"{self.docker_image!r} isolation. Use an API model with "
                f"docker_image, or drop docker_image for CLI models."
            )

        if self.docker_image:
            from kiss.agents.sorcar.docker_manager import DockerManager

            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                if self.printer:
                    _printer = self.printer

                    def _docker_stream(text: str) -> None:
                        _printer.print(text, type="bash_stream")

                    docker_mgr.stream_callback = _docker_stream
                try:
                    return self.perform_task(tools or [], attachments=attachments)
                finally:
                    self.docker_manager = None
        return self.perform_task(tools or [], attachments=attachments)
