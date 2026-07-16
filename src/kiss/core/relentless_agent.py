# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
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
- **At step {step_threshold}: you MUST call finish(success=False, is_continue=True, \
summary="precise chronologically-ordered list of things the agent did \
with the reason for doing that along with relevant code snippets")** or \
if the task is not complete and you are at risk of running out of steps or context length.
- Work dir: {work_dir}
- Current process PID: {current_pid} — NEVER kill this process.
"""

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
- Call finish(result="detailed summary of work done so far").
"""

#: Maximum size (in characters, ~15K tokens) of the accumulated
#: prior-attempt summaries embedded in ``CONTINUATION_PROMPT``.  Without a
#: cap, every continuation session starts with ALL previous summaries and
#: the prompt grows unboundedly — each successive session begins with less
#: context headroom, making repeated context exhaustion progressively
#: worse.
MAX_PROGRESS_CHARS = 60_000


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
    # Reserve room for the truncation note, the omission note, and the
    # separators around them so the RETURNED text never exceeds
    # ``MAX_PROGRESS_CHARS`` (a true hard cap).
    budget = MAX_PROGRESS_CHARS - 200
    sections = [f"### Attempt {i + 1}\n{s}" for i, s in enumerate(summaries)]
    kept: list[str] = []
    total = 0
    for section in reversed(sections):
        if len(section) > budget:
            # Even the newest summary alone must not blow the
            # continuation prompt (the whole point of the cap is that a
            # fresh session starts with context headroom).
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
    """Join prior session summaries into "### Previous Session N" markdown sections."""
    return "\n\n---\n\n".join(
        f"### Previous Session {i + 1}\n{s}" for i, s in enumerate(summaries)
    )


def _build_exhaustion_summary(summaries: list[str], banner: str) -> str:
    """Compose the merged failure summary emitted on sub-session exhaustion.

    The exhaustion banner (``"Task failed after N sub-sessions"``) is
    appended AFTER a "### Previous Session N" section when any prior
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

    # ``_reset`` assigns the resolved absolute path on every ``run()``;
    # this default makes ``self.work_dir`` safe to read on a freshly
    # constructed agent (mirroring ``Base.model_name``).
    work_dir: str = ""

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
        self.max_steps = max_steps if max_steps is not None else 100
        self.max_budget = max_budget if max_budget is not None else 200.0
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
        # See :attr:`kiss.core.kiss_agent.KISSAgent.pre_step_hook`.
        # Propagated to every inner per-session executor created in
        # :meth:`perform_task`, so subclasses (e.g. ``SorcarAgent``)
        # can hook into model calls without re-implementing the
        # per-session executor.
        self.pre_step_hook: Callable[..., None] | None = None
        self.set_printer(printer, verbose=verbose)

    def _accumulate_usage(self, agent: Base) -> None:
        """Fold a sub-agent's budget, tokens and steps into the running totals."""
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
        summary = ""
        summaries: list[str] = []
        current_pid = str(os.getpid())
        # Report ``self.work_dir`` verbatim — in worktree mode this is the
        # real on-disk ``<repo>/.kiss-worktrees/<slug>/...`` path, matching
        # what ``pwd`` / ``os.getcwd()`` return inside the agent's tools.
        important_instructions = IMPORTANT_INSTRUCTIONS.format(
            step_threshold=str(self.max_steps - 2),
            work_dir=self.work_dir,
            current_pid=current_pid,
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
            # Propagate any pre-step hook installed on the relentless
            # parent (e.g. ``SorcarAgent.run``'s pending-user-messages
            # drain) to the per-session inner executor — that's the
            # layer that actually calls the model.
            executor.pre_step_hook = getattr(self, "pre_step_hook", None)
            # Enforce the parent task's TOTAL budget from inside the
            # executor's step loop — the executor's own ``budget_used``
            # never sees spend attributed to the parent mid-session by
            # parallel sub-agents (``_attribute_sub_usage``), so without
            # this hook the session would keep running long after the
            # task's budget was exhausted.
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
                    model_config=self.model_config,
                    printer=self.printer,
                    verbose=self.verbose,
                    attachments=attachments if session == 0 else None,
                )
            except BudgetExceededError:
                # A budget limit is a hard stop.  Never launch the LLM
                # trajectory summarizer here: doing so would make more
                # paid model calls after the configured limit.  Account
                # for the live executor exactly once, clear the pointer,
                # and preserve the typed error for task-runner/UI handling.
                self._current_executor = None
                self._accumulate_usage(executor)
                raise
            except Exception as exc:
                logger.debug("Exception caught", exc_info=True)
                # A context-window overflow is always recoverable via the
                # trajectory-summarizer/continuation path, even when it
                # carries a ``__cause__`` (the provider's rejection is
                # chained by ``KISSAgent._run_agentic_loop``).  Other
                # chained or non-KISS errors stay terminal.  A first-step
                # overflow still hard-fails: continuing would replay the
                # same oversized prompt forever.
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
                    error_result = finish(False, False, str(exc))
                    if self.printer:
                        self.printer.print(
                            error_result,
                            type="result",
                            step_count=executor.step_count,
                            total_tokens=executor.total_tokens_used,
                            cost=f"${executor.budget_used:.4f}",
                        )
                    return error_result
                trajectory_path: Path | None = None
                try:
                    tmp_dir = Path(self.work_dir) / "tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    trajectory_path = tmp_dir / f"trajectory_{session}.json"
                    trajectory_path.write_text(executor.get_trajectory(), encoding="utf-8")
                    _stop_ev = getattr(self.printer, "stop_event", None) if self.printer else None
                    # Lazy import avoids the package cycle
                    # relentless_agent -> sorcar.__init__ -> sorcar_agent ->
                    # relentless_agent during a fresh direct import.
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
                                # ``SUMMARIZER_PROMPT`` uses
                                # ``{trajectory_path}`` as the
                                # placeholder; the argument key MUST
                                # match or the literal placeholder
                                # leaks into the LLM prompt.
                                "trajectory_path": str(trajectory_path),
                            },
                            max_steps=self.max_steps,
                            max_budget=summarizer_budget,
                            # Inherit the parent's model routing
                            # (e.g. custom ``base_url``/``api_key``)
                            # so the summarizer talks to the same
                            # provider as the executor — otherwise it
                            # silently falls back to the default
                            # OpenAI client and its cost vanishes
                            # from this agent's accounting.
                            model_config=self.model_config,
                            # Inherit verbose/printer so the
                            # summarizer's streaming behaviour
                            # matches the parent's (no surprise
                            # ``ConsolePrinter`` when the parent
                            # asked for ``verbose=False``).
                            printer=self.printer,
                            verbose=self.verbose,
                            # The summarizer is an INTERNAL helper: its
                            # "# Summarizer" prompt must never surface
                            # as a user-visible ``type="prompt"`` event
                            # in the shared printer's event stream
                            # (users reported it as an unexpected
                            # prompt message in the task events).
                            print_prompts=False,
                        )
                    finally:
                        # The summarizer's spend MUST be folded into
                        # the parent's running totals BEFORE we leave
                        # this block — otherwise every model call the
                        # summarizer made (which costs real money on
                        # paid providers) silently disappears from
                        # ``self.budget_used`` and the user is told
                        # they spent less than they actually did.
                        # This must run even when ``summarizer_agent.run``
                        # raises (e.g. budget exceeded mid-summary) so
                        # the partial spend is still attributed.
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
                result = finish(False, True, summary_text)

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
                    # ``summaries`` holds every PRIOR session's summary
                    # (collected from ``is_continue=True`` returns).  Prepend
                    # them as historical context BEFORE the terminal session's
                    # summary, which stays the primary payload summary.
                    prior_section = _prior_sessions_section(summaries)
                    if final_summary:
                        payload["summary"] = (
                            f"{prior_section}\n\n---\n\n### Final Session\n"
                            f"{final_summary}"
                        )
                    else:
                        # An empty terminal summary must still be
                        # split-able by the front-end
                        # (``splitMultiSessionSummary`` requires the
                        # ``\n\n---\n\n`` separator).  Use a placeholder
                        # so the merged Result panel isn't collapsed
                        # into a single "Previous Sessions" block.
                        payload["summary"] = (
                            f"{prior_section}\n\n---\n\n### Final Session\n"
                            "(no summary)"
                        )
                    result = yaml.dump(payload, sort_keys=False)
                    # The inner ``KISSAgent`` already emitted a per-session
                    # ``type="result"`` event carrying ONLY the last session's
                    # summary and status. When prior sessions exist, the
                    # front-end Result panel would otherwise miss the merged
                    # context (and, for a terminal failure, still show a stale
                    # "Continue" banner from an earlier session).  Broadcast
                    # the merged payload so the front-end can supersede the
                    # per-session panel with a "Previous Sessions" + "Result"
                    # pair via ``splitMultiSessionSummary`` in ``main.js``.
                    self._emit_merged_result_event(payload)
                return result

            summary = payload.get("summary", "")
            if summary:  # pragma: no branch
                summaries.append(summary)

                progress_section = CONTINUATION_PROMPT.format(
                    progress_text=_capped_progress_text(summaries),
                    continuation_number=session + 1,
                )
        # Sub-session budget exhausted without a terminal ``is_continue=False``
        # (or successful) return.  The last inner session's ``type="result"``
        # event carries ``is_continue=True``, so without this broadcast the
        # front-end Result panel would render a stale "Status: Continue" for
        # what is actually a terminal failure.
        banner = f"Task failed after {self.max_sub_sessions} sub-sessions"
        self._emit_merged_result_event(
            {
                "success": False,
                "is_continue": False,
                "summary": _build_exhaustion_summary(summaries, banner),
            }
        )
        # Tag the exception so downstream error handlers
        # (e.g. ``TaskRunner``'s ``except Exception`` block) know a
        # terminal ``type="result"`` event has ALREADY been broadcast
        # for this failure and skip their own generic broadcast —
        # otherwise the front-end would render a duplicate FAILED
        # Result panel with no summary.
        err = KISSError(banner)
        err.terminal_result_broadcast = True  # type: ignore[attr-defined]
        raise err

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
        # ``self.total_steps`` / ``total_tokens_used`` / ``budget_used``
        # are the CUMULATIVE aggregates across every sub-session.  The
        # printer's per-task offset attributes (set at the start of
        # each sub-session in :meth:`perform_task` to the aggregate
        # BEFORE that session) would be added on top of these totals
        # by ``JsonPrinter._broadcast_result`` — double-counting the
        # prior sessions.  Zero them for the duration of the print and
        # restore afterwards so subsequent inner-session emits (if any)
        # still get the correct offset applied.
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
    ) -> str:
        """Run the agent with the provided tools.

        Args:
            model_name: LLM model to use. Defaults to "claude-opus-4-6".
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: System-level instructions passed to the underlying LLM
                via model_config. Defaults to empty string (no system instructions).
            max_steps: Maximum steps per sub-session. Defaults to 100.
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
        args = arguments or {}
        self.task_description = substitute_prompt_args(prompt_template, args)

        if self.docker_image:
            from kiss.docker.docker_manager import DockerManager

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
