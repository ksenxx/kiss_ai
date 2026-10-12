# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar agent with both coding tools and browser automation."""

from __future__ import annotations

import argparse
import difflib
import functools
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml

from kiss.agents.sorcar.agent_dispatch import (
    kill_jobs_of,
    live_agent_jobs,
    make_agent_job_tool,
    make_run_agent_tool,
    make_run_parallel_tool,
)
from kiss.agents.sorcar.commit_message import fallback_commit_message
from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool
from kiss.agents.sorcar.fanout_guard import (
    is_implementation_task,
)
from kiss.agents.sorcar.git_worktree import (
    _PRECOMMIT_FIX_LINES,
    _PRESERVE_MARKER_ATTEMPTS,
    _PRESERVE_MARKER_RETRY_DELAY,
    _WORKTREE_SUBDIR,
    GitWorktree,
    GitWorktreeOps,
    MergeConflictResolver,
    MergeResult,
    _manual_merge_cmd,
    _merge_fix_steps,
    _path_inside,
    _reclaim_process_lock,
    _repo_key,
    _rescue_failed_warning,
    _rescue_ignored_files,
    _WorktreeCleanupOutcome,
    inside_kiss_worktree,
    kiss_worktree_root,
    repo_lock,
)
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.relentless_agent import (
    _OFFSET_PUBLISH_LOCK,
    RelentlessAgent,
    _session_usage,
    resolve_work_dir,
)
from kiss.agents.sorcar.sea_commands import model_sea
from kiss.agents.sorcar.sea_settings import (
    SeaError,
    alias_free_profile,
)
from kiss.agents.sorcar.skills import make_skill_tool
from kiss.agents.sorcar.task_classifier import (
    TaskClassification,
    classification_enabled,
    classify_task,
)
from kiss.agents.sorcar.useful_tools import (
    BackgroundJob,
    UsefulTools,
    _absolutize,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.core.memoryfield.tools import MemoryTools
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import (
    MODEL_INFO,
    OPENAI_COMPATIBLE_PROVIDERS,
    _match_openai_compatible_provider,
    _strip_provider_prefix,
    get_default_model,
    model_runs_task_to_completion,
)
from kiss.core.models.model_info import model as _model_factory
from kiss.core.printer import Printer
from kiss.core.tool_verdict import Verdict
from kiss.core.utils import substitute_prompt_args

logger = logging.getLogger(__name__)


BROWSER_TOOL_NAMES: frozenset[str] = frozenset({
    "go_to_url", "click", "type_text", "press_key", "scroll", "screenshot",
    "get_page_content", "show_browser", "close_browser",
})
"""Names of the tools :meth:`WebUseTool.get_tools` returns."""

MEMORY_TOOL_NAMES: frozenset[str] = frozenset({
    "memory_search", "memory_pull", "memory_read", "memory_write", "memory_list",
    "memory_delete", "memory_refresh",
})
"""Names of the tools :meth:`MemoryTools.tools` returns."""

MCP_AUTH_TOOL_NAMES: frozenset[str] = frozenset({
    "connect_mcp_server", "finish_mcp_server_connect",
})
"""Names of the MCP sign-in tools; a profile naming them also gets the
tools of every configured MCP server."""

TOOL_GROUPS: dict[str, frozenset[str]] = {
    # Run commands and read files (and their output).
    "shell": frozenset({"Bash", "bash_job", "Read", "run_commands_parallel"}),
    # Change files — in a git worktree the agent creates and merges itself.
    "edit": frozenset({"Edit", "Write", "create_worktree", "commit_and_merge"}),
    # Drive the Chromium browser.
    "browser": BROWSER_TOOL_NAMES,
    # Persistent memory pages.
    "memory": MEMORY_TOOL_NAMES,
    # Sub-agents: channel/cron/SEA dispatch and the parallel fan-out.
    "agents": frozenset({"run_agent", "agent_job", "run_parallel", "number_of_cores"}),
    # MCP servers: their tools plus the OAuth sign-in pair.
    "mcp": MCP_AUTH_TOOL_NAMES,
    # Project and user skills.
    "skills": frozenset({"skill"}),
    # Talk with the user.
    "user": frozenset({"ask_user_question", "talk"}),
    # Typed classification through the decisions model.
    "decide": frozenset({"decide"}),
    # Steer the run itself.
    "control": frozenset({"summary", "set_model"}),
}
"""One tool profile per group of built-in tools.

Groups are the building blocks of composite profiles: a profile name
may join any number of :data:`TOOL_PROFILES` keys with ``+``
(``"shell+edit+browser"``), and :func:`resolve_tool_profile` returns the
union of their tool sets.  The ``mcp`` and ``skills`` groups gate tools
that exist only when servers or skills are configured.
"""

TOOL_PROFILES: dict[str, frozenset[str] | None] = {
    # ``None``: every tool the agent can build (today's default).
    "full": None,
    # Reduced reviewer set: it can inspect the tree, run commands (Bash
    # is unrestricted, so this is not a sandbox), browse the web to
    # check facts and documentation, recall memory and speak to the
    # user, but has no file editing, memory writing, agent dispatch or
    # fan-out tools.
    "review": frozenset({
        *TOOL_GROUPS["shell"], *TOOL_GROUPS["browser"], "memory_search",
        "memory_pull", "memory_read", "memory_list", "decide", "summary", "talk",
    }),
    **TOOL_GROUPS,
    # Shell runner that also talks with the user: ``shell`` plus asking
    # and speaking, decide, summary and switching its own model (no file
    # editing, browser, memory, agent dispatch or fan-out).
    "assistant": (
        TOOL_GROUPS["shell"] | TOOL_GROUPS["user"] | TOOL_GROUPS["decide"]
        | TOOL_GROUPS["control"]
    ),
    # Single command runner (the bundled ``/sh`` agent): Bash and nothing else.
    "bash": frozenset({"Bash"}),
    # No built-in tool at all: ``finish`` plus whatever the agent
    # SEA's ``tools()`` supplies (the bundled ``/ask`` agent).
    "none": frozenset(),
}
"""Tool profiles an agent can run with (``finish`` is always added).

Every tool schema is re-sent on every model step, so a reviewer that
carries the editing, channel, cron and fan-out tools pays for ~20
schemas it never calls.  The fan-out engine gives reviewer-marked
children the ``review`` profile; a parent may name a profile explicitly
through ``run_parallel(..., tool_profile=...)``, and a top-level run
through ``run(tool_profile=...)`` (the ``tool_profile`` parameter of
:func:`kiss.server.sorcar.run` / the ``run_agent`` tool, or an agent
script's ``tool_profile`` setting).

A profile name is either one key or several keys joined with ``+``
(``"shell+edit+memory"``); see :func:`resolve_tool_profile`.
"""

PROFILE_SEPARATOR = "+"
"""Joins the parts of a composite tool profile name."""



def canonical_tool_profile(name: str) -> str:
    """Return *name* with every part a :data:`TOOL_PROFILES` key, or raise.

    Args:
        name: A profile name: one key or alias, or several joined with
            ``+``.  Whitespace around a part is ignored; the empty name
            (no profile chosen) is returned unchanged.

    Returns:
        The parts, aliases replaced by their key
        (:data:`~kiss.agents.sorcar.sea_settings.PROFILE_ALIASES`),
        joined with ``+``.

    Raises:
        ValueError: If a part is neither a key nor an alias.  The message
            lists the keys and, when one is spelled closely enough, asks
            whether that was meant.
    """
    canonical = alias_free_profile(name)
    parts = canonical.split(PROFILE_SEPARATOR) if canonical else []
    unknown = [part for part in parts if part not in TOOL_PROFILES]
    if unknown:
        close = difflib.get_close_matches(unknown[0], list(TOOL_PROFILES), n=1, cutoff=0.7)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"tool_profile must be one of {', '.join(TOOL_PROFILES)} "
            f"(several joined with '+'), got {name!r}.{hint}"
        )
    return canonical


def resolve_tool_profile(name: str) -> frozenset[str] | None:
    """Return the tool names a (possibly composite) profile *name* allows.

    *name* is one :data:`TOOL_PROFILES` key (or an alias of one) or
    several joined with ``+``; the result is the union of
    their tool sets.  ``None`` means every tool the agent can build: the
    ``full`` profile, alone or as a part of a composite, and the empty
    name (no profile chosen).

    Args:
        name: The profile name, e.g. ``"review"`` or ``"shell+edit+browser"``.

    Returns:
        The allowed tool names, or ``None`` for everything.

    Raises:
        ValueError: If a part is not a :data:`TOOL_PROFILES` key or alias
            (see :func:`canonical_tool_profile`).
    """
    canonical = canonical_tool_profile(name)
    parts = canonical.split(PROFILE_SEPARATOR) if canonical else []
    allowed: frozenset[str] = frozenset()
    for part in parts:
        tools = TOOL_PROFILES[part]
        if tools is None:
            return None
        allowed |= tools
    return None if not parts else allowed


RESTRICTED_PROFILE_NOTE = """

# Restricted tool profile: {profile}
This sub-agent has only these tools plus finish: {tools}. Rules above that
require any tool not listed here (e.g. Write/Edit files, tmp/PROGRESS.md,
browser research, memory writes, run_parallel, run_agent) do not apply: do
not attempt them. Report everything in finish(summary_in_html=...).
"""

WEB_TOOLS_OFF_NOTE = """

# Web tools are off
The user disabled web tools in the settings: there is no browser tool
(go_to_url, click, type_text, screenshot, get_page_content, show_browser).
The Web Research rules above do not apply: do not attempt Internet research
through them or through curl/wget substitutes; answer from local files and
your own knowledge, and say so when a fact could be outdated.
"""


def summary(description: str) -> str:
    """Every 10 steps: summarize your steps since the last `summary` call.

    Args:
        description: Natural language summary in 5-10 sentences of
            what the agent since the last call to `summary`, written in
            Markdown format (use bullet lists for the steps, and
            ``**bold**`` / backtick code spans).

    Returns:
        A short confirmation string.
    """
    del description
    return ""


def _memory_root_for_run(
    append_basic_tools: bool,
    docker_image: str | None,
    model_name: str,
    caller_system_instruction: bool = False,
    use_memory_override: bool | None = None,
) -> Path | None:
    """The memory directory this run should use, or None for no memory.

    Memory rides with the built-in toolset, so four gates precede the
    user's ``use_memory`` setting (see :func:`_memory_settings`):

    * ``append_basic_tools=False`` strips the run down to ``finish`` plus
      the caller's tools — promising ``memory_*`` tools in the prompt
      would be a lie.
    * Docker runs replace Bash/Read/Edit/Write with container-backed
      tools; the memory tools execute in the host process on host paths,
      so registering them would hand a containerized task read/write
      access to host memory outside the ``docker_image`` boundary.
    * Run-to-completion CLI models (``cc/*``, ``codex/*``) never see
      KISS-registered tools, so they get neither the tools nor a protocol
      demanding them.
    * A caller-supplied ``model_config["system_instruction"]`` replaces
      the whole composed prompt (``KISSAgent.run`` only ``setdefault``-s
      it), so ``MEMORY_PROTOCOL`` would never reach the model; the tools
      must not be registered without the protocol that governs them.

    The gates are hard invariants — an explicit *use_memory_override*
    never bypasses them.  Past the gates, a boolean override is the
    caller's per-run choice and wins over both the ``KISS_USE_MEMORY``
    environment variable and the stored ``use_memory`` setting; ``None``
    (the default) keeps the :func:`_memory_settings` resolution.

    Args:
        append_basic_tools: Whether the run builds the built-in toolset.
        docker_image: The run's Docker image, if any.
        model_name: The resolved model name the run will use.
        caller_system_instruction: Whether the caller's ``model_config``
            carries its own ``system_instruction``.
        use_memory_override: Per-run memory toggle — ``True`` enables
            (subject to the gates above), ``False`` disables, ``None``
            falls back to the environment/config default.

    Returns:
        The memory root when every gate and the effective toggle allow
        it, else None.
    """
    if not append_basic_tools or docker_image or caller_system_instruction:
        return None
    if model_runs_task_to_completion(model_name):
        return None
    enabled, root = _memory_settings()
    if use_memory_override is not None:
        enabled = use_memory_override
    return root if enabled else None


def _memory_settings() -> tuple[bool, Path]:
    """Return whether persistent agent memory is enabled and where it lives.

    The toggle is the ``use_memory`` key of ``$KISS_HOME/config.json`` (see
    :data:`kiss.core.vscode_config.DEFAULTS`, default on).  The
    ``KISS_USE_MEMORY`` environment variable, when non-empty, wins over
    the stored value — ``0``/``false``/``no``/``off`` (any case) disable,
    anything else enables — so one process or test can flip memory
    without editing the config file.  Pages live in the config's
    ``memory_dir`` when set, else ``$KISS_HOME/memories``
    (``$KISS_HOME/memories``).

    Returns:
        ``(enabled, root)`` where *root* is the memory page directory
        (created lazily on first write by
        :class:`kiss.core.memoryfield.pages.MemoryDir`).
    """
    from kiss.core.config import kiss_home
    from kiss.core.vscode_config import load_config

    cfg = load_config()
    env = os.environ.get("KISS_USE_MEMORY", "").strip().lower()
    if env:
        enabled = env not in ("0", "false", "no", "off")
    else:
        enabled = bool(cfg.get("use_memory", True))
    raw_dir = str(cfg.get("memory_dir", "")).strip()
    root = Path(raw_dir).expanduser() if raw_dir else kiss_home() / "memories"
    return enabled, root


def _repo_memory_domains(work_dir: str) -> dict[str, str]:
    """The domain memories a run in *work_dir* attaches: the memory of its repository.

    A repository's memory is the sub-directory of the general memory named
    after the repository's top-level directory (slugified).  The main
    checkout, its sub-directories and its linked worktrees share one name:
    ``git rev-parse --git-common-dir`` points at the main checkout's
    ``.git`` even from a worktree, and that directory's parent is the
    checkout.  When the common dir is not a ``.git`` directory (a
    submodule's ``.git/modules/<name>``, a ``--separate-git-dir`` layout),
    the current checkout's ``--show-toplevel`` names the repository
    instead.  Pages about the repository's code, architecture and
    conventions live there rather than in the general memory of daily work.

    Args:
        work_dir: The run's effective working directory
            (:func:`resolve_work_dir`).  It may not exist yet (the run
            creates it); the nearest existing ancestor decides.  A
            directory outside any git work tree attaches no domain memory.

    Returns:
        ``{memory_name: description}`` for :class:`MemoryTools`'s *domains*.
    """
    from kiss.core.memoryfield.pages import slugify
    from kiss.core.memoryfield.tools import GENERAL

    cwd = Path(work_dir)
    while not cwd.is_dir():
        if cwd.parent == cwd:
            return {}
        cwd = cwd.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return {}  # not a repository, or a bare one (no work tree)
    common_dir = (cwd / lines[0]).resolve()
    repo = common_dir.parent if common_dir.name == ".git" else Path(lines[1]).resolve()
    name = slugify(repo.name)
    if name == GENERAL:
        name = f"{GENERAL}-repo"  # the general memory's own name is reserved
    return {name: f"memory of the repository {repo}"}


def _generate_commit_message(
    commit_dir: Path,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Generate a commit message for staged changes using an LLM.

    Gets the staged diff and delegates to
    :func:`~kiss.agents.sorcar.commit_message.generate_commit_message_from_diff`.
    When *user_prompt* is provided, it is forwarded so the user's
    task prompt is incorporated into the commit message.  When
    *task_result* is provided, the task's result summary is appended
    to the commit message as well.

    Args:
        commit_dir: The directory containing staged changes.
        user_prompt: The user's task prompt that produced these
            staged changes, or ``None`` when not available.
        task_result: The task's result summary, or ``None`` when
            not available.

    Returns:
        A commit message string.
    """
    from kiss.agents.sorcar.commit_message import generate_commit_message_from_diff
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    diff_text = GitWorktreeOps.staged_diff(commit_dir)
    return generate_commit_message_from_diff(
        diff_text, user_prompt=user_prompt, task_result=task_result,
    )


def auto_commit_changes(
    commit_dir: Path,
    user_prompt: str | None,
    message_fn: Callable[[Path, str | None, str | None], str],
    notify_fn: Callable[[str, str], None] | None = None,
    task_result: str | None = None,
) -> bool:
    """Stage all changes, generate a commit message, and commit.

    Stages once so *message_fn* can compute the diff, runs
    *message_fn* (typically a slow LLM call) to generate the commit
    subject/body, then re-stages immediately before the commit so
    any file that appeared in the worktree during the LLM call
    (e.g. ``PROGRESS.md`` rewrites, macOS ``.DS_Store`` materializing
    after an ``open`` of the report, an editor side-channel saving
    swap files) is included in the same commit.  Without the second
    ``stage_all`` those late-arriving files would be left
    uncommitted, ``_finalize_worktree`` would see them via
    ``has_uncommitted_changes`` and abort the auto-merge with the
    misleading "pre-commit hook may have rejected" warning
    (observed in production on 2026-06-26 07:23:14 for worktree
    ``kiss_wt-1782483430-cb03445c`` even though the repo had no
    custom pre-commit hooks installed).

    Falls back to a generic commit message when *message_fn* raises
    (e.g. the LLM-based generator is unavailable).

    Args:
        commit_dir: Directory whose changes are staged and committed.
        user_prompt: The user's task prompt, woven into the commit
            message (or its fallback), or ``None`` when unavailable.
        message_fn: Callable producing a commit message from
            ``(commit_dir, user_prompt, task_result)``.
        task_result: The task's result summary, appended to the
            commit message (or its fallback) under a ``Result:``
            heading, or ``None`` when unavailable.
        notify_fn: Optional UI callback invoked at two life-cycle
            points so the chat webview can render toasts:

            - ``notify_fn("generating", "")`` immediately before
              *message_fn* runs (typically a slow LLM call) so the
              user sees "Generating commit message" while the LLM
              works.
            - ``notify_fn("committed", subject)`` immediately after
              a successful commit, where *subject* is the first
              non-empty line of the committed message.

            Both hooks are SKIPPED when there is nothing to commit
            (no staged diff after the initial ``stage_all``), so the
            webview never sees a misleading "Generating commit
            message" toast without a follow-up.  When
            ``commit_staged`` returns ``False`` after *message_fn*
            (e.g. a pre-commit hook rejected the commit),
            ``notify_fn("failed", "")`` is invoked instead so the
            sticky "generating" toast always gets a terminal update.

            All ``notify_fn`` exceptions are swallowed so a broken
            UI hook can never block the commit itself.

    Returns:
        True if a commit was created, False if nothing to commit.
    """
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps

    GitWorktreeOps.stage_all(commit_dir)
    if not GitWorktreeOps.has_staged_changes(commit_dir):
        return False
    _safe_notify(notify_fn, "generating", "")
    try:
        msg = message_fn(commit_dir, user_prompt, task_result)
    except Exception:
        logger.debug(
            "LLM commit message generation failed; using fallback", exc_info=True,
        )
        msg = fallback_commit_message(user_prompt, task_result)
    GitWorktreeOps.stage_all(commit_dir)
    committed = GitWorktreeOps.commit_staged(commit_dir, msg)
    if committed:
        _safe_notify(notify_fn, "committed", _commit_subject(msg))
    else:
        # Terminal notification for the failure path: without it the
        # sticky "Generating commit message" toast (emitted above)
        # would linger in the webview forever after e.g. a pre-commit
        # hook rejection.
        _safe_notify(notify_fn, "failed", "")
    return committed


def _commit_subject(message: str) -> str:
    """Return the first non-empty line of a commit *message*.

    Used as the subject the chat webview renders inside the
    "Committed <subject>" toast.  Falls back to an empty string when
    the message has no printable line (defensive: in practice the
    fallback message always starts with ``kiss:``).
    """
    for raw in message.splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _safe_notify(
    notify_fn: Callable[[str, str], None] | None,
    stage: str,
    subject: str,
) -> None:
    """Invoke *notify_fn* swallowing any exception.

    Errors in the optional UI hook must never prevent the commit
    itself (and must never poison the surrounding ``except`` block
    that the LLM-failure fallback relies on).
    """
    if notify_fn is None:
        return
    try:
        notify_fn(stage, subject)
    except Exception:
        logger.debug("auto_commit_changes notify_fn raised", exc_info=True)


def _agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return ``(budget_used, total_tokens_used, total_steps)`` for *agent*.

    A :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent`
    derives the triple from one append-only usage ledger; it MUST be
    read once through ``usage_snapshot()``, which sums a single ledger
    reference.  Three separate property reads each sum the ledger
    afresh, so concurrent banks between them tear the triple — a final
    abandoned-child reclaim that read the OLD budget with the NEW
    tokens/steps then discarded the child and permanently lost its
    spend from the parent's accounting.  The per-attribute fallback
    serves agent-shaped objects whose fields are plain attributes.

    A bare :class:`KISSAgent` also exposes ``usage_snapshot()``, but
    its third element is the SESSION dimension ``step_count`` — not
    the folded ``total_steps`` this reader reports — so it goes
    through the attribute fallback on purpose (its per-field property
    reads are each one atomic snapshot load; only ledger-bearing
    agents need the cross-field snapshot here, because only their
    per-property reads re-sum a concurrently growing ledger).
    """
    snapshot = getattr(agent, "usage_snapshot", None)
    if callable(snapshot) and not isinstance(agent, KISSAgent):
        budget, tokens, steps = cast("tuple[float, int, int]", snapshot())
        return float(budget or 0.0), int(tokens or 0), int(steps or 0)
    return (
        float(getattr(agent, "budget_used", 0.0) or 0.0),
        int(getattr(agent, "total_tokens_used", 0) or 0),
        int(getattr(agent, "total_steps", 0) or 0),
    )


def _broadcast_subagent_done(
    printer: Any, tab_ids: list[str], model: str = "",
) -> None:
    """Broadcast ``subagentDone`` for each tab id so the frontend can
    stop the running indicator on the sub-agent tab.

    A sub-agent that switched models with ``set_model`` also has to hand
    its tab's model picker back to *model* — the model the task was
    launched with — for the same reason a top-level task does.  Errors
    are swallowed (the broadcast is best-effort UI signalling).

    Args:
        printer: The printer to broadcast through.
        tab_ids: The sub-agent's tab plus any tabs viewing it.
        model: The model the sub-agent was launched with, restored into
            those tabs' pickers.
    """
    broadcast = getattr(printer, "broadcast", None)
    if broadcast is None:
        return
    restore = getattr(printer, "restore_model_pick", None)
    for vid in tab_ids:
        try:
            broadcast({"type": "subagentDone", "tab_id": vid, "tabId": ""})
            if callable(restore) and model:
                restore(model, vid)
        except Exception:
            pass


def _notify_subagent_done(
    printer: Any, sub_task_id: str | None, sub_tab_id: str, model: str = "",
) -> None:
    """Broadcast ``subagentDone`` to every tab watching a finished sub-agent.

    The targets are the tabs subscribed to *sub_task_id* through the
    printer's fan-out registry plus the sub-agent's own synthetic
    *sub_tab_id*.  An empty *sub_task_id* (no ``task_history`` row was
    allocated, so no ``new_tab`` was ever broadcast) fans out to the
    synthetic tab only.  Shared by ``run_parallel`` children, ``run_agent``
    dispatches, the merge agent and the task-update agent.

    Args:
        printer: The parent task's printer.
        sub_task_id: The sub-agent's persisted task id, or ``None``.
        sub_tab_id: The sub-agent's synthetic tab id (may be empty).
        model: The model the sub-agent was launched with, restored into
            the watching tabs' pickers.
    """
    viewer_ids: list[str] = []
    fanout = getattr(printer, "_fanout_targets", None)
    if callable(fanout) and sub_task_id:
        found = fanout(sub_task_id)
        if isinstance(found, list):
            viewer_ids = [v for v in found if v]
    if sub_tab_id and sub_tab_id not in viewer_ids:
        viewer_ids.append(sub_tab_id)
    _broadcast_subagent_done(printer, viewer_ids, model)


class _ClassifierSpend(NamedTuple):
    """One pre-run task classification's complete, immutable spend.

    ``SorcarAgent._classify_task_once`` publishes the WHOLE outcome —
    the retry-stable fold key together with all three spend dimensions
    — as one instance with a single ``STORE_ATTR``, so an
    asynchronously injected stop can only leave the agent with the
    complete outcome or with no outcome at all, never with a torn
    subset (round-6 finding 3: a stop between separate key and spend
    stores made the mandatory fold consume a zero triple and clear the
    transaction, silently dropping real provider spend).
    """

    key: str
    budget: float
    tokens: int
    steps: int


def subagent_parent_tab_id_of(parent_agent: Any) -> str:
    """Return the frontend tab id children of *parent_agent* hang off.

    The daemon's helper sub-agents (task update, merge-conflict
    resolver, ``run_agent`` dispatches) stamp this as their
    ``parent_tab_id``; it is :meth:`SorcarAgent._subagent_parent_tab_id`
    when the parent has it, else the parent's raw ``_tab_id`` (a
    duck-typed parent).  A parent that is itself a sub-agent carries a
    synthetic ``_tab_id`` no webview has, so stamping that raw id
    would make every surface drop the child's ``new_tab``.

    Args:
        parent_agent: The agent spawning the child.

    Returns:
        The tab id, or ``""`` when the parent runs headless.
    """
    resolve_tab = getattr(parent_agent, "_subagent_parent_tab_id", None)
    if callable(resolve_tab):
        return str(resolve_tab() or "")
    return str(getattr(parent_agent, "_tab_id", "") or "")


def _persisted_task_id(agent: Any) -> str:
    """Return *agent*'s persisted ``task_history`` row id, or ``""``.

    Only :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    and its subclasses persist a row (and expose the ``last_task_id``
    accessor that reads it under the agent's lock), and even they have
    no id before their first ``run``, so every caller must tolerate
    ``""``.

    Args:
        agent: Any agent object, or ``None``.

    Returns:
        The row id, or ``""`` when the agent has none.
    """
    task_id = getattr(agent, "last_task_id", "") if agent is not None else ""
    return task_id if isinstance(task_id, str) else ""


def _executor_usage(agent: Any) -> tuple[float, int, int]:
    """Return the in-flight executor session's ``(budget, tokens, steps)``.

    :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent` folds a
    session executor's spend into the agent's totals only when the
    session ends, so mid-session the live spend is visible only on
    ``agent._current_executor``.  The single reader of that executor's
    counters (the executor's step counter is ``step_count``, not
    ``total_steps``, an easy copy to get wrong).

    Args:
        agent: The agent whose live executor to read.

    Returns:
        The executor's spend, or ``(0.0, 0, 0)`` when no session is in
        flight.
    """
    executor = getattr(agent, "_current_executor", None)
    if executor is None:
        return 0.0, 0, 0
    # ONE coherent snapshot (see _session_usage): this function is
    # polled from other threads (the daemon's live usage readers)
    # while the executor thread updates the counters, and three
    # separate property reads could pair one response's tokens with
    # the pre-response cost.
    return _session_usage(executor)


def _live_agent_usage(agent: Any) -> tuple[float, int, int]:
    """Return live ``(budget, tokens, steps)`` for *agent*, including its
    in-flight executor session (see :func:`_executor_usage`).

    A :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent`
    answers through ``live_usage_snapshot()``, which counts a session
    that is being banked at this very moment exactly once; the
    attribute-based sum serves agent-shaped objects.
    """
    if isinstance(agent, RelentlessAgent):
        return agent.live_usage_snapshot()
    budget, tokens, steps = _agent_usage(agent)
    live_budget, live_tokens, live_steps = _executor_usage(agent)
    return budget + live_budget, tokens + live_tokens, steps + live_steps


def _attribute_sub_usage(
    agent: RelentlessAgent,
    budget: float,
    tokens: int,
    steps: int,
    key: str | None = None,
    seq: int = 0,
    epoch: Any = None,
) -> None:
    """Attribute sub-agents' cost, tokens, and steps to the parent *agent*.

    Without this, sub-agent budgets would be invisible to the parent
    agent's global accounting and UI.  Also updates the printer offsets
    so the live status line in the current sub-session reflects the
    additional spend immediately (the offsets are otherwise
    snapshotted only at session start).

    The delta is committed through the agent's
    :meth:`RelentlessAgent._attribute_usage` — ONE atomic append of an
    immutable record carrying all three dimensions to the agent's
    append-only usage ledger.  No writer can overwrite or lose another
    writer's record: this function is called concurrently by the agent
    thread (a ``talk`` synthesis bank) and by server threads (a
    finished sub-task's attribution from the dispatch job thread, the
    merge-conflict resolver), and it can also cross
    a :meth:`RelentlessAgent._reset` — three separate property stores
    used to let the reset land between them and publish an impossible
    mixed state (zero budget, pre-reset tokens/steps), whereas the
    single record now lands wholly in the old epoch (discarded with
    it) or wholly in the new one.  The append takes no lock, so no
    caller can deadlock here, even after an injected stop.

    Args:
        agent: The parent :class:`RelentlessAgent` receiving the
            attribution.
        budget: USD spend to add.
        tokens: Token count to add.
        steps: Step count to add.
        key: Stable transaction source for a retryable adjustment (an
            abandoned-child reclaim); ``None`` for one-shot
            attributions (fan-out totals, TTS), which need no dedup
            identity because they are appended at most once.
        seq: Per-*key* monotonic sequence number (a reclaim's
            generation); ignored for one-shot attributions.
        epoch: The ledger epoch object the commit is bound to (see
            ``RelentlessAgent._attribute_usage``), or ``None`` for the
            agent's current epoch.  A reclaim passes the epoch
            captured when the abandoned child was registered, so a
            concurrent reset can never divert a prior task's spend
            into the new task's ledger.
    """
    agent._attribute_usage(budget, tokens, steps, key=key, seq=seq, epoch=epoch)
    printer: Any = agent.printer
    if printer is not None:
        try:
            # Snapshot and publish under the lock the agent thread's
            # session-start rebase (``_set_printer_offsets``) also takes,
            # so publishes land in snapshot order and the offsets never
            # decrease.
            with _OFFSET_PUBLISH_LOCK:
                # One coherent triple (see _agent_usage): separate property
                # reads could tear across a concurrent snapshot publish.
                budget_total, tokens_total, steps_total = _agent_usage(agent)
                # The offsets belong to the PARENT's task.  This runs on
                # whatever thread finished the child (a fan-out's tool
                # thread, but also the ``/update`` worker or a server
                # thread), so a thread-keyed setter would file them under
                # that thread's task; name the parent's task when it has one.
                task_id = str(getattr(agent, "last_task_id", "") or "")
                set_offsets = getattr(printer, "set_usage_offsets", None)
                if task_id and callable(set_offsets):
                    set_offsets(task_id, budget_total, tokens_total, steps_total)
                else:
                    printer.budget_offset = budget_total
                    printer.tokens_offset = tokens_total
                    printer.steps_offset = steps_total
        except Exception:
            pass


def _attribute_tts_usage(agent: Any, usage: dict[str, Any]) -> None:
    """Attribute a ``talk``-tool TTS synthesis call's spend to *agent*.

    ``synthesize_talk_audio`` runs a throwaway single-shot
    ``TalkSynthesisAgent`` (gpt-audio-1.5, whose audio output bills
    $64/M tokens) whose ``budget_used`` would otherwise vanish from the
    task's accounting — the reported per-task cost would lie low (July
    2026 cost audit).  Steps are NOT attributed: the synthesis is one
    non-agentic model call, not an agent step.

    Args:
        agent: The Sorcar agent whose task accounting receives the spend.
        usage: The ``usage_out`` dict filled by ``synthesize_talk_audio``
            (``budget_used`` USD, ``total_tokens_used``); empty when the
            synthesis never issued an API call.
    """
    budget = float(usage.get("budget_used", 0.0) or 0.0)
    tokens = int(usage.get("total_tokens_used", 0) or 0)
    if budget > 0 or tokens > 0:
        _attribute_sub_usage(agent, budget, tokens, 0)


_FACTORY_DEFAULT_BASE_URLS: frozenset[str] = frozenset(
    provider.base_url.rstrip("/") for provider in OPENAI_COMPATIBLE_PROVIDERS
)


_PROVIDER_SPECIFIC_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "openai": frozenset({"reasoning_effort", "use_responses_api"}),
    "anthropic": frozenset({"thinking"}),
    "gemini": frozenset({"thinking_config"}),
}


def _model_family(model_name: str) -> str:
    """Return the provider family *model_name* routes to in the factory.

    Mirrors the routing order of :func:`kiss.core.models.model_info.model`:
    OpenAI-compatible providers first, then Gemini, then Anthropic.

    Args:
        model_name: A model name, possibly carrying a harbor-style
            ``provider/`` prefix.

    Returns:
        One of ``"openai"``, ``"gemini"``, ``"anthropic"``, or
        ``"other"``.
    """
    name = _strip_provider_prefix(model_name)
    if _match_openai_compatible_provider(name) is not None:
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    if name.startswith("claude-"):
        return "anthropic"
    return "other"


def _sanitize_model_config_for_switch(
    config: dict[str, Any], old_model_name: str, new_model_name: str,
) -> dict[str, Any]:
    """Drop source-provider request options that the target cannot accept.

    ``set_model`` copies the old adapter's complete ``model_config``
    onto the new one.  Provider-specific request options (Anthropic's
    ``thinking``, Gemini's ``thinking_config``, OpenAI's
    ``reasoning_effort`` / ``use_responses_api``) survive that copy and
    are then sent as unsupported SDK kwargs by the target adapter, so
    the switch reports success but the next model request fails.  When
    the provider family changes, remove every known provider-specific
    key that does not belong to the target family.

    Args:
        config: The config dict to sanitize (mutated in place).
        old_model_name: The model name the config came from.
        new_model_name: The model name the config is being given to.

    Returns:
        The same *config* dict, for chaining.
    """
    new_family = _model_family(new_model_name)
    if new_family == _model_family(old_model_name):
        return config
    for family, keys in _PROVIDER_SPECIFIC_CONFIG_KEYS.items():
        if family == new_family:
            continue
        for key in keys:
            config.pop(key, None)
    return config


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
    """Agent with coding tools, browser automation and git-worktree isolation.

    A task starts in the caller's ``work_dir``.  The agent's
    ``create_worktree`` tool moves the work dir into a fresh
    ``<repo>/.kiss-worktrees/kiss_wt-*`` worktree of the repository the
    path belongs to (one worktree per repository, each on its own
    ``kiss/wt-*`` branch; see :mod:`kiss.agents.sorcar.git_worktree`),
    and ``commit_and_merge`` commits every worktree's changes and
    squash-merges each branch into the branch the task started from.
    The write guard (:meth:`_guard_worktree`) refuses ``Write`` /
    ``Edit`` of a tracked checkout path that no worktree isolates, and
    the first ``finish`` with an unmerged worktree is refused once.  A
    worktree the task leaves unmerged is kept for the user's decision:
    the server's merge / discard / leave-as-is bar calls :meth:`merge`,
    :meth:`discard` and :meth:`leave_as_is`, and the next task on the
    same agent retires it first (:meth:`_retire_previous_worktree`).

    Worktrees are not associated with a chat: branch names use a unique
    time + random suffix, and there is no cross-process state
    restoration based on chat session.

    Attributes:
        _wts: The current/pending worktrees keyed by repository root,
            in creation order; empty when idle.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.web_use_tool: WebUseTool | None = None
        # The run's file/shell tools; kept so ``create_worktree``, which
        # moves the work directory mid-run, can re-point them.
        self._useful_tools: UsefulTools | None = None
        self._wts: dict[Path, GitWorktree] = {}
        self._stash_pop_warning: str | None = None
        self._merge_conflict_warning: str | None = None
        self._warning_lock: threading.Lock = threading.Lock()
        # Start of the current task block, stamped by the daemon's task
        # runner (``cmd["_start_ms"]``) and read back when a tab rebinds.
        self._task_start_ms: int = 0
        self._last_preserve_outcome: _WorktreeCleanupOutcome | None = None
        # Set by ``_do_merge``: the last merge conflicted and the merge
        # SEA resolved it (``merge`` reports this in its message).
        self._last_merge_resolved_by_agent: bool = False
        # Why ``create_worktree`` cannot isolate anything in this run
        # (a Docker container): the write guard then lets the task edit
        # the checkout directly.  Reset per run.
        self._worktree_unavailable: str | None = None
        # Repositories whose worktree setup failed (detached HEAD, no
        # commits, a git failure): the task edits them in place.
        self._unisolated_repos: set[Path] = set()
        # Repository root -> worktree directory of the calling tasks'
        # worktrees (:meth:`inherit_worktrees`); never created here.
        self._inherited_worktrees: dict[Path, Path] = {}
        # One-shot ``finish`` gate: the first ``finish`` with a worktree
        # still pending is refused once with "call commit_and_merge".
        self._worktree_finish_gate_used: bool = False
        # Per-run settings taken from :meth:`run`'s arguments; the
        # defaults are what a run that never reached ``run`` reports.
        self._launch_model_name: str = ""
        self._ask_user_question_callback: Callable[[str], str] | None = None
        self._inherited_tools: list[Callable[..., Any]] = []
        self._tools_hook: (
            Callable[[list[Callable[..., Any]]], list[Callable[..., Any]]] | None
        ) = None
        self._base_system_prompt: str = ""
        self._system_prompt_suffix: str = ""
        # Set by the fan-out engine / task runner on a sub-agent
        # (parent task and tab ids, reviewer marker); None for a
        # top-level agent.  Declared here, not on ChatSorcarAgent,
        # because this class reads it (tool profile, budget share,
        # browser profile, sub-agent tab ids).
        self._subagent_info: dict[str, object] | None = None
        # Frontend tab this agent's events belong to.  The fan-out
        # engine assigns each sub-agent its own synthetic tab id.
        self._tab_id: str = ""
        # Persistent agent memory (kiss.core.memoryfield), built per
        # run by :meth:`run` when the ``use_memory`` config flag (or the
        # KISS_USE_MEMORY environment variable) enables it; None keeps
        # the run memory-free.  :meth:`_get_tools` registers its tools.
        self._memory_tools: MemoryTools | None = None
        # Whether this session's one-time ``finish`` rejection for
        # still-running ``run_agent`` jobs has been used (see
        # :meth:`_block_finish_with_live_jobs`).
        self._live_jobs_finish_gate_used = False
        # Per-run memory toggle (:meth:`run`'s *use_memory*), kept on
        # self so the ``run_parallel`` fan-out — which executes DURING
        # the run — forwards the same override to every sub-agent.
        self._use_memory_override: bool | None = None
        self._use_web_tools: bool = True
        # The daemon's BrowserTabService (kiss.server.browser_tab) when
        # running under kiss-web: show_browser() then puts the page in
        # the Browser tab on every surface instead of a local window.
        self._live_browser: Any = None
        self._is_parallel: bool = True
        # The caller's extra tools of the current run (``run(tools=...)``:
        # a SEA's ``tools()`` additions, plus whatever this
        # agent itself inherited as a sub-task).  Kept on self so a
        # ``run_agent`` sub-task dispatched DURING the run can take
        # them over (``task_runner`` reads them off the parent agent).
        self._extra_tools: list[Callable[..., Any]] = []
        # The names of every tool the current run offers the model
        # (the executor holds the callables); the worktree guard reads
        # it to know whether ``create_worktree`` is available.
        self._run_tool_names: frozenset[str] = frozenset()
        # The ``appendToPrompt`` suffix of the current run (see
        # ``run(prompt_suffix=...)``), kept on self for the same reason.
        self._prompt_suffix: str = ""
        # Background jobs started by ``Bash(background=True)``, kept on
        # the agent (not the per-run UsefulTools) so a follow-up prompt
        # in the same chat can still wait on, tail or kill them.
        self._background_jobs: dict[str, BackgroundJob] = {}
        # Explicit tool profile named by the fan-out (a TOOL_PROFILES key)
        # or "" to let _tool_profile() decide from the reviewer marker.
        self._tool_profile_name: str = ""
        # Pre-run task classification state (see
        # :meth:`_classify_task_once`).  ``_classification_attempted``
        # makes the classifier run at most once per task even when a
        # driver pre-seeded the verdict (:meth:`classify_task_for_run`)
        # before :meth:`run` consults it for the system prompt; one
        # immutable ``_classifier_spend`` record holds the classifier's
        # spend until :meth:`_fold_classifier_usage` banks it into the
        # run totals.
        self._task_classification: TaskClassification | None = None
        self._classification_attempted: bool = False
        # True when the verdict was established by an external driver
        # (the server's task runner via :meth:`classify_task_for_run`)
        # for the CURRENT submission: internal per-run resets then keep
        # the verdict so every subtask of the submission reuses it and
        # the driver's own worktree bookkeeping never disagrees with
        # the agent's.  Cleared by the next classify_task_for_run call.
        self._classification_preseeded: bool = False
        # The UNFOLDED classifier spend: one immutable record carrying
        # the retry-stable fold key AND all three spend dimensions,
        # published with a single store (see :class:`_ClassifierSpend`)
        # — never a partial subset.  The fold appends its ledger record
        # under the record's key BEFORE clearing this field, so a
        # stop-interrupted fold retried later re-appends the SAME key
        # and read-side dedup makes the spend count exactly once.
        self._classifier_spend: _ClassifierSpend | None = None

    def _subagent_budget_share(self, num_tasks: int) -> float | None:
        """Return the ``max_budget`` each parallel sub-agent may spend.

        Splits this task's REMAINING budget — ``max_budget`` minus the
        spend already attributed to this agent minus the live executor
        session's own spend — evenly across *num_tasks* sub-agents PLUS
        one reserved parent share.  Reserving that share leaves the main
        agent enough budget to process the results and finish; importantly,
        even a one-item fan-out cannot consume the parent's entire remainder.

        Args:
            num_tasks: Number of parallel sub-agent tasks about to spawn.

        Returns:
            The per-sub-agent budget share in USD, or ``None`` when this
            agent has no budget context yet (``run``/``_reset`` never
            ran) — the sub-agents then fall back to their default
            budget.

        Raises:
            BudgetExceededError: If the task has no remaining budget.
        """
        raw_max_budget = getattr(self, "max_budget", None)
        if raw_max_budget is None:
            return None
        max_budget = float(raw_max_budget)
        remaining = max_budget - self._live_budget_used()
        if remaining <= 0:
            raise BudgetExceededError(
                f"Agent {self.name} has no remaining budget for parallel "
                f"sub-agents (${max_budget - remaining:.4f} / "
                f"${max_budget:.2f})."
            )
        if num_tasks <= 0:
            return remaining
        return remaining / (num_tasks + 1)

    def _is_reviewer_subagent(self) -> bool:
        """Return whether this agent runs inside a reviewer's sub-tree.

        The fan-out engine stamps ``_subagent_info["reviewer"]`` on a
        child whose task is a review task or whose parent is itself a
        reviewer, so the flag covers the whole sub-tree under a
        reviewer.

        Returns:
            True for a reviewer sub-agent or any of its descendants.
        """
        return bool((self._subagent_info or {}).get("reviewer", False))

    def _subagent_parent_tab_id(self) -> str:
        """Return the frontend tab id sub-agents should call their parent.

        Normally this agent's own ``_tab_id``.  When this agent is
        itself a sub-agent (nested ``run_parallel``, or a ``run_agent``
        dispatch), its ``_tab_id`` is a synthetic id — the fan-out's
        invented one, or the daemon dispatch's ``api-…`` id — which no
        webview has opened.  Every webview opens a sub-agent's tab
        under the deterministic id ``{parent_tab_id}__sub_{task_id}``
        (``subagentTabIdFor`` in ``media/main.js``; ``task`` stands in
        for a missing parent), and the daemon replays it under the
        same id, so that id is derived here: a sub-agent that fans out
        right away — before any client's ``resumeSession`` for its tab
        has arrived — has no viewer registered yet, and parenting its
        children under the synthetic id would keep every surface from
        opening their tabs.  Once viewers ARE registered in the
        printer's fan-out registry they are authoritative, for a
        top-level agent as much as for a sub-agent: the registry lists
        the OPEN tabs showing this agent's task (a closed tab is
        unsubscribed at once, see ``_drop_tab_state``), so a chat the
        user closed while it ran and reopened from history under
        another tab id parents later spawns under that new tab — the
        one every surface shows — and a sub-agent of such a chat is
        addressed as ``{new_tab}__sub_{task_id}``.  A sub-agent's own
        synthetic id is never a candidate (a ``run_agent`` dispatch
        registers it as a subscriber too, see ``subscribe_tab``).

        Returns:
            The tab id, or ``""`` when running headless.
        """
        tab_id = self._tab_id
        info = self._subagent_info
        own_task_id = _persisted_task_id(self)
        if not own_task_id:
            return tab_id
        if info is None:
            own = tab_id
        else:
            parent_tab_id = str(info.get("parent_tab_id") or "") or "task"
            own = f"{parent_tab_id}__sub_{own_task_id}"
        fanout = getattr(self.printer, "_fanout_targets", None)
        viewer_ids = sorted(
            str(v) for v in (fanout(own_task_id) if fanout else [])
            if v and (v != tab_id or info is None)
        )
        if not viewer_ids or own in viewer_ids:
            return own
        return viewer_ids[0]


    def _docker_bash(
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = 50000,
    ) -> str:
        """Run *command* in the task's container, honouring both limits.

        ``DockerManager.Bash`` honours a timeout and truncates its
        output; both are forwarded so the model can raise the
        30-second cap for a slow build or ask for more than the
        default slice of a large output — the limits the non-docker
        ``UsefulTools.Bash`` exposes.

        Args:
            command: The bash command to run.
            description: A brief description of the command.
            timeout_seconds: Timeout in seconds for the command.
            max_output_chars: Maximum characters in output before truncation.

        Returns:
            The output of the command.

        Raises:
            KISSError: If no docker manager is attached to this agent.
        """
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return str(
            self.docker_manager.Bash(
                command, description, timeout_seconds, max_output_chars,
            )
        )

    def _tool_profile(self, task: str = "") -> str:
        """Return the tool profile this agent runs with.

        A profile named explicitly by the fan-out (``_tool_profile_name``)
        wins.  Otherwise a reviewer sub-agent gets the reduced ``review``
        profile when ``DEFAULT_CONFIG.tool_profiles`` is on and its task
        does not ask for changes (:func:`is_implementation_task`), and
        everything else gets ``full``.

        Args:
            task: The task text to judge; defaults to this agent's
                ``task_description``.

        Returns:
            A :data:`TOOL_PROFILES` key or a ``+``-joined composite of
            keys (see :func:`resolve_tool_profile`).
        """
        if self._tool_profile_name:
            try:
                return canonical_tool_profile(self._tool_profile_name)
            except ValueError:
                pass  # An unknown name falls through to the default rule.
        task = task or self.task_description
        if (
            DEFAULT_CONFIG.tool_profiles
            and self._is_reviewer_subagent()
            and not is_implementation_task(task)
        ):
            return "review"
        return "full"

    def _get_tools(self) -> list:
        """Build tool list, using DockerTools when docker_manager is active.

        Must be called after docker_manager is set up (i.e., from perform_task,
        not from run() before super().run()).

        The list is cut down to the agent's tool profile
        (:meth:`_tool_profile`, resolved by :func:`resolve_tool_profile`
        so composites such as ``shell+edit+browser`` work): the browser,
        skill, MCP, channel-dispatch and fan-out tools are built only
        where the profile names them, so a ``review`` or ``shell``
        sub-agent's every step carries only the schemas it can use.
        """
        allowed = resolve_tool_profile(self._tool_profile())

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
            from kiss.agents.sorcar import cron_agent

            if cron_agent.is_unattended(self):
                # A cron run (or its sub-task) has no one to answer; a
                # blocked question would only stall until the timeout.
                return (
                    "Error: this task runs unattended (scheduled automation) and "
                    "nobody can answer. Do not ask again: proceed on the most "
                    "reasonable assumption, or report the blocker in your final "
                    "summary and finish."
                )
            if self._ask_user_question_callback is not None:
                return str(self._ask_user_question_callback(question))
            return "(ask_user_question not available in this environment)"

        def talk(language: str, text: str, emotion: str = "") -> str:
            """Speak text aloud to the user through their device speakers.

            Broadcasts a text-to-speech request to every client tab open
            for the running task (across all connected devices); each
            client plays the text on its default speaker system using
            the given language's voice.  Use this to respond aloud when
            the user speaks to the running task.

            Write *text* the way a warm, engaged human actually talks —
            NEVER like a robot reading a report.  Use contractions
            ("I'm", "let's", "that's"), short varied sentences, and
            natural interjections ("Alright,", "Oh nice —", "Hmm,",
            "Okay, so...").  Punctuation drives the delivery: questions
            rise, exclamations add energy, an ellipsis trails off, and
            sentence breaks become natural breathing pauses.  Pick an
            *emotion* that matches the vibe of the moment instead of
            leaving it flat.

            Args:
                language: BCP-47 language tag for the speech voice
                    (e.g. "en-US", "es", "fr-FR").
                text: The text to synthesize and play aloud, written in
                    a natural, conversational, emotionally expressive
                    style (contractions, interjections, punctuation).
                emotion: Optional vibe for the delivery; the client
                    shapes speech rate and pitch to match.  One of
                    "cheerful", "excited", "playful", "curious", "warm",
                    "proud", "calm", "empathetic", "reassuring",
                    "apologetic", "serious", or "sad".  Empty means
                    neutral (the client may still infer a vibe from the
                    punctuation and wording of *text*).

            Returns:
                A confirmation message, or a note that audio playback is
                unavailable in this environment.
            """
            broadcast = getattr(self.printer, "broadcast", None)
            if not callable(broadcast):
                return "(talk not available in this environment)"
            payload: dict[str, Any] = {
                "type": "talk",
                "language": language,
                "text": text,
                "emotion": emotion,
                "talkId": uuid.uuid4().hex,
            }
            tts_usage: dict[str, Any] = {}
            try:
                from kiss.core.speech_synthesis import synthesize_talk_audio

                synthesized = synthesize_talk_audio(
                    text, language, emotion, usage_out=tts_usage,
                )
            except Exception:
                synthesized = None
            finally:
                # Also on a Stop (KeyboardInterrupt) that unwinds through
                # here: the synthesis already banked its spend in
                # ``tts_usage`` and the task must still pay for it.
                _attribute_tts_usage(self, tts_usage)
            if synthesized:
                payload["audioB64"], payload["audioMime"] = synthesized
            broadcast(payload)
            return f"Spoke to the user in language {language!r}."

        if self.docker_manager:
            from kiss.agents.sorcar.docker_tools import DockerTools

            docker_tools = DockerTools(self._docker_bash)
            self.docker_manager.stop_event = self._stop_event

            def Bash(  # noqa: N802
                command: str,
                description: str,
                timeout_seconds: int = 30,
                max_output_chars: int = 50000,
                background: bool = False,
            ) -> str:
                """Runs a bash command in the task's Docker container and returns its output.

                The command starts in the container's working directory;
                in a container kiss started that is the task's work dir,
                bind-mounted at its host path.
                Background jobs (``background=True``) are not available in
                Docker mode: start long commands yourself with
                ``nohup cmd > /tmp/out.log 2>&1 < /dev/null &`` and poll
                the log file with ``tail``.

                Args:
                    command: The bash command to run.
                    description: A brief description of the command.
                    timeout_seconds: Timeout in seconds for the command.
                    max_output_chars: Maximum characters in output before truncation.
                    background: Must stay false in Docker mode (see above).

                Returns:
                    The output of the command.
                """
                if background:
                    return (
                        "Error: background=True is not available in Docker mode. "
                        "Run the command as `nohup cmd > /tmp/out.log 2>&1 "
                        "< /dev/null &` and poll the log with `tail`."
                    )
                return self._docker_bash(
                    command, description, timeout_seconds, max_output_chars,
                )

            tools: list = [
                Bash, self.docker_manager.run_commands_parallel,
                docker_tools.Read, docker_tools.Edit, docker_tools.Write,
            ]
        else:
            useful_tools = UsefulTools(
                stream_callback=_stream,
                stop_event=self._stop_event,
                work_dir=self.work_dir,
                jobs=self._background_jobs,
            )
            self._useful_tools = useful_tools
            tools = [
                useful_tools.Bash, useful_tools.bash_job,
                useful_tools.run_commands_parallel,
                useful_tools.Read, useful_tools.Edit, useful_tools.Write,
            ]
            tools.extend([self.create_worktree, self.commit_and_merge])
            # The Read dedupe assumes an earlier output is still in the
            # model's context; the executor calls this when it is not.
            self.context_reset_hook = useful_tools.forget_reads
        if (
            (allowed is None or allowed & BROWSER_TOOL_NAMES)
            and self._use_web_tools
            and self.web_use_tool is None
        ):
            # Sub-agents run concurrently, so they get a throwaway profile
            # instead of contending for the shared profile's Chromium lock.
            self.web_use_tool = WebUseTool(
                work_dir=self.work_dir,
                ephemeral=self._subagent_info is not None,
                live_browser=self._live_browser,
            )
            tools.extend(self.web_use_tool.get_tools())
        def number_of_cores() -> int:
            """Return the number of CPU cores available on the current machine.

            Useful for choosing how many splits to give
            ``run_commands_parallel`` (``max_workers``) or how many
            tasks to fan out at once.

            Returns:
                The number of CPU cores available to the process,
                falling back to ``1`` when it cannot be determined.
            """
            return os.process_cpu_count() or 1

        def set_model(model_name: str) -> str:
            """Change only this running agent's LLM model dynamically.

            The tabs watching this task show the new model in their
            picker for as long as the task runs, then revert to the
            user's own choice.  The switch is display-only: it never
            persists ``last_model``, so the user's picker preference
            survives untouched — only an explicit picker selection
            updates that.

            Args:
                model_name: New LLM model name (for example
                    ``"gpt-5.5"``, ``"claude-sonnet-4-8"``,
                    ``"gemini-3.5-flash"``).

            Returns:
                A human-readable confirmation string describing the
                change (or a "no change" message when the requested
                model is already active).
            """
            from kiss.core.models.model_info import (
                model_runs_task_to_completion,
            )

            if self.docker_image and model_runs_task_to_completion(model_name):
                return (
                    f"Cannot switch to {model_name}: it is a CLI agent "
                    "that runs natively on the host, which would bypass "
                    "this task's docker_image isolation. Pick an API model."
                )
            target = self._current_executor or self
            old_model = getattr(target, "model", None)
            if old_model is None:
                self.model_name = model_name
                self._show_model_in_picker(model_name)
                return (
                    f"Model deferred-changed to {model_name} "
                    "(no live model yet)."
                )
            if old_model.model_name == model_name:
                return f"Model is already {model_name}; no change."

            new_config: dict[str, Any] = dict(old_model.model_config or {})
            old_info = MODEL_INFO.get(old_model.model_name)
            if (
                old_info is not None
                and new_config.get("reasoning_effort") == old_info.thinking
            ):
                new_config.pop("reasoning_effort", None)
            _sanitize_model_config_for_switch(
                new_config, old_model.model_name, model_name,
            )
            old_base_url = getattr(old_model, "base_url", None)
            old_api_key = getattr(old_model, "api_key", None)
            if "use_responses_api" in new_config:
                # `use_responses_api` forces /responses delegation, which
                # only some OpenAI-compatible vendors support.  Keep it
                # only when the switch stays on the SAME vendor endpoint.
                from kiss.core.models.model_info import (
                    openai_compatible_provider_for_base_url,
                )

                old_vendor = openai_compatible_provider_for_base_url(
                    old_base_url or ""
                )
                new_vendor = _match_openai_compatible_provider(
                    _strip_provider_prefix(model_name)
                )
                if old_vendor is None or old_vendor is not new_vendor:
                    new_config.pop("use_responses_api", None)
            if old_base_url and "base_url" not in new_config:
                normalized = old_base_url.rstrip("/")
                if normalized in _FACTORY_DEFAULT_BASE_URLS:
                    # Standard provider endpoint: preserve routing (and
                    # crucially the possibly task-specific api_key) only
                    # when the target model routes to the SAME provider
                    # default — otherwise the factory would silently
                    # replace a per-task key with the process-global one.
                    target_provider = _match_openai_compatible_provider(
                        _strip_provider_prefix(model_name)
                    )
                    preserve = (
                        target_provider is not None
                        and target_provider.base_url.rstrip("/") == normalized
                    )
                else:
                    # Custom endpoint: always carry it (and its key) over.
                    preserve = True
                if preserve:
                    new_config["base_url"] = old_base_url
                    if old_api_key is not None:
                        new_config["api_key"] = old_api_key
            new_model = _model_factory(
                model_name,
                model_config=new_config or None,
                token_callback=old_model.token_callback,
                thinking_callback=old_model.thinking_callback,
            )
            old_family = _model_family(old_model.model_name)
            if (
                old_api_key
                and old_family in ("gemini", "anthropic")
                and old_family == _model_family(model_name)
                and hasattr(new_model, "api_key")
            ):
                # Native same-provider switch (Gemini→Gemini,
                # Claude→Claude): the factory always injects the
                # process-global key, which would silently replace a
                # task-specific one.  Both native adapters build their
                # SDK client from self.api_key inside initialize(), so
                # overriding BEFORE initialize() routes requests with
                # the task's credential.  setattr: the attribute lives
                # on the concrete adapters, not the Model base class
                # (the hasattr guard above ensures it exists).
                setattr(new_model, "api_key", old_api_key)  # noqa: B010
            new_model.initialize("")
            new_model.conversation = old_model.conversation
            new_model.usage_info_for_messages = old_model.usage_info_for_messages
            old_sigs = getattr(old_model, "_thought_signatures", None)
            new_sigs = getattr(new_model, "_thought_signatures", None)
            if isinstance(old_sigs, dict) and isinstance(new_sigs, dict):
                # Gemini-to-Gemini switch: the conversation references
                # historical tool-call ids whose thought signatures live
                # only in this side map (initialize() cleared the new
                # model's copy); without them signature-enforcing Gemini
                # models reject the next request.
                new_sigs.update(old_sigs)

            previous_name = old_model.model_name
            target.model = new_model  # type: ignore[attr-defined, union-attr]
            target.model_name = model_name
            self.model_name = model_name
            if getattr(target, "function_map", None):
                target._cached_tools_schema = new_model._build_openai_tools_schema(  # type: ignore[attr-defined, union-attr]
                    target.function_map,
                )
            self._show_model_in_picker(model_name)
            return f"Model changed from {previous_name} to {model_name}."

        if self._memory_tools is not None:
            tools.extend(self._memory_tools.tools())
        if allowed is None or "skill" in allowed:
            skill_tool = make_skill_tool(self.work_dir or ".")
            if skill_tool is not None:
                tools.append(skill_tool)
        # The MCP servers' tools carry server-specific names, so they
        # are kept as a block, unfiltered, wherever the profile names
        # the sign-in pair (the ``mcp`` group).
        mcp_tools: list = []
        if allowed is None or allowed & MCP_AUTH_TOOL_NAMES:
            try:
                from kiss.agents.sorcar.mcp_oauth import make_mcp_auth_tools
                from kiss.agents.sorcar.mcp_servers import make_mcp_tools

                mcp_tools.extend(make_mcp_tools(self.work_dir or "."))
                tools.extend(make_mcp_auth_tools(self.work_dir or "."))
            except Exception:
                logger.warning("MCP tool setup failed", exc_info=True)
        if allowed is None or "run_agent" in allowed:
            # Scheduled automations (cron) are not a built-in tool: the
            # agent dispatches them via run_agent(agent="cron", ...), which
            # runs kiss.agents.sorcar.cron_agent as a SEA.  Passing
            # self makes each dispatched sub-task's cost/tokens/steps fold
            # into THIS task's accounting, so the end-of-task cost shown
            # to the user includes run_agent sub-tasks.
            tools.append(make_run_agent_tool(self.work_dir or "", self))
            tools.append(make_agent_job_tool(self))
        tools.append(ask_user_question)
        tools.append(talk)
        tools.append(set_model)
        # Typed classification / routing / scoring through OpenRouter's
        # decisions endpoint (Jev).  Offered only when the settings
        # panel's "Use Jev" checkbox (``classify_with_decisions``) is on
        # and it can actually run: an OpenRouter key is configured and
        # the catalog has the model, otherwise every call would fail and
        # the tool would only cost prompt tokens.  Its spend folds into
        # this task's accounting like ``talk``'s synthesis.
        if decisions_tool_available():
            tools.append(make_decide_tool(self))
        # No-op tool letting the model periodically condense its recent
        # activity.  Chat-webview runs react to the persisted
        # ``tool_call`` event by nesting and collapsing the preceding
        # event panels (see ``media/main.js``); outside a webview the
        # call is a harmless no-op.  The every-N-steps cadence is
        # requested by the SYSTEM.md instructions and this tool's
        # docstring only — there is no mechanical enforcement.
        tools.append(summary)
        if self._is_parallel and (allowed is None or "run_parallel" in allowed):
            tools.append(make_run_parallel_tool(self.work_dir or "", self))
            tools.append(number_of_cores)
        if allowed is not None:
            tools = [tool for tool in tools if tool.__name__ in allowed]
        return tools + mcp_tools

    def _show_model_in_picker(self, model_name: str) -> None:
        """Display *model_name* in the picker of every tab watching this task.

        The override lasts only while the task runs — the daemon puts
        the user's own pick back when the task ends — so this is purely
        a live view of what the agent is running right now.  Purely
        cosmetic, hence best-effort: printers without the capability
        (plain console runs) and transport errors are ignored rather
        than allowed to fail the ``set_model`` tool call.

        The printer resolves the watching tabs itself via its
        transient all-watching-tabs primitive
        (``JsonPrinter._transient_targets``, shared with
        ``broadcast_transient``); the agent only supplies its ids —
        ``_last_task_id`` keeps the fan-out working when the call is
        made off the run thread or near teardown, after the printer's
        thread-local task id has been cleared.  The model pick goes
        through ``broadcast_agent_model_pick`` rather than the plain
        primitive because the printer must also remember each target
        for ``restore_model_pick``.

        Args:
            model_name: The model the agent just switched to.
        """
        show = getattr(self.printer, "broadcast_agent_model_pick", None)
        if not callable(show):
            return
        try:
            show(model_name, self._tab_id, _persisted_task_id(self) or None)
        except Exception:
            logger.warning("model picker update failed", exc_info=True)

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
        # ``run(append_basic_tools=False)`` strips the agent down to
        # ``finish`` (added by ``RelentlessAgent.perform_task``) plus
        # the caller's *tools*: the built-in toolset is never built, so
        # no web profile, MCP server, or run_agent/run_parallel wiring
        # is set up either.
        built_in = self._get_tools() if self._append_basic_tools else []
        # Tools inherited from the dispatching task come last, and only
        # under names this run does not have yet (the caller may lack a
        # built-in this run has, e.g. ``number_of_cores`` without
        # ``is_parallel``).  Everything after the built-ins is this
        # run's effective extra toolset — what ITS ``run_agent``
        # sub-tasks inherit in turn.
        extra = list(tools)
        names = {tool.__name__ for tool in built_in + extra}
        for tool in self._inherited_tools:
            if tool.__name__ not in names:
                extra.append(tool)
                names.add(tool.__name__)
        all_tools = built_in + extra
        if self._tools_hook is not None:
            # The SEA's ``tools()`` sees the whole toolset and returns
            # the run's.  Of two tools with one name (the SEA adding a
            # tool its parent, running the same SEA, passed down) the
            # later one, the SEA's own, stands.  Whatever it kept or
            # added beyond the built-ins is what this run's sub-tasks
            # inherit.
            by_name = {tool.__name__: tool for tool in self._tools_hook(all_tools)}
            all_tools = list(by_name.values())
            extra = [tool for tool in all_tools if tool not in built_in]
        self._extra_tools = extra
        self._run_tool_names = frozenset(tool.__name__ for tool in all_tools)
        # Always install the steering hooks: they are self-guarding
        # no-ops when no follow-up channel exists (a printer without
        # the duck-typed ``drain_pending_user_messages`` bridge), and
        # the server UI's printer bridge must be drained when present.
        self.pre_step_hook = self._drain_pending_user_messages
        self.tool_call_guard = self._guard_tool_call
        self._live_jobs_finish_gate_used = False
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
        # :meth:`run` resolved the model once for the whole run; a
        # direct ``RelentlessAgent.run`` call (tests) resolves here.
        self._launch_model_name = self._resolve_model_name(model_name)
        super()._reset(
            model_name=self._launch_model_name,
            max_sub_sessions=max_sub_sessions,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir or ".",
            docker_image=docker_image,
            printer=printer,
            verbose=verbose if verbose is not None else False,
        )

    @staticmethod
    def _resolve_model_name(model_name: str | None) -> str:
        """The model a run asked to use *model_name* actually runs with.

        The caller's model, else the user's last-selected model, else
        the configured default.  The fallback re-reads the user's
        config, which another thread (the model picker) may change at
        any time, so a run resolves it ONCE — :meth:`run` stores the
        result in ``_launch_model_name`` and passes it to the
        classifier, the memory-root gate and ``_reset`` — and
        ``ChatSorcarAgent.run``, which records the run's settings
        before :meth:`run` executes, resolves first and passes the
        result down as *model_name*.

        The picker persists a model-picker SEA (``autorouter``,
        ``bestrouter``; :func:`kiss.agents.sorcar.sea_commands.model_sea`)
        like any pick, but it is not a model: the daemon resolves it into
        a SEA run before any agent sees it, and a run launched
        without the daemon cannot route, so here it counts as no pick at
        all.

        Args:
            model_name: The caller-supplied model name, possibly None.

        Returns:
            The resolved model name.
        """
        if model_name:
            return model_name
        last_model = _load_last_model()
        if last_model and model_sea(last_model) is not None:
            last_model = ""
        return last_model or get_default_model()

    def _system_prompt_task_settings(self) -> dict[str, str]:
        """Extend the base settings with this agent's parallel mode.

        Returns:
            The base label → value pairs plus "Parallel mode".
        """
        settings = super()._system_prompt_task_settings()
        settings["Parallel mode"] = (
            "parallel" if self._is_parallel else "sequential"
        )
        return settings

    def _reset_task_classification(self) -> None:
        """Forget the classification state of the run that just ended.

        Called by :meth:`_fold_classifier_usage` in :meth:`run`'s
        cleanup (and by :meth:`classify_task_for_run` before it seeds a
        new verdict) so a stale verdict never leaks into the next task
        on a reused agent instance.  Unfolded classifier spend is
        dropped with the verdict: folding it into an unrelated later
        run would corrupt that run's accounting worse than losing the
        (≤ classifier-cap) telemetry.

        A verdict pre-seeded by :meth:`classify_task_for_run` survives:
        the external driver established it for the current submission,
        and internal resets (one per subtask run) must not discard it.
        """
        if self._classification_preseeded:
            return
        self._classification_attempted = False
        self._task_classification = None
        self._classifier_spend = None

    def classify_task_for_run(
        self,
        model_name: str | None,
        task: str,
        model_config: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> TaskClassification | None:
        """Classify *task* now and pre-seed the verdict for coming runs.

        For external drivers that classify BEFORE calling :meth:`run`
        (the server's task runner shows a "Classifying task…" phase).
        The pre-seeded verdict is then reused by the run itself (system
        prompt selection) and by every later subtask of the same
        submission, so the driver and the agent can never disagree.
        The seed lasts until the next ``classify_task_for_run`` call on
        this agent.

        Args:
            model_name: The model the run will use, possibly None.
            task: The (already substituted) task prompt about to run.
            model_config: The model configuration the run will use.
            enabled: Per-run override of the persisted
                ``classify_tasks`` setting — the ``classifyTasks`` wire
                field of the ``run`` command (the *auto_classify*
                parameter of :func:`kiss.server.sorcar.run`).  ``True``
                forces classification on, ``False`` skips it (the run
                then behaves exactly as it would without a
                classifier), and ``None`` (the default) follows the
                config.  The ``KISS_DISABLE_TASK_CLASSIFIER``
                environment kill switch wins over any override.

        Returns:
            The task's classification, or ``None`` when classification
            is disabled or failed.
        """
        self._classification_preseeded = False
        self._reset_task_classification()
        verdict = self._classify_task_once(
            self._resolve_model_name(model_name), task, model_config, enabled_override=enabled,
        )
        self._classification_preseeded = True
        return verdict

    def _classify_task_once(
        self,
        model_name: str,
        task: str,
        model_config: dict[str, Any] | None,
        arguments: dict[str, str] | None = None,
        enabled_override: bool | None = None,
    ) -> TaskClassification | None:
        """Classify *task* at most once per run and return the verdict.

        The first call per run (see :meth:`_reset_task_classification`)
        performs the classification — when
        :func:`~kiss.agents.sorcar.task_classifier.classification_enabled`
        allows it — with the same resolved model and model config the
        main run will use, and publishes the classifier's spend as one
        ``_classifier_spend`` record for :meth:`_fold_classifier_usage`
        to bank.  Every later call returns the cached verdict, so a
        driver that classified ahead of :meth:`run`
        (:meth:`classify_task_for_run`) and :meth:`run` (system prompt
        selection) share one classification.

        Args:
            model_name: The run's resolved model name (see
                :meth:`_resolve_model_name`).
            task: The task prompt template about to run.
            model_config: The caller-supplied model configuration.
            arguments: The caller-supplied prompt-template arguments;
                substituted into *task* before classification so the
                classifier sees the prompt the run will actually
                execute, not the raw ``{placeholder}`` template.
            enabled_override: Per-run override of the persisted
                ``classify_tasks`` config key (see
                :meth:`classify_task_for_run`); ``None`` follows the
                config.

        Returns:
            The task's classification, or ``None`` when classification
            is disabled or failed (the run then behaves exactly as it
            would without a classifier).
        """
        if self._classification_attempted:
            return self._task_classification
        self._classification_attempted = True
        if not classification_enabled(enabled_override):
            return None
        outcome = classify_task(
            task=substitute_prompt_args(task, arguments),
            model_name=model_name,
            model_config=model_config,
        )
        # ONE immutable publication: the retry-stable fold key and the
        # whole spend triple become visible together with a single
        # store, so an injected stop leaves either the complete outcome
        # or no outcome — never a key paired with a zero triple that
        # the mandatory fold would consume and clear (round-6
        # finding 3).  A stop between the model call returning and
        # this store loses the whole outcome, which is the documented
        # all-or-nothing semantics: an unpublished classification was
        # never committed, so nothing partial can leak into totals.
        self._classifier_spend = _ClassifierSpend(
            f"classifier:{uuid.uuid4().hex}",
            outcome.budget_used,
            outcome.tokens_used,
            outcome.steps,
        )
        self._task_classification = outcome.classification
        if outcome.classification is not None:
            logger.info(
                "Task classified: is_simple=%s", outcome.classification.is_simple,
            )
        return self._task_classification

    def _fold_classifier_usage(self) -> None:
        """Bank the pre-run classifier's spend into this run's totals.

        Runs from :meth:`run`'s ``finally``, AFTER ``super().run`` — the
        classifier executes before ``RelentlessAgent._reset`` zeroes the
        cumulative counters, so folding earlier would be erased.  Also
        clears the classification state so a reused agent instance
        starts its next run fresh.

        The fold is one :meth:`RelentlessAgent._attribute_usage`
        ledger append: it runs UNCONDITIONALLY on every unwind path,
        including after a stop-injected ``KeyboardInterrupt`` unwound a
        session bank mid-commit.  The append is lock-free, so it always
        terminates — an earlier lock-based design deadlocked ``run``'s
        mandatory ``finally`` forever on a lock the injection leaked.

        The fold consumes ``_classifier_spend`` — ONE immutable record
        carrying the stable transaction key and the whole spend triple
        (see :class:`_ClassifierSpend`), so it can never observe a
        torn subset of the outcome (round-6 finding 3).  The keyed
        append happens BEFORE the record is cleared, so a stop
        injected anywhere in this method is retry-safe: a stop before
        the append leaves the record intact for a later fold to
        commit; a stop after it leaves it intact too, and the retry's
        same-key re-append deduplicates on read — exactly once either
        way (round-4 finding 2b: the previous unkeyed append-once
        design lost the spend on a pre-append stop and could not be
        retried safely).  :meth:`run`'s ``finally`` republishes the
        totals when the ledger moved, so nothing is reported here.
        """
        spend = self._classifier_spend
        if spend is not None:
            self._attribute_usage(
                spend.budget,
                spend.tokens,
                spend.steps,
                key=spend.key,
            )
            self._classifier_spend = None
        self._reset_task_classification()

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
        is_parallel: bool = True,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
        base_system_prompt: str = "",
        append_basic_tools: bool = True,
        inherited_tools: list[Callable[..., Any]] | None = None,
        llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
        use_memory: bool | None = None,
        tool_profile: str = "",
        live_browser: Any = None,
        prompt_suffix: str = "",
        system_prompt_hook: Callable[[str], str] | None = None,
        tools_hook: Callable[[list[Callable[..., Any]]], list[Callable[..., Any]]] | None = None,
    ) -> str:
        """Run the assistant agent with coding tools and browser automation.

        A worktree still pending from an earlier run of this agent is
        retired first (:meth:`_retire_previous_worktree`: committed to
        its own branch, never merged), so the task starts in *work_dir*
        and its ``create_worktree`` tool makes a fresh worktree.  The
        warnings left by the retirement are flushed to *printer*.  A
        run that ends with a worktree unmerged leaves its branch marked
        preserved (:meth:`GitWorktreeOps.save_preserve_marker`), so no
        orphan reclaim in another process merges it behind the user's
        back.  An unexpected exception becomes a ``success: false``
        result; a :class:`KISSError` or :class:`SeaError` (a setup
        failure the caller must see as a diagnostic) propagates.

        Args:
            model_name: LLM model to use. Defaults to config value.
            prompt_template: Task prompt template with format placeholders.
            arguments: Dictionary of values to fill prompt_template placeholders.
            system_prompt: system prompt to be appended to the actual system
                prompt.  Also forwarded to every sub-agent spawned via
                ``run_parallel``, appended to each sub-agent's own base
                system prompt — like *base_system_prompt*, so the extra
                instructions constrain the whole task tree.
            tools: List of tools to be added in addition to bash and web tools.
            max_steps: Maximum steps per sub-session. Defaults to 10000.
            max_budget: Maximum budget in USD. Defaults to config value.
            work_dir: Working directory for the agent. Defaults to artifact_dir/kiss_workdir.
            printer: Printer instance for output display.
            max_sub_sessions: Maximum continuation sub-sessions. Defaults to config value.
            docker_image: Docker image name to run tools inside a container.
            web_tools: Whether to include browser/web tools. Defaults to True.
            live_browser: The daemon's ``BrowserTabService``; when given,
                ``show_browser()`` opens the page in the Browser tab on
                every surface (forwarded to every sub-agent).  None (no
                daemon) shows a local window instead.
                Set to False for terminal-only environments.
            prompt_suffix: The caller-supplied text (the daemon's
                ``appendToPrompt`` wire field, e.g. a ``run_agent``
                call's ``add_to_prompt`` option) that the caller has
                ALREADY appended to *prompt_template*; it is not added
                again here.  Recorded as ``_prompt_suffix`` so a ``run_agent``
                sub-task dispatched during the run inherits it as its
                own ``add_to_prompt`` option (see
                ``agent_dispatch.inherit_from_parent``).  Defaults to
                "" (the run has no suffix).  Last in the signature so
                every earlier argument keeps its position.
            is_parallel: Whether to include the run_parallel tool. Defaults to True.
                When True, the agent can spawn parallel sub-agents for independent tasks.
            verbose: Whether to print output to console. Defaults to config verbose setting.
            current_editor_file: Path to the currently active editor file, appended to prompt.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.
            ask_user_question_callback: Optional callback used by the ask_user_question
                tool to collect a text response from the user.
            base_system_prompt: Custom base system prompt.  When non-blank it
                REPLACES the default ``SYSTEM.md`` system prompt
                (:data:`kiss.core.base.SYSTEM_PROMPT`) for this agent and for
                every sub-agent it spawns via ``run_parallel``.  The
                *system_prompt* suffix, the active-editor-file line, and the
                per-run operational instructions (work dir, PID,
                ``$KISS_HOME/AGENTS.md``) are still appended.  Blank (default)
                keeps the default system prompt.  Appended after the
                historical positional arguments so every one of them
                keeps its position.
            append_basic_tools: Whether :meth:`perform_task` prepends the
                built-in basic toolset (:meth:`_get_tools`: Bash, Read,
                Edit, Write, browser tools, run_agent,
                ask_user_question, talk, set_model, decide,
                run_parallel, ...) to the caller's *tools*.  Defaults
                to True.  When False
                the agent runs with ONLY the ``finish`` tool (added by
                ``RelentlessAgent.perform_task``) and the caller's
                *tools* — *web_tools* and *is_parallel* then have no
                effect, since the tools they toggle are never built.
            inherited_tools: Extra tools taken over from the task that
                dispatched this run (a ``run_agent`` sub-task gets the
                caller's SEA ``tools()``, see ``task_runner``).  Added
                after the built-in toolset and *tools*; one whose name
                this run already has (a
                built-in the caller lacked, or one of *tools*) is
                skipped rather than registered twice.  ``None``
                (default) adds nothing.
            llm_call_hook: Optional hook forwarded to the underlying
                :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
                sub-session this agent runs (see that docstring): called
                before every LLM call with the new messages about to be
                sent, and its return value replaces them.  Applies to
                this agent only, not to ``run_parallel`` sub-agents.
                Defaults to None (no hook).
            tool_call_hook: Optional hook forwarded to the underlying
                :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
                sub-session this agent runs (see that docstring): called
                before every tool call with the tool's name and
                arguments and returns a
                :class:`~kiss.core.tool_verdict.Verdict` (``ALLOW``, or
                ``refuse(text)``: the call is suppressed and the model
                reads *text* as its result).  Applies to this agent only, not to
                ``run_parallel`` sub-agents.  Defaults to None (no
                hook).
            use_memory: Per-run persistent-memory toggle
                (:mod:`kiss.core.memoryfield`).  ``True`` gives the
                run the ``memory_*`` tools and the ``MEMORY_PROTOCOL``
                prompt block, ``False`` withholds them, and ``None``
                (the default) falls back to the ``KISS_USE_MEMORY``
                environment variable / the stored ``use_memory``
                setting (see :func:`_memory_settings`).  A boolean
                never bypasses the hard gates of
                :func:`_memory_root_for_run` (stripped basic tools,
                Docker runs, run-to-completion CLI models, a caller
                ``model_config["system_instruction"]``).  Forwarded to
                every ``run_parallel`` sub-agent, so one override
                governs the whole task tree.
            tool_profile: Name of the tool profile this run's built-in
                toolset is cut down to — a key of :data:`TOOL_PROFILES`
                (the composites ``"full"``, ``"review"``, ``"assistant"``,
                ``"bash"`` or the groups ``"shell"``, ``"edit"``,
                ``"browser"``, ``"memory"``, ``"agents"``, ``"mcp"``,
                ``"skills"``, ``"user"``, ``"decide"``, ``"control"``),
                several keys joined with ``+`` (``"shell+edit+browser"``
                keeps the union of their tools; see
                :func:`resolve_tool_profile`), or ``""`` (the default)
                to let :meth:`_tool_profile` decide (``full`` for a
                top-level task, ``review`` for a reviewer sub-agent).
                Applies to this agent only: ``run_parallel`` children
                pick their own profile.
            system_prompt_hook: The SEA's ``system_prompt`` method
                (:func:`kiss.agents.sorcar.sea_commands.base_system_prompt`):
                called once with the assembled system prompt (base or
                *base_system_prompt*, plus *system_prompt*) and its
                return value is the run's system prompt, verbatim.  Not
                forwarded to sub-agents (their own SEA layers apply
                theirs; they inherit *base_system_prompt* and
                *system_prompt* as given).  ``None`` (default) changes
                nothing.
            tools_hook: The SEA's ``tools`` method
                (:func:`kiss.agents.sorcar.sea_commands.base_tools`):
                called once by :meth:`perform_task` with the built-in
                toolset plus *tools* and *inherited_tools*, and its
                return value is the run's toolset.  ``None`` (default)
                changes nothing.

        Returns:
            YAML string with 'success' and 'summary' keys.

        Raises:
            ValueError: If *tool_profile* is neither ``""`` nor a
                ``+``-joined list of :data:`TOOL_PROFILES` keys.
        """
        self._tool_profile_name = canonical_tool_profile(tool_profile)
        self._ask_user_question_callback = ask_user_question_callback
        self._use_web_tools = web_tools
        self._live_browser = live_browser
        self._use_memory_override = use_memory
        self._is_parallel = is_parallel
        self._append_basic_tools = append_basic_tools
        self._inherited_tools = list(inherited_tools or [])
        self._tools_hook = tools_hook
        self._prompt_suffix = prompt_suffix if prompt_suffix else ""
        # Stored on self (not just a local) so the ``run_parallel``
        # fan-out — which executes DURING ``super().run`` below — can
        # forward the same base system prompt to every sub-agent.
        self._base_system_prompt = (
            base_system_prompt if base_system_prompt.strip() else ""
        )
        # Stored on self for the same reason: the fan-out forwards the
        # append-only *system_prompt* suffix to every sub-agent, so a
        # run's extra system instructions constrain its whole task
        # tree, exactly like a *base_system_prompt* replacement.
        self._system_prompt_suffix = system_prompt if system_prompt else ""
        self.web_use_tool = None
        self._memory_tools = None
        self._begin_task(printer)
        # Resolved ONCE for the whole run (see _resolve_model_name):
        # the classifier, the memory-root gate and ``_reset`` all get
        # this string, so they cannot disagree about the model.
        model_name = self._launch_model_name = self._resolve_model_name(model_name)
        try:
            try:
                # Pre-run task classification (idempotent per run: a driver
                # may have pre-seeded the verdict).
                # A task the classifier deems
                # simple — no software development, no Internet search —
                # runs on the reduced SYSTEM_LITE.md prompt; everything
                # else (including a failed or disabled classification)
                # keeps the full SYSTEM.md.  A caller-supplied
                # *base_system_prompt* replaces both.
                classification = self._classify_task_once(
                    model_name, prompt_template, model_config,
                    arguments=arguments,
                )
                default_base_prompt = (
                    SYSTEM_PROMPT_LITE
                    if classification is not None and classification.is_simple
                    else SYSTEM_PROMPT
                )
                system_instructions = (
                    (self._base_system_prompt or default_base_prompt)
                    + (system_prompt if system_prompt else "")
                )
                if system_prompt_hook is not None:
                    # The SEA's return is the run's system prompt, as its
                    # ``prompt()`` return is the run's prompt.  Neither is
                    # forwarded to sub-agents: they inherit the caller's
                    # *base_system_prompt* and *system_prompt* and their own
                    # SEA layers shape their prompt (the daemon applies
                    # them), so an appended rule is
                    # stated once per run without any deduplication.
                    system_instructions = system_prompt_hook(system_instructions)
                memory_root = _memory_root_for_run(
                    self._append_basic_tools,
                    docker_image,
                    model_name,
                    caller_system_instruction=bool(
                        (model_config or {}).get("system_instruction")
                    ),
                    use_memory_override=use_memory,
                )
                if memory_root is not None:
                    self._memory_tools = MemoryTools(
                        memory_root, domains=_repo_memory_domains(resolve_work_dir(work_dir))
                    )
                profile = self._tool_profile(prompt_template)
                # No note without a built-in toolset to cut down: a run with
                # ``append_basic_tools=False`` has only ``finish`` and the
                # caller's tools, whatever profile it names.
                allowed = resolve_tool_profile(profile)
                if allowed is not None and self._append_basic_tools:
                    # The note lists what :meth:`_get_tools` will build,
                    # never a tool this run cannot have: the docker toolset
                    # has no job registry (see the docker ``Bash`` shim),
                    # the browser needs the settings panel's "Use web
                    # tools", the fan-out needs parallel mode, memory needs
                    # a memory root, ``decide`` the decisions model and
                    # ``skill`` a configured user or project skill.
                    offered = set(allowed)
                    if docker_image:
                        offered.discard("bash_job")
                    if not web_tools:
                        offered -= BROWSER_TOOL_NAMES
                    if not is_parallel:
                        offered -= {"run_parallel", "number_of_cores"}
                    if memory_root is None:
                        offered -= MEMORY_TOOL_NAMES
                    if not decisions_tool_available():
                        offered.discard("decide")
                    if "skill" in offered and make_skill_tool(resolve_work_dir(work_dir)) is None:
                        offered.discard("skill")
                    listed = sorted(offered)
                    if allowed & MCP_AUTH_TOOL_NAMES:
                        listed.append("the tools of every configured MCP server")
                    system_instructions += RESTRICTED_PROFILE_NOTE.format(
                        profile=profile, tools=", ".join(listed),
                    )
                    if not web_tools and allowed & BROWSER_TOOL_NAMES:
                        system_instructions += WEB_TOOLS_OFF_NOTE
                elif self._append_basic_tools and not web_tools:
                    # The settings panel's "Use web tools" is off: the
                    # browser tools are not built (:meth:`_get_tools`), so
                    # the static Web Research rules above must not send the
                    # model after go_to_url() or a curl substitute.
                    system_instructions += WEB_TOOLS_OFF_NOTE
                if self._memory_tools is not None:
                    system_instructions += "\n\n" + self._memory_tools.protocol()
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
                    llm_call_hook=llm_call_hook,
                    tool_call_hook=tool_call_hook,
                )
            finally:
                # No sub-task outlives its caller: a run_agent job the run
                # neither waited for nor killed is stopped here, before the
                # run's terminal status goes out.  The cancels are sent
                # before its bounded join, and a stop's injected interrupt
                # landing in that join must not skip the cleanup below: it
                # is held and re-raised after it when the run was otherwise
                # returning normally (the stop must still be honoured), or
                # dropped when the run is already unwinding on an exception.
                unwinding = sys.exc_info()[1] is not None
                interrupted: BaseException | None = None
                try:
                    if kill_jobs_of(self):
                        logger.info("stopped run_agent jobs still running at the end of the task")
                except BaseException as exc:  # noqa: BLE001 — held, see above
                    logger.warning("interrupted while waiting for cancelled run_agent jobs")
                    interrupted = None if unwinding else exc
                # A worktree the run leaves unmerged awaits the user's
                # decision: marked on disk so no orphan reclaim in another
                # process merges its branch behind the user's back.
                self._mark_unmerged_worktree_preserved()
                self._fold_classifier_usage()
                if self.web_use_tool:
                    self.web_use_tool.close()
                self.web_use_tool = None
                self._memory_tools = None
                self._ask_user_question_callback = None
                self.pre_step_hook = None
                self.tool_call_guard = None
                # The hook is a bound method of this run's UsefulTools;
                # left in place it would keep that instance (its read
                # cache, stream callback, stop event) alive for the next
                # run and be called by a docker run that never installs one.
                self.context_reset_hook = None
                # The run's last word on its spend, always: a sub-task's
                # fold can land on its own thread at any point after the
                # run's last event (between a session's final event and its
                # bank, during the join above, or ahead of the classifier
                # fold), so no snapshot taken here can tell whether that
                # event already carried it.  The persisted row reads the
                # same totals after ``run`` returns.
                self._emit_usage_totals()
                if interrupted is not None:
                    raise interrupted
        except (KISSError, SeaError):
            # A broken SEA method (its ``system_prompt`` or ``tools``
            # run inside the agent) is a setup failure the caller must
            # see as a diagnostic, not a summarised task result.
            raise
        except Exception as exc:
            # The task's failure and the cleanup's alike: the caller
            # gets a result, not a traceback.
            return str(
                yaml.dump(
                    {
                        "success": False,
                        "summary": f"Task failed with error: {exc}",
                    }
                )
            )

    def _begin_task(self, printer: Printer | None) -> None:
        """Clear the way for a new task before anything of it is recorded.

        Binds *printer* (so the retirement below can notify), resets the
        per-run worktree flags, retires a worktree still pending from an
        earlier run of this agent (committed to its own branch and kept
        for review, never merged: the new task starts in the caller's
        ``work_dir`` and ``create_worktree`` makes it a fresh worktree)
        and flushes the warnings that left.  Called by :meth:`run`, and
        before it by ``ChatSorcarAgent.run`` — ahead of the chat state
        the retirement's commit message is generated from
        (:meth:`_commit_context`) being replaced by the new task's; the
        second call then finds nothing to do.

        Args:
            printer: The run's printer, or ``None`` to keep the current one.
        """
        if printer is not None:
            self.set_printer(printer)
        self._worktree_unavailable = None
        self._unisolated_repos = set()
        self._worktree_finish_gate_used = False
        if self._wts:
            self._retire_previous_worktree()
        self._flush_warnings(printer)

    def _drain_pending_user_messages(self, model: Any) -> None:
        """Append any queued follow-up prompts to *model*'s conversation.

        Called once at the top of every model step (wired in via
        :attr:`kiss.core.kiss_agent.KISSAgent.pre_step_hook`).  Drains
        the run's queued follow-up prompts through the printer's
        duck-typed ``drain_pending_user_messages`` bridge (the server
        keeps them on the task's registered agent state, keyed by the
        calling thread's task id) and pushes each entry into *model*'s
        conversation as a ``user`` role message.  Each entry is
        wrapped as ``User says: <message>. Take the message into
        account and finish your task.`` so the model treats it as a
        mid-task steering instruction rather than a bare trajectory
        line.  The bridge empties the queue on every drain so the same
        queued message is never injected twice, and emits a durable
        ``recordOnly`` echo for any message whose live echo could not
        be attributed to a task id at queueing time.

        Args:
            model: The live model whose conversation receives the
                queued user messages.
        """
        drain = getattr(self.printer, "drain_pending_user_messages", None)
        queued: list[str] = drain() if drain is not None else []
        for msg in queued:
            model.add_message_to_conversation(
                "user",
                f"User says: {msg}. "
                "Take the message into account and finish your task.",
            )

    def _block_finish_when_user_message_pending(
        self, name: str, args: dict[str, Any],
    ) -> str | None:
        """Reject ``finish`` while a queued user follow-up is undrained.

        The server accepts ``appendUserMessage`` while a model call is
        in flight, but queued messages are drained only by the
        pre-step hook at the TOP of a step.  Without this guard, a
        prompt queued after the last drain would be silently discarded
        when the in-flight response calls ``finish`` — the user sees
        their follow-up echoed in the UI even though the agent never
        saw it.  Blocking the ``finish`` forces one more step, whose
        pre-step drain injects the queued message.

        Args:
            name: The tool name the model is calling.
            args: The tool call arguments (unused).

        Returns:
            ``None`` to allow the call, or a rejection message when
            ``finish`` was attempted with steering input still queued.
        """
        del args
        if name != "finish":
            return None
        has_pending = getattr(self.printer, "has_pending_user_messages", None)
        if has_pending is None or not has_pending():
            return None
        return (
            "Error: finish rejected — the user sent a new message while "
            "you were working. It will be appended to the conversation "
            "at the start of your next step; take it into account "
            "before finishing."
        )

    def _block_finish_with_live_jobs(self, name: str, args: dict[str, Any]) -> str | None:
        """Reject ``finish`` once while a ``run_agent`` job is still running.

        A job is a sub-task the model started with ``wait="false"`` or
        one whose ``run_agent`` call returned at its ``timeout``; its
        result would be lost and the sub-task killed by :meth:`run`'s
        cleanup if the task ended now.  The first ``finish`` with live
        jobs is rejected with their ids so the model can wait for or
        kill them; a second ``finish`` passes, so a model that means it
        is never trapped.

        Args:
            name: The tool name the model is calling.
            args: The tool call arguments (unused).

        Returns:
            ``None`` to allow the call, or the one-time rejection.
        """
        del args
        if name != "finish" or self._live_jobs_finish_gate_used:
            return None
        live = live_agent_jobs(self)
        if not live:
            return None
        self._live_jobs_finish_gate_used = True
        now = time.monotonic()
        jobs = ", ".join(
            f"{job.job_id} ({job.name}, running {now - job.started:.0f}s)" for job in live
        )
        return (
            f"Error: finish rejected — run_agent jobs are still running: {jobs}. "
            f"agent_job(id, 'wait') collects a job's result and 'kill' stops it; "
            f"finishing again kills every job still running."
        )

    @property
    def _wt(self) -> GitWorktree | None:
        """The pending worktree the work dir is in, else the first one; ``None`` when idle."""
        work_dir = Path(self.work_dir).resolve() if self.work_dir else None
        for wt in self._wts.values():
            if work_dir is not None and _path_inside(work_dir, wt.wt_dir) is not None:
                return wt
        return next(iter(self._wts.values()), None)

    @_wt.setter
    def _wt(self, wt: GitWorktree | None) -> None:
        """Make *wt* the only pending worktree (``None``: none)."""
        self._wts = {} if wt is None else {_repo_key(wt): wt}

    @property
    def worktree(self) -> GitWorktree | None:
        """The first worktree pending merge/discard, or ``None``."""
        return self._wt

    @property
    def worktrees(self) -> list[GitWorktree]:
        """Every worktree pending merge/discard, in creation order."""
        return list(self._wts.values())

    @property
    def _repo_root(self) -> Path | None:
        """Git repo root path, or ``None`` if not in a repo."""
        return self._wt.repo_root if self._wt else None

    @property
    def _wt_branch(self) -> str | None:
        """Branch name of the current/pending worktree task."""
        return self._wt.branch if self._wt else None

    @property
    def _original_branch(self) -> str | None:
        """The branch the user was on when the task started."""
        return self._wt.original_branch if self._wt else None

    @property
    def _wt_pending(self) -> bool:
        """Whether a worktree task is pending merge/discard."""
        return bool(self._wts)

    @property
    def _wt_dir(self) -> Path | None:
        """Worktree directory path."""
        return self._wt.wt_dir if self._wt else None

    @property
    def _wt_work_dir(self) -> Path | None:
        """The task's working directory inside the worktree.

        ``wt_dir`` plus the original work dir's offset from the repo
        root; differs from ``wt_dir`` when the task was launched from a
        subdirectory of the repo.
        """
        return None if self._wt is None else self._wt_work_dir_of(self._wt)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _enter_work_dir(self, work_dir: str) -> None:
        """Point this agent and its live tools at *work_dir*.

        Args:
            work_dir: The directory Bash commands run in and relative
                paths resolve against from now on.
        """
        self.work_dir = work_dir
        for tools in (self._useful_tools, self.web_use_tool):
            if tools is not None:
                tools.work_dir = work_dir

    def _wt_for_repo(self, repo: Path) -> GitWorktree | None:
        """This task's worktree of the repository rooted at *repo*, if any."""
        return self._wts.get(repo.resolve())

    def _work_dir_in_checkout_of(self, repo: Path) -> bool:
        """Whether the work dir is inside *repo*'s checkout (not a worktree).

        True for the repository the task was launched in: creating its
        worktree moves the work dir into it.  False for every other
        repository the task touches, and for a work dir that is itself
        inside a worktree (a sub-agent of a worktree task).

        Args:
            repo: A git repository root.
        """
        work_dir = self.work_dir or ""
        if not work_dir or inside_kiss_worktree(work_dir):
            return False
        return _path_inside(Path(work_dir).resolve(), repo) is not None

    def create_worktree(self, path: str = "") -> str:
        """Create a git worktree for the repository of *path* (default: the work dir).

        Call this once per repository, before the first change (create,
        edit, write, or make a folder) to a Git-tracked file or folder
        of that repository.  The task's edits to the repository then
        land on a private ``kiss/wt-*`` branch checked out under
        ``<repo>/.kiss-worktrees/`` instead of the user's checkout.  For
        the repository containing the work dir the work dir moves into
        the worktree, so relative paths and Bash commands use it from
        then on; for any other repository, edit its files under the
        returned worktree directory (``<repo>/x.py`` becomes
        ``<worktree>/x.py``).  Call ``commit_and_merge`` when the work is
        done: it merges every worktree.  A no-op when the repository
        already has a worktree or *path* is not inside a git repository.

        Args:
            path: A file or directory inside the repository to isolate;
                relative to the work dir.  Empty: the work dir itself.

        Returns:
            The worktree directory and branch (and the new work dir when
            it moved), or the reason no worktree was created (edit the
            files in place then).
        """
        work_dir = self.work_dir or ""
        if self.docker_manager is not None:
            self._worktree_unavailable = "the task runs in a Docker container"
            return (
                "No worktree: the task runs in a Docker container, so edit "
                f"the files in {work_dir} directly."
            )
        target = Path(_absolutize(path, work_dir)).resolve() if path else Path(work_dir).resolve()
        anchor = target
        while not anchor.is_dir():
            anchor = anchor.parent
        repo = GitWorktreeOps.discover_repo(anchor) if work_dir or path else None
        existing = next(
            (wt for wt in self.worktrees if _path_inside(target, wt.wt_dir) is not None),
            self._wt_for_repo(repo) if repo is not None else None,
        )
        if existing is not None:
            if _path_inside(Path(work_dir).resolve(), existing.wt_dir) is not None:
                return (
                    f"A worktree is already active: work dir {self._wt_work_dir_of(existing)} "
                    f"on branch '{existing.branch}'. Keep working there."
                )
            return (
                f"A worktree is already active for {existing.repo_root}: {existing.wt_dir} "
                f"on branch '{existing.branch}'. Edit that repository's files there."
            )
        if inside_kiss_worktree(target):
            return (
                f"No new worktree needed: {target} is already inside the "
                "worktree of the calling task. Edit the files in place; the "
                "calling task commits and merges them."
            )
        if repo is None:
            return (
                f"No worktree: {target if work_dir or path else 'the work dir'} is not "
                "inside a git repository. Edit the files in place, or pass a path "
                "inside the repository to isolate."
            )
        callers = self._isolating_worktree_dir(repo)
        if callers is not None:
            return (
                f"No new worktree needed: the calling task already isolates "
                f"{repo} in the worktree {callers}. Edit that repository's files "
                "there; the calling task commits and merges them."
            )
        moves_work_dir = self._work_dir_in_checkout_of(repo)
        wt_work_dir = self._try_setup_worktree(repo, work_dir if moves_work_dir else str(repo))
        if wt_work_dir is None:
            self._unisolated_repos.add(repo.resolve())
            return (
                f"No worktree could be created for {repo} (detached HEAD, no "
                "commits, or a git failure; see the log). Edit its files in "
                "place."
            )
        wt = self._wt_for_repo(repo)
        assert wt is not None
        if moves_work_dir:
            self._enter_work_dir(str(wt_work_dir))
        self._worktree_finish_gate_used = False
        printer = getattr(self, "printer", None)
        if printer is not None and hasattr(printer, "broadcast"):
            event: dict[str, object] = {
                "type": "worktree_created",
                "worktreeDir": str(wt.wt_dir),
                "branch": wt.branch,
            }
            if moves_work_dir:
                event["worktreeWorkDir"] = str(wt_work_dir)
            self._broadcast_to_watchers(printer, event)
        created = (
            f"Created worktree {wt.wt_dir} on branch '{wt.branch}' "
            f"(from '{wt.original_branch}')."
        )
        if moves_work_dir:
            return (
                f"{created} The work dir is now {wt_work_dir}: relative paths "
                "and Bash commands use it. Call commit_and_merge when the work "
                "is done."
            )
        return (
            f"{created} The work dir is unchanged ({work_dir}): edit the files "
            f"of {repo} under {wt.wt_dir} (absolute paths). Call "
            "commit_and_merge when the work is done."
        )

    @staticmethod
    def _wt_work_dir_of(wt: GitWorktree) -> Path:
        """The task's work directory inside *wt* (its root for a legacy snapshot)."""
        return wt.work_dir or wt.wt_dir

    def commit_and_merge(self) -> str:
        """Commit every worktree's changes and merge them into their original branches.

        For each worktree created by ``create_worktree``, commits every
        change on its task branch (with a generated commit message),
        squash-merges the branch into the branch the task started from,
        deletes the branch and removes the worktree; the work dir
        returns to the user's checkout.  A merge conflict is resolved
        by the merge SEA running as a sub-agent; when it cannot, that
        branch is kept and the manual steps are returned.  Call this
        after the work is complete and verified, before ``finish``.

        Returns:
            What was merged, or why a merge did not complete.
        """
        if not self._wts:
            return (
                "No worktree is active for this task (create_worktree was not "
                "called, or its branches were already merged); nothing to commit "
                "or merge."
            )
        from kiss.agents.sorcar.merge_conflict_resolver import resolve_merge_conflict

        messages = []
        for wt in self.worktrees:
            branch = wt.branch
            message = self._merge_one(wt, conflict_resolver=resolve_merge_conflict)
            merged = self._wt_for_repo(wt.repo_root) is None
            if not merged and not wt.wt_dir.exists():
                # The branch holds the commits but its directory is
                # gone: give the task its working copy back so it can
                # fix the problem (a rejected hook, a conflict the merge
                # agent could not resolve) and retry, instead of editing
                # the user's checkout while the worktree counts as active.
                message += self._reattach_after_failed_merge(wt)
            printer = getattr(self, "printer", None)
            if printer is not None and hasattr(printer, "broadcast"):
                self._broadcast_to_watchers(printer, {
                    "type": "worktree_result",
                    "action": "merge",
                    "branch": branch,
                    "worktreeDir": str(wt.wt_dir),
                    "success": merged,
                    "message": message,
                })
            messages.append(message)
        text = "\n\n".join(messages)
        if not self._wts:
            return f"{text} The work dir is {self.work_dir} again."
        return text

    def _reattach_after_failed_merge(self, wt: GitWorktree) -> str:
        """Check *wt* out again after a failed merge removed its directory.

        Git restores tracked files only, so the task's (possibly empty
        or ignored) work subdirectory and the node_modules links are
        recreated the way the initial setup did them; the work dir
        re-enters the worktree when it had been moved there.

        Args:
            wt: The still-pending worktree whose directory is gone.

        Returns:
            Text telling the agent where to continue, to append to the
            merge message.
        """
        if not GitWorktreeOps.reattach(wt.repo_root, wt.branch, wt.wt_dir):
            return (
                f"\n\nThe worktree could not be checked out again; the work is "
                f"on branch '{wt.branch}' for the user to merge or discard. Make no "
                "further edits and finish."
            )
        wt_work_dir = self._wt_work_dir_of(wt)
        wt_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            GitWorktreeOps.link_node_modules(wt.repo_root, wt.wt_dir)
        except OSError:
            logger.warning("Failed to link node_modules into worktree", exc_info=True)
        if self._work_dir_in_checkout_of(wt.repo_root):
            self._enter_work_dir(str(wt_work_dir))
            return (
                f"\n\nThe worktree is checked out again at {wt.wt_dir}; the work "
                f"dir is {self.work_dir}. Fix the problem there and call "
                "commit_and_merge again."
            )
        return (
            f"\n\nThe worktree is checked out again at {wt.wt_dir}. Fix the "
            "problem there and call commit_and_merge again."
        )

    def _guard_worktree(self, name: str, args: dict[str, Any]) -> str | None:
        """Worktree guardrail on every tool call.

        * ``Write`` / ``Edit`` of a path inside a git repository's
          checkout that is not git-ignored is refused: when the
          repository is isolated in a worktree the work dir is not in
          (another repository's worktree, or one a calling task made)
          the refusal names the path to use inside the worktree;
          when no worktree isolates it the refusal asks for
          ``create_worktree`` of that repository first.  Paths inside
          a worktree, checkout paths of the work dir's own worktree
          (the tools remap them), git-ignored paths, and repositories
          whose worktree setup failed pass.
        * The first ``finish`` while a worktree is unmerged is refused
          once, asking for ``commit_and_merge``; a second ``finish``
          passes so the agent is never trapped.

        Args:
            name: The tool being called.
            args: Its arguments.

        Returns:
            The refusal text, or ``None`` to let the call through.
        """
        if name == "finish":
            if not self._wts or self._worktree_finish_gate_used:
                return None
            self._worktree_finish_gate_used = True
            branches = ", ".join(f"'{wt.branch}'" for wt in self.worktrees)
            noun = "worktree" if len(self._wts) == 1 else "worktrees"
            return (
                f"The {noun} on branch {branches} has not been merged. "
                "Call commit_and_merge() first, then finish again (or finish again "
                "as is to leave the branch for the user to merge or discard)."
            )
        if name not in ("Write", "Edit") or not isinstance(args.get("file_path"), str):
            return None
        if self._worktree_unavailable is not None:
            return None
        # A run without the tool (a WorkerSea, a profile without the
        # edit group's worktree tools) edits the tree in place.
        if "create_worktree" not in self._run_tool_names or self.docker_manager is not None:
            return None
        path = Path(_absolutize(args["file_path"], self.work_dir)).resolve()
        if inside_kiss_worktree(path):
            return None
        anchor = path
        while not anchor.is_dir():
            anchor = anchor.parent
        repo = GitWorktreeOps.discover_repo(anchor)
        if (
            repo is None
            or repo.resolve() in self._unisolated_repos
            or GitWorktreeOps.is_ignored(repo, path)
        ):
            return None
        wt_dir = self._isolating_worktree_dir(repo)
        offset = _path_inside(path, repo)
        if wt_dir is not None and offset is not None:
            if _path_inside(Path(self.work_dir or "").resolve(), wt_dir) is not None:
                # The tools remap a checkout path of the work dir's own
                # worktree into it (``_active_worktree_remap``).
                return None
            return (
                f"Error: {name} of {path} refused: the repository {repo} is "
                f"isolated in the worktree {wt_dir}; {name} {wt_dir / offset} instead."
            )
        call = "create_worktree()" if self._work_dir_in_checkout_of(repo) else (
            f"create_worktree({str(repo)!r})"
        )
        return (
            f"Error: {name} of {path} refused: it is inside the git repository "
            f"{repo} and is not git-ignored. Call {call} first, "
            f"then retry the {name}."
        )

    def _isolating_worktree_dir(self, repo: Path) -> Path | None:
        """The worktree directory holding this task's copy of *repo*, if any.

        A worktree this task created for *repo*, a calling task's
        worktree of *repo* (:meth:`inherit_worktrees`), or the enclosing
        worktree of the work dir when it lies inside a worktree of
        *repo* (a sub-agent of a worktree task started without the
        daemon's inheritance).

        Args:
            repo: A git repository root.
        """
        own = self._wt_for_repo(repo)
        if own is not None:
            return own.wt_dir
        inherited = self._inherited_worktrees.get(repo.resolve())
        if inherited is not None:
            return inherited
        enclosing = kiss_worktree_root(self.work_dir or "")
        if enclosing is not None and enclosing.parent.parent.resolve() == repo.resolve():
            return enclosing
        return None

    def inherit_worktrees(self, worktrees: dict[Path, Path]) -> None:
        """Adopt the calling tasks' worktrees before this agent runs as a sub-agent.

        A sub-agent never creates a worktree for a repository a calling
        task (any ancestor) has already isolated: ``create_worktree``
        points at the ancestor's worktree instead, and the write guard
        maps the repository's checkout paths into it.  The ancestors
        commit and merge those worktrees.

        Args:
            worktrees: Repository root -> worktree directory, as
                :meth:`isolated_repos` of the calling agent returns it.
        """
        self._inherited_worktrees = {
            repo.resolve(): wt_dir for repo, wt_dir in worktrees.items()
        }

    def isolated_repos(self) -> dict[Path, Path]:
        """Repository root -> worktree directory for every repo this task isolates.

        Its own worktrees and the ones it inherited from calling tasks,
        so a sub-agent of a sub-agent still sees the top task's
        worktrees.
        """
        return {
            **self._inherited_worktrees,
            **{repo: wt.wt_dir for repo, wt in self._wts.items()},
        }

    # ------------------------------------------------------------------
    # Worktree creation
    # ------------------------------------------------------------------

    def _live_worktree_branches(self) -> set[str]:
        """Return the set of ``kiss/wt-*`` branches owned by live agents.

        Union of *self*'s current worktree branch (if any) and every
        other live agent's branch as tracked by the server's agent-state
        registry, reached through the printer's duck-typed
        ``live_worktree_branches`` bridge.  Excluded from the orphan
        reclaim in :meth:`_try_setup_worktree` so a concurrent live
        agent's active worktree is never adopted, merged, or removed.
        """
        branches = {wt.branch for wt in self._wts.values()}
        live = getattr(
            getattr(self, "printer", None), "live_worktree_branches", None,
        )
        if live is not None:
            branches.update(live())
        return branches

    def _try_setup_worktree(self, repo: Path, work_dir_str: str) -> Path | None:
        """Create a worktree branch for the current task.

        Side effect: registers the worktree in ``self._wts`` on success.

        Args:
            repo: Git repo root path.
            work_dir_str: The task's current work directory inside *repo*.

        Returns:
            The worktree-relative work directory, or ``None`` when no
            worktree can be created (detached HEAD, no commits, a git
            failure).
        """
        # The whole baseline snapshot — reading the main tree's
        # checked-out branch and copying its dirty state — must be one
        # consistent read of the SHARED main worktree.  ``repo_lock``
        # only excludes threads of this process; a peer process's
        # merge/reclaim/discard stashes, checks out and resets that same
        # tree under the cross-process flock, so the capture takes the
        # flock too (repo_lock -> flock order, like every other taker).
        with repo_lock(repo), _reclaim_process_lock(repo):
            original_branch = GitWorktreeOps.current_branch(repo)
            if original_branch is None:
                logger.warning("Detached HEAD, no worktree for %s", repo)
                return None
            try:
                offset = Path(work_dir_str).resolve().relative_to(repo.resolve())
            except ValueError:  # pragma: no cover
                logger.warning("work_dir not inside repo, no worktree")
                return None
            try:
                GitWorktreeOps.prepare_repo(repo)
                GitWorktreeOps.ensure_scratch_merge_driver(repo)
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning("Failed to update git exclude", exc_info=True)
            # Reclaim before sweep: reclaim may merge and delete orphan
            # branches, and sweep purges leftover config sections whose
            # branches were just removed.
            try:
                GitWorktreeOps.reclaim_orphaned_worktrees(
                    repo, exclude_branches=self._live_worktree_branches(),
                )
                GitWorktreeOps.sweep_orphaned_state(repo)
            except Exception:  # pragma: no cover — unexpected git failure
                logger.warning("Orphan-worktree maintenance failed", exc_info=True)
            branch = GitWorktreeOps.new_task_branch(repo)
            wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
            if not GitWorktreeOps.create(repo, branch, wt_dir):
                # pragma: no cover — git worktree add failure
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None
            if not GitWorktreeOps.save_original_branch(repo, branch, original_branch):
                # pragma: no cover — git config failure
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None
            try:
                dirty_copied = GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            except OSError:
                logger.warning(
                    "Failed to copy dirty state into worktree", exc_info=True,
                )
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                return None
            baseline_commit: str | None = None
            if dirty_copied:
                GitWorktreeOps.stage_all(wt_dir)
                if GitWorktreeOps.commit_staged(
                    wt_dir, "kiss: baseline from dirty state", no_verify=True,
                ):
                    baseline_commit = GitWorktreeOps.head_sha(wt_dir)
                    if baseline_commit:
                        GitWorktreeOps.save_baseline_commit(repo, branch, baseline_commit)
                elif GitWorktreeOps.has_uncommitted_changes(wt_dir):
                    logger.warning("Baseline commit failed in new worktree")
                    GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                    return None
            wt_work_dir = wt_dir / offset
            self._wts[repo.resolve()] = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch=original_branch,
                wt_dir=wt_dir,
                baseline_commit=baseline_commit,
                work_dir=wt_work_dir,
            )
            wt_work_dir.mkdir(parents=True, exist_ok=True)
            try:
                GitWorktreeOps.link_node_modules(repo, wt_dir)
            except OSError:
                logger.warning("Failed to link node_modules into worktree", exc_info=True)
            return wt_work_dir

    def _restore_work_dir_after_teardown(self, wt: GitWorktree) -> None:
        """Point ``work_dir`` back at the parent repository once *wt* is gone.

        The sub-directory offset the task was launched from is kept, so
        a run started in ``<wt>/pkg`` ends in ``<repo>/pkg``.  A
        ``work_dir`` outside *wt* is left alone.

        Args:
            wt: The worktree whose directory was just removed.
        """
        if not self.work_dir:
            return
        try:
            offset = Path(self.work_dir).resolve().relative_to(wt.wt_dir.resolve())
        except ValueError:
            return
        self._enter_work_dir(str(wt.repo_root.resolve() / offset))

    # ------------------------------------------------------------------
    # Commit / merge engine
    # ------------------------------------------------------------------

    def _commit_worktree(self, wt: GitWorktree | None = None) -> bool:
        """Commit any uncommitted changes in a worktree.

        Args:
            wt: The worktree to commit; ``None`` for the first pending one.

        Stages all changes once, generates a commit message from the
        staged diff, then commits the already-staged changes.  Falls
        back to a generic commit message when the LLM-based message
        generator is unavailable.  Emits two best-effort UI
        notifications (``"Generating commit message"``, ``"Committed
        <subject>"``) through :meth:`_broadcast_commit_notification`.

        Returns:
            True if a commit was created, False if nothing to commit.
        """
        if wt is None:
            wt = self._wt
        if wt is None or not wt.wt_dir.exists():
            return False
        commit_run_id = f"autocommit-{time.time_ns()}"
        user_prompt, task_result = self._commit_context()
        return auto_commit_changes(
            wt.wt_dir,
            user_prompt,
            _generate_commit_message,
            notify_fn=functools.partial(
                self._broadcast_commit_notification, commit_run_id,
            ),
            task_result=task_result,
        )

    def _commit_context(self) -> tuple[str | None, str | None]:
        """The task text and result a worktree commit message is generated from.

        ``(None, None)`` here: :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`,
        which keeps the chat's last prompt and result summary, overrides it.
        """
        return None, None

    def _broadcast_commit_notification(
        self, notification_id: str, stage: str, subject: str,
    ) -> None:
        """Broadcast a commit life-cycle notification to the webview.

        Hook used by :func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`.
        Silently no-ops when no printer with a ``broadcast`` method is
        attached, and never raises into the caller.

        Args:
            notification_id: Toast id shared across the stages of ONE
                commit, so the webview updates the toast in place.
            stage: ``"generating"``, ``"committed"`` or ``"failed"``.
            subject: First non-empty line of the commit message (the
                toast body for ``"committed"``).
        """
        printer = getattr(self, "printer", None)
        if printer is None or not hasattr(printer, "broadcast"):
            return
        severity = "info"
        if stage == "generating":
            message = "Generating commit message"
        elif stage == "committed":
            message = f"Committed {subject}" if subject else "Committed"
        elif stage == "failed":
            message = (
                "Commit failed (a pre-commit hook may have "
                "rejected it); the worktree is preserved"
            )
            severity = "warning"
        else:
            return
        event: dict[str, object] = {
            "type": "notification",
            "id": notification_id,
            "severity": severity,
            "message": message,
        }
        if stage == "generating":
            event["sticky"] = True
        try:
            self._broadcast_to_watchers(printer, event)
        except Exception:  # pragma: no cover — best-effort UI hook
            logger.debug("commit notification broadcast failed", exc_info=True)

    def _broadcast_to_watchers(self, printer: Any, event: dict[str, object]) -> None:
        """Broadcast *event* so it reaches this agent's tab.

        The printer's ``broadcast_transient`` primitive resolves every
        tab currently watching the task and stamps one copy per tab,
        falling back to this agent's tab; printers without it get one
        copy stamped with the tab id directly.

        Args:
            printer: A ``broadcast``-capable printer.
            event: The event to route; must not carry ``tabId``.
        """
        transient = getattr(printer, "broadcast_transient", None)
        if callable(transient):
            # ``last_task_id``: the ``task_history`` row of a ChatSorcarAgent.
            task_id = str(getattr(self, "last_task_id", "") or "")
            transient(event, task_id=task_id or None, tab_id=self._tab_id)
            return
        printer.broadcast({**event, "tabId": self._tab_id})

    def _commit_and_clean_worktree(
        self, wt: GitWorktree,
    ) -> tuple[_WorktreeCleanupOutcome, str]:
        """Commit *wt*'s changes, then remove the worktree and prune.

        After the LLM-driven commit, a single-shot retry runs
        :meth:`GitWorktreeOps.commit_all` with a generic message to
        catch a file appearing between the staging and the commit.

        Args:
            wt: The worktree to commit and clean up.

        Returns:
            A ``(outcome, leftover)`` pair.  ``leftover`` is the raw
            ``git status --porcelain`` output when the outcome is
            :attr:`_WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED`
            (e.g. a pre-commit hook rejection), otherwise ``""``.  The
            worktree directory is removed only on
            :attr:`_WorktreeCleanupOutcome.COMMITTED_AND_REMOVED`; the
            preserved outcomes leave it in place so no work is lost.
        """
        if wt.wt_dir.exists():
            self._commit_worktree(wt)
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                GitWorktreeOps.commit_all(wt.wt_dir, "kiss: commit late-arriving changes")
            if GitWorktreeOps.has_uncommitted_changes(wt.wt_dir):
                leftover = GitWorktreeOps.status_porcelain(wt.wt_dir)
                return _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED, leftover
            # ``git add -A`` skips git-ignored task output; without
            # this rescue the removal below would destroy it.  Fails
            # closed: the worktree is preserved when a file could not
            # be landed in the main repo.
            if not _rescue_ignored_files(wt):
                return _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED, ""
        GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
        self._restore_work_dir_after_teardown(wt)
        return _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, ""

    def _finalize_worktree(self, wt: GitWorktree | None = None) -> bool:
        """Commit, remove worktree, prune; log why a worktree was preserved.

        Args:
            wt: The worktree to finalize; ``None`` for the first pending one.

        Returns:
            True if the worktree was cleaned up successfully.  False if
            uncommitted changes remain after both the commit and the
            retry, or ignored output could not be rescued — the
            worktree directory is preserved so no work is lost.
        """
        if wt is None:
            wt = self._wt
        assert wt is not None
        outcome, leftover = self._commit_and_clean_worktree(wt)
        self._last_preserve_outcome = outcome
        if outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED:
            logger.warning(
                "Worktree has uncommitted changes after commit and retry "
                "(a pre-commit hook rejection, a commit failure, or a "
                "concurrent write); preserving %s\ngit status --porcelain:\n%s",
                wt.wt_dir, leftover,
            )
            return False
        if outcome is _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED:
            logger.warning(
                "Git-ignored task output in worktree '%s' could not be "
                "rescued into the main repository; preserving %s",
                wt.branch, wt.wt_dir,
            )
            return False
        return True

    def _do_merge(
        self,
        wt: GitWorktree,
        conflict_resolver: MergeConflictResolver | None = None,
    ) -> tuple[MergeResult, str, str]:
        """Stash, checkout, squash-merge, pop for a worktree branch.

        Serialized under ``repo_lock`` and the cross-process flock so
        concurrent tabs and processes cannot interleave operations on
        the main repository.

        Args:
            wt: The worktree state to merge.
            conflict_resolver: Called as ``(self, wt, user_prompt,
                task_result)`` when the squash merge conflicts, with
                the main tree clean and checked out on the original
                branch, still under the merge locks; it must leave the
                tree clean again unless it returns
                :attr:`MergeResult.SUCCESS` with the merge committed.
                ``_last_merge_resolved_by_agent`` records a success.

        Returns:
            ``(result, stash_warning, cleanup_warning)`` where *result*
            is the merge outcome, *stash_warning* is non-empty if
            stash-pop failed, and *cleanup_warning* is non-empty when
            the merge succeeded but the task branch could not be
            deleted.  A dirty main tree whose ``git stash push`` failed
            returns :attr:`MergeResult.STASH_FAILED` without touching
            the repo.
        """
        stash_warning = ""
        cleanup_warning = ""
        self._last_merge_resolved_by_agent = False
        if wt.original_branch is None:
            return (MergeResult.CHECKOUT_FAILED, "", "")
        with repo_lock(wt.repo_root), _reclaim_process_lock(wt.repo_root):
            try:
                GitWorktreeOps.ensure_scratch_merge_driver(wt.repo_root)
            except Exception:  # pragma: no cover — filesystem permission error
                logger.warning("Failed to install scratch merge driver", exc_info=True)
            did_stash = GitWorktreeOps.stash_if_dirty(wt.repo_root)
            if not did_stash and GitWorktreeOps.has_uncommitted_changes(wt.repo_root):
                return (MergeResult.STASH_FAILED, "", "")
            current = GitWorktreeOps.current_branch(wt.repo_root)
            if current != wt.original_branch:
                ok, err = GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
                if not ok:
                    logger.warning("Cannot checkout '%s': %s", wt.original_branch, err)
                    if did_stash and not GitWorktreeOps.stash_pop(wt.repo_root):
                        stash_warning = (
                            "Your uncommitted changes were stashed "
                            "before the failed checkout and could not "
                            "be auto-restored. Run 'git stash pop' to "
                            "recover them."
                        )
                    return (MergeResult.CHECKOUT_FAILED, stash_warning, "")

            user_prompt, task_result = self._commit_context()
            result = GitWorktreeOps.squash_merge(
                wt.repo_root, wt.branch, wt.baseline_commit,
                user_prompt=user_prompt, task_result=task_result,
            )
            if result == MergeResult.CONFLICT and conflict_resolver is not None:
                try:
                    result = conflict_resolver(self, wt, user_prompt, task_result)
                except BaseException:
                    # The resolver runs an agent for minutes, so a user
                    # stop lands here.  It restored the clean tree
                    # before re-raising; give the user's stashed edits
                    # back now, because nothing below runs.
                    if did_stash and not GitWorktreeOps.stash_pop(wt.repo_root):
                        self._set_warnings(stash=(
                            "Your uncommitted changes were stashed before "
                            "the interrupted merge and could not be "
                            "auto-restored. Run 'git stash pop' to "
                            "recover them."
                        ))
                    raise
                self._last_merge_resolved_by_agent = result == MergeResult.SUCCESS
            if did_stash:
                if result == MergeResult.SUCCESS:
                    if not GitWorktreeOps.stash_pop(wt.repo_root):
                        stash_warning = (
                            "Your uncommitted changes could not be "
                            "auto-restored after merging the worktree "
                            f"('{wt.branch}'). Run 'git stash pop' to "
                            "recover them."
                        )
                        logger.warning("git stash pop failed after merge of '%s'", wt.branch)
                else:
                    stash_warning = (
                        "Your uncommitted changes were saved before "
                        "the merge attempt and are safe in "
                        "'git stash'. After resolving the merge, "
                        "run 'git stash pop' to restore them."
                    )

            if result == MergeResult.SUCCESS and not GitWorktreeOps.delete_branch(
                wt.repo_root, wt.branch
            ):
                logger.warning(
                    "Merged branch '%s' could not be deleted; it still exists in %s",
                    wt.branch, wt.repo_root,
                )
                cleanup_warning = (
                    f"Merged branch '{wt.branch}' could not be deleted "
                    "and still exists. Run "
                    f"'git branch -D {shlex.quote(wt.branch)}' to "
                    "remove it."
                )

        return (result, stash_warning, cleanup_warning)

    def merge(self, conflict_resolver: MergeConflictResolver | None = None) -> str:
        """Merge every pending task branch into its original branch.

        Every step is idempotent — safe to re-run after a crash.
        Commits any uncommitted changes in each worktree before merging
        it.  If a main working tree has uncommitted changes, they are
        stashed before the merge and restored afterward.  A worktree
        whose merge fails stays pending; the others are still merged.

        Args:
            conflict_resolver: Resolves a merge conflict and completes
                the merge (see :meth:`_do_merge`) instead of this method
                returning the manual-resolution steps.

        Returns:
            One success or error message per worktree, blank-line
            separated.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if not self._wts:
            raise RuntimeError("No pending worktree task to merge")
        return "\n\n".join(
            self._merge_one(wt, conflict_resolver) for wt in self.worktrees
        )

    def _merge_one(
        self, wt: GitWorktree, conflict_resolver: MergeConflictResolver | None,
    ) -> str:
        """Merge one pending worktree (see :meth:`merge`); drop it on success.

        Args:
            wt: A pending worktree of this agent.
            conflict_resolver: See :meth:`merge`.

        Returns:
            The success or error message.
        """
        if wt.original_branch is None:
            merge_cmd = _manual_merge_cmd(wt)
            return (
                "Cannot merge: original branch is unknown (likely due to a "
                "crash during setup).  Please specify the target branch "
                "manually:\n"
                f"    git checkout <branch> && {merge_cmd}"
            )

        if not self._finalize_worktree(wt):
            if self._last_preserve_outcome is _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED:
                return (
                    "Cannot merge: " + _rescue_failed_warning(wt)
                    + "\n\nFix the destination in the main repository, "
                    "then retry the merge."
                )
            return (
                f"Cannot merge: committing '{wt.branch}' failed "
                "(a pre-commit hook may have rejected the commit). "
                f"The worktree is preserved at: {wt.wt_dir}\n\n"
                "Fix the issue, then commit manually:\n"
                f"    cd {shlex.quote(str(wt.wt_dir))}\n"
                "    git add -A && git commit -m 'agent work'\n\n"
                "Then retry the merge."
            )

        result, stash_warning, cleanup_warning = self._do_merge(
            wt, conflict_resolver=conflict_resolver,
        )
        stash_suffix = ""
        if self._last_merge_resolved_by_agent:
            stash_suffix = " The merge agent resolved the conflicts."
        if stash_warning:
            stash_suffix += "\n\n⚠️  " + stash_warning
        if cleanup_warning:
            stash_suffix += "\n\n⚠️  " + cleanup_warning

        if result == MergeResult.CHECKOUT_FAILED:
            return (
                f"Cannot checkout '{wt.original_branch}'.\n"
                "Fix the issue and retry the merge, or discard the branch."
                + stash_suffix
            )

        if result == MergeResult.STASH_FAILED:
            return (
                "Cannot merge: your uncommitted changes in the main "
                "repository could not be stashed (git stash push "
                "failed), so the merge was aborted to avoid mixing "
                "them into the merge commit.\n"
                "Commit or clean up the changes, then retry the merge, "
                "or discard the branch." + stash_suffix
            )

        if result == MergeResult.SUCCESS:
            self._wts.pop(_repo_key(wt), None)
            return f"Successfully merged branch '{wt.branch}'." + stash_suffix

        stash_step = ""
        if stash_warning:
            stash_step = "    git stash pop  # restore your uncommitted changes\n"

        if result == MergeResult.MERGE_FAILED:
            return (
                f"Merge of '{wt.branch}' applied cleanly but the commit "
                "failed (a pre-commit hook may have rejected it). "
                "The branch is kept — retry manually:\n"
                + _merge_fix_steps(wt, _PRECOMMIT_FIX_LINES)
                + "\n" + stash_step + "\nOr discard the branch." + stash_suffix
            )

        return (
            "Merge conflict detected.  Resolve manually:\n"
            + _merge_fix_steps(
                wt,
                "    # resolve conflicts in your editor\n"
                "    git add .\n"
                "    git commit\n",
            )
            + "\n" + stash_step + "\nOr discard the branch."
        )

    def leave_as_is(self) -> str:
        """Detach from the pending worktrees, leaving them untouched on disk.

        Serves the "Do nothing" button of the post-task worktree bar:
        each branch, its worktree directory, and any uncommitted
        changes inside it are left exactly as the task ended.  A
        preserve-for-review marker is persisted on every branch so
        :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` in a future
        process also refuses to silently publish it.

        Returns:
            Messages telling the user where the untouched worktrees are.

        Raises:
            RuntimeError: If no worktree task is pending, or if a
                preserve marker could not be written (that worktree
                stays pending and the bar keeps its controls).
        """
        if not self._wts:
            raise RuntimeError("No pending worktree task to leave as is")
        messages = []
        for wt in self.worktrees:
            if not GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch):
                raise RuntimeError(
                    f"Could not record the keep decision for branch "
                    f"'{wt.branch}' (writing the preserve marker to git "
                    "config failed); the worktree is still pending. Retry, "
                    "or merge/discard it instead."
                )
            self._wts.pop(_repo_key(wt), None)
            messages.append(
                f"Left branch '{wt.branch}' and its worktree at "
                f"{wt.wt_dir} as they are. Merge or discard them "
                "manually when you are ready."
            )
        return "\n\n".join(messages)

    def discard(self, *, rescue_ignored: bool = False) -> str:
        """Throw away every pending task branch and worktree, checkout original.

        Every step is idempotent — safe to call multiple times.

        Args:
            rescue_ignored: When True, git-ignored files the task
                created in a worktree are copied into its main
                repository (never overwriting existing files) before
                the directory is removed.  The automatic discard of a
                worktree without changes passes True, because the
                changed-files probe cannot see ignored files.

        Returns:
            One confirmation message per worktree (including a warning
            if checkout to the original branch failed), or a "Discard
            deferred" message when ignored output could not be rescued.

        Raises:
            RuntimeError: If no worktree task is pending.
        """
        if not self._wts:
            raise RuntimeError("No pending worktree task to discard")
        return "\n\n".join(
            self._discard_one(wt, rescue_ignored) for wt in self.worktrees
        )

    def _discard_one(self, wt: GitWorktree, rescue_ignored: bool) -> str:
        """Discard one pending worktree (see :meth:`discard`).

        Args:
            wt: A pending worktree of this agent.
            rescue_ignored: See :meth:`discard`.

        Returns:
            The confirmation or "Discard deferred" message.
        """
        checkout_warning = ""
        delete_warning = ""
        with repo_lock(wt.repo_root), _reclaim_process_lock(wt.repo_root):
            if rescue_ignored and wt.wt_dir.exists() and not _rescue_ignored_files(wt):
                logger.warning(
                    "Deferring discard of worktree '%s': its "
                    "git-ignored output could not be rescued",
                    wt.branch,
                )
                return (
                    f"Discard deferred: git-ignored files created "
                    f"by the task in branch '{wt.branch}' could not "
                    f"be copied into the main repository, so the "
                    f"worktree at {wt.wt_dir} was kept — it holds "
                    "the only copy of those files."
                )
            GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)
            self._restore_work_dir_after_teardown(wt)
            if wt.original_branch:
                ok, err = GitWorktreeOps.checkout(wt.repo_root, wt.original_branch)
                if not ok:
                    checkout_warning = f"\n⚠️  Could not checkout '{wt.original_branch}': {err}"
            if not GitWorktreeOps.delete_branch(wt.repo_root, wt.branch):
                delete_warning = (
                    f"\n⚠️  Branch '{wt.branch}' could not be deleted "
                    "and still exists.  Switch to a different branch "
                    f"(e.g. 'git checkout <other>') and run "
                    f"'git branch -D {shlex.quote(wt.branch)}' to remove it."
                )
        self._wts.pop(_repo_key(wt), None)
        if delete_warning:
            return f"Partially discarded branch '{wt.branch}'.{checkout_warning}{delete_warning}"
        return f"Discarded branch '{wt.branch}'.{checkout_warning}"

    # ------------------------------------------------------------------
    # Pending worktrees of earlier runs
    # ------------------------------------------------------------------

    def _preserve_pending_worktree_for_review(self) -> bool:
        """Commit pending worktree changes onto the branch, no merge.

        Called before a new task, a new chat, or a tab disposal when a
        worktree of an earlier run is still pending: the partial work
        is committed to its ``kiss/wt-*`` branch (recoverable via ``git
        checkout <branch>``), the worktree directory is removed and
        pruned, and the branch is never squash-merged into the user's
        original branch behind their back.  When the commit does not
        happen (a hook refusing it) the directory is kept instead,
        because it is then the only copy of the work.

        Records the cleanup outcome in ``_last_preserve_outcome``.

        Returns:
            True when a pending worktree was preserved (or had no
            uncommitted work and was just cleaned up).  False when
            there was nothing to do.  A preserved worktree whose keep
            decision could not be made durable (:meth:`_keep_for_review`)
            still returns True but stays pending.
        """
        self._last_preserve_outcome = None
        if not self._wts:
            return False
        # A stale warning from an earlier run must not hide this one's.
        self._set_warnings(merge="")
        for wt in self.worktrees:
            self._preserve_one_for_review(wt)
        return True

    def _preserve_one_for_review(self, wt: GitWorktree) -> None:
        """Preserve one pending worktree (see :meth:`_preserve_pending_worktree_for_review`).

        Args:
            wt: A pending worktree of this agent.
        """
        outcome, leftover = self._commit_and_clean_worktree(wt)
        self._last_preserve_outcome = outcome
        if outcome is _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED:
            logger.warning(
                "Worktree '%s' has uncommitted changes after "
                "preserve-for-review (likely a pre-commit hook "
                "rejection); preserving worktree directory %s\n"
                "git status --porcelain:\n%s",
                wt.branch, wt.wt_dir, leftover,
            )
            self.add_warning(
                f"Branch '{wt.branch}' still has uncommitted changes — the "
                f"commit was refused, most likely by a pre-commit hook. Its "
                f"worktree directory {wt.wt_dir} was kept so the work is not "
                f"lost; recover it there. git status --porcelain:\n{leftover}"
            )
        elif outcome is _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED:
            logger.warning(
                "Git-ignored output of worktree '%s' could not be "
                "rescued; preserving worktree directory %s",
                wt.branch, wt.wt_dir,
            )
            self.add_warning(_rescue_failed_warning(wt))
        else:
            self.add_warning(
                f"The unmerged work of the previous task was committed to "
                f"branch '{wt.branch}' (not merged into "
                f"'{wt.original_branch}'). Merge or delete it when you are "
                "ready."
            )
        if not self._keep_for_review(wt):
            # The worktree is preserved on disk, but the keep decision
            # is not durable yet: keep the claim so a later retry (or an
            # explicit merge/discard) can still make it so.
            return
        self._wts.pop(_repo_key(wt), None)

    def _keep_for_review(self, wt: GitWorktree) -> bool:
        """Make the "keep *wt* for manual review" decision durable.

        The ``kiss-preserve`` marker is the ONLY thing that stops
        :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` — running in
        a fresh process after this one has exited — from squash-merging
        and deleting the parked work, so it fails closed: when the
        ``git config`` write fails the caller must keep ``self._wt``
        and a warning tells the user why the worktree is still pending.

        Args:
            wt: The worktree being kept for review.

        Returns:
            True when the marker is on disk and the in-memory claim may
            be dropped; False when it is not and the claim must stay.
        """
        if GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch):
            return True
        logger.warning(
            "Could not record the keep-for-review marker for branch '%s'; "
            "keeping its worktree %s pending on this agent",
            wt.branch, wt.wt_dir,
        )
        self.add_warning(
            f"The keep decision for branch '{wt.branch}' could not be "
            "recorded in git config, so its worktree is still pending on "
            "this tab: merge, discard, or leave it as is explicitly once "
            "git config is writable again."
        )
        return False

    def _retire_previous_worktree(self) -> None:
        """Clear the way for a new task without publishing the old one.

        A new task must not inherit the previous task's branch, and the
        user typing a new prompt is not a decision about the old
        task's unmerged work: the pending worktree is committed to its
        own ``kiss/wt-*`` branch and kept for review
        (:meth:`_preserve_pending_worktree_for_review`), never merged
        into the user's branch behind their back.

        When the retire kept the worktree but could not make that
        decision durable (see :meth:`_keep_for_review`), ``self._wt``
        is still set on return.
        """
        self._preserve_pending_worktree_for_review()

    def retire_for_disposal(self) -> bool:
        """Retire the pending worktree because its tab is going away.

        Returns:
            True when this agent no longer holds a worktree claim and
            the caller may drop it; False when the claim is retained
            (the keep decision could not be made durable) and the
            caller must keep this agent reachable so a later disposal
            attempt can retry.
        """
        self._retire_previous_worktree()
        return not self._wts

    def new_chat(self) -> None:
        """Start afresh (VS Code's 'Clear'), preserving any pending worktree.

        The warning the preserve sets is flushed to the attached printer
        now: if the user never runs another task on this agent, no later
        :meth:`run` would flush it.
        """
        self._retire_previous_worktree()
        self._flush_warnings(getattr(self, "printer", None))

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _set_warnings(self, stash: str | None = None, merge: str | None = None) -> None:
        """Set pending warning attribute(s) under ``_warning_lock``.

        Args:
            stash: New ``_stash_pop_warning`` value, or ``None`` to
                leave it unchanged.
            merge: New ``_merge_conflict_warning`` value, or ``None`` to
                leave it unchanged.
        """
        with self._warning_lock:
            if stash is not None:
                self._stash_pop_warning = stash
            if merge is not None:
                self._merge_conflict_warning = merge

    def add_warning(self, text: str, *, prepend: bool = False) -> None:
        """Combine *text* with the pending merge warning atomically.

        Args:
            text: The warning to add.  Joined to the pending warning
                with a newline; becomes the whole warning when none is
                pending.
            prepend: Put *text* before the pending warning instead of
                after it.
        """
        with self._warning_lock:
            current = self._merge_conflict_warning
            if not current:
                combined = text
            elif prepend:
                combined = f"{text}\n{current}"
            else:
                combined = f"{current}\n{text}"
            self._merge_conflict_warning = combined

    def _flush_warnings(self, printer: Any) -> None:
        """Broadcast and clear any pending stash/merge warnings.

        The take-and-clear of both warning attributes happens
        atomically under ``_warning_lock`` before any broadcast, so
        two concurrent flushes never broadcast the same warning twice
        and a warning set concurrently survives for the next flush.
        A broadcast that fails puts the warning back.

        Args:
            printer: An object with a ``broadcast(event)`` method, or
                any other value (ignored when ``broadcast`` is absent).
        """
        if printer is None or not hasattr(printer, "broadcast"):
            return
        with self._warning_lock:
            stash_warning = self._stash_pop_warning
            self._stash_pop_warning = None
            merge_warning = self._merge_conflict_warning
            self._merge_conflict_warning = None
        if stash_warning:
            try:
                self._broadcast_to_watchers(
                    printer, {"type": "warning", "message": stash_warning},
                )
            except Exception:
                logger.debug("stash warning broadcast failed", exc_info=True)
                with self._warning_lock:
                    current = self._stash_pop_warning
                    self._stash_pop_warning = (
                        f"{stash_warning}\n{current}" if current else stash_warning
                    )
        if merge_warning:
            try:
                self._broadcast_to_watchers(
                    printer, {"type": "warning", "message": merge_warning},
                )
            except Exception:
                logger.debug("merge warning broadcast failed", exc_info=True)
                self.add_warning(merge_warning, prepend=True)

    def _mark_unmerged_worktree_preserved(self) -> None:
        """Record on disk that the pending worktree awaits the user's decision.

        Called when a run ends with ``create_worktree``'s branch still
        unmerged; the marker keeps
        :meth:`GitWorktreeOps.reclaim_orphaned_worktrees` of any later
        process from committing and squash-merging the branch.  The
        write is serialised with the other git-config transactions on
        the repo (``repo_lock`` then the cross-process reclaim lock)
        and retried a few times, since a concurrent ``git`` command
        can hold ``.git/config.lock`` briefly; when it still fails only
        the in-memory claim protects the branch within this process.
        """
        for wt in self.worktrees:
            marked = False
            for _ in range(_PRESERVE_MARKER_ATTEMPTS):
                # One ``git config`` per attempt under the locks; the
                # retry delay is slept with both released, so the other
                # takers (this process's merges and setups, peer daemons
                # waiting on the flock) are not parked for it.
                with repo_lock(wt.repo_root), _reclaim_process_lock(wt.repo_root):
                    marked = GitWorktreeOps.save_preserve_marker(wt.repo_root, wt.branch)
                if marked:
                    break
                time.sleep(_PRESERVE_MARKER_RETRY_DELAY)
            if not marked:
                logger.warning(
                    "Could not mark branch '%s' as preserved; a reclaim in another "
                    "process may merge it", wt.branch,
                )

    def _guard_tool_call(self, name: str, args: dict[str, Any]) -> str | None:
        """The run's tool-call guard: every reason a call is rejected, first one wins."""
        return (
            self._block_finish_when_user_message_pending(name, args)
            or self._block_finish_with_live_jobs(name, args)
            or self._guard_worktree(name, args)
        )


def _budget_arg(text: str) -> float:
    """Parse and validate a ``--max-budget`` command-line value.

    Plain ``float`` would accept ``nan`` and infinities, and the budget
    checks compare with ``>=`` / ``<= 0`` — a NaN cap makes both
    comparisons false and silently disables budget enforcement, so
    non-finite and non-positive values are rejected here.

    Args:
        text: The raw command-line value.

    Returns:
        The budget as a positive finite float.

    Raises:
        argparse.ArgumentTypeError: If *text* is not a number or is not
            a positive finite value.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid budget value: {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"budget must be a positive finite number, got {text!r}"
        )
    return value


def _ask_user_in_terminal(question: str) -> str:
    """Print *question* on the terminal and return the user's typed reply.

    Used as the ``ask_user_question_callback`` of :func:`main` so the
    agent's ``ask_user_question`` tool works when the agent runs from a
    shell instead of the kiss-web UI.

    Args:
        question: The question the agent wants the user to answer.

    Returns:
        The line the user typed, or an empty string on end-of-file.
    """
    print(f"\n{question}")
    try:
        return input("> ")
    except EOFError:
        return ""


def main() -> None:
    """Run a :class:`SorcarAgent` on a task given on the command line.

    Installed as the ``sorcar`` console script.  The task comes from
    exactly one of two required, mutually exclusive options: ``-t
    TASK`` runs the given string, and ``-f FILE`` runs the file's
    content as the task.  The agent works in ``$KISS_WORKDIR`` —
    exported by the ``~/.local/bin/sorcar`` wrapper the VS Code
    extension installs, so the agent acts on the directory the user
    invoked ``sorcar`` from — falling back to the current directory.
    Exits with status 0 when the agent reports success and 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="sorcar",
        description="Run the KISS SorcarAgent on a task.",
        epilog='example: sorcar -t "Summarize README.md"',
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-t",
        "--task",
        default=None,
        help="the task for the agent",
    )
    source.add_argument(
        "-f",
        "--file",
        default=None,
        help="file whose content is used as the task",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="LLM model name (default: best model for the configured API keys)",
    )
    parser.add_argument(
        "-b",
        "--max-budget",
        type=_budget_arg,
        default=None,
        help="maximum budget in USD for the task",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="directory the agent works in (default: $KISS_WORKDIR or the"
        " current directory)",
    )
    args = parser.parse_args()

    if args.file is not None:
        try:
            task = Path(args.file).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            # UnicodeError too: a non-UTF-8 file must surface as the
            # same status-2 usage error as an unreadable one, not as an
            # uncaught UnicodeDecodeError traceback with exit status 1.
            parser.error(f"cannot read task file {args.file!r}: {exc}")
        if not task:
            parser.error(f"task file {args.file!r} is empty")
    else:
        task = args.task.strip()
        if not task:
            parser.error(
                'task must not be empty, e.g.: sorcar -t "Summarize README.md"'
            )

    model_name = args.model or get_default_model()
    if model_name == "No model":
        parser.exit(
            1,
            "sorcar: no model available — set at least one API key"
            " (e.g. ANTHROPIC_API_KEY) in the environment\n",
        )

    work_dir = args.work_dir or os.environ.get("KISS_WORKDIR") or os.getcwd()
    # Interactive terminals get the verbose console printer, which
    # already displays the formatted result at the end of the run —
    # printing the raw YAML again would show it twice.  Piped/redirected
    # stdout gets exactly the raw YAML result and nothing else.
    verbose = sys.stdout.isatty()
    # The daemon's agent class (imported here: its module imports this
    # one), minus the chat history: ``_skip_persistence`` keeps the
    # run's result out of ``history.db``.
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    agent = ChatSorcarAgent("Sorcar CLI")
    result = agent.run(
        model_name=model_name,
        prompt_template=task,
        work_dir=work_dir,
        max_budget=args.max_budget,
        verbose=verbose,
        ask_user_question_callback=_ask_user_in_terminal,
        _skip_persistence=True,
    )
    if not verbose:
        print(result)
    for wt in agent.worktrees:
        sys.stderr.write(
            f"sorcar: the task left its work unmerged on branch '{wt.branch}' "
            f"(worktree {wt.wt_dir}); merge or delete it when you are ready.\n"
        )
    try:
        success = bool(yaml.safe_load(result).get("success"))
    except Exception:
        success = False
    raise SystemExit(0 if success else 1)
