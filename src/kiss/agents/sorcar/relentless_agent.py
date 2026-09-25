# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Base relentless agent with smart continuation for long tasks."""

from __future__ import annotations

import getpass
import html
import logging
import os
import platform
import re
import socket
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, NoReturn, cast
from uuid import uuid4

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


class _UsageEvent(NamedTuple):
    """One immutable, append-once record in a task's usage ledger.

    The whole accounting state lives in an APPEND-ONLY list of these
    records (see :attr:`_UsageLedger.records`): committing a
    transaction is one ``list.append`` — a single atomic C call, so an
    asynchronously injected exception (``PyThreadState_SetAsyncExc``,
    used by the server's stop watchdog) can only land before or after
    the append, never inside it.  The list is never swapped, cut or
    reordered within its epoch (round-7 hardening), so THE APPEND IS
    THE WHOLE COMMIT: the instant it returns, the record is visible to
    every later reader, forever — there is no confirmation step whose
    interruption could strand a committed record, and no compaction
    window in which a committed record is invisible.

    Deduplication identity (``source``, ``seq``):

    * ``source=None`` marks a ONE-SHOT adjustment (a fan-out totals
      bank, a TTS synthesis bank, a property-setter overwrite delta).
      Its producer appends it at most once, so it needs no identity:
      every keyless record counts, and compaction folds it without
      retaining anything.
    * A RETRYABLE transaction carries its producer's stable ``source``
      plus a per-source sequence number that is MONOTONIC in append
      order: a session bank is ``(_session_key(executor), 0)``, an
      abandoned-child reclaim is ``("reclaim:<txn>", generation)``,
      a classifier fold is ``("classifier:<invocation>", 0)``.  A
      retry after an injected stop re-appends the SAME ``(source,
      seq)`` with the same values, and readers count each ``(source,
      seq)`` once (first record wins) while summing
      (:func:`_ledger_totals`).  The monotonicity contract — a
      producer never appends ``(source, seq)`` after it has appended
      ``(source, seq')`` with ``seq' > seq``, except as an identical
      retry of an already-appended record — is what lets the folded
      dedup state be one MAX seq per source (:class:`_SeenMap`,
      O(#producers)) instead of one retained key per event.
    """

    source: str | None
    seq: int
    budget: float
    tokens: int
    steps: int


class _SeenMap:
    """Immutable max-``seq``-per-``source`` map for folded ledger records.

    Compaction must remember, for every retryable source it has ever
    folded, the highest sequence number folded — so a late retry
    duplicate of a folded record still deduplicates.  Round 6 showed
    that keeping this as one flat ``frozenset`` of every historical
    key made compaction quadratic (each fold copied the whole set).
    This map instead stores the entries as a tuple of plain dict
    LEVELS, newest last:

    * :meth:`including` adds a fold's newly-seen sources as one new
      level and then coalesces adjacent levels while the older of the
      two holds at most twice the entries of the newer — the classic
      geometric-merge discipline, which keeps the level count
      O(log #sources) and bounds the TOTAL merge work over an epoch by
      O(#sources * log #sources).  No fold ever copies the whole
      historical state again.
    * :meth:`get` scans levels newest-first; per-source seqs are
      monotonic and newer levels hold newer folds, so the first hit is
      the maximum, and merges let the newer level's entries win.

    Instances are immutable after construction (levels are private to
    the compaction that built them until published inside a
    :class:`_LedgerView`), so readers on any thread can consult a
    published map without locks.
    """

    __slots__ = ("levels", "size")

    def __init__(self, levels: tuple[dict[str, int], ...] = ()) -> None:
        """Wrap *levels* (newest last); each dict is owned exclusively.

        Args:
            levels: Max-seq dicts, oldest first.  Callers hand over
                ownership: the dicts are never mutated afterwards.
        """
        self.levels = levels
        self.size = sum(len(level) for level in levels)

    def get(self, source: str) -> int | None:
        """Return the highest folded seq for *source*, or ``None``.

        Args:
            source: The producer's stable transaction source.

        Returns:
            The maximum folded sequence number, or ``None`` when no
            record of *source* has been folded.
        """
        for level in reversed(self.levels):
            seq = level.get(source)
            if seq is not None:
                return seq
        return None

    def including(self, delta: dict[str, int]) -> _SeenMap:
        """Return a new map that also covers *delta* (newest entries).

        Args:
            delta: Newly folded ``source -> max seq`` entries; the
                caller hands over ownership of the dict.

        Returns:
            ``self`` when *delta* is empty, else a new immutable map.
        """
        if not delta:
            return self
        levels = [*self.levels, delta]
        while len(levels) >= 2 and len(levels[-2]) <= 2 * len(levels[-1]):
            newer = levels.pop()
            older = levels.pop()
            merged = dict(older)
            # Newer entries win: per-source seqs are monotonic, so the
            # newer level's seq for a shared source is the maximum.
            merged.update(newer)
            levels.append(merged)
        return _SeenMap(tuple(levels))


class _LedgerView(NamedTuple):
    """The immutable folded state of one usage-ledger epoch.

    Compaction (:meth:`RelentlessAgent._maybe_compact`) sums
    ``records[:fold_index]`` into one of these and publishes it with a
    SINGLE ``STORE_ATTR`` — the only mutation compaction performs, so
    an asynchronously injected stop anywhere in compaction leaves
    either the old complete view or the new complete view, never a
    torn or record-losing intermediate state.  ``seen`` carries the
    max folded seq per retryable source (see :class:`_SeenMap`) so a
    retry duplicate of a folded record still counts once.
    """

    budget: float
    tokens: int
    steps: int
    fold_index: int
    seen: _SeenMap


_EMPTY_VIEW = _LedgerView(0.0, 0, 0, 0, _SeenMap())

#: Compaction threshold: when an epoch's unfolded suffix
#: (``records[view.fold_index:]``) reaches this length, the next
#: commit folds it into the view.  Reads are therefore O(threshold)
#: — never O(records ever appended) — and repeated adjustments (one
#: ledger append + one totals read each) cost O(n * threshold)
#: instead of quadratic O(n^2).
_COMPACTION_THRESHOLD = 200


class _UsageLedger:
    """One accounting epoch: an append-only record list plus a folded view.

    The agent's ``_usage_ledger`` attribute always points at one of
    these; :meth:`RelentlessAgent.reset_usage` swaps in a fresh
    instance with a single ``STORE_ATTR`` (the epoch boundary).  The
    instance itself is the EPOCH TOKEN: abandoned-subagent items
    record it at registration and their reclaims commit INTO that
    exact object (:meth:`RelentlessAgent._attribute_usage`'s *epoch*
    argument), so a reset racing a reclaim can never route a prior
    epoch's spend into the current one.

    Mutation protocol (lock-free for writers, single-store for
    compaction):

    * A writer commits by appending one immutable :class:`_UsageEvent`
      to :attr:`records` — one atomic ``list.append``, which IS the
      whole commit.  :attr:`records` is never replaced, truncated or
      reordered within the epoch, so a committed record can never
      become invisible and a writer can never append into a discarded
      list.
    * Compaction publishes one new :class:`_LedgerView` — a single
      ``STORE_ATTR`` — and touches nothing else.  A reader loads
      ``view`` once and sums ``records[view.fold_index:]`` on top of
      it; the view's totals cover exactly ``records[:fold_index]``, a
      fixed prefix of an append-only list, so EVERY interleaving of
      reads, commits, compactions and injected stops observes exact
      totals.
    * Concurrent compactions are serialized by a non-blocking
      ``compaction_lock.acquire(False)``: a loser simply skips (a
      later commit retries).  The lock is never blocked on, so no
      unwind order can deadlock; if an injected stop leaks it, this
      epoch merely stops compacting (reads degrade to O(records),
      still exact) and the next epoch starts fresh.

    Memory: the epoch retains every record it ever banked (a small
    immutable tuple each) plus one max-seq entry per retryable
    producer.  Both are linear in real work done — a record per
    committed transaction, a ``seen`` entry per session / reclaim
    transaction / classifier invocation — and the epoch is dropped
    whole at the next ``reset_usage()``.  Read time stays
    O(:data:`_COMPACTION_THRESHOLD`) and cumulative compaction time is
    O(n log n) in the number of events (see :class:`_SeenMap`), never
    the round-6 quadratic full-key-set copy.
    """

    __slots__ = ("view", "records", "compaction_lock")

    def __init__(self) -> None:
        """Create an empty epoch."""
        self.view: _LedgerView = _EMPTY_VIEW
        self.records: list[_UsageEvent] = []
        self.compaction_lock = threading.Lock()


def _session_key(agent: Base) -> str:
    """Return the stable, process-unique banking key for *agent*.

    Assigned on first use through ``dict.setdefault`` — one atomic C
    call — so a banking retry (``perform_task``'s ``except
    BaseException`` re-bank after an injected stop) and a concurrent
    server-thread reclaim always observe the SAME key for one session,
    and their duplicate ledger records deduplicate on read.  A
    ``uuid4`` hex never collides across executors, unlike ``id()``,
    which the allocator reuses after garbage collection; the key also
    stays valid after the executor is collected, so banked executors
    are never pinned in memory.
    """
    key = agent.__dict__.get("_usage_session_key")
    if key is None:
        key = agent.__dict__.setdefault("_usage_session_key", uuid4().hex)
    return str(key)


def _ledger_totals(ledger: _UsageLedger) -> tuple[float, int, int]:
    """Sum one ledger epoch into ``(budget, tokens, steps)``.

    Loads ``ledger.view`` ONCE, then slices the unfolded suffix
    ``records[view.fold_index:]`` (one atomic C call) and sums it on
    top of the view's totals.  The view's totals cover EXACTLY
    ``records[:fold_index]``, a fixed prefix of a list that is only
    ever appended to — never swapped, cut or reordered — so the pair
    is coherent under every interleaving: a concurrent compaction
    publishing a newer view does not disturb this read (it uses the
    view it loaded), and a concurrent append only extends the suffix
    with complete transactions.

    Duplicate retryable records — retries re-appending the same
    ``(source, seq)`` — count once: a suffix record is skipped when
    its seq is already covered by the view's folded max
    (``view.seen``) or by an earlier suffix record of the same source
    (per-source seqs are monotonic in append order, so a max
    comparison is a membership test).  Keyless one-shot records all
    count; their producers append them at most once.

    Cost: O(len(suffix)) time per read; compaction keeps that below
    roughly :data:`_COMPACTION_THRESHOLD` (plus whatever raced in
    since the last fold), so a read never scales with the number of
    records EVER appended to the epoch.
    """
    view = ledger.view
    budget = view.budget
    tokens = view.tokens
    steps = view.steps
    seen = view.seen
    local: dict[str, int] = {}
    for event in ledger.records[view.fold_index:]:
        source = event.source
        if source is not None:
            folded = seen.get(source)
            if folded is not None and event.seq <= folded:
                continue
            prev = local.get(source)
            if prev is not None and event.seq <= prev:
                continue
            local[source] = event.seq
        budget += event.budget
        tokens += event.tokens
        steps += event.steps
    return budget, tokens, steps

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
{work_dir_line}- Current process PID: {current_pid} — NEVER kill this process.
"""

#: The ``IMPORTANT_INSTRUCTIONS`` work-dir line.  A container run from an
#: image bind-mounts ``work_dir`` at the same path and starts the
#: container there, so the line holds inside the container too.  It is
#: omitted only when the tools run in a container that does not mount
#: ``work_dir`` (an attached ``container:<id>`` owned by the caller):
#: naming the host path would then point the model at files it cannot reach.
WORK_DIR_LINE = "- Work dir: {work_dir}\n"

TASK_SETTINGS_HEADER = "\n# Task Settings\n"

#: Consecutive continuation sessions that made no progress — no tool
#: call other than ``finish``, or a summary identical to the previous
#: session's — after which :meth:`RelentlessAgent.perform_task` stops
#: instead of spending the remaining sub-sessions on the same stall.
MAX_ZERO_PROGRESS_SESSIONS = 2

#: Budget cap (USD) a run falls back to when the caller states none.
DEFAULT_MAX_BUDGET = 200.0

#: Model a run falls back to when the caller states none.
DEFAULT_MODEL_NAME = "claude-opus-4-6"

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


# The usage-info block ``KISSAgent._execute_step`` appends to every
# model-role trajectory message; noise in a partial result.  Only the
# LAST fenced block is the framework's: the model's own narration may
# contain ```text fences too.
_USAGE_BLOCK_RE = re.compile(r"\n```text\n(?:(?!```text\n).)*```\n?$", re.DOTALL)

# How much of the executor's trajectory a partial result quotes.
PARTIAL_RESULT_MAX_STEPS = 8
PARTIAL_RESULT_MAX_CHARS_PER_STEP = 600


def _partial_result_html(
    executor: KISSAgent | None,
    exc: BaseException,
    budget_used: float,
    max_budget: float,
    total_steps: int,
) -> str:
    """Describe the work *executor* did before *exc* ended its run, as HTML.

    Used when a sub-agent runs out of budget: instead of the bare
    ``Task failed`` the parent used to get — losing every step the
    child took — the parent receives the child's own account of its
    progress: the model's most recent trajectory messages (its
    narration plus the tool calls it made), oldest first.

    Args:
        executor: The session that was running when the budget ran out,
            or ``None`` when it ran out between sessions.
        exc: The budget error that ended the run.
        budget_used: The sub-agent's cumulative spend in USD (all
            sessions, nested sub-agents included).
        max_budget: The sub-agent's budget cap in USD.
        total_steps: The sub-agent's cumulative step count.

    Returns:
        An HTML fragment: a heading, a one-paragraph explanation, and
        the quoted trajectory tail (when there is one).
    """
    heading = f"<h3>Partial result: {html.escape(str(exc))}</h3>"
    if executor is None:
        return heading + (
            "<p>The sub-agent's budget ran out between sessions, before "
            "the next session could start.  Only the previous sessions' "
            "summaries above exist; the task is incomplete.</p>"
        )
    steps = [
        _USAGE_BLOCK_RE.sub("", str(m["content"])).strip()
        for m in executor.messages
        if m.get("role") == "model"
    ]
    tail = steps[-PARTIAL_RESULT_MAX_STEPS:]
    items = []
    for text in tail:
        if len(text) > PARTIAL_RESULT_MAX_CHARS_PER_STEP:
            text = text[:PARTIAL_RESULT_MAX_CHARS_PER_STEP] + " …"
        items.append(f"<li><pre>{html.escape(text)}</pre></li>")
    omitted = len(steps) - len(tail)
    parts = [
        heading,
        f"<p>The sub-agent spent ${budget_used:.4f} of its "
        f"${max_budget:.4f} budget in {total_steps} steps "
        "and was stopped before it called finish.  Below is its own account "
        "of the work so far, oldest first; it is incomplete and unverified.</p>",
    ]
    if items:
        if omitted > 0:
            parts.append(f"<p>({omitted} earlier steps omitted.)</p>")
        parts.append("<ol>" + "".join(items) + "</ol>")
    return "".join(parts)


class RelentlessAgent(Base):
    """Base agent with auto-continuation for long tasks."""

    work_dir: str = ""

    # The current usage-ledger epoch (see _UsageLedger).  Annotation
    # only: every access goes through _usage_ledger_object, which
    # creates the epoch lazily because Base.__init__ zeroes the
    # counters through the property setters BEFORE this class's
    # __init__ body runs.
    _usage_ledger: _UsageLedger

    def __init__(self, name: str) -> None:
        """Initialize the agent and its usage ledger.

        Args:
            name: The name identifier for the agent.
        """
        super().__init__(name)
        # The append-only accounting ledger (see :class:`_UsageEvent`
        # and :meth:`_accumulate_usage`).  The writers run on different
        # threads of the SAME agent: the agent thread
        # (:meth:`_accumulate_usage` at session end,
        # ``_attribute_sub_usage`` when a fan-out or a ``talk``
        # synthesis banks its spend, :meth:`_reset` at run start,
        # ``_fold_classifier_usage`` in ``SorcarAgent.run``'s
        # ``finally``) and server threads
        # (``reclaim_abandoned_subagents`` from worktree cleanup /
        # teardown / discard).  Each writer commits with one atomic
        # ``list.append`` of an immutable record, so no lock is needed
        # — nothing can deadlock on a stop-injected
        # ``KeyboardInterrupt`` and no writer can overwrite another's
        # commit.
        #
        # The ledger exists from construction, not only after
        # ``_reset``: the classifier-usage fold in ``SorcarAgent.run``'s
        # ``finally`` (and any other pre-``_reset`` failure path) may
        # touch the cumulative counters before the first session
        # starts.  ``Base.__init__`` above already zeroed the counters
        # through the property setters (zero deltas append nothing);
        # this swap just pins the canonical empty ledger.
        self.reset_usage()

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
        self.model_name = model_name if model_name is not None else DEFAULT_MODEL_NAME
        self.verbose = verbose
        # One atomic ledger swap resets the three counters AND the
        # banked-session marks together (see reset_usage): a
        # server-thread attribution in flight
        # (reclaim_abandoned_subagents) lands wholly in the old epoch
        # (discarded with it) or wholly in the new one — never a mixed
        # state, and never a torn triple.
        self.reset_usage()
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

    @property
    def budget_used(self) -> float:
        """Cumulative USD spend banked so far (see :class:`_UsageEvent`)."""
        return _ledger_totals(self._usage_ledger_object())[0]

    @budget_used.setter
    def budget_used(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, value: float,
    ) -> None:
        # Overrides Base's plain attribute with a property on purpose:
        # the counter is DERIVED from the append-only ledger, so an
        # absolute overwrite appends the delta that makes the derived
        # total equal *value*.  A concurrent attribution's append
        # linearizes AFTER the overwrite (set-then-add) — never a torn
        # or lost state.  Concurrent absolute overwrites of the SAME
        # field are not serialized against each other; the production
        # writers (Base.__init__, the server's pre-run zeroing for
        # plain agents) are single-threaded, and multi-field zeroing
        # goes through the atomic reset_usage instead.
        ledger = self._usage_ledger_object()
        delta = float(value) - _ledger_totals(ledger)[0]
        if delta:
            self._commit_usage_event(
                ledger,
                _UsageEvent(None, 0, delta, 0, 0),
            )

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens banked so far (see :class:`_UsageEvent`)."""
        return _ledger_totals(self._usage_ledger_object())[1]

    @total_tokens_used.setter
    def total_tokens_used(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, value: int,
    ) -> None:
        # Deliberate property-over-attribute override; see budget_used.
        ledger = self._usage_ledger_object()
        delta = int(value) - _ledger_totals(ledger)[1]
        if delta:
            self._commit_usage_event(
                ledger,
                _UsageEvent(None, 0, 0.0, delta, 0),
            )

    @property
    def total_steps(self) -> int:
        """Cumulative steps banked so far (see :class:`_UsageEvent`)."""
        return _ledger_totals(self._usage_ledger_object())[2]

    @total_steps.setter
    def total_steps(self, value: int) -> None:
        # Deliberate property-over-attribute override; see budget_used.
        ledger = self._usage_ledger_object()
        delta = int(value) - _ledger_totals(ledger)[2]
        if delta:
            self._commit_usage_event(
                ledger,
                _UsageEvent(None, 0, 0.0, 0, delta),
            )

    def _usage_ledger_object(self) -> _UsageLedger:
        """Return the current ledger epoch, creating it lazily.

        Lazy because ``Base.__init__`` zeroes the counters through the
        property setters before ``RelentlessAgent.__init__`` runs.
        ``dict.setdefault`` is one atomic C call, so two racing first
        touches always share one epoch.
        """
        ledger = self.__dict__.get("_usage_ledger")
        if ledger is None:
            ledger = self.__dict__.setdefault("_usage_ledger", _UsageLedger())
        return cast("_UsageLedger", ledger)

    def _usage_events(self) -> list[_UsageEvent]:
        """Return the current epoch's append-only record list.

        The list contains EVERY record banked in the epoch, including
        the prefix already folded into the published view; totals must
        always be read through :meth:`usage_snapshot` /
        :func:`_ledger_totals`, never by summing this list.
        """
        return self._usage_ledger_object().records

    def _usage_epoch(self) -> _UsageLedger:
        """Return the current epoch token (the ledger object itself).

        An adjustment producer that outlives task boundaries — the
        abandoned-subagent bookkeeping in ``sorcar_agent`` — records
        this token when it registers a source and later commits INTO
        the recorded object (:meth:`_attribute_usage`'s *epoch*
        argument), so spend from a source created under a PRIOR epoch
        can never be attributed to the current one (``reset_usage``
        swaps the token, and a late commit lands in the discarded
        object).
        """
        return self._usage_ledger_object()

    def _commit_usage_event(self, ledger: _UsageLedger, event: _UsageEvent) -> None:
        """Append *event* to *ledger* — one atomic, irrevocable commit.

        The single ``list.append`` IS the whole commit: *ledger*'s
        records list is never swapped, truncated or reordered within
        its epoch, so the record is visible to every later reader the
        instant the append returns and stays visible forever.  There
        is no confirmation step, no re-append, and no drain — an
        asynchronously injected stop lands strictly before the commit
        (no trace; a keyed producer's retry re-appends the identical
        record) or strictly after it (the record is durably counted;
        a keyed retry deduplicates on read).  A reset swaps the whole
        ledger object, never this list, so a commit racing a reset
        linearizes before it and dies with the old epoch — by design.

        After the commit, this opportunistically folds the epoch when
        the unfolded suffix has grown past the threshold; folding is
        pure maintenance (see :meth:`_maybe_compact`) and interrupting
        it loses nothing.
        """
        ledger.records.append(event)
        if len(ledger.records) - ledger.view.fold_index >= _COMPACTION_THRESHOLD:
            self._maybe_compact(ledger)

    def _maybe_compact(self, ledger: _UsageLedger) -> None:
        """Fold *ledger*'s unfolded suffix into its view, bounding reads.

        Serialized by a NON-BLOCKING ``compaction_lock.acquire(False)``
        — a losing caller returns immediately (some later commit will
        compact), so no unwind order can block or deadlock.  Under the
        lock, the fold sums ``records[view.fold_index:n]`` exactly as
        a reader would (dedup against ``view.seen`` plus per-source
        max within the slice) and then publishes ONE new
        :class:`_LedgerView` with a single ``STORE_ATTR``.  The
        records list is NOT touched, so:

        * a reader that loaded the old view keeps summing the old
          suffix — exact;
        * a reader that loads the new view sums the shorter suffix on
          top of the folded totals — exact;
        * an injected stop anywhere in this method leaves the old
          complete view published and every record still in place —
          nothing is ever lost, and the next commit simply re-folds.

        Cumulative fold time over an epoch is O(n log n) in the number
        of events: each event is summed in exactly one fold (folds
        cover disjoint slices) and the folded per-source max-seq state
        merges geometrically (:class:`_SeenMap`) instead of copying
        the whole historical key set each time (the round-6 quadratic
        finding).  A stop that leaks the lock merely disables
        compaction for this epoch: reads degrade to O(records) but
        stay exact, and the next epoch gets a fresh lock.
        """
        if not ledger.compaction_lock.acquire(blocking=False):
            return
        try:
            view = ledger.view
            records = ledger.records
            n = len(records)
            budget = view.budget
            tokens = view.tokens
            steps = view.steps
            seen = view.seen
            delta: dict[str, int] = {}
            for event in records[view.fold_index:n]:
                source = event.source
                if source is not None:
                    folded = seen.get(source)
                    if folded is not None and event.seq <= folded:
                        continue
                    prev = delta.get(source)
                    if prev is not None and event.seq <= prev:
                        continue
                    delta[source] = event.seq
                budget += event.budget
                tokens += event.tokens
                steps += event.steps
            ledger.view = _LedgerView(
                budget, tokens, steps, n, seen.including(delta),
            )
        finally:
            ledger.compaction_lock.release()

    def reset_usage(self) -> None:
        """Reset the whole accounting state to zero in ONE atomic store.

        Swaps in a fresh ledger epoch with a single ``STORE_ATTR`` —
        atomic under both thread interleaving and asynchronously
        injected exceptions.  A writer that already loaded the OLD
        epoch appends its record there: the transaction linearizes
        BEFORE the reset and is discarded with the old epoch.  A writer
        that loads the ledger after the swap lands in the new epoch
        with its whole triple.  Either way every observable state is
        coherent — never a mix of pre- and post-reset dimensions, and
        never a torn triple.  A session bank retried across the swap
        counts exactly once: its duplicate record carries the same
        session key, and the old epoch's record is no longer summed.

        The swap is also the EPOCH BOUNDARY for adjustment sources
        that outlive a run: abandoned-subagent items are tagged with
        the epoch token (:meth:`_usage_epoch`) at registration and
        their reclaims commit into that exact object, so a prior
        epoch's late spend settles in the discarded ledger and is
        never banked into the new epoch (an explicit, documented
        undercount versus real provider spend — see
        ``SorcarAgent.reclaim_abandoned_subagents``).

        Also the coherent replacement for zeroing the three counter
        properties one by one (the server's ``_zero_usage_counters``),
        which could otherwise interleave with a racing attribution.
        """
        self._usage_ledger = _UsageLedger()

    def usage_snapshot(self) -> tuple[float, int, int]:
        """Return one coherent ``(budget_used, total_tokens_used, total_steps)``.

        Sums ONE ledger epoch (see :func:`_ledger_totals`), so the
        triple always describes a single prefix of the append-only
        history.  Reading the three properties separately instead can
        TEAR: each property read sums the ledger afresh, and a
        concurrent append or reset between two of those reads yields
        an impossible mix (e.g. the old budget with the new
        tokens/steps) — a final abandoned-child reclaim that reads
        such a mix permanently loses the unseen dimension from the
        parent's accounting.
        """
        return _ledger_totals(self._usage_ledger_object())

    def _attribute_usage(
        self,
        budget: float,
        tokens: int,
        steps: int,
        key: str | None = None,
        seq: int = 0,
        epoch: Any = None,
    ) -> None:
        """Add a usage delta to the cumulative counters as ONE record.

        The single atomic ``list.append`` keeps the three dimensions
        coherent (no reader can ever observe the budget grown but not
        the tokens) and cannot lose to or erase any other usage writer
        — including :meth:`_reset` (the record either lands in the old
        epoch, linearizing before the reset, or in the new one) and a
        server-thread reclaim.  Never blocks: there is no lock to leak.
        A zero delta appends nothing, so an empty classifier fold and
        repeated zero adjustments do not grow the ledger — and a
        keyed retry of a zero-delta transaction is symmetric (neither
        attempt appends).

        Args:
            budget: USD spend to add.
            tokens: Token count to add.
            steps: Step count to add.
            key: Stable transaction source for a RETRYABLE adjustment
                (abandoned-child reclaim, classifier fold): a retry
                re-appends the same ``(key, seq)`` and readers count
                it once.  ``None`` marks a one-shot adjustment
                (fan-out totals, TTS, property-setter deltas), which
                is appended at most once by its producer and therefore
                needs no dedup identity.
            seq: Per-*key* sequence number, MONOTONIC in commit order
                (an abandoned-child reclaim passes its generation);
                ignored for one-shot adjustments.
            epoch: The ledger epoch object captured at the start of
                the producer's transaction
                (:meth:`_usage_epoch`), or ``None`` for the current
                epoch.  Binding the commit to the captured object
                closes the check-then-commit race with a concurrent
                ``reset_usage()``: a commit for a superseded epoch
                lands in that discarded object instead of being
                misattributed to the new task's ledger.
        """
        if not budget and not tokens and not steps:
            return
        ledger = (
            epoch
            if isinstance(epoch, _UsageLedger)
            else self._usage_ledger_object()
        )
        self._commit_usage_event(
            ledger,
            _UsageEvent(
                key,
                int(seq) if key is not None else 0,
                float(budget),
                int(tokens),
                int(steps),
            ),
        )

    def _accumulate_usage(self, agent: Base) -> None:
        """Bank a finished sub-agent's budget, tokens and steps exactly once.

        Exactly-once per *agent* under BOTH asynchronous interruption
        and concurrency, with no lock and no compare-and-swap:

        * ``perform_task``'s session try block banks the executor on
          its success path and its ``except BaseException`` handler
          banks it again before re-raising, so a stop-injected
          ``KeyboardInterrupt`` (``PyThreadState_SetAsyncExc``,
          delivered at an arbitrary bytecode boundary) can land
          ANYWHERE inside the first call.  The commit is one atomic
          ``list.append`` of an immutable :class:`_UsageEvent`: an
          injection before the append leaves no trace (the retry banks
          the full amount once) and an injection after it can at worst
          make the retry append a DUPLICATE record with the SAME
          session key (:func:`_session_key` is retry-stable), which
          readers count once (:func:`_ledger_totals`).
        * A server-thread reclaim (``reclaim_abandoned_subagents``)
          racing the agent thread's bank of the same executor likewise
          appends at most one extra same-key record — deduplicated on
          read, so no interleaving double-counts or loses the spend.

        The pre-append ledger scan (folded seen map plus the unfolded
        suffix) is an
        optimization, not a correctness requirement: it keeps repeated
        reclaims of an already-banked session from growing the ledger.
        See ``test_conc2026_relentless_usage_double_bank.py``, which
        proves the exactly-once property by injecting at every opcode
        boundary and by racing concurrent bankers.

        The banked triple itself is read through the executor's
        ``usage_snapshot()`` when it has one (``KISSAgent`` publishes
        its whole triple as one immutable snapshot), so a concurrent
        or interrupted response-accounting update on the executor can
        never hand this bank a torn source triple.
        """
        key = _session_key(agent)
        ledger = self._usage_ledger_object()
        view = ledger.view
        if view.seen.get(key) is not None or any(
            event.source == key
            for event in ledger.records[view.fold_index:]
        ):
            return
        snapshot = getattr(agent, "usage_snapshot", None)
        if callable(snapshot):
            budget, tokens, steps = cast("tuple[float, int, int]", snapshot())
        else:
            budget = agent.budget_used
            tokens = agent.total_tokens_used
            steps = agent.step_count
        self._commit_usage_event(
            ledger,
            _UsageEvent(
                key,
                0,
                float(budget or 0.0),
                int(tokens or 0),
                int(steps or 0),
            ),
        )

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

        Each sub-session is a fresh :class:`KISSAgent`; one that returns
        ``finish(is_continue=True, ...)`` hands its summary to the next.
        The ``IMPORTANT_INSTRUCTIONS`` suffix names the work dir when the
        tools can reach it: on the host, or in a container that bind-mounts
        it (every container kiss starts from an image does; see
        :data:`WORK_DIR_LINE`).

        Args:
            tools: List of callable tools available to the agent during execution.
            attachments: Optional file attachments (images, PDFs) for the initial prompt.

        Returns:
            YAML string with 'success' and 'summary' keys on successful completion.

        Raises:
            KISSError: If the task fails after exhausting all sub-sessions,
                or after :data:`MAX_ZERO_PROGRESS_SESSIONS` consecutive
                continuations that made no progress (no tool call other
                than ``finish``, or a summary identical to the previous
                session's).
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
        previous_summary: str | None = None  # the last continuation's, even if empty
        zero_progress_streak = 0
        current_pid = str(os.getpid())
        work_dir_visible = (
            self.docker_manager is None or self.work_dir in self.docker_manager.volumes
        )
        important_instructions = IMPORTANT_INSTRUCTIONS.format(
            work_dir_line=(
                WORK_DIR_LINE.format(work_dir=self.work_dir) if work_dir_visible else ""
            ),
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
            # One coherent snapshot for the whole session prologue: a
            # server-thread reclaim can publish between two separate
            # property reads and tear the triple (see usage_snapshot).
            budget_banked, tokens_banked, steps_banked = self.usage_snapshot()
            remaining_budget = self.max_budget - budget_banked
            if remaining_budget <= 0:
                exhausted = BudgetExceededError(
                    f"Agent {self.name} budget exhausted "
                    f"(${budget_banked:.4f} / ${self.max_budget:.2f})."
                )
                partial = self._budget_exhausted_result(None, summaries, exhausted)
                if partial is None:
                    raise exhausted
                return partial
            if self.printer:
                self.printer.tokens_offset = tokens_banked  # type: ignore[attr-defined]
                self.printer.budget_offset = budget_banked  # type: ignore[attr-defined]
                self.printer.steps_offset = steps_banked  # type: ignore[attr-defined]
            logger.info(
                "Session %d start: agent=%s budget_remaining=$%.4f "
                "total_tokens=%d total_steps=%d",
                session,
                self.name,
                remaining_budget,
                tokens_banked,
                steps_banked,
            )
            executor = KISSAgent(f"{self.name} Session-{session}")
            executor.pre_step_hook = getattr(self, "pre_step_hook", None)
            executor.tool_call_guard = getattr(self, "tool_call_guard", None)
            context_reset_hook = getattr(self, "context_reset_hook", None)
            executor.context_reset_hook = context_reset_hook
            if session > 0 and context_reset_hook is not None:
                # A new session starts from an empty context: nothing
                # shown to the previous session's model is visible now.
                context_reset_hook()
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
                self._current_executor = None
                self._accumulate_usage(executor)
            except BudgetExceededError as exc:
                self._current_executor = None
                self._accumulate_usage(executor)
                partial = self._budget_exhausted_result(executor, summaries, exc)
                if partial is None:
                    raise
                return partial
            except Exception as exc:
                logger.debug("Exception caught", exc_info=True)
                # Bank the failed session's spend BEFORE any recovery
                # work: a stop (``KeyboardInterrupt``) landing inside
                # the trajectory summarizer below must not lose it —
                # the runner's terminal ``result`` event and the
                # persisted ``task_history`` row read these counters
                # (``_subtask_metrics``). ``_summarize_failed_session``
                # relies on this: its budget math subtracts only
                # ``self.budget_used``, which now includes the failed
                # executor's spend.
                self._current_executor = None
                self._accumulate_usage(executor)
                is_context_overflow = isinstance(exc, ContextWindowExceededError)
                if (
                    (
                        not is_context_overflow
                        and (exc.__cause__ is not None or not isinstance(exc, KISSError))
                    )
                    or executor.step_count <= 1
                ):
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
            except BaseException:
                # A stop-injected ``KeyboardInterrupt`` (or any other
                # non-``Exception`` exit) must not lose the live
                # session's spend: the runner's terminal
                # stopped/failed ``result`` event and the persisted
                # ``task_history`` row read these counters
                # (``_subtask_metrics``), which otherwise report a
                # $0.0000 cost for a task that burned real money.
                self._current_executor = None
                self._accumulate_usage(executor)
                raise

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
            # Zero-progress guard: a continuation that called no tool but
            # ``finish``, or that repeated the previous session's summary
            # verbatim (an empty one included — ``summaries`` below keeps
            # only non-empty ones), did nothing the next session could
            # build on.
            stalled = executor.tool_calls_made == 0 or summary == previous_summary
            previous_summary = summary
            zero_progress_streak = zero_progress_streak + 1 if stalled else 0
            if summary:
                summaries.append(summary)

                progress_section = CONTINUATION_PROMPT.format(
                    progress_text=_capped_progress_text(summaries),
                    continuation_number=session + 1,
                )
            if zero_progress_streak >= MAX_ZERO_PROGRESS_SESSIONS:
                self._fail_after_sessions(
                    summaries,
                    f"Task stopped after {zero_progress_streak} consecutive "
                    "sub-sessions with no progress (no tool calls or an "
                    "unchanged summary)",
                )
        self._fail_after_sessions(
            summaries, f"Task failed after {self.max_sub_sessions} sub-sessions",
        )

    def _fail_after_sessions(self, summaries: list[str], banner: str) -> NoReturn:
        """End the run with *banner* after its sub-sessions stopped paying off.

        Shared by the two ways :meth:`perform_task` gives up without a
        terminal ``finish``: every sub-session was used, or
        :data:`MAX_ZERO_PROGRESS_SESSIONS` continuations in a row made
        no progress.  Emits the merged ``result`` event (the prior
        sessions' summaries followed by *banner*, see
        :func:`_build_exhaustion_summary`) and raises a :class:`KISSError`
        flagged ``terminal_result_broadcast`` so the runner does not
        emit a second terminal event.

        Args:
            summaries: Prior sessions' summaries, oldest first.
            banner: The short reason shown as the terminal result.

        Raises:
            KISSError: Always, carrying *banner*.
        """
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

    def _budget_exhausted_result(
        self,
        executor: KISSAgent | None,
        summaries: list[str],
        exc: BudgetExceededError,
    ) -> str | None:
        """Turn a sub-agent's budget exhaustion into a partial result.

        A top-level task keeps raising *exc*: the server, the CLI and
        the result panel all report "budget exceeded" from it.  A
        sub-agent (a ``run_parallel`` child or a ``run_agent``
        dispatch, marked by ``_subagent_info``) instead returns a
        ``finish(success=False, is_continue=False, ...)`` result whose
        summary quotes what it did (:func:`_partial_result_html`),
        preceded by any prior sessions' summaries — so the parent can
        use the work instead of receiving a bare "Task failed".  The
        merged result event is emitted here because the executor
        raised before it could emit its own.

        Args:
            executor: The session that ran out of budget, or ``None``
                when the budget ran out between sessions.
            summaries: Prior sessions' summaries, oldest first.
            exc: The budget error.

        Returns:
            The partial result string for a sub-agent, ``None`` for a
            top-level task (the caller re-raises *exc*).
        """
        if getattr(self, "_subagent_info", None) is None:
            return None
        # Cumulative figures (the executor's spend is already banked,
        # and nested sub-agents' spend is attributed here, not to the
        # executor).
        budget_used, _tokens, total_steps = self.usage_snapshot()
        payload = {
            "success": False,
            "is_continue": False,
            "summary": _build_exhaustion_summary(
                summaries,
                _partial_result_html(
                    executor, exc, budget_used, self.max_budget, total_steps,
                ),
            ),
        }
        self._emit_merged_result_event(payload)
        result: str = yaml.dump(payload, sort_keys=False)
        return result

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
            # The caller (``perform_task``'s failure handler) banked
            # the failed executor's spend into ``self.budget_used``
            # before calling here, so the remaining budget is a plain
            # subtraction — subtracting ``executor.budget_used`` again
            # would double-count it.
            summarizer_budget = max(0.01, self.max_budget - self.budget_used)
            summarizer_agent = KISSAgent(f"{self.name} Summarizer")
            try:
                summarizer_result = summarizer_agent.run(
                    model_name=self.model_name,
                    prompt_template=SUMMARIZER_PROMPT,
                    tools=[shell_tools.Read, shell_tools.Bash, shell_tools.bash_job],
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
        # The printer adds its per-task offsets to every result event's
        # totals, but this agent's snapshot is already CUMULATIVE, so
        # the raw values passed down are the snapshot MINUS the current
        # offsets.  The previous design zeroed the offsets around the
        # print and restored them afterwards; round-4 finding 5 showed
        # an asynchronously injected stop could land after the zeroing
        # and skip (or interrupt) the restoration, leaving the task's
        # offsets zeroed.  Reading the offsets without ever mutating
        # them removes that failure mode entirely: there is no shared
        # state to restore, so no injection point can corrupt it.
        tokens_offset = int(getattr(self.printer, "tokens_offset", 0) or 0)
        budget_offset = float(getattr(self.printer, "budget_offset", 0.0) or 0.0)
        steps_offset = int(getattr(self.printer, "steps_offset", 0) or 0)
        # One coherent triple (see usage_snapshot): three separate
        # property reads could tear across a concurrent bank.
        budget, tokens, steps = self.usage_snapshot()
        self.printer.print(
            yaml.dump(payload, sort_keys=False),
            type="result",
            step_count=steps - steps_offset,
            total_tokens=tokens - tokens_offset,
            cost=f"${budget - budget_offset:.4f}",
        )

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

            # The work dir is bind-mounted at its host path and is the
            # container's working directory, so relative paths and the
            # ``Work dir`` prompt line mean the same thing inside and out.
            # An attached ``container:<id>`` ignores both (see DockerManager).
            with DockerManager(
                self.docker_image,
                workdir=self.work_dir,
                volumes={self.work_dir: self.work_dir},
            ) as docker_mgr:
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
