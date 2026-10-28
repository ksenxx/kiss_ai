# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixes C/D (2026-10-13): run cleanup under a second stop, interrupt-safe locks, marker file.

* ``SorcarAgent.run``'s cleanup must run every step when the runner's
  second ``KeyboardInterrupt`` injection lands in one of them or
  between two of them: the final ``usage_info`` event still goes out
  and the per-run hooks are dropped (``_finish_run``, literally nested
  ``try``/``finally`` blocks).
* The task-thread takers of ``_OFFSET_PUBLISH_LOCK``, ``TASK_USAGE_LOCK``
  and ``task_classifier._cache_lock`` go through ``locked_call``, which
  cannot leak the lock to an interrupt landing right after the
  acquisition (a plain ``with`` can: CPython gh-74174) and waits for a
  slow holder instead of timing out past it.
* ``GitWorktreeOps.save_preserve_marker`` writes a file of its own, so
  a ``.git/config.lock`` held by any concurrent git command, however
  long, cannot lose the marker; a fresh process honours it.

CPython delivers an asynchronous exception only at a check point:
function entry (``RESUME``), a loop's ``JUMP_BACKWARD``, or right after
a ``CALL``.  The cleanup sweep therefore raises, through a
``sys.settrace`` opcode hook, at the instruction after every call the
cleanup makes: the deterministic equivalent of
``PyThreadState_SetAsyncExc`` landing at each of its check points.
The lock tests use ``PyThreadState_SetAsyncExc`` itself.
"""

from __future__ import annotations

import dis
import functools
import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import task_classifier
from kiss.agents.sorcar._concurrency import locked_call
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.persistence import TASK_USAGE_LOCK
from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK, RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.test_budget_enforcement_e2e import (
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)
from kiss.tests.lock_injection import hammer
from kiss.tests.server.conftest import init_repo, run_git
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_CHEAP = (10, 5)


class _FinishHandler(BaseHTTPRequestHandler):
    """Model script: ``finish`` on the first call."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        arguments = json.dumps({
            "success": True, "is_continue": False, "summary_in_html": "<p>done</p>",
        })
        _send_json(self, _tool_call_response("finish", arguments, *_CHEAP))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _CapturePrinter(JsonPrinter):
    """A real ``JsonPrinter`` whose broadcast events are collected."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:  # type: ignore[override]
        self.events.append(dict(event))


@pytest.fixture
def model_url() -> Iterator[str]:
    """A local chat-completions server whose model finishes at once."""
    server, url = _start_server(_FinishHandler)
    try:
        yield url
    finally:
        server.shutdown()


_FINISH_RUN_CODE = SorcarAgent._finish_run.__code__


def _interrupt_on_entry(step_name: str, seen: list[int], printer: _CapturePrinter) -> Callable:
    """Build a trace function that raises ``KeyboardInterrupt`` entering *step_name*.

    Records in *seen* how many events *printer* had broadcast at that
    moment, so the test can tell which events came after the interrupt.
    A trace function that raises is unset by the interpreter, so it fires once.
    """
    def tracer(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code.co_name == step_name:
            seen.append(len(printer.events))
            raise KeyboardInterrupt("second stop injection")
        return None
    return tracer


def _interrupt_at_offset(offset: int, seen: list[int], printer: _CapturePrinter) -> Callable:
    """Build a trace function that raises ``KeyboardInterrupt`` at *offset* of ``_finish_run``.

    The opcode hook fires before the instruction at *offset* runs, so
    raising there models the interrupt delivered by the check point of
    the instruction before it.
    """
    def tracer(frame: Any, event: str, arg: Any) -> Callable | None:
        if frame.f_code is not _FINISH_RUN_CODE:
            return None
        frame.f_trace_opcodes = True
        if event == "opcode" and frame.f_lasti == offset and not seen:
            seen.append(len(printer.events))
            raise KeyboardInterrupt("second stop injection")
        return tracer
    return tracer


def _record_offsets(offsets: list[int]) -> Callable:
    """Build a trace function that records the instruction offsets ``_finish_run`` executes."""
    def tracer(frame: Any, event: str, arg: Any) -> Callable | None:
        if frame.f_code is not _FINISH_RUN_CODE:
            return None
        frame.f_trace_opcodes = True
        if event == "opcode":
            offsets.append(frame.f_lasti)
        return tracer
    return tracer


def _run_traced(
    tracer: Callable, printer: _CapturePrinter, tmp_path: Path, model_url: str,
    *, interrupted: bool,
) -> SorcarAgent:
    """Run a ``SorcarAgent`` under *tracer*; an *interrupted* run must end in ``KeyboardInterrupt``.

    Returns the agent, for the caller's checks on its post-run state.
    """
    agent = SorcarAgent("audit1013-cleanup")
    sys.settrace(tracer)
    try:
        if interrupted:
            with pytest.raises(KeyboardInterrupt):
                _run(agent, printer, tmp_path, model_url)
        else:
            _run(agent, printer, tmp_path, model_url)
    finally:
        sys.settrace(None)
    return agent


def _run(agent: SorcarAgent, printer: _CapturePrinter, tmp_path: Path, model_url: str) -> None:
    """One scripted run: the model finishes at once."""
    agent.run(
        model_name="gpt-4o-mini",
        prompt_template="finish at once",
        max_steps=3,
        max_budget=1.0,
        append_basic_tools=False,
        use_memory=False,
        web_tools=False,
        work_dir=str(tmp_path),
        printer=printer,
        verbose=False,
        model_config={"base_url": model_url, "api_key": "test-key"},
    )


def _assert_cleanup_completed(
    agent: SorcarAgent, printer: _CapturePrinter, seen: list[int],
) -> None:
    """Every cleanup step ran: the run's last word is its usage event, hooks and browser dropped."""
    assert seen, "the run never reached the interrupt point"
    last = printer.events[-1]
    assert last["type"] == "usage_info", "the run's final usage_info was skipped by the interrupt"
    assert last["total_steps"] >= 1
    assert agent.context_reset_hook is None
    assert agent.pre_step_hook is None
    assert agent.tool_call_guard is None
    assert agent.web_use_tool is None


class TestRunCleanupSurvivesSecondInterrupt:
    """Every cleanup step runs wherever the second interrupt lands."""

    @pytest.mark.parametrize(
        "step_name", ["_mark_unmerged_worktree_preserved", "_fold_classifier_usage"],
    )
    def test_interrupt_inside_a_step(
        self, step_name: str, tmp_path: Path, model_url: str,
    ) -> None:
        seen: list[int] = []
        printer = _CapturePrinter()
        agent = _run_traced(
            _interrupt_on_entry(step_name, seen, printer), printer, tmp_path, model_url,
            interrupted=True,
        )
        _assert_cleanup_completed(agent, printer, seen)
        assert len(printer.events) > seen[0], "the usage event predates the interrupt"

    def test_interrupt_at_every_check_point_between_steps(
        self, tmp_path: Path, model_url: str,
    ) -> None:
        """An interrupt delivered right after any call ``_finish_run`` makes
        (its only check points besides function entry) still runs the rest:
        the former step loop left its backedge unprotected."""
        executed: list[int] = []
        _run_traced(
            _record_offsets(executed), _CapturePrinter(), tmp_path, model_url,
            interrupted=False,
        )
        calls = {
            ins.offset for ins in dis.get_instructions(_FINISH_RUN_CODE)
            if ins.opname.startswith("CALL")
        }
        targets = sorted({
            executed[i + 1] for i, offset in enumerate(executed[:-1]) if offset in calls
        })
        assert len(targets) >= 6, f"expected one check point per cleanup step, got {targets}"
        for offset in targets:
            seen: list[int] = []
            printer = _CapturePrinter()
            agent = _run_traced(
                _interrupt_at_offset(offset, seen, printer), printer, tmp_path, model_url,
                interrupted=True,
            )
            _assert_cleanup_completed(agent, printer, seen)


def _noop() -> None:
    pass


class TestLockedCall:
    """Interrupts cannot leak the shared locks; a live holder is waited for."""

    def _hammer(self, lock: threading.RLock, call: Callable[[], object]) -> None:
        """Inject interrupts into a thread looping *call*; the lock stays free."""
        hammer(call, {"lock": lock})

    def test_helper_never_leaks(self) -> None:
        lock = threading.RLock()
        self._hammer(lock, functools.partial(locked_call, lock, _noop))

    def test_set_printer_offsets(self) -> None:
        # The console printer's offsets are plain thread-local fields;
        # ``JsonPrinter``'s setters take its own lock through
        # ``locked_call`` (hammered in the server's printer injection test).
        agent = RelentlessAgent("audit1013-offsets")
        agent.printer = ConsolePrinter(file=io.StringIO())
        self._hammer(_OFFSET_PUBLISH_LOCK, agent._set_printer_offsets)

    def test_emit_usage_totals(self) -> None:
        agent = RelentlessAgent("audit1013-totals")
        agent.printer = _CapturePrinter()
        self._hammer(TASK_USAGE_LOCK, agent._emit_usage_totals)

    def test_classifier_cache(self) -> None:
        # The cache file lives under the session's temporary KISS_HOME.
        self._hammer(task_classifier._cache_lock, task_classifier.clear_classification_cache)

    def test_slow_holder_is_waited_for(self) -> None:
        """A holder still alive (a side channel's charge) is never bypassed:
        the run's final usage event goes out after it, never past it."""
        agent = RelentlessAgent("audit1013-wait")
        printer = _CapturePrinter()
        agent.printer = printer
        released_at: list[float] = []
        held = threading.Event()

        def hold() -> None:
            with TASK_USAGE_LOCK:
                held.set()
                time.sleep(0.5)
                released_at.append(time.monotonic())

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(10), "the holder never took the lock"
        agent._emit_usage_totals()
        emitted_at = time.monotonic()
        holder.join()
        assert [e["type"] for e in printer.events] == ["usage_info"]
        assert emitted_at >= released_at[0]


def _init(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    init_repo(repo)
    return repo


def _marker_in_fresh_process(repo: Path, branch: str) -> bool:
    """Whether a process with no volatile claim sees *branch* preserved."""
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from pathlib import Path\n"
            "from kiss.agents.sorcar.git_worktree import GitWorktreeOps\n"
            "print(GitWorktreeOps.load_preserve_marker(Path(sys.argv[1]), sys.argv[2]))",
            str(repo), branch,
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip() == "True"


class TestPreserveMarkerIsAFile:
    """The marker needs no git lock, so no git command can defeat it."""

    def test_marker_lands_while_config_lock_is_held(self, tmp_path: Path) -> None:
        repo = _init(tmp_path)
        (repo / ".git" / "config.lock").write_text("")  # held for the whole test
        started = time.monotonic()
        assert GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-x")
        assert time.monotonic() - started < 1.0
        assert GitWorktreeOps._preserve_marker_on_disk(repo, "kiss/wt-x")
        assert _marker_in_fresh_process(repo, "kiss/wt-x")

    def test_marker_fails_closed_when_its_directory_is_unwritable(self, tmp_path: Path) -> None:
        repo = _init(tmp_path)
        (repo / ".git" / "kiss-preserve").write_text("")  # a file where the directory goes
        assert not GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-y")
        assert not GitWorktreeOps._preserve_marker_on_disk(repo, "kiss/wt-y")
        # The volatile claim still protects the branch within this process.
        assert GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-y")
        assert not _marker_in_fresh_process(repo, "kiss/wt-y")

    def test_legacy_config_marker_is_still_honoured(self, tmp_path: Path) -> None:
        repo = _init(tmp_path)
        run_git(repo, "config", "branch.kiss/wt-z.kiss-preserve", "1")
        assert GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-z")

    def test_marker_goes_away_with_the_branch(self, tmp_path: Path) -> None:
        repo = _init(tmp_path)
        run_git(repo, "branch", "kiss/wt-w")
        assert GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-w")
        assert GitWorktreeOps.delete_branch(repo, "kiss/wt-w")
        assert not GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-w")

    def test_volatile_claim_goes_away_with_the_branch(self, tmp_path: Path) -> None:
        """A claim held only in memory (marker unwritable) dies with its branch.

        Otherwise a later branch of the same name in the same repository
        would be born preserved-for-review and never reclaimed.
        """
        repo = _init(tmp_path)
        run_git(repo, "branch", "kiss/wt-v")
        (repo / ".git" / "kiss-preserve").write_text("")  # a file where the directory goes
        assert not GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-v")
        assert GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-v")
        assert GitWorktreeOps.delete_branch(repo, "kiss/wt-v")
        assert not GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-v")
        run_git(repo, "branch", "kiss/wt-v")
        assert not GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-v")


_OWNER = """
import sys
from pathlib import Path
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

repo = Path(sys.argv[1])
agent = ChatSorcarAgent("audit1013-owner")
assert agent._try_setup_worktree(repo, str(repo)) is not None
wt = agent._wt
(wt.wt_dir / "agent.txt").write_text("unaccepted work\\n", encoding="utf-8")
lock = repo / ".git" / "config.lock"
lock.write_text("")  # a git command holding it for the owner's whole exit
agent._retire_previous_worktree()
assert agent._wt is None, "the retire did not park the worktree"
print("BRANCH " + wt.branch)
"""


class TestPreservedWorkSurvivesOwnerExit:
    """The reclaim of a later process never publishes work parked under a config lock."""

    def test_fresh_process_reclaim_skips_the_parked_branch(self) -> None:
        home = IsolatedKissHome("kiss-audit1013-owner-exit-")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(_OWNER), str(home.repo)],
                capture_output=True, text=True, cwd=str(home.repo), timeout=180,
                env=dict(os.environ, KISS_HOME=str(home.kiss_home)),
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
            branch = [ln for ln in proc.stdout.splitlines() if ln.startswith("BRANCH ")][-1][7:]
            (home.repo / ".git" / "config.lock").unlink()
            assert GitWorktreeOps.reclaim_orphaned_worktrees(
                home.repo, exclude_branches=set(),
            ) == 0
            assert GitWorktreeOps.branch_exists(home.repo, branch)
            assert not (home.repo / "agent.txt").exists(), "unaccepted work was published"
        finally:
            home.cleanup()
