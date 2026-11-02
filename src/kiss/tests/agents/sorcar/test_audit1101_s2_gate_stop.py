# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F2 and F3: the coding gate's host commands and stages end on a Stop.

* F2: :func:`run_on_host` killed the process group at its deadline and then
  drained the pipe with no bound, so a descendant that escaped into a
  session of its own and kept stdout held the gate until it exited; an
  exception cutting the wait short (a Stop landing as ``KeyboardInterrupt``)
  left the command running.  The reap after the kill is bounded, the pipe is
  closed rather than read to its end, the output read so far is kept, and
  every exceptional exit kills the group.
* F3: :meth:`CodingGate.run_stages` joined the stage threads with bare
  ``join()`` calls, inside which an injected Stop cannot land, and the
  stages had no way to learn of it.  The wait is sliced, and a Stop (the
  task's stop event, or an exception cutting the wait short) cancels the
  stages: the verifier's command dies with its process group and the
  reviewer's model call ends at once.

Real bash, real threads, the framework's own exception injection.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.core import stop_signal
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    serve,
    text_body,
)
from kiss.tests.conftest import posix_only

pytestmark = posix_only("process groups and bash")


def _group_alive(pid: int) -> bool:
    """Whether any process of the group led by *pid* is still running."""
    return subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, check=False,
                          timeout=10).returncode == 0


def _pid_from(path: Path, within: float = 5.0) -> int:
    """Return the pid a command wrote into *path*, waiting up to *within* seconds for it."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        text = path.read_text() if path.exists() else ""
        if text.strip():
            return int(text)
        time.sleep(0.02)
    raise AssertionError(f"no pid written to {path}")


def _inject_after(thread: threading.Thread, seconds: float) -> tuple[threading.Thread, list[int]]:
    """Inject ``KeyboardInterrupt`` into *thread* after *seconds*, as the Stop watchdog does.

    Returns the injector thread and the list that receives the number of threads hit
    (0 when *thread* had already ended); the test joins the injector before it ends.
    """
    hits: list[int] = []

    def inject() -> None:
        time.sleep(seconds)
        assert thread.ident is not None
        hits.append(inject_async_exception(thread.ident, KeyboardInterrupt))
    injector = threading.Thread(target=inject, daemon=True)
    injector.start()
    return injector, hits


# A descendant that leaves the command's process group for a session of its own
# and keeps the inherited stdout; macOS has no setsid(1), so Python's os.setsid.
_ESCAPED_SLEEPER = (f"{shlex.quote(sys.executable)} -c 'import os; os.setsid(); "
                    "os.execvp(\"sleep\", [\"sleep\", \"6\"])'")


def test_host_timeout_is_bounded_by_an_escaped_descendant_holding_stdout(tmp_path: Path) -> None:
    started = time.monotonic()
    code, output = run_on_host(f"echo partial; {_ESCAPED_SLEEPER} & sleep 30", str(tmp_path), 1)
    elapsed = time.monotonic() - started
    assert code is None and output == "partial\n"
    # The deadline, then the bounded reap (REAP_GRACE_SECONDS): not the descendant's 6 s.
    assert elapsed < 4.5, elapsed
    code, output = run_on_host("echo partial; exec > /dev/null; sleep 30", str(tmp_path), 1)
    assert code is None and output == "partial\n"  # output kept after the command closed stdout
    assert run_on_host("echo done", str(tmp_path), 5) == (0, "done\n")
    assert run_on_host("echo err >&2; exit 3", str(tmp_path), 5) == (3, "err\n")


def test_an_interrupted_host_command_dies_with_its_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = run_on_host("echo $$ > pid; sleep 30", str(tmp_path), 30)
        except KeyboardInterrupt:
            outcome["interrupted"] = time.monotonic()

    worker = threading.Thread(target=run)
    worker.start()
    pid = _pid_from(pidfile)
    injected_at = time.monotonic()
    injector, hits = _inject_after(worker, 0.0)
    worker.join(5)
    injector.join(2)
    assert not worker.is_alive() and hits == [1], (outcome, hits)
    assert "interrupted" in outcome and outcome["interrupted"] - injected_at < 2, outcome
    deadline = time.monotonic() + 3
    while _group_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _group_alive(pid)


def test_a_stop_while_the_verifier_runs_ends_the_wait_and_the_verifier(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("echo $$ > pid\nsleep 30\n")
    gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                      str(tmp_path), "tmp")
    outcome: dict[str, Any] = {}

    def finish() -> None:
        try:
            outcome["verdicts"] = gate.run_stages({"summary_in_html": "done"}, "task")
        except KeyboardInterrupt:
            outcome["interrupted"] = time.monotonic()

    worker = threading.Thread(target=finish)
    worker.start()
    pid = _pid_from(tmp_path / "pid")
    injected_at = time.monotonic()
    injector, hits = _inject_after(worker, 0.0)
    worker.join(5)
    injector.join(2)
    assert not worker.is_alive() and hits == [1], (outcome, hits)
    assert "interrupted" in outcome and outcome["interrupted"] - injected_at < 2, outcome
    # The cancelled verifier's command is killed with its group, promptly.
    deadline = time.monotonic() + 3
    while _group_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _group_alive(pid)
    stage = next((t for t in threading.enumerate() if t.name == "gate-verifier"), None)
    if stage is not None:
        stage.join(3)
        assert not stage.is_alive()


class _SlowHandler(ScriptedHandler):
    """A reviewer that answers only once released (or after ten seconds)."""

    arrived = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:  # noqa: N802
        _SlowHandler.arrived.set()
        _SlowHandler.release.wait(10)
        super().do_POST()


def test_the_task_stop_event_cancels_a_pending_review(
    tmp_path: Path,
) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("exit 0\n")
    _SlowHandler.arrived.clear()
    _SlowHandler.release.clear()
    stop = threading.Event()
    outcome: dict[str, Any] = {}
    with serve([text_body("VERDICT: OK")], handler_base=_SlowHandler) as (url, requests):
        gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                          str(tmp_path), "tmp", reviewer=MODEL,
                          model_config={"base_url": url, "api_key": "local"})
        gate.observe("Edit", {"file_path": str(tmp_path / "a.py"), "old_string": "a",
                              "new_string": "b"})
        (tmp_path / "a.py").write_text("b\n")

        def finish() -> None:
            # The task thread's stop event, as the daemon binds it on a task.
            stop_signal.set_thread_stop_event(stop)
            try:
                outcome["verdicts"] = gate.run_stages({"summary_in_html": "done"}, "task")
            except KeyboardInterrupt:
                outcome["interrupted"] = True
            finally:
                outcome["ended"] = time.monotonic()

        worker = threading.Thread(target=finish)
        worker.start()
        assert _SlowHandler.arrived.wait(10), "the review did not reach the reviewer"
        assert not requests  # the reviewer has not answered
        # Stop pressed: the event is set first; the watchdog injects a second later.
        stopped_at = time.monotonic()
        stop.set()
        injector, _ = _inject_after(worker, 1.0)  # hits nothing when the event alone sufficed
        worker.join(8)
        try:
            assert not worker.is_alive(), outcome
            # The pending model call ended on the event, well before the server answered.
            assert outcome["ended"] - stopped_at < 4, outcome
            stage = next((t for t in threading.enumerate() if t.name == "gate-review"), None)
            if stage is not None:
                stage.join(3)
                assert not stage.is_alive()
        finally:
            _SlowHandler.release.set()
            injector.join(3)
    assert gate.reviews == 0


def test_bare_stage_results_are_unchanged_without_a_stop(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("echo checked; exit 1\n")
    gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                      str(tmp_path), "tmp")
    verdicts = gate.run_stages({"summary_in_html": "done"}, "task")
    assert list(verdicts) == ["verifier"]
    assert verdicts["verifier"] is not None and "exited 1" in verdicts["verifier"]
    assert gate.last_verifier_run == (1, "checked\n")
    assert not stop_signal.stop_requested()
    assert os.path.exists(tmp_path / "tmp" / "verify.sh")
