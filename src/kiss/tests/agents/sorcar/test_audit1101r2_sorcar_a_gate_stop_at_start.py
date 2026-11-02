# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 round 2 (sorcar_a F5): a Stop while the gate's stages start.

:meth:`CodingGate.run_stages` started its stage threads before the
``try`` whose cancellation handler sets their stop event, so a Stop
landing after the first stage started (while the next one was being
started) left that stage running: its verifier kept going after the
caller had returned on the Stop.  The threads now start inside the
protected region, so every exceptional exit after any stage started
cancels it.

Real bash, real threads, the framework's own exception injection; a
trace of the caller's thread chooses the moment the Stop lands and
replaces nothing.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, text_body
from kiss.tests.agents.sorcar.test_audit1101_s2_gate_stop import _group_alive, _pid_from
from kiss.tests.conftest import posix_only

pytestmark = posix_only("process groups and bash")


class _StopAfterTheFirstStart:
    """A trace of the caller's thread: the Stop lands once the first stage thread has started.

    At the next line of ``run_stages`` reached while a ``gate-verifier``
    thread is alive (and its command wrote *pidfile*: it is running), the
    trace turns itself off and injects ``KeyboardInterrupt`` into the
    thread, as the Stop watchdog does.

    Args:
        pidfile: Where the verifier's command writes its pid.
    """

    def __init__(self, pidfile: Path) -> None:
        self.pidfile = pidfile
        self.injected = 0

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event == "call":
            return self if frame.f_code is CodingGate.run_stages.__code__ else None
        if event == "line" and any(
            thread.name == "gate-verifier" and thread.is_alive() for thread in threading.enumerate()
        ):
            _pid_from(self.pidfile)
            sys.settrace(None)
            # Counted first: the injected exception may land before the call returns.
            self.injected += 1
            inject_async_exception(threading.get_ident(), KeyboardInterrupt)
            return None
        return self


def test_a_stop_as_the_stages_start_cancels_the_started_stage(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text(
        "echo $$ > pid\nwhile [ ! -e release ]; do sleep 0.05; done\necho escaped > late\n"
    )
    outcome: dict[str, Any] = {}
    trace = _StopAfterTheFirstStart(tmp_path / "pid")
    with serve([text_body("VERDICT: OK")]) as (url, _requests):
        gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                          str(tmp_path), "tmp", reviewer=MODEL,
                          model_config={"base_url": url, "api_key": "local"})
        gate.observe("Edit", {"file_path": str(tmp_path / "a.py"), "old_string": "a",
                              "new_string": "b"})
        (tmp_path / "a.py").write_text("b\n")
        assert gate.review_wanted()  # two stages: the Stop lands as the second one starts

        def finish() -> None:
            sys.settrace(trace)
            try:
                outcome["verdicts"] = gate.run_stages({"summary_in_html": "done"}, "task")
            except KeyboardInterrupt:
                outcome["interrupted"] = True
            finally:
                sys.settrace(None)

        worker = threading.Thread(target=finish)
        worker.start()
        worker.join(10)
        try:
            assert not worker.is_alive() and trace.injected == 1, (outcome, trace.injected)
            assert outcome == {"interrupted": True}, outcome
            # The started verifier is cancelled with the caller: its command dies with its group.
            pid = _pid_from(tmp_path / "pid")
            deadline = time.monotonic() + 3
            while _group_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _group_alive(pid), "the verifier outlived the Stop: it was not cancelled"
        finally:
            (tmp_path / "release").touch()
            for thread in threading.enumerate():
                if thread.name.startswith("gate-"):
                    thread.join(5)
                    assert not thread.is_alive(), thread.name
    time.sleep(0.3)
    assert not (tmp_path / "late").exists(), "the verifier wrote after the caller's Stop"
