"""A Stop landing right after rsi7d's replay job is launched still stops the replay.

``_dispatch_replay`` used to start its job with ``agent_dispatch.start_agent_job``
before entering the ``try`` whose ``except BaseException`` kills it.  A Stop (an
asynchronously injected ``KeyboardInterrupt``) delivered after the job was
registered and its thread started, but before ``start_agent_job`` returned, left
the replay running uncancelled in the job registry; a replay with no calling task
(``current_agent()`` is ``None``, e.g. the standalone ``uv run python -c`` route
``_legacy_daemon_error`` names) has no run end to collect it either.

The Stop is a real ``KeyboardInterrupt`` raised by a tracer on the ``return``
event of ``agent_dispatch.launch_agent_job``, the window in question; it is the
same delivery model as ``PyThreadState_SetAsyncExc``.  No daemon runs, so the
job's dispatch ends with a connection error on its own: what tells the two
behaviours apart is whether the replay's stop was requested.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.rsi7d import rsi7d_sea
from kiss.agents.sorcar import agent_dispatch


def _stop_after_launch(frame: Any, event: str, arg: Any) -> Any:
    """Line tracer raising ``KeyboardInterrupt`` as ``launch_agent_job`` returns."""
    if frame.f_code is not agent_dispatch.launch_agent_job.__code__:
        return None
    if event == "return":
        raise KeyboardInterrupt("injected stop")
    return _stop_after_launch


def test_stop_right_after_the_launch_cancels_the_ownerless_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interrupted replay's job is cancelled and its thread ends, not left running."""
    monkeypatch.chdir(tmp_path)
    before = set(agent_dispatch.agent_jobs_of(None))
    sys.settrace(_stop_after_launch)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected stop"):
            rsi7d_sea._dispatch_replay(
                "demo", str(tmp_path / "demo_sea.py"), "say hi", str(tmp_path),
                "gpt-6-luna", 0.01, 60.0,
            )
    finally:
        sys.settrace(None)
    jobs = [job for job_id, job in agent_dispatch.agent_jobs_of(None).items()
            if job_id not in before]
    assert len(jobs) == 1, "the replay job was never launched; nothing was tested"
    job = jobs[0]
    try:
        assert job.cancel.is_set(), "the Stop left the replay running uncancelled"
        assert agent_dispatch.join_agent_job(job, 30.0)
        assert job.finished
    finally:
        agent_dispatch.forget_agent_job(job)
