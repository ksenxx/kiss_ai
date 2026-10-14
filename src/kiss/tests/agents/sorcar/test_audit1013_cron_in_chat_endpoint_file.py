# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (E2): ``in_chat`` cron jobs reach the recorded daemon endpoint.

``cron_agent._run_in_chat_job`` passed the module-local
``_daemon_endpoint_file`` global to ``daemon_client.run``.  A
dispatched cron session runs its ``cron_job`` tool from a fresh
synthetic copy of the module (``load_sea``), whose global is never set
by the scheduler: ``cron_job("run_now", ...)`` on an ``in_chat`` job
fell back to the environment's endpoint resolution instead of the
hosting daemon's recorded file (prompt jobs already read the canonical
module's value through ``agent_dispatch._daemon_endpoint_file``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.sea_settings import execute_python_file


def _job() -> dict[str, object]:
    return {"id": "abcd1234", "name": "hi", "prompt": "say hi", "in_chat": True, "max_budget": 0}


@pytest.mark.parametrize("synthetic_copy", [False, True])
def test_in_chat_job_error_names_the_recorded_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_copy: bool,
) -> None:
    """The canonical module's recorded endpoint wins over the environment's, copy or not."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    recorded = tmp_path / "recorded.json"
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "env.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(recorded))
    module = execute_python_file(cron_agent.__file__) if synthetic_copy else vars(cron_agent)
    if synthetic_copy:
        assert module["_daemon_endpoint_file"] is None  # the copy's own global is unset
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status, text = module["_run_in_chat_job"](_job(), "say hi", run_dir, 5.0)
    assert status == "error"
    assert text is not None
    assert f"Cannot connect to the sorcar daemon: no endpoint file at {recorded}" in text
