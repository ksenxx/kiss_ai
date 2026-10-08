# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``cron_agent`` prompt jobs reach the SAME daemon endpoint ``run_agent`` uses.

Redundancy found by the 2026-09-02 sorcar-infra audit:
``cron_agent._run_prompt_job`` once re-implemented the endpoint
precedence that :func:`daemon_client._resolve_endpoint_file` already
owns, just to name the endpoint in its "cannot reach the daemon" error.
A prompt job is now launched as a generated SEA through the
``run_agent`` tool (:func:`agent_dispatch.make_run_agent_tool`), whose
dispatch resolves the endpoint file exactly once: the file recorded by
:func:`cron_agent.start_scheduler_thread` (read through the canonical
module by :func:`agent_dispatch._daemon_endpoint_file`), then
``KISS_SORCAR_LOCAL``, then ``$KISS_HOME/sorcar-local.json``.  These
tests run the real (unreachable-daemon) path for all three precedence
levels and check the recorded error names exactly the endpoint file the
client tried.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent, daemon_client


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    return tmp_path


def _job() -> dict[str, object]:
    return {"id": "abcd1234", "name": "hi", "prompt": "say hi", "max_budget": 0}


def _run(tmp_path: Path) -> str:
    work_dir = tmp_path / "run"
    work_dir.mkdir(exist_ok=True)
    status, text = cron_agent._run_prompt_job(
        _job(), work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert status == "error"
    assert text is not None
    return text


def test_error_names_recorded_daemon_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = tmp_path / "recorded.json"
    # The env endpoint must lose to the one the hosting daemon recorded.
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "env.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(endpoint))
    text = _run(tmp_path)
    assert f"Cannot connect to the sorcar daemon: no endpoint file at {endpoint}" in text
    assert str(daemon_client._resolve_endpoint_file(str(endpoint))) == str(endpoint)


def test_error_names_env_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = tmp_path / "env.json"
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(endpoint))
    text = _run(tmp_path)
    assert f"Cannot connect to the sorcar daemon: no endpoint file at {endpoint}" in text


def test_error_names_default_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KISS_SORCAR_LOCAL", raising=False)
    text = _run(tmp_path)
    expected = daemon_client._resolve_endpoint_file(None)
    assert expected == tmp_path / "sorcar-local.json"
    assert f"Cannot connect to the sorcar daemon: no endpoint file at {expected}" in text
