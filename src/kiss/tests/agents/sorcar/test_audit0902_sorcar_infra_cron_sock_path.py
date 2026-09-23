# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``cron_agent`` prompt jobs reach the SAME daemon socket ``run_agent`` uses.

Redundancy found by the 2026-09-02 sorcar-infra audit:
``cron_agent._run_prompt_job`` once re-implemented the socket-path
precedence that :func:`daemon_client._resolve_sock_path` already owns,
just to name the socket in its "cannot reach the daemon" error.  A
prompt job is now launched as a generated SEA through the ``run_agent``
tool (:func:`agent_dispatch.make_run_agent_tool`), whose dispatch
resolves the socket exactly once: the daemon socket recorded by
:func:`cron_agent.start_scheduler_thread` (read through the canonical
module by :func:`agent_dispatch._daemon_sock_path`), then
``KISS_SORCAR_SOCK``, then ``$KISS_HOME/sorcar.sock``.  These tests run
the real (unreachable-daemon) path for all three precedence levels and
check the recorded error names exactly the socket the client tried.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent, daemon_client


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    return tmp_path


def _job() -> dict[str, object]:
    return {"id": "abcd1234", "name": "hi", "prompt": "say hi", "max_budget": 0}


def _run(tmp_path: Path) -> str:
    work_dir = tmp_path / "run"
    work_dir.mkdir(exist_ok=True)
    status, text = cron_agent._run_prompt_job(_job(), work_dir)
    assert status == "error"
    assert text is not None
    return text


def test_error_names_recorded_daemon_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sock = tmp_path / "recorded.sock"
    # The env socket must lose to the one the hosting daemon recorded.
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(tmp_path / "env.sock"))
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", str(sock))
    text = _run(tmp_path)
    assert f"Cannot connect to the sorcar daemon at {sock}:" in text
    assert str(daemon_client._resolve_sock_path(str(sock))) == str(sock)


def test_error_names_env_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sock = tmp_path / "env.sock"
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(sock))
    text = _run(tmp_path)
    assert f"Cannot connect to the sorcar daemon at {sock}:" in text


def test_error_names_default_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KISS_SORCAR_SOCK", raising=False)
    text = _run(tmp_path)
    expected = daemon_client._resolve_sock_path(None)
    assert expected == tmp_path / "sorcar.sock"
    assert f"Cannot connect to the sorcar daemon at {expected}:" in text
