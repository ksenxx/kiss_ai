# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``cron_agent`` names the SAME daemon socket ``daemon_client`` connects to.

Redundancy found by the 2026-09-02 sorcar-infra audit:
``cron_agent._run_prompt_job`` re-implemented the socket-path
precedence (explicit argument, then ``KISS_SORCAR_SOCK``, then
``$KISS_HOME/sorcar.sock``) that :func:`daemon_client._resolve_sock_path`
already owns, just to name the socket in its "cannot reach the daemon"
error.  Two copies of a precedence rule drift; the message now uses the
client's resolver.  These tests run the real (unreachable-daemon) path
for all three precedence levels and check the message names exactly
the socket the client tried.
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
    return {"id": "abcd1234", "prompt": "say hi", "max_budget": 0}


def test_error_names_explicit_socket(tmp_path: Path) -> None:
    sock = tmp_path / "explicit.sock"
    status, text = cron_agent._run_prompt_job(_job(), sock_path=str(sock))
    assert status == "error"
    assert text is not None
    assert f"cannot reach the kiss-web daemon at {sock}:" in text
    assert str(daemon_client._resolve_sock_path(str(sock))) == str(sock)


def test_error_names_env_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sock = tmp_path / "env.sock"
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(sock))
    status, text = cron_agent._run_prompt_job(_job())
    assert status == "error"
    assert text is not None
    assert f"cannot reach the kiss-web daemon at {sock}:" in text


def test_error_names_default_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KISS_SORCAR_SOCK", raising=False)
    status, text = cron_agent._run_prompt_job(_job())
    assert status == "error"
    assert text is not None
    expected = daemon_client._resolve_sock_path(None)
    assert expected == tmp_path / "sorcar.sock"
    assert f"cannot reach the kiss-web daemon at {expected}:" in text
