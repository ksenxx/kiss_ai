# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run_parallel`` call whose children cannot start returns the error alone.

``run_agents_parallel`` makes one plan per call and starts one
``run_agent`` job per task from it: the checks are the same for every
task, so when the plan fails (an empty first task, an unresolvable
agent) nothing has started and the error alone comes back; the SEA is
loaded once however many tasks there are.  A real daemon stand-in
(:class:`RecordingDaemon`) answers every child that does start.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.agent_dispatch import run_agents_parallel
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

SEA = '''from kiss.agents.seas.base.base_sea import BaseSea


class HelperSea(BaseSea):
    def description(self):
        return "a helper"
'''


@pytest.fixture()
def daemon(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingDaemon]:
    stand_in = RecordingDaemon(cost=0.25, tokens=10, steps=1, chat_id="chat-child")
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(stand_in.endpoint_file))
    try:
        yield stand_in
    finally:
        stand_in.close()


def test_first_start_error_starts_nothing(tmp_path: Path, daemon: RecordingDaemon) -> None:
    script = tmp_path / "helper_sea.py"
    script.write_text(SEA, encoding="utf-8")
    out = run_agents_parallel(str(tmp_path), ["   ", "second"], str(script), timeout="30")
    assert out == "Error: task must be a non-empty string."
    assert daemon.run_commands == []


def test_unresolvable_agent_starts_nothing(tmp_path: Path, daemon: RecordingDaemon) -> None:
    out = run_agents_parallel(str(tmp_path), ["first", "second"], "no-such-agent-xyz", timeout="30")
    assert out.startswith("Error:"), out
    assert daemon.run_commands == []


def test_sea_is_loaded_once_for_every_task(tmp_path: Path, daemon: RecordingDaemon) -> None:
    """The plan is the same for every task, so the SEA file is executed once per call."""
    loads = tmp_path / "loads.txt"
    script = tmp_path / "helper_sea.py"
    script.write_text(
        f"open({str(loads)!r}, 'a').write('load\\n')\n{SEA}", encoding="utf-8",
    )
    out = run_agents_parallel(str(tmp_path), ["a", "b", "c"], str(script), timeout="30")
    assert len(yaml.safe_load(out)) == 3
    assert sorted(call["prompt"] for call in daemon.run_commands) == ["a", "b", "c"]
    assert loads.read_text(encoding="utf-8").count("load") == 1
