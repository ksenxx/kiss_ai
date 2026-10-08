# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run_parallel`` child that cannot start is one entry of the result, not the whole result.

``run_agents_parallel`` starts one ``run_agent`` job per task.  When a
later task's start fails (here: an empty task text), the children
already started are still awaited and their results returned in task
order, with the start error standing in for the failed task; when the
first task fails nothing has started and the error alone comes back.
A real daemon stand-in (:class:`RecordingDaemon`) answers every child
that does start.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, cron_agent
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


def test_later_start_error_is_that_tasks_entry(tmp_path: Path, daemon: RecordingDaemon) -> None:
    script = tmp_path / "helper_sea.py"
    script.write_text(SEA, encoding="utf-8")
    out = run_agents_parallel(str(tmp_path), ["first", "   ", "third"], str(script), timeout="30")
    results = yaml.safe_load(out)
    assert len(results) == 3
    assert results[1] == "Error: task must be a non-empty string."
    for entry in (results[0], results[2]):
        assert yaml.safe_load(entry)["success"] is True, entry
    assert sorted(call["prompt"] for call in daemon.run_commands) == ["first", "third"]
    assert agent_dispatch.agent_jobs_of(None) == {}, "no job may stay registered"


def test_first_start_error_starts_nothing(tmp_path: Path, daemon: RecordingDaemon) -> None:
    script = tmp_path / "helper_sea.py"
    script.write_text(SEA, encoding="utf-8")
    out = run_agents_parallel(str(tmp_path), ["   ", "second"], str(script), timeout="30")
    assert out == "Error: task must be a non-empty string."
    assert daemon.run_commands == []
