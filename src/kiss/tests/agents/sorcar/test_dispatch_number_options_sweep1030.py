# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``run_agent``'s ``max_budget`` / ``timeout`` arguments: every shape a caller can pass.

Audit 2026-10-30 (S6), SIMP: ``_parse_number`` lost its ``"no"``
sentinel (a string stored in the float variable to mean "not a
number").  These pin the parse through the real ``run_agent`` tool,
whose planning refuses a bad number before anything is dispatched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH, make_run_agent_tool


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, "max_budget must be a number, got True."),
        (False, "max_budget must be a number, got False."),
        ("cheap", "max_budget must be a number, got 'cheap'."),
        ([1], "max_budget must be a number, got [1]."),
        (10**400, f"max_budget must be a number, got {10**400!r}."),
        ("1e999", "max_budget must be a positive finite number, got '1e999'."),
        (" -2 ", "max_budget must be a positive finite number, got ' -2 '."),
        (0, "max_budget must be a positive finite number, got 0."),
    ],
)
def test_a_bad_number_is_refused_before_dispatch(tmp_path: Path, value: Any, error: str) -> None:
    run_agent = make_run_agent_tool(str(tmp_path))
    assert run_agent("task", DEFAULT_AGENT_PATH, max_budget=value) == f"Error: {error}"


def test_a_good_number_and_a_blank_one_pass_the_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    run_agent = make_run_agent_tool(str(tmp_path))
    # Past the parse the call is dispatched, and there is no daemon.
    for value in (" 2.5 ", 3, 0.25, "", "   ", None):
        out = run_agent("task", DEFAULT_AGENT_PATH, max_budget=value, timeout=value)
        assert out.startswith("Error: the sorcar agent task could not run: Cannot connect"), out
