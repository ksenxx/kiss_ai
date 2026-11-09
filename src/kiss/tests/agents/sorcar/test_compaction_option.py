# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``compaction`` run option of the Sorcar agents.

A real :class:`RelentlessAgent` runs against the scripted local model
server: the option it was given is the one every per-session executor
runs with (``KISSAgent.run(compaction=...)``), so a run given
``compaction=False`` keeps every old tool output whole where the default
run compacts them.  A headless ``run_sea`` of a SEA whose ``settings()``
returns ``compaction: False`` does the same through
:meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`, and so does
the global Compaction setting saved off in ``$KISS_HOME/config.json``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.headless import run_sea
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import COMPACTION_START_TOKENS, STUB_PREFIX
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

URL_ENV = "COMPACTION_OPTION_SEA_URL"
COMPACTION_ENV = "COMPACTION_OPTION_SEA_SETTING"
# What ``big_output`` returns; a stub keeps only its first 200 characters.
WHOLE_OUTPUT = "line\n" * 4000

SEA_SOURCE = f'''\
import os
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea


def big_output() -> str:
    """Return a 20k-character tool output."""
    return "line\\n" * 4000


class ProbeSea(WorkerSea):
    """Fixture SEA on the local server; ``compaction: False`` when the environment says so."""

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        settings = settings | {{
            "model": {MODEL!r},
            "tool_profile": "none",
            "auto_classify": False,
            "model_config": {{
                "base_url": os.environ[{URL_ENV!r}], "api_key": "local", "plain_io": True,
            }},
        }}
        if os.environ.get({COMPACTION_ENV!r}) == "off":
            settings["compaction"] = False
        return settings

    def tools(self, tools: list[Any]) -> list[Any]:
        return [*tools, big_output]
'''


def big_output() -> str:
    """Return a 20k-character tool output."""
    return WHOLE_OUTPUT


def _script(steps_per_session: int) -> list[bytes]:
    """Two sessions: the first exhausts its steps on big outputs, the second finishes.

    The last call of session 1 reports a context of
    ``COMPACTION_START_TOKENS``, so the executor considers a compaction
    before the next model call of that session; the summarizer then
    writes the continuation and session 2 calls ``probe`` once and
    finishes.
    """
    script = [tool_call_body("big_output", {}, 3000) for _ in range(steps_per_session - 1)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    # One more call in session 1: the compaction, if any, happens before it.
    script.append(tool_call_body("probe", {}, COMPACTION_START_TOKENS))
    script.append(tool_call_body("finish", {"result": "<p>continue</p>"}, 500))
    script.append(tool_call_body("probe", {}, 500))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    return script


def _run(compaction: bool | None) -> tuple[list[bool], list[int], list[dict[str, Any]]]:
    """Run the two-session task; return the executors' options, the stub counts and requests.

    *compaction* ``None`` leaves the keyword out, so the default applies.
    """
    seen: list[bool] = []
    stubs: list[int] = []
    agent = RelentlessAgent("compaction-option-test")

    def probe() -> str:
        """Record the live session executor's option and how many outputs are stubs."""
        executor = agent._current_executor
        assert executor is not None
        seen.append(executor.compaction)
        stubs.append(sum(
            str(m.get("content", "")).startswith(STUB_PREFIX)
            for m in executor.model.conversation if m.get("role") == "tool"
        ))
        return "probed"

    kwargs: dict[str, Any] = {} if compaction is None else {"compaction": compaction}
    script = _script(31)
    with tempfile.TemporaryDirectory() as td, serve(script) as (url, requests):
        result = agent.run(
            model_name=MODEL,
            prompt_template="Call big_output repeatedly.",
            tools=[big_output, probe],
            max_steps=32,
            max_sub_sessions=2,
            max_budget=50.0,
            work_dir=td,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
            **kwargs,
        )
    assert "success: true" in result
    assert len(requests) == len(script)
    assert agent.compaction is (compaction is not False)
    return seen, stubs, list(requests)


@pytest.mark.parametrize("compaction", [None, True], ids=["default", "explicit"])
def test_default_run_compacts_in_the_session_executor(compaction: bool | None) -> None:
    """The default is on, and ``compaction=True`` spells the same thing out."""
    assert DEFAULT_CONFIG.tool_output_compaction is True
    seen, stubs, requests = _run(compaction)
    # Both sessions ran with the option on; session 1 compacted its old
    # outputs before the probe call, session 2 started from a fresh context.
    assert seen == [True, True]
    assert stubs[0] > 0
    assert stubs[1] == 0
    probe_request = requests[31]
    sent = [m for m in probe_request["messages"] if m.get("role") == "tool"]
    assert any(str(m["content"]).startswith(STUB_PREFIX) for m in sent)


def test_run_option_off_reaches_every_session_executor() -> None:
    assert DEFAULT_CONFIG.tool_output_compaction is True
    seen, stubs, requests = _run(False)
    assert seen == [False, False]
    assert stubs == [0, 0]
    # The whole first session went to the model uncompacted.
    probe_request = requests[31]
    sent = [m for m in probe_request["messages"] if m.get("role") == "tool"]
    assert len(sent) == 31
    assert all(WHOLE_OUTPUT in str(m["content"]) for m in sent)


def _headless_tool_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, global_switch: bool = True,
) -> list[str]:
    """Run the fixture SEA headless; return the tool results the finish request carried.

    *global_switch* is the persisted global Compaction setting of the
    isolated ``KISS_HOME`` (the settings panel's checkbox); the live
    value ``run_sea`` applies from it is restored at teardown.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    (home / "config.json").write_text(json.dumps({"compaction": global_switch}))
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_compaction", True)
    sea_path = tmp_path / "sea" / "probe_sea.py"
    sea_path.parent.mkdir()
    sea_path.write_text(SEA_SOURCE, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    script = [tool_call_body("big_output", {}, 3000) for _ in range(30)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    with serve(script) as (url, requests):
        monkeypatch.setenv(URL_ENV, url)
        monkeypatch.setenv(COMPACTION_ENV, setting)
        outcome = run_sea(
            sea_path, "Call big_output repeatedly.", work_dir=str(work), max_budget=5.0,
            save_trajectories=False,
        )
    assert outcome.error == "" and outcome.success, outcome
    assert len(requests) == len(script)
    sent = [m for m in requests[-1]["messages"] if m.get("role") == "tool"]
    assert len(sent) == 31
    return [str(m["content"]) for m in sent]


def test_headless_sea_setting_turns_compaction_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DEFAULT_CONFIG.tool_output_compaction is True
    results = _headless_tool_results(tmp_path, monkeypatch, "off")
    assert all(WHOLE_OUTPUT in text for text in results)


def test_headless_sea_without_the_setting_compacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _headless_tool_results(tmp_path, monkeypatch, "on")
    assert results[0].startswith(STUB_PREFIX)
    assert not results[-1].startswith(STUB_PREFIX)


def test_headless_run_honours_the_persisted_global_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global Compaction setting saved off turns a headless run's compaction off."""
    results = _headless_tool_results(tmp_path, monkeypatch, "on", global_switch=False)
    assert DEFAULT_CONFIG.tool_output_compaction is False
    assert all(WHOLE_OUTPUT in text for text in results)
