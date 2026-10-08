# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Every sub-task result starts with the configuration it ran with (P3 / A4).

``reports/sea-run-agent-semantics-and-automation-2026-10-04.md``: a
``run_agent`` or ``run_parallel`` result used to be ``{success,
summary}``; nothing told the calling model that the SEA
replaced the ``tool_profile`` it asked for, which values the sub-task
inherited, or what budget it ran with, and nothing persisted that
record for ``rsi7d`` to mine.  Now the daemon folds the effective
configuration into the task's ``task_settings`` event, the client
reads it back, and both tools put a ``ran:`` line first.

Driven over a real daemon on a loopback endpoint with the LLM boundary
stubbed (:class:`DaemonLocalHarness`), so the wire fields
(``provenance``, ``timeout``), the daemon's record and the persisted
event are the production ones.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.agent_dispatch import RunOptions, _run_agent, dispatch_result
from kiss.agents.sorcar.run_config import note_pinned, run_config_line, with_run_config
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness

pytestmark = pytest.mark.usefixtures("stubbed_agent_model")

MODELS = get_available_models() or ["parent-model-x", "other-model"]
PARENT_MODEL = MODELS[0]
OTHER_MODEL = MODELS[-1]
"""Models the daemon accepts (the LLM boundary is stubbed, so none is ever called)."""

SEA_SOURCE = textwrap.dedent("""
    from kiss.agents.seas.base.base_sea import BaseSea

    class Sea(BaseSea):
        def description(self):
            return "Echo test SEA: pins the shell tool profile and a budget."

        def settings(self, settings):
            return settings | {"tool_profile": "shell", "max_budget": 0.75, "use_memory": False}


""")

LOCKED_SEA_SOURCE = textwrap.dedent("""
    from kiss.agents.seas.base.base_sea import BaseSea

    class Sea(BaseSea):
        def description(self):
            return "Echo test SEA that locks its tool profile."

        def settings(self, settings):
            return settings | {"tool_profile": "shell", "locked": ["tool_profile"]}


""")


def _parent(repo: str) -> SorcarAgent:
    """A calling agent with the live state a run leaves behind (no model endpoint needed)."""
    parent = SorcarAgent("run-config-parent")
    parent.model_name = PARENT_MODEL
    parent._launch_model_name = PARENT_MODEL
    parent.max_budget = 4.0
    parent.budget_used = 1.0
    setattr(parent, "_chat_id", "chat-parent")  # noqa: B010 - private attribute of the real agent
    parent.work_dir = repo
    return parent


class RunConfigEchoTest(DaemonLocalHarness):
    """The ``ran:`` line and the persisted record, through the real daemon."""

    def setUp(self) -> None:
        super().setUp()
        self._saved_env = os.environ.get("KISS_SORCAR_LOCAL")
        os.environ["KISS_SORCAR_LOCAL"] = str(self.endpoint_file)
        if not get_available_models():
            self.skipTest("the daemon accepts a run only with a configured model")
        self.sea = Path(self.tmpdir) / "echo_sea.py"
        self.sea.write_text(SEA_SOURCE)
        self.plain_sea = Path(self.tmpdir) / "plain_sea.py"
        self.plain_sea.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 'A plain SEA.'
""")
        self.locked_sea = Path(self.tmpdir) / "locked_sea.py"
        self.locked_sea.write_text(LOCKED_SEA_SOURCE)
        self._stub_model()

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_LOCAL", None)
        else:
            os.environ["KISS_SORCAR_LOCAL"] = self._saved_env
        super().tearDown()

    def _stub_model(self) -> None:
        """Stub the LLM boundary: every run succeeds at once with a fixed summary."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            if printer is not None:
                printer.print(raw, type="result", step_count=1, total_tokens=5, cost="$0.0010")
            return raw

        self._parent_class.run = stub_run

    def _persisted_settings(self, task_id: str) -> dict[str, Any]:
        db = _persistence._get_db()
        with _persistence._rw_lock.read_lock():
            events = _persistence._fetch_events_for_task_id(db, task_id)
        settings = [e["settings"] for e in events if e.get("type") == "task_settings"]
        assert len(settings) == 1, events
        assert isinstance(settings[0], dict)
        return settings[0]

    def test_run_agent_result_starts_with_the_effective_configuration(self) -> None:
        """The explicit ``review`` wins over the script's ``shell``; the script beats the
        inherited budget."""
        parent = _parent(self.repo)
        out = _run_agent(
            self.repo,
            "say hi",
            agent=str(self.sea),
            tool_profile="review",
            timeout="45",
            parent_agent=parent,
        )
        parsed = yaml.safe_load(out)
        assert list(parsed) == ["ran", "success", "summary"], out
        assert parsed["success"] is True and parsed["summary"] == "done"
        ran = parsed["ran"]
        assert ran.startswith(f"echo model={PARENT_MODEL} tools=review budget=$0.75 "), (
            ran
        )
        assert "timeout=45s" in ran
        # Inherited from the parent: its model, its chat and half its remaining budget.
        inherited = ran.split("inherited=")[1].split(" ")[0].split(",")
        assert {"model", "chat_id", "max_budget"} <= set(inherited), ran
        # Only the inherited budget share was replaced; the explicit profile stood.
        assert ran.endswith("pinned=max_budget(1.5->0.75)"), ran

    def test_locked_setting_refuses_a_differing_explicit_argument(self) -> None:
        """A locked ``tool_profile`` is an error for ``review``, fine when asked for ``shell``."""
        parent = _parent(self.repo)
        out = _run_agent(
            self.repo,
            "say hi",
            agent=str(self.locked_sea),
            tool_profile="review",
            parent_agent=parent,
        )
        assert out == "Error: locked: the script locks tool_profile='shell' (asked for 'review')"
        out = _run_agent(
            self.repo,
            "say hi",
            agent=str(self.locked_sea),
            tool_profile="shell",
            parent_agent=parent,
        )
        assert yaml.safe_load(out)["ran"].startswith("locked model=")
        # The daemon enforces the lock too: a run command marked explicit is refused.
        result = dispatch_result(
            "locked",
            "say hi",
            str(self.locked_sea),
            self.repo,
            "",
            None,
            30.0,
            parent_agent=parent,
            inherit=True,
            options=RunOptions(tool_profile="review"),
            settings={"tool_profile": "shell", "locked": ["tool_profile"]},
        )
        assert not isinstance(result, str) and result.success is False, result
        assert "the script locks tool_profile='shell' (asked for 'review')" in result.text

    def test_dispatch_result_carries_the_settings_the_daemon_persisted(self) -> None:
        """``TaskResult.settings`` is the persisted ``task_settings`` payload."""
        parent = _parent(self.repo)
        result = dispatch_result(
            "echo",
            "say hi",
            str(self.sea),
            self.repo,
            "",
            None,
            30.0,
            parent_agent=parent,
            inherit=True,
            options=RunOptions(tool_profile="review", use_memory=True),
            settings={"tool_profile": "shell", "max_budget": 0.75, "use_memory": False},
        )
        assert not isinstance(result, str), result
        assert result.task_id
        settings = result.settings
        assert settings["sea"] == "echo"
        assert settings["channel"] is False
        assert settings["tool_profile"] == "review"
        assert settings["timeout"] == 30.0
        assert settings["max_budget"] == 0.75
        assert settings["model"] == PARENT_MODEL
        assert "chat_id" in settings["inherited"] and "model" in settings["inherited"]
        # The explicit profile and memory flag stood; the inherited budget share did not.
        assert settings["pinned"] == {"max_budget": [1.5, 0.75]}
        persisted = self._persisted_settings(result.task_id)
        for key in ("sea", "channel", "tool_profile", "timeout", "inherited", "pinned"):
            assert persisted[key] == settings[key], (key, persisted, settings)

    def test_plain_dispatch_records_no_sea_and_nothing_pinned(self) -> None:
        """Without a replacing script the line says so, and explicit values stay explicit."""
        parent = _parent(self.repo)
        result = dispatch_result(
            "plain",
            "say hi",
            str(self.plain_sea),
            self.repo,
            OTHER_MODEL,
            0.2,
            30.0,
            parent_agent=parent,
            inherit=True,
        )
        assert not isinstance(result, str), result
        settings = result.settings
        assert settings["sea"] == "plain" and settings["pinned"] == {}, result
        assert settings["model"] == OTHER_MODEL and settings["max_budget"] == 0.2
        assert "model" not in settings["inherited"]
        assert "max_budget" not in settings["inherited"]
        assert "chat_id" in settings["inherited"]
        line = run_config_line({**settings, "timeout": 30.0})
        assert line.startswith(f"plain model={OTHER_MODEL} tools=full budget=$0.20 ")
        assert line.endswith("pinned=none")



def test_with_run_config_prefixes_a_non_mapping_result() -> None:
    """A result that is not YAML mapping gets the line prefixed instead of re-dumped."""
    settings = {"sea": "x", "model": "m"}
    assert with_run_config("plain text", settings) == (
        "ran: x model=m tools=full budget=none timeout=none "
        "inherited=none pinned=none\nplain text"
    )
    assert with_run_config("- a\n- b\n", settings).startswith("ran: x model=m")
    assert with_run_config("key: [unclosed", settings).startswith("ran: x model=m")
    out = with_run_config("ran: old\nsuccess: true\n", settings)
    assert yaml.safe_load(out) == {
        "ran": run_config_line(settings),
        "success": True,
    }


def test_run_config_line_shortens_long_values() -> None:
    """Overridden values longer than 40 characters are cut; empty ones read ``empty``."""
    long = "x" * 60
    line = run_config_line({"pinned": {"work_dir": ["", long]}, "max_budget": 2})
    assert "work_dir(empty->" + "x" * 37 + "...)" in line
    assert "budget=$2.00" in line
    assert run_config_line({"inherited": []}).endswith("inherited=none pinned=none")


def test_note_pinned_reduces_dicts_to_their_keys_and_skips_unasked() -> None:
    """``model_config`` may hold an API key: only its key names are recorded."""
    pinned: dict[str, list[Any]] = {}
    note_pinned(
        pinned, "model_config", {"api_key": "SECRET", "base_url": "o"}, {"base_url": "n"}
    )
    note_pinned(pinned, "model", None, "m")
    note_pinned(pinned, "work_dir", "", "/x")
    note_pinned(pinned, "use_memory", False, False)
    note_pinned(pinned, "use_web_tools", False, True)
    assert pinned == {
        "model_config": ["dict(api_key, base_url)", "dict(base_url)"],
        "use_web_tools": [False, True],
    }
    assert "SECRET" not in run_config_line({"pinned": pinned})


def test_locked_conflicts_covers_dispatcher_keys_paths_and_inheritance(tmp_path: Path) -> None:
    """``timeout`` and ``inherit`` lock; relative ``work_dir`` compares resolved; locks inherit.

    A SEA extends another by Python inheritance: the derived class's
    ``settings`` receives the base's dict, so a base's lock survives
    unless the derived class rewrites ``locked`` — and joins its own
    lock by extending the list it was handed.
    """
    from kiss.agents.seas.base.base_sea import BaseSea
    from kiss.agents.sorcar.sea_commands import base_settings, declared_settings
    from kiss.agents.sorcar.sea_settings import (
        SeaError,
        locked_conflicts,
        resolve_settings,
    )

    settings = {
        "timeout": 10.0,
        "tool_profile": "bash",
        "work_dir": ".",
        "locked": ["timeout", "tool_profile", "work_dir"],
    }
    assert locked_conflicts(settings, {"timeout": 20.0}) == (
        "the script locks timeout=10.0 (asked for 20.0)"
    )
    assert locked_conflicts(settings, {"tool_profile": "review"}) == (
        "the script locks tool_profile='bash' (asked for 'review')"
    )
    assert locked_conflicts(settings, {"timeout": 10.0, "tool_profile": "bash"}) == ""
    # The same directory spelled relative and absolute is no clash; another one is.
    assert locked_conflicts(settings, {"work_dir": str(tmp_path)}, str(tmp_path)) == ""
    assert locked_conflicts(settings, {"work_dir": "sub"}, str(tmp_path)).startswith(
        "the script locks work_dir='.'"
    )
    # A lock set by a base survives the derived SEA that leaves ``locked``
    # alone; a derived SEA that extends the list it is handed joins its own.
    class LockingBase(BaseSea):
        def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            return settings | {"tool_profile": "bash", "locked": ["tool_profile"]}

    class KeepsLock(LockingBase):
        def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            return settings | {"max_budget": 1}

    class JoinsLock(LockingBase):
        def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
            return settings | {"max_budget": 1, "locked": [*settings["locked"], "max_budget"]}

    assert declared_settings([KeepsLock()]) == {
        "tool_profile": "bash", "locked": ["tool_profile"], "max_budget": 1,
    }
    assert base_settings([KeepsLock()])["locked"] == ["tool_profile"]
    assert base_settings([JoinsLock()])["locked"] == ["max_budget", "tool_profile"]
    assert locked_conflicts(base_settings([JoinsLock()]), {"max_budget": 2}) == (
        "the script locks max_budget=1.0 (asked for 2)"
    )
    with pytest.raises(
        SeaError, match=r"settings\(\)\['locked'\] may only name settings keys"
    ):
        resolve_settings({"locked": ["kind", "nope"]})
    with pytest.raises(SeaError, match="must be a positive number of seconds, got 0"):
        resolve_settings({"timeout": 0})
