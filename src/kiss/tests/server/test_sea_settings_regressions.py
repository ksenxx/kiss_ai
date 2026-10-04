# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the review findings of the SEA ``settings()`` refactor.

Each test pins one defect a read-only review demonstrated against the
first cut of the ``settings()`` contract:

* a ``toolProfile`` of ``" none "`` (whitespace around the name) still
  resolved as a profile but no longer suppressed the built-in and
  inherited tools;
* a ``settings()`` value whose own methods raise (an untrusted ``str``
  or number subclass) escaped :func:`resolve_settings` as the raw
  exception instead of a :exc:`SettingsError` naming the source;
* an unknown settings key was accepted silently when its value was
  ``None``;
* a ``/xxx text`` run of a ``channel``-preset SEA worked in the
  project directory instead of ``~/.kiss/channel_work``, unlike the
  same SEA dispatched through ``run_agent``;
* the terminal persistence of a slash-command run stored the stripped
  SEA task instead of the raw ``/xxx text`` the user typed.

The daemon tests drive a real daemon (:class:`DaemonRunApiHarness`)
with a stub executor loop; nothing is mocked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.persistence import _get_db, _rw_lock
from kiss.agents.sorcar.sea_settings import SettingsError, resolve_settings
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import agent_state, sorcar
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness
from kiss.tests.server.test_sea_command_dispatch import _seed_seas_md


class _RaisingStr(str):
    """A ``str`` whose ``strip`` raises, as untrusted script code may return."""

    __slots__ = ()

    def strip(self, chars: str | None = None) -> str:  # type: ignore[override]
        raise RuntimeError("strip exploded")


class _RaisingNumber(int):
    """An ``int`` whose ``__float__`` raises."""

    __slots__ = ()

    def __float__(self) -> float:
        raise RuntimeError("float exploded")


class _OverflowingNumber(int):
    """An ``int`` too large for a float: ``float()`` raises ``OverflowError``."""

    __slots__ = ()

    def __float__(self) -> float:
        raise OverflowError("too big")


def test_removed_prompt_settings_name_their_replacement() -> None:
    # ``prompt`` and ``system_prompt`` are functions now, not settings:
    # the old spellings fail with the replacement, not as unknown keys.
    namespace: dict[str, Any] = {"settings": lambda: {"prompt": _RaisingStr("x")}}
    with pytest.raises(
        SettingsError, match=r"settings\(\)\['prompt'\] is no longer a setting: define",
    ):
        resolve_settings(namespace)
    namespace = {"settings": lambda: {"system_prompt": "x"}}
    with pytest.raises(SettingsError, match=r"define `def system_prompt\(\) -> str` instead"):
        resolve_settings(namespace)


def test_broken_numeric_value_is_a_settings_error() -> None:
    namespace = {"settings": lambda: {"timeout": _RaisingNumber(5)}}
    with pytest.raises(SettingsError, match=r"settings\(\)\['timeout'\] returned a broken value"):
        resolve_settings(namespace)


def test_overflowing_numeric_value_is_reported_as_non_finite() -> None:
    namespace = {"settings": lambda: {"max_budget": _OverflowingNumber(1)}}
    with pytest.raises(SettingsError, match=r"max_budget'\] must return a finite number"):
        resolve_settings(namespace)


def test_finite_numbers_are_returned_as_floats() -> None:
    resolved = resolve_settings({"settings": lambda: {"timeout": 7, "max_budget": 2}})
    assert resolved["timeout"] == 7.0 and isinstance(resolved["timeout"], float)
    assert resolved["max_budget"] == 2.0 and math.isfinite(resolved["max_budget"])


def test_unknown_key_with_none_value_is_rejected() -> None:
    namespace = {"settings": lambda: {"preset": "worker", "tiemout": None}}
    with pytest.raises(SettingsError, match="unknown key 'tiemout'"):
        resolve_settings(namespace)


def test_known_key_with_none_value_is_dropped() -> None:
    resolved = resolve_settings({"settings": lambda: {"preset": "worker", "timeout": None}})
    assert "timeout" not in resolved
    assert resolved["preset"] == "worker"


def _history_tasks() -> list[str]:
    """Return the ``task`` text of every persisted task row, oldest first."""
    with _rw_lock.read_lock():
        rows = _get_db().execute(
            "SELECT task FROM task_history ORDER BY timestamp ASC, rowid ASC",
        ).fetchall()
    return [str(r["task"]) for r in rows]


def _probe_tool(note: str) -> str:
    """Record *note*; the parent script under test adds this tool to its run."""
    return note


def _child_tool(note: str) -> str:
    """Record *note*; the child script under test adds this tool to its run."""
    return note


class SeaSettingsDaemonRegressionTest(DaemonRunApiHarness):
    """Daemon-level regressions: tool-profile whitespace, channel work dir, history row."""

    def setUp(self) -> None:
        super().setUp()
        sea_commands._reset_for_tests()
        self.addCleanup(sea_commands._reset_for_tests)
        self.addCleanup((kiss_home() / "SEAS.md").unlink, missing_ok=True)

    def _record_runs(
        self, runs: list[dict[str, Any]], on_run: Callable[[str, str], None] | None = None,
    ) -> None:
        """Replace the executor LLM loop with a stub recording tools and the api tab.

        *on_run*, when given, is called inside the stub with the task's
        id and prompt before the stub returns, so a test can dispatch a
        child task while its parent is still running.
        """
        registry = self.server._vscode_server.tab_registry

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            if "task_description" not in arguments:
                return "result: prior progress\n"
            prompt = str(arguments["task_description"])
            runs.append({
                "prompt": prompt,
                "tool_names": [t.__name__ for t in (kwargs.get("tools") or [])],
                "work_dirs": [
                    str(entry["workDir"]) for entry in registry.snapshot()
                    if entry["tabId"].startswith("api-")
                ],
            })
            if on_run is not None:
                # ``self_agent`` is the inner executor the running
                # ``SorcarAgent`` (the state's ``agent``) delegates to.
                task_ids = [
                    str(task_id) for task_id, state in agent_state.agent_states.items()
                    if getattr(state.agent, "_current_executor", None) is self_agent
                ]
                assert len(task_ids) == 1, task_ids
                on_run(task_ids[0], prompt)
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
                )
            return raw

        KISSAgent.run = stub_run  # type: ignore[assignment,method-assign]

    def test_none_profile_with_whitespace_strips_the_run_to_the_script_tools(self) -> None:
        """``toolProfile=" none "`` behaves like ``"none"``: no built-in, no inherited tools.

        A parent whose script adds ``_probe_tool`` dispatches, while it
        runs, a child with ``inheritTools`` and the padded profile; the
        child's script adds ``_child_tool``.  The child must get exactly
        its own tool plus ``finish`` — the inherited ``_probe_tool`` is
        what an unstripped profile let through.
        """
        parent = Path(self.tmpdir) / "probe_agent.py"
        parent.write_text(
            "def add_to_tools():\n"
            "    from kiss.tests.server.test_sea_settings_regressions import _probe_tool\n"
            "    return [_probe_tool]\n",
        )
        child = Path(self.tmpdir) / "child_agent.py"
        child.write_text(
            "def add_to_tools():\n"
            "    from kiss.tests.server.test_sea_settings_regressions import _child_tool\n"
            "    return [_child_tool]\n",
        )
        runs: list[dict[str, Any]] = []
        child_results: list[Any] = []

        def dispatch_child(task_id: str, prompt: str) -> None:
            if "PARENT" not in prompt:
                return
            child_results.append(sorcar.run(
                "CHILD", work_dir=self.repo, use_worktree=False, auto_commit=False,
                extension_agent_path=str(child), tool_profile=" none ",
                parent_task_id=task_id, inherit_tools=True,
                endpoint_file=self.endpoint_file, timeout=60,
            ))

        self._record_runs(runs, dispatch_child)
        result = sorcar.run(
            "PARENT", work_dir=self.repo, use_worktree=False, auto_commit=False,
            extension_agent_path=str(parent),
            endpoint_file=self.endpoint_file, timeout=120,
        )
        assert result.success is True, result
        assert len(child_results) == 1 and child_results[0].success is True, child_results
        by_prompt = {run["prompt"]: run for run in runs}
        assert set(by_prompt) == {"# Task\nPARENT", "# Task\nCHILD"}, runs
        # The parent runs the built-in toolset plus its own tool.
        parent_tools = by_prompt["# Task\nPARENT"]["tool_names"]
        assert "_probe_tool" in parent_tools and "Bash" in parent_tools, parent_tools
        # The child's padded ``none`` profile keeps out both the
        # built-ins and the parent's ``_probe_tool``.
        child_tools = by_prompt["# Task\nCHILD"]["tool_names"]
        assert sorted(child_tools) == ["_child_tool", "finish"], child_tools

    def test_channel_preset_slash_command_runs_in_the_channel_scratch_dir(self) -> None:
        """``/chan text`` works in ``~/.kiss/channel_work``, as ``run_agent`` would."""
        _seed_seas_md(
            Path(self.tmpdir) / "user-seas", "chan",
            "def description():\n    return 'a channel'\n"
            "def settings():\n    return {'preset': 'channel'}\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/chan say hi", work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        scratch = str(kiss_home() / "channel_work")
        # The tab's registry entry (what every surface shows as the
        # run's directory) is re-pinned to the channel scratch directory.
        assert runs[0]["work_dirs"] == [scratch], runs[0]
        assert Path(scratch).is_dir()

    def test_session_preset_slash_command_keeps_the_project_dir(self) -> None:
        """A plain SEA's slash run stays in the calling project."""
        _seed_seas_md(
            Path(self.tmpdir) / "user-seas", "plain",
            "def description():\n    return 'plain'\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/plain do it", work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert runs[0]["work_dirs"] == [self.repo], runs[0]

    def test_slash_run_history_row_keeps_the_raw_prompt_after_the_run(self) -> None:
        """The persisted row reads ``/plain do it`` once the run has ended, not ``do it``."""
        _seed_seas_md(
            Path(self.tmpdir) / "user-seas", "plain",
            "def description():\n    return 'plain'\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/plain do it", work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert _history_tasks() == ["/plain do it"], _history_tasks()
