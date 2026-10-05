# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA / ``run_agent`` semantics after the 2026-10-04 simplification.

``reports/archive/sea-run-agent-2026-10-04/sea-run-agent-semantics-assessment-2026-10-04.md``
§3 proposed
nine changes; each has a test here that drives the real code paths
(the settings loader, the daemon-side ``apply_agent_overrides``, the
dispatcher with a captured ``daemon_client.run``, a real
``VSCodeServer`` run for the classifier, and the fan-out engine):

* P1 — one ``kind`` axis (``session`` / ``worker`` / ``channel``): a
  kind is a dict of defaults under the explicit keys; the daemon and
  the dispatcher key on ``kind: "channel"`` (scratch ``work_dir``,
  workspace, preamble, no inheritance).
* P2 — ``run_parallel`` refuses, with a message, a script or an
  ``options`` object pinning what a fan-out child cannot honour.
* P3 — ``add_to_prompt`` is not a setting; ``{task_id}`` is
  substituted in what ``prompt(task)`` returns.
* P4 — the classifier never demotes a ``use_worktree`` a script pinned.
* P5 — ``run_agent`` and ``run_parallel`` share one argument order;
  ``workspace`` and ``tool_profile`` moved (into ``options`` / out of it).
* P6 — ``cron`` is the registered command ``/cron``; ``agent`` resolves
  by three rules and an unknown name gets a did-you-mean.
* P7 — ``model: ""`` in ``settings()`` means "no override".
* P8 — ``run_agent(..., wait="false")`` returns a job id; ``agent_job``
  waits on, reports or kills the job.
* P9 — ``/<name> check`` reports what a run of the SEA would use.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent, daemon_client, sea_commands
from kiss.agents.sorcar.agent_dispatch import fanout_conflict, make_run_agent_tool
from kiss.agents.sorcar.agent_file import (
    CHANNEL_PREAMBLE,
    DISPATCHER_SETTINGS,
    SETTING_FIELDS,
    apply_agent_overrides,
    channel_workspace,
    is_channel,
    load_layers,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_commands import SeaScriptError, sea_layers, sea_settings
from kiss.agents.sorcar.sea_settings import (
    KINDS,
    SETTING_TYPES,
    WORKER_DEFAULTS,
    SettingsError,
    kind_defaults,
    resolve_settings,
)
from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES
from kiss.agents.sorcar.task_classifier import clear_classification_cache
from kiss.core import config as config_module
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    isolated = IsolatedKissHome(prefix="kiss-sea-proposals-")
    isolated.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Capture every ``daemon_client.run`` call's keyword arguments instead of dispatching."""
    calls: list[dict[str, Any]] = []

    def fake_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        calls.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.0, tokens=1, steps=1, chat_id="c",
        )

    monkeypatch.setattr(daemon_client, "run", fake_run)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    return calls


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _bare_agent(work_dir: Path) -> ChatSorcarAgent:
    """A parent agent with the attributes ``run()`` sets before ``_get_tools``."""
    agent = ChatSorcarAgent("proposals-parent")
    agent._use_web_tools = False
    agent._is_parallel = True
    agent._use_memory_override = None
    agent._append_basic_tools = True
    agent.web_use_tool = None
    agent._memory_tools = None
    agent.work_dir = str(work_dir)
    agent.docker_manager = None
    agent.printer = None
    return agent


# ---------------------------------------------------------------------------
# P1 / P2 (2026-10-04 round two) — one ``kind`` axis: session | worker | channel
# ---------------------------------------------------------------------------


def test_kinds_are_pure_dicts_and_channel_is_worker_plus_work_dir(
    home: IsolatedKissHome,
) -> None:
    table = kind_defaults()
    assert tuple(table) == KINDS == ("session", "worker", "channel")
    assert table["session"] == {}
    assert table["worker"] == WORKER_DEFAULTS
    assert table["channel"] == {**WORKER_DEFAULTS, "work_dir": str(home.kiss_home / "channel_work")}
    # A kind is defaults only: an explicit key wins; the kind itself is kept.
    resolved = resolve_settings({"settings": lambda: {"kind": "channel", "work_dir": "/w"}})
    assert resolved["kind"] == "channel" and resolved["work_dir"] == "/w"
    assert resolve_settings({"settings": lambda: {}})["kind"] == "session"
    with pytest.raises(
        SettingsError,
        match="settings\\(\\)\\['kind'\\] must be one of session, worker, channel; got 'agent'",
    ):
        resolve_settings({"settings": lambda: {"kind": "agent"}})
    # The former second axis is gone: ``preset`` is a renamed key, ``inherit`` a removed one.
    with pytest.raises(
        SettingsError, match="key 'preset' was renamed to 'kind'; run `uv run sea lint --fix`",
    ):
        resolve_settings({"settings": lambda: {"preset": "worker"}})
    with pytest.raises(
        SettingsError, match="key 'inherit' was removed: a `channel` run never inherits",
    ):
        resolve_settings({"settings": lambda: {"inherit": False}})
    assert "preset" not in SETTING_TYPES and "inherit" not in SETTING_TYPES
    # None of the script-only keys travels the wire.
    assert set(DISPATCHER_SETTINGS) == {"kind", "extends", "timeout", "locked", "hidden"}
    assert not set(DISPATCHER_SETTINGS) & set(SETTING_FIELDS)


def test_daemon_keys_on_the_channel_kind(home: IsolatedKissHome, tmp_path: Path) -> None:
    """Workspace, preamble, scratch work_dir and the extends refusal follow ``kind: "channel"``."""
    channel = _write(
        tmp_path / "chan" / "chan_sea.py",
        "def description():\n    return 'p'\ndef settings():\n    return {'kind': 'channel'}\n",
    )
    worker = _write(
        tmp_path / "worker" / "worker_sea.py",
        "def description():\n    return 'w'\ndef settings():\n    return {'kind': 'worker'}\n",
    )
    cmd: dict[str, Any] = {"agentPath": str(channel), "prompt": "p", "workspace": "acct"}
    layers = load_layers(cmd)
    assert is_channel(layers)
    assert channel_workspace(cmd, layers) == "acct"
    assert "workDir" in apply_agent_overrides(cmd, layers)
    assert cmd["workDir"] == str(home.kiss_home / "channel_work")
    preamble = CHANNEL_PREAMBLE.format(name="chan")
    assert cmd["appendToSystemPrompt"].startswith(preamble)
    layers = load_layers({"agentPath": str(worker)})
    assert not is_channel(layers) and channel_workspace({"agentPath": str(worker)}, layers) == ""
    derived = _write(
        tmp_path / "derived" / "derived_sea.py",
        f"def description():\n    return 'd'\n"
        f"def settings():\n    return {{'extends': {str(channel)!r}}}\n",
    )
    with pytest.raises(SeaScriptError, match="cannot extend the channel agent script"):
        sea_layers(derived)


def test_dispatcher_inherits_unless_channel_or_inherit_false(
    captured: list[dict[str, Any]], tmp_path: Path, home: IsolatedKissHome,
) -> None:
    """The kind and the call's ``inherit`` option decide the dispatch; a channel never inherits."""
    plain = _write(tmp_path / "plain_sea.py", "def settings():\n    return {}\n")
    channel = _write(
        tmp_path / "chan_sea.py", "def settings():\n    return {'kind': 'channel'}\n",
    )
    run_agent = make_run_agent_tool(str(home.repo))
    run_agent("t", str(plain))
    assert captured[-1]["inherit_tools"] is True
    assert captured[-1]["work_dir"] == str(home.repo)
    run_agent("t", str(plain), options='{"inherit": false}')
    assert captured[-1]["inherit_tools"] is False
    assert captured[-1]["work_dir"] == str(home.repo)
    n = len(captured)
    run_agent("t", str(channel))
    assert captured[-1]["inherit_tools"] is False
    assert captured[-1]["work_dir"] == str(home.kiss_home / "channel_work")
    assert captured[-1]["workspace"] == ""
    run_agent("t", str(channel), options='{"workspace": " acct "}')
    assert captured[-1]["workspace"] == "acct"
    out = run_agent("t", str(channel), options='{"inherit": true}')
    assert out == "Error: chan: a channel agent never inherits from the calling task"
    assert len(captured) == n + 2  # the refused call never dispatched


# ---------------------------------------------------------------------------
# P2 — run_parallel refuses what a child cannot honour
# ---------------------------------------------------------------------------


def test_fanout_conflict_is_one_rule_for_settings_and_options() -> None:
    assert fanout_conflict({}) == ""
    unpinned = {"use_worktree": False, "auto_commit": False, "auto_classify": False}
    assert fanout_conflict(unpinned) == ""
    channel = fanout_conflict({"kind": "channel"})
    assert channel.startswith("is a channel agent")
    for key in ("use_worktree", "auto_commit", "auto_classify"):
        assert f"pins {key}: true" in fanout_conflict({key: True})
    assert "pins chat_id" in fanout_conflict({"chat_id": ""})
    assert "names a workspace" in fanout_conflict({"workspace": "acct"})
    # ``timeout`` and ``work_dir`` are not refused: ignored and honoured respectively.
    assert fanout_conflict({"timeout": 60, "work_dir": "/w"}) == ""


def test_run_parallel_refuses_pinned_settings_and_options_loudly(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = home.repo
    _write(repo / "wt_sea.py", "def settings():\n    return {'use_worktree': True}\n")
    _write(repo / "chat_sea.py", "def settings():\n    return {'chat_id': 'c-9'}\n")
    _write(
        repo / "fine_sea.py",
        "def settings():\n    return {'timeout': 5, 'use_worktree': False}\n",
    )
    agent = _bare_agent(repo)
    fanned: list[dict[str, Any]] = []

    def fake_fanout(tasks: list[str], **kwargs: Any) -> list[str]:
        fanned.append({"tasks": tasks, **kwargs})
        return ["- success: true\n  summary: ok"] * len(tasks)

    monkeypatch.setattr("kiss.agents.sorcar.sorcar_agent.run_tasks_parallel", fake_fanout)
    monkeypatch.setattr(agent, "reclaim_abandoned_subagents", lambda: None)
    run_parallel = next(t for t in agent._get_tools() if t.__name__ == "run_parallel")
    out = run_parallel('["a"]', agent="wt_sea.py")
    assert out.startswith("Error: wt pins use_worktree: true, which a run_parallel child")
    assert "use run_agent" in out
    out = run_parallel('["a"]', agent="chat_sea.py")
    assert out.startswith(
        "Error: chat pins chat_id, but a run_parallel child runs in the caller's chat",
    )
    out = run_parallel('["a"]', agent="ntfy")
    assert out == "Error: ntfy is a channel agent, which run_parallel cannot run; use run_agent"
    out = run_parallel('["a"]', options='{"auto_commit": true}')
    assert out.startswith("Error: run_parallel options pins auto_commit: true")
    out = run_parallel('["a"]', options='{"workspace": "acct"}')
    assert out.startswith("Error: run_parallel options names a workspace")
    assert fanned == []
    # A script pinning only what a child does anyway (or a timeout) runs;
    # an explicit option wins over its settings, so one a child cannot
    # honour is refused even when the script pins the opposite.
    out = run_parallel(
        '["a", "b"]', agent="fine_sea.py", max_budget="0.5",
        options='{"work_dir": "sub", "use_worktree": true}',
    )
    assert out.startswith("Error: run_parallel options pins use_worktree: true")
    out = run_parallel(
        '["a", "b"]', agent="fine_sea.py", max_budget="0.5", options='{"work_dir": "sub"}',
    )
    assert "success: true" in out
    (call,) = fanned
    assert call["max_budget"] == 0.5
    assert call["work_dir"] == str(repo / "sub")
    assert call["tool_profile"] == ""


# ---------------------------------------------------------------------------
# P3 — add_to_prompt folded into prompt(task)
# ---------------------------------------------------------------------------


def test_add_to_prompt_setting_is_rejected_and_task_id_lives_in_prompt(tmp_path: Path) -> None:
    old = _write(tmp_path / "old_sea.py", "def settings():\n    return {'add_to_prompt': 'x'}\n")
    with pytest.raises(SeaScriptError, match=r"settings\(\) has an unknown key 'add_to_prompt'"):
        sea_settings(old)
    assert "add_to_prompt" not in SETTING_TYPES
    new = _write(
        tmp_path / "new_sea.py",
        "def prompt(task):\n    return task + ' (about task {task_id})'\n",
    )
    cmd: dict[str, Any] = {"agentPath": str(new), "prompt": "why?", "parentTaskId": "T-7",
                           "appendToPrompt": "caller {task_id}"}
    assert apply_agent_overrides(cmd) == {"prompt"}
    assert cmd["prompt"] == "why? (about task T-7)"
    assert cmd["appendToPrompt"] == "caller {task_id}"
    # The non-empty check applies to the text AFTER substitution: with no
    # parent task a prompt made only of the placeholder is empty.
    only_id = _write(tmp_path / "only_id_sea.py", "def prompt(task):\n    return '{task_id}'\n")
    layers = sea_layers(only_id)
    assert sea_commands.sea_prompt(layers, "hello", "T-7") == "T-7"
    with pytest.raises(SeaScriptError, match="prompt\\(\\) of agent script .* non-empty string"):
        sea_commands.sea_prompt(layers, "hello", "")
    # The option form of an appended text stays available to a caller.
    assert agent_dispatch.OPTION_TYPES["add_to_prompt"] is str
    assert run_tool_doc_mentions("add_to_prompt")


def run_tool_doc_mentions(key: str) -> bool:
    """Return whether the ``run_agent`` tool's docstring names *key*."""
    return key in (make_run_agent_tool("").__doc__ or "")


# ---------------------------------------------------------------------------
# P5 / P6 / P7 — aligned signatures, one resolution rule, "" model
# ---------------------------------------------------------------------------


def test_run_agent_and_run_parallel_share_one_argument_order(home: IsolatedKissHome) -> None:
    tools = {t.__name__: t for t in _bare_agent(home.repo)._get_tools()}
    run_agent = list(inspect.signature(tools["run_agent"]).parameters)
    run_parallel = list(inspect.signature(tools["run_parallel"]).parameters)
    assert run_agent == [
        "task", "agent", "model", "tool_profile", "max_budget", "timeout", "options", "wait",
    ]
    assert run_parallel == [
        "tasks", "agent", "model", "tool_profile", "max_budget", "timeout", "max_workers",
        "options",
    ]
    assert run_agent[1:6] == run_parallel[1:6]
    agent_job = list(inspect.signature(tools["agent_job"]).parameters)
    assert agent_job == ["job_id", "action", "timeout_seconds"]
    assert TOOL_PROFILES["agents"] == {"run_agent", "agent_job", "run_parallel", "number_of_cores"}
    # ``tool_profile`` is an argument of both and may repeat as an option key.
    assert "tool_profile" in agent_dispatch.OPTION_TYPES
    assert "workspace" in agent_dispatch.OPTION_TYPES and "kind" not in agent_dispatch.OPTION_TYPES


def test_cron_is_a_registered_command_and_agent_resolves_by_three_rules(
    home: IsolatedKissHome,
) -> None:
    commands = sea_commands.refresh_registry()
    assert "cron" in commands and "ntfy" in commands
    assert sea_commands.get_command("cron") == Path(cron_agent.__file__).resolve()
    assert sea_commands.help_text_if_command("/cron help") == cron_agent.description()
    assert sea_commands.slash_command_task("/cron list the jobs") == (
        "list the jobs", Path(cron_agent.__file__).resolve(),
    )
    resolve = agent_dispatch.resolve_agent
    assert resolve("", "") == (agent_dispatch.DEFAULT_AGENT_PATH, "sorcar")
    assert resolve("general", "") == (agent_dispatch.DEFAULT_AGENT_PATH, "sorcar")
    assert str(resolve("reviewer", "")).startswith("Error: 'reviewer' is not an agent.")
    assert resolve("cron", "") == (str(Path(cron_agent.__file__).resolve()), "cron")
    assert resolve("CRON", "") == (str(Path(cron_agent.__file__).resolve()), "cron")
    resolved = resolve("Home-Assistant", "")
    assert isinstance(resolved, tuple) and resolved[1] == "homeassistant"
    assert Path(resolved[0]).parts[-2:] == ("homeassistant", "homeassistant_sea.py")
    resolved = resolve("write_paper", "")
    assert isinstance(resolved, tuple) and resolved[1] == "write_paper"
    assert Path(resolved[0]).parts[-2:] == ("write_paper", "write_paper_sea.py")
    error = resolve("cronn", "")
    assert isinstance(error, str)
    assert error.startswith(
        "Error: unknown agent 'cronn' — not a registered slash command and not a path",
    )
    assert "Did you mean 'cron'?" in error and "Commands: " in error
    assert "Available channels" not in error


def test_an_empty_model_setting_means_no_override(
    captured: list[dict[str, Any]], tmp_path: Path,
) -> None:
    blank = _write(tmp_path / "blank_sea.py", "def settings():\n    return {'model': ''}\n")
    assert "model" not in sea_settings(blank)
    none = _write(tmp_path / "none_sea.py", "def settings():\n    return {'model': None}\n")
    assert sea_settings(none) == sea_settings(blank) == {"kind": "session"}
    named = _write(tmp_path / "named_sea.py", "def settings():\n    return {'model': 'm-1'}\n")
    assert sea_settings(named)["model"] == "m-1"
    # Other empty strings keep their meaning (``chat_id: ""`` is a fresh chat).
    fresh = _write(tmp_path / "fresh_sea.py", "def settings():\n    return {'chat_id': ''}\n")
    assert sea_settings(fresh)["chat_id"] == ""


# ---------------------------------------------------------------------------
# P9 — /<name> check
# ---------------------------------------------------------------------------


def test_slash_check_reports_the_effective_run_or_the_first_error(
    home: IsolatedKissHome, tmp_path: Path,
) -> None:
    folder = tmp_path / "seas"
    _write(
        folder / "good" / "good_sea.py",
        "def description():\n    return 'Good things.'\n"
        "def settings():\n    return {'kind': 'worker', 'model': 'm-1', 'timeout': 60}\n"
        "def prompt(task):\n    return '[good] ' + task + ' #{task_id}'\n"
        "def add_to_system_prompt():\n    return 'protocol'\n"
        "def probe(x: str) -> str:\n    \"\"\"Probe x.\"\"\"\n    return x\n"
        "def add_to_tools():\n    return [probe]\n",
    )
    _write(
        folder / "broken" / "broken_sea.py",
        "def description():\n    return 'Broken.'\n"
        "def settings():\n    return {'kind': 'nope'}\n",
    )
    (home.kiss_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    report = sea_commands.help_text_if_command("/good check")
    assert report is not None
    lines = report.splitlines()
    assert lines[0] == "/good: Good things."
    assert lines[1] == "layers: good"
    assert lines[2] == "kind: worker"
    settings = json.loads(lines[3].removeprefix("settings: "))
    assert settings["model"] == "m-1" and settings["timeout"] == 60
    assert settings["use_worktree"] is False
    assert lines[4] == "model: m-1"
    assert lines[5] == "tools added: probe"
    assert lines[6] == "getters and hooks defined: add_to_system_prompt, prompt"
    assert lines[7] == "prompt for <the task text>: [good] <the task text> #<task id>"
    broken = sea_commands.help_text_if_command("/broken CHECK")
    assert broken is not None and broken.startswith("/broken is broken: agent script ")
    assert "settings()['kind'] must be one of session, worker, channel; got 'nope'" in broken
    # The picker getters are validated too, and a tool without a
    # ``__name__`` (a partial) is reported by its type, not a crash.
    picker = _write(
        tmp_path / "picker_sea.py",
        "import functools\n"
        "def description():\n    return 'Picker.'\n"
        "def register_as_model():\n    return True\n"
        "def on_picked_as_model(work_dir):\n    return 'ok'\n"
        "def add_to_tools():\n    return [functools.partial(print, 'x')]\n",
    )
    report = sea_commands.sea_check("picker", picker)
    assert "tools added: partial" in report
    assert "getters and hooks defined: register_as_model, on_picked_as_model" in report
    bad_register = _write(
        tmp_path / "bad_register_sea.py",
        "def description():\n    return 'Bad.'\n"
        "def register_as_model():\n    return 'yes'\n",
    )
    assert "register_as_model() of agent script " in sea_commands.sea_check("b", bad_register)
    assert "must return a bool, got str" in sea_commands.sea_check("b", bad_register)
    for body in (
        "def on_picked_as_model():\n    return 17\n",
        "def on_picked_as_model(*, work_dir):\n    return 'ok'\n",
    ):
        bad_picked = _write(
            tmp_path / "bad_picked_sea.py", "def description():\n    return 'Bad.'\n" + body,
        )
        report = sea_commands.sea_check("b", bad_picked)
        assert "must take one positional argument (work_dir)" in report, report
    not_callable = _write(
        tmp_path / "bad_picked_sea.py",
        "def description():\n    return 'Bad.'\non_picked_as_model = None\n",
    )
    report = sea_commands.sea_check("b", not_callable)
    assert report.startswith("/b is broken: on_picked_as_model of agent script "), report
    assert "must be a callable, got NoneType" in report
    # A builtin without an introspectable signature is callable with one
    # argument at pick time; the check accepts it instead of crashing.
    builtin = _write(
        tmp_path / "builtin_picked_sea.py",
        "def description():\n    return 'Builtin.'\non_picked_as_model = str\n",
    )
    assert sea_commands.sea_check("b", builtin).startswith("/b: Builtin.")
    # ``check`` is reserved like ``help``: it never runs the SEA.
    assert sea_commands.slash_command_task("/good check") is None
    assert sea_commands.slash_command_task("/good check this") == (
        "check this", (folder / "good" / "good_sea.py").resolve(),
    )
    assert sea_commands.help_text_if_command("/nosuch check") is None


# ---------------------------------------------------------------------------
# P8 — wait="false" and agent_job, against a real daemon
# ---------------------------------------------------------------------------


class _Daemon:
    """A private kiss-web daemon in the isolated home (the real dispatch target)."""

    def __init__(self, home: IsolatedKissHome) -> None:
        import asyncio

        from kiss.server.web_server import RemoteAccessServer

        self.endpoint_file = str(home.tmpdir / "sorcar-local.json")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._server = RemoteAccessServer(
            local_endpoint_file=self.endpoint_file, work_dir=str(home.repo),
        )
        started = asyncio.run_coroutine_threadsafe(self._server.start_private_async(), self._loop)
        started.result(timeout=30)

    def stop(self) -> None:
        import asyncio

        asyncio.run_coroutine_threadsafe(self._server.stop_async(), self._loop).result(timeout=15)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


_JOB_ID = re.compile(r"job (agent-[0-9a-f]{8})")


def test_run_agent_wait_false_returns_a_job_the_parent_waits_on_or_kills(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent starts two background children; one finishes, one is killed."""
    from kiss.server import sorcar

    daemon = _Daemon(home)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", daemon.endpoint_file)
    release = threading.Event()
    parent_steps: list[str] = []
    children_seen: list[str] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        # A child inherits the parent's chat, so its context quotes the
        # parent's task; its own task follows the ``# Task (work on it
        # now)`` marker.
        if re.search(r"# Task[^\n]*\s+CHILD-FAST", text):
            children_seen.append("fast")
            return finish_response("fast child done")
        if re.search(r"# Task[^\n]*\s+CHILD-SLOW", text):
            # Busy in short Bash steps (never wedged inside one model
            # call), so the daemon's cooperative stop ends it promptly
            # and the kill is CONFIRMED by its terminal status.
            if "slow" not in children_seen:
                children_seen.append("slow")
            if release.is_set():
                return finish_response("slow child done")
            return tool_call_response("Bash", {"command": "sleep 0.3"})
        parent_steps.append(text)
        step = len(parent_steps)
        if step == 1:
            return tool_call_response(
                "run_agent", {"task": "CHILD-FAST say done", "wait": "false", "timeout": "60"},
            )
        if step == 2:
            return tool_call_response(
                "run_agent", {"task": "CHILD-SLOW wait for me", "wait": "false", "timeout": "60"},
            )
        ids = _JOB_ID.findall(text)
        if step == 3:
            return tool_call_response(
                "agent_job", {"job_id": ids[0], "action": "wait", "timeout_seconds": "60"},
            )
        if step == 4:
            return tool_call_response("agent_job", {"job_id": ids[1], "action": "tail"})
        if step == 5:
            return tool_call_response("agent_job", {"job_id": ids[1], "action": "kill"})
        if step == 6:
            return tool_call_response("agent_job", {"job_id": "agent-00000000", "action": "tail"})
        return finish_response("parent done")

    model = StandInModelServer(responder)
    try:
        result = sorcar.run(
            "PARENT start two background agents",
            work_dir=str(home.repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            use_worktree=False,
            auto_commit=False,
            endpoint_file=daemon.endpoint_file,
            timeout=300,
        )
    finally:
        release.set()
        model.stop()
        daemon.stop()
    assert result.success is True, result
    assert "parent done" in result.text
    assert sorted(children_seen) == ["fast", "slow"]
    # Step 2 saw the first job notice; step 3 the second.
    assert "Started the sorcar agent task as job agent-" in parent_steps[1]
    assert "agent_job(" in parent_steps[1]
    assert "Wait for or kill it before finishing" in parent_steps[1]
    # Step 4 saw the fast child's YAML result from ``wait``.
    assert "success: true" in parent_steps[3] and "fast child done" in parent_steps[3]
    # Step 5 saw the slow job still running (``tail``).
    still_running = r"Job agent-[0-9a-f]{8} \(sorcar agent task\) is still running\."
    assert re.search(still_running, parent_steps[4])
    # Step 6 saw the kill's outcome; step 7 the unknown-job error.
    assert "was stopped before it finished" in parent_steps[5]
    assert "Error: unknown agent job 'agent-00000000'; this task's jobs: agent-" in parent_steps[6]


def test_agent_job_validates_its_arguments_and_sees_only_its_owners_jobs() -> None:
    owner, stranger = object(), object()
    agent_job = agent_dispatch.make_agent_job_tool(owner)
    assert agent_job("nope") == "Error: unknown agent job 'nope'; this task's jobs: none."
    job = agent_dispatch.AgentJob(
        "agent-test0001", "x", owner, threading.Event(), threading.Event(),
        threading.Thread(target=lambda: None),
    )
    job.thread.start()
    job.thread.join()
    job.result = "done"
    with agent_dispatch._AGENT_JOBS_LOCK:
        agent_dispatch._AGENT_JOBS[job.job_id] = job
    try:
        assert agent_job("agent-test0001", "tail") == "done"
        assert agent_job("agent-test0001", "wait", "x") == (
            "Error: timeout_seconds must be a number, got 'x'."
        )
        for bad in ("inf", "nan", "-1"):
            assert agent_job("agent-test0001", "wait", bad) == (
                f"Error: timeout_seconds must be a finite non-negative number, got {bad!r}."
            )
        assert agent_job("agent-test0001", "wait", "1") == "done"
        assert agent_job("agent-test0001", "pause") == (
            "Error: action must be tail, wait or kill, got 'pause'."
        )
        # Another agent's tool neither sees nor lists the job.
        assert agent_dispatch.make_agent_job_tool(stranger)("agent-test0001") == (
            "Error: unknown agent job 'agent-test0001'; this task's jobs: none."
        )
        assert agent_dispatch.agent_jobs_of(owner) == {"agent-test0001": job}
    finally:
        with agent_dispatch._AGENT_JOBS_LOCK:
            del agent_dispatch._AGENT_JOBS[job.job_id]
    bad_wait = make_run_agent_tool("")("t", wait="maybe")
    assert bad_wait == "Error: wait must be true or false, got 'maybe'."


# ---------------------------------------------------------------------------
# P4 — the classifier never demotes a script-pinned worktree
# ---------------------------------------------------------------------------

_CLASSIFIER_MARKER = "You are a task classifier"
_KEY_FIELDS = (
    "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY", "ZAI_API_KEY", "MOONSHOT_API_KEY",
)


def _verdict_body(is_development: bool) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kiss-classifier", "object": "chat.completion", "created": 0,
        "model": STANDIN_MODEL,
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({
                "is_simple": not is_development, "is_development": is_development,
            })},
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


class ClassifierNeverDemotesAPinnedWorktreeTest(unittest.TestCase):
    """The same non-development verdict demotes a tab default but not a SEA's pin."""

    def setUp(self) -> None:
        from kiss.agents.sorcar import worktree_pool

        self._saved_env = {
            name: os.environ.get(name)
            for name in (worktree_pool._DISABLE_ENV, "KISS_DISABLE_TASK_CLASSIFIER")
        }
        os.environ[worktree_pool._DISABLE_ENV] = "1"
        os.environ["KISS_DISABLE_TASK_CLASSIFIER"] = "0"
        self.home = IsolatedKissHome(prefix="kiss-sea-pinned-wt-")
        self.home.write_config(
            auto_commit_mode=False, is_worktree=True, max_budget=5.0,
            use_web_browser=False, classify_tasks=True,
        )
        clear_classification_cache()
        keys = config_module.DEFAULT_CONFIG
        self._saved_keys = {name: getattr(keys, name) for name in _KEY_FIELDS}
        for name in _KEY_FIELDS:
            setattr(keys, name, "")
        keys.OPENAI_API_KEY = "kiss-pinned-standin-key"
        self.classifier_calls = 0
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.home.repo)
        self.pinned = _write(
            self.home.repo / "pinned_sea.py",
            "def settings():\n    return {'use_worktree': True}\n",
        )

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
            agent = state.agent
            if agent is not None and getattr(agent, "_wt", None) is not None:
                try:
                    agent.discard()
                except Exception:
                    pass
        self.standin.stop()
        keys = config_module.DEFAULT_CONFIG
        for name, value in self._saved_keys.items():
            setattr(keys, name, value)
        clear_classification_cache()
        self.home.cleanup()
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        if _CLASSIFIER_MARKER in request_text(request):
            self.classifier_calls += 1
            return _verdict_body(is_development=False)
        return finish_response("said hello")

    def _run(self, tab_id: str, prompt: str, agent_path: str = "") -> None:
        cmd: dict[str, Any] = {
            "type": "run", "tabId": tab_id, "prompt": prompt, "model": STANDIN_MODEL,
            "workDir": str(self.home.repo), "useWorktree": True, "isParallel": False,
            "autoCommit": False, "useWebTools": False, "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        }
        if agent_path:
            cmd["agentPath"] = agent_path
        self.server._run_task(cmd)
        deadline = time.time() + 60
        while time.time() < deadline and not self.printer.events_of_type("result"):
            time.sleep(0.05)

    def test_tab_default_is_demoted_but_a_sea_pin_stands(self) -> None:
        self._run("tab-default", "say hello")
        self.assertEqual(self.classifier_calls, 1)
        self.assertEqual(self.printer.events_of_type("worktree_created"), [])
        with self.printer._capture_lock:
            self.printer.captured.clear()
        clear_classification_cache()
        self._run("tab-pinned", "say hello again", agent_path=str(self.pinned))
        self.assertEqual(self.classifier_calls, 2)
        created = self.printer.events_of_type("worktree_created")
        self.assertEqual(len(created), 1, self.printer.events_of_type("result"))
        results = self.printer.events_of_type("result")
        self.assertTrue(results and results[-1].get("success") is not False, results)
