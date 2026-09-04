# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the immediate agent dispatch tool.

Everything runs against the real installed channel modules and the
real agent-script loader — no mocks or test doubles (``monkeypatch``
is used only to isolate environment variables, the working directory,
and the cron module's daemon-socket default between tests, and to
capture the daemon submission that a live dispatch would perform).  Branches
not exercised here, and why they need no doubles-based tests:

- ``run_agent``'s successful dispatch path submits a task to the
  kiss-web daemon and needs a live LLM endpoint (unavailable and
  non-deterministic in unit tests); the dispatch plumbing up to the
  daemon socket is covered via the unreachable-daemon path (and, with
  a real daemon stand-in, in
  ``kiss.tests.agents.sorcar.test_dispatch_timeout``), and the
  agent-script contract the daemon applies is covered directly
  through ``apply_agent_overrides``.
- ``_package_dir``'s package-absent branches would require
  uninstalling ``kiss.agents.third_party_agents`` from the test
  environment.
- ``_run_agent``'s no-agent-class guard is unreachable for any
  installed channel (``test_every_channel_module_is_dispatchable``
  proves the contract holds for all of them).

The agent-script loader tests that never touch a channel (pure
kiss.agents.sorcar + kiss.server closure) moved to
``kiss.tests.server.test_agent_dispatch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    _agent_class,
    _daemon_sock_path,
    available_channels,
    get_tools,
    make_run_agent_tool,
)
from kiss.server.agent_file import apply_agent_overrides

# The standalone tool (no calling-task work directory): relative agent
# paths resolve against the process working directory and path-mode
# sub-tasks run in ``$KISS_HOME/agent_work``.  The closure captures
# only the work-dir string, so one instance is safe across tests.
run_agent = make_run_agent_tool("")


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate KISS_HOME, the daemon sockets, and the workspace env var.

    ``KISS_SORCAR_SOCK`` points into an empty temp dir so a dispatch
    can never reach a real daemon that happens to be running on this
    machine, and the cron module's recorded daemon socket is reset so
    a scheduler started elsewhere cannot redirect the dispatch.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_SOCK", str(tmp_path / "no-daemon.sock"))
    monkeypatch.delenv("KISS_CHANNEL_WORKSPACE", raising=False)
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
    return tmp_path


def test_available_channels_discovery() -> None:
    channels = available_channels()
    for expected in ("slack", "telegram", "discord", "email", "ntfy"):
        assert expected in channels
    # Infrastructure and private modules are not user-facing channels.
    for hidden in ("a2a", "openai_compat", "channel_cli", "backend_utils"):
        assert hidden not in channels
    assert channels == sorted(channels)


def test_docstring_lists_channels() -> None:
    doc = run_agent.__doc__ or ""
    assert "{channels}" not in doc
    assert "slack" in doc and "telegram" in doc


def test_unknown_agent_error() -> None:
    out = run_agent("no_such_channel", "say hi")
    assert out.startswith("Error: unknown agent")
    assert "not a path to a .py agent script" in out
    assert "slack" in out


def test_channel_name_is_normalized() -> None:
    # Case/whitespace variants still resolve; the unreachable daemon
    # then fails the dispatch cleanly instead of "unknown agent".
    out = run_agent("  NTFY ", "say hi")
    assert "unknown agent" not in out
    assert out.startswith("Error: the ntfy agent task could not run:")


def test_empty_task_error() -> None:
    assert run_agent("slack", "   ") == (
        "Error: task must be a non-empty string."
    )


def test_bad_budget_error() -> None:
    out = run_agent("slack", "say hi", max_budget="cheap")
    assert out == "Error: max_budget must be a number, got 'cheap'."
    for bad in ("nan", "inf", "0", "-2"):
        out = run_agent("slack", "say hi", max_budget=bad)
        assert out == (
            f"Error: max_budget must be a positive finite number, "
            f"got {bad!r}."
        )


def test_channel_alias_normalization() -> None:
    # The SYSTEM.md directive names channels with natural spelling;
    # case, spaces, hyphens, and underscores must all resolve.
    for alias, canonical in (
        ("Home Assistant", "homeassistant"),
        ("phone control", "phone_control"),
        ("SLACK", "slack"),
    ):
        out = run_agent(alias, "say hi")
        assert "unknown agent" not in out
        assert out.startswith(
            f"Error: the {canonical} agent task could not run:"
        )


def test_hyphenated_alias_is_a_channel_not_a_path() -> None:
    # A hyphen is a channel-name separator, not a path marker: the
    # alias resolves to the channel even though "-" appears in it.
    out = run_agent("nextcloud-talk", "say hi")
    assert "unknown agent" not in out
    assert out.startswith(
        "Error: the nextcloud_talk agent task could not run:"
    )


def test_channel_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    out = run_agent("ntfy", "say hi", max_budget="1.5")
    assert out.startswith("Error: the ntfy agent task could not run:")
    assert "no-daemon.sock" in out
    # The workspace env var (unset before the call) is unset again.
    import os

    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # Channel dispatches run in the channel agents' shared work
    # directory (the same default their poll-mode runner uses).
    assert (tmp_path / "channel_work").is_dir()
    assert not (tmp_path / "agent_work").exists()


def test_dispatch_pins_tab_scope_to_calling_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every dispatch scopes the sub-task's tab to the CALLING work dir.

    A ``run_agent`` sub-task runs in a channel/cron/agent scratch
    directory (``work_dir``) but its tab must show in the calling
    workspace's tab bar, so the dispatch forwards the calling task's
    work directory as ``daemon_client.run``'s ``scope_work_dir``.  The
    real dispatch path is exercised up to the daemon-client boundary;
    only that boundary call is captured, to read the argument the
    dispatch computed.
    """
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        captured.append(kwargs)
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)

    caller = tmp_path / "caller_project"
    caller.mkdir()
    tool = make_run_agent_tool(str(caller))

    # Channel mode: executes in the shared channel_work scratch dir,
    # but the tab is scoped to the caller's project.  Every mode also
    # forwards the parsed dispatch timeout (the 300-s default when the
    # tool's ``timeout`` argument is empty) and opts in to the
    # stop-on-timeout cascade — a timed-out channel sub-task must not
    # outlive its workspace reservation.
    captured.clear()
    tool("ntfy", "say hi")
    assert captured[0]["work_dir"] == str(tmp_path / "channel_work")
    assert captured[0]["scope_work_dir"] == str(caller)
    assert (
        captured[0]["timeout"]
        == agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS
    )
    assert captured[0]["stop_on_timeout"] is True

    # Cron mode: executes in the cron work dir, scoped to the caller;
    # an explicit ``timeout`` argument is parsed and forwarded.
    captured.clear()
    tool("cron", "run 'echo hi' every 5 minutes", timeout="42.5")
    assert captured[0]["work_dir"] == cron_agent.get_work_dir()
    assert captured[0]["scope_work_dir"] == str(caller)
    assert captured[0]["timeout"] == 42.5
    assert captured[0]["stop_on_timeout"] is True

    # Path mode: executes in the caller's project (scope == work_dir).
    script = caller / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    captured.clear()
    tool(str(script), "say hi")
    assert captured[0]["work_dir"] == str(caller)
    assert captured[0]["scope_work_dir"] == str(caller)
    assert (
        captured[0]["timeout"]
        == agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS
    )
    assert captured[0]["stop_on_timeout"] is True


def test_cron_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    # "cron" (any case/spacing) routes to the built-in cron agent
    # script, not to channel lookup: the dispatch fails only on the
    # unreachable daemon and runs in the cron work directory.
    out = run_agent("  Cron ", "run 'echo hi' every 5 minutes")
    assert "unknown agent" not in out
    assert out.startswith("Error: the cron agent task could not run:")
    assert "no-daemon.sock" in out
    # The cron dispatch runs in the cron agent's own work directory.
    assert (tmp_path / "cron" / "work").is_dir()
    assert not (tmp_path / "channel_work").exists()
    assert not (tmp_path / "agent_work").exists()


def test_docstring_and_error_mention_cron() -> None:
    assert "cron" in (run_agent.__doc__ or "")
    # "cron" is not a third-party channel — it must never appear in
    # the channel list, only via its dedicated dispatch branch.
    assert "cron" not in available_channels()
    # A mistyped agent name gets a hint about the built-in cron agent.
    out = run_agent("no_such_channel", "say hi")
    assert out.startswith("Error: unknown agent")
    assert "cron" in out


def test_path_mode_missing_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_agent.py"
    out = run_agent(str(missing), "say hi")
    assert out.startswith("Error: agent script")
    assert "does not exist" in out


def test_path_mode_non_python_file_error(tmp_path: Path) -> None:
    not_py = tmp_path / "agent.txt"
    not_py.write_text("hello")
    out = run_agent(str(not_py), "say hi")
    assert out.startswith("Error: agent script")
    assert "is not a Python (.py) file" in out


def test_path_mode_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    # A valid agent-script path takes the path branch (no channel
    # lookup, no workspace handling) and fails cleanly on the
    # unreachable daemon, named by the script's file stem.
    import os

    script = tmp_path / "my_researcher.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    out = run_agent(str(script), "say hi", workspace="ignored-ws")
    assert out.startswith(
        "Error: the my_researcher agent task could not run:"
    )
    assert "no-daemon.sock" in out
    # Path mode never touches the channel workspace env var.
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # The standalone tool runs path-mode sub-tasks in agent_work.
    assert (tmp_path / "agent_work").is_dir()
    assert not (tmp_path / "channel_work").exists()


def test_path_mode_detected_by_py_suffix_and_separator(
    tmp_path: Path,
) -> None:
    # ".py" suffix without a separator is path mode, not a channel.
    out = run_agent("slack_agent.py", "say hi")
    assert out.startswith("Error: agent script")
    # A separator without a ".py" suffix is path mode too — rejected
    # with the loader's .py diagnostic rather than "unknown agent".
    out = run_agent(str(tmp_path / "somedir" / "agent"), "say hi")
    assert out.startswith("Error: agent script")
    assert "is not a Python (.py) file" in out


def test_relative_path_resolves_against_captured_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tool runs in the daemon process, whose CWD is unrelated to
    # the user's project: a relative agent path must resolve against
    # the CALLING task's work directory captured by the factory.
    project = tmp_path / "project"
    (project / "agents").mkdir(parents=True)
    script = project / "agents" / "reviewer.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    elsewhere = tmp_path / "daemon_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    tool = make_run_agent_tool(str(project))
    out = tool("agents/reviewer.py", "say hi")
    # The script was found (under the project, not under the CWD) and
    # the dispatch failed only on the unreachable daemon.
    assert out.startswith("Error: the reviewer agent task could not run:")
    assert "no-daemon.sock" in out
    # A missing relative path names the project-anchored resolution.
    out = tool("agents/nope.py", "say hi")
    assert out.startswith("Error: agent script")
    assert str(project / "agents" / "nope.py") in out
    assert "does not exist" in out


def test_path_mode_runs_in_captured_work_dir(tmp_path: Path) -> None:
    # A path-named agent's sub-task runs in the calling task's work
    # directory — no agent_work/channel_work scratch dir is created.
    project = tmp_path / "project"
    project.mkdir()
    script = project / "helper.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    out = make_run_agent_tool(str(project))(str(script), "say hi")
    assert out.startswith("Error: the helper agent task could not run:")
    assert not (tmp_path / "agent_work").exists()
    assert not (tmp_path / "channel_work").exists()


def test_standalone_relative_path_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a captured work directory (standalone tool), a relative
    # path resolves against the process working directory.
    script = tmp_path / "local_agent.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    monkeypatch.chdir(tmp_path)
    out = run_agent("local_agent.py", "say hi")
    assert out.startswith("Error: the local_agent agent task could not run:")


def test_dispatch_uses_launcher_workspace_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The workspace env var follows the launcher's reference-counting
    # registry (shared with the channel CLIs): while a launch with a
    # DIFFERENT workspace is active a dispatch refuses to overwrite
    # the exported value (it would hand the running session the wrong
    # account's credentials) and fails loudly after its bounded wait;
    # the last exit removes the env var — a pre-existing value counts
    # as stale, exactly as in
    # kiss.agents.third_party_agents._kiss_web_launcher.
    import os

    from kiss.agents.third_party_agents._kiss_web_launcher import (
        _enter_workspace,
        _exit_workspace,
    )

    monkeypatch.setenv("KISS_CHANNEL_WORKSPACE", "stale-ws")
    monkeypatch.setattr(agent_dispatch, "WORKSPACE_WAIT_TIMEOUT_SECONDS", 0.2)
    assert _enter_workspace("other-ws")  # a concurrent dispatch is active
    try:
        out = run_agent("ntfy", "say hi", workspace="my-ws")
        assert out.startswith("Error: workspace 'my-ws' could not be activated")
        # The concurrent dispatch is still active; its workspace was
        # never overwritten.
        assert os.environ["KISS_CHANNEL_WORKSPACE"] == "other-ws"
        # A dispatch SHARING the active workspace proceeds normally
        # (and fails only at the unreachable daemon socket).
        out = run_agent("ntfy", "say hi", workspace="other-ws")
        assert out.startswith("Error: the ntfy agent task could not run:")
        assert os.environ["KISS_CHANNEL_WORKSPACE"] == "other-ws"
    finally:
        _exit_workspace("other-ws")
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ


def test_dispatch_uses_recorded_daemon_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inside the kiss-web daemon the cron scheduler records the
    # daemon's own UDS at boot; the dispatch must target it even when
    # KISS_SORCAR_SOCK points elsewhere — in path mode too.
    recorded = tmp_path / "recorded-daemon.sock"
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", str(recorded))
    assert _daemon_sock_path() == str(recorded)
    out = run_agent("ntfy", "say hi")
    assert "recorded-daemon.sock" in out
    script = tmp_path / "probe_agent.py"
    script.write_text("def get_model() -> str:\n    return 'm'\n")
    out = run_agent(str(script), "say hi")
    assert "recorded-daemon.sock" in out


def test_agent_class_resolution() -> None:
    import kiss.agents.third_party_agents.slack_agent as slack_agent

    cls = _agent_class(slack_agent)
    assert cls is not None and cls.__name__ == "SlackAgent"
    # A module defining no BaseChannelAgent subclass of its own
    # (imported classes do not count) resolves to None.
    assert _agent_class(agent_dispatch) is None


def test_every_channel_module_is_dispatchable() -> None:
    import importlib

    for channel in available_channels():
        module = importlib.import_module(
            f"kiss.agents.third_party_agents.{channel}_agent"
        )
        cls = _agent_class(module)
        assert cls is not None, channel
        assert isinstance(
            getattr(cls, "channel_system_prompt", None), str,
        ), channel
        assert module.__file__ and Path(module.__file__).is_file(), channel
        assert callable(getattr(module, "get_tools", None)), channel


def test_channel_module_is_a_valid_agent_script() -> None:
    # The exact contract the dispatch relies on: passing a channel
    # module as ``extension_agent_path`` makes the daemon use the module as its
    # own tools file (its ``get_tools()`` returns the tool list).
    import kiss.agents.third_party_agents.ntfy_agent as ntfy_agent

    cmd = {"agentPath": ntfy_agent.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"toolsFile"}
    assert cmd["toolsFile"] == ntfy_agent.__file__


def test_get_tools_and_sorcar_wiring() -> None:
    tools = get_tools()
    assert len(tools) == 1
    assert tools[0].__name__ == "run_agent"
    assert "slack" in (tools[0].__doc__ or "")
    # The module lives in the sorcar package and never imports from
    # kiss.agents.third_party_agents at module scope (soft plugin).
    source_text = Path(agent_dispatch.__file__).read_text(encoding="utf-8")
    assert "/agents/sorcar/" in agent_dispatch.__file__
    for line in source_text.splitlines():
        assert not line.startswith("from kiss.agents.third_party_agents")
        assert not line.startswith("import kiss.agents.third_party_agents")
    # The default Sorcar toolset registers the tool, bound to the
    # calling task's work directory AND the calling agent itself, so
    # each dispatched sub-task's spend is folded into the calling
    # task's cost accounting.
    agent_source = Path(agent_dispatch.__file__).parent / "sorcar_agent.py"
    assert (
        'tools.append(make_run_agent_tool(self.work_dir or "", self))'
        in agent_source.read_text(encoding="utf-8")
    )
    # The system prompt directs the agent to dispatch immediately.
    system_md = Path(agent_dispatch.__file__).parents[2] / "SYSTEM.md"
    assert "run_agent" in system_md.read_text(encoding="utf-8")
