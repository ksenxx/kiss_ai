# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the immediate channel-agent dispatch tool.

Everything runs against the real installed channel modules and the
real agent-script loader — no mocks or test doubles (``monkeypatch``
is used only to isolate environment variables and the cron module's
daemon-socket default between tests).  Branches not exercised here,
and why they need no doubles-based tests:

- ``run_channel_agent``'s successful and timed-out dispatch paths
  submit a task to the kiss-web daemon and need a live LLM endpoint
  (unavailable and non-deterministic in unit tests); the dispatch
  plumbing up to the daemon socket is covered via the
  unreachable-daemon path, and the agent-script contract the daemon
  applies is covered directly through ``apply_agent_overrides``.
- ``_package_dir``'s package-absent branches would require
  uninstalling ``kiss.agents.third_party_agents`` from the test
  environment.
- ``run_channel_agent``'s no-agent-class guard is unreachable for any
  installed channel (``test_every_channel_module_is_dispatchable``
  proves the contract holds for all of them).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from kiss.agents.sorcar import channel_agents, cron_agent
from kiss.agents.sorcar.channel_agents import (
    _agent_class,
    _daemon_sock_path,
    available_channels,
    get_tools,
    run_channel_agent,
)
from kiss.server.agent_file import AgentFileError, apply_agent_overrides


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
    doc = run_channel_agent.__doc__ or ""
    assert "{channels}" not in doc
    assert "slack" in doc and "telegram" in doc


def test_unknown_channel_error() -> None:
    out = run_channel_agent("no_such_channel", "say hi")
    assert out.startswith("Error: unknown channel")
    assert "slack" in out


def test_channel_name_is_normalized() -> None:
    # Case/whitespace variants still resolve; the unreachable daemon
    # then fails the dispatch cleanly instead of "unknown channel".
    out = run_channel_agent("  NTFY ", "say hi")
    assert "unknown channel" not in out
    assert out.startswith("Error: the ntfy agent task could not run:")


def test_empty_task_error() -> None:
    assert run_channel_agent("slack", "   ") == (
        "Error: task must be a non-empty string."
    )


def test_bad_budget_error() -> None:
    out = run_channel_agent("slack", "say hi", max_budget="cheap")
    assert out == "Error: max_budget must be a number, got 'cheap'."
    for bad in ("nan", "inf", "0", "-2"):
        out = run_channel_agent("slack", "say hi", max_budget=bad)
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
        ("nextcloud-talk", "nextcloud_talk"),
        ("SLACK", "slack"),
    ):
        out = run_channel_agent(alias, "say hi")
        assert "unknown channel" not in out
        assert out.startswith(
            f"Error: the {canonical} agent task could not run:"
        )


def test_dispatch_unreachable_daemon_is_a_clean_error(tmp_path: Path) -> None:
    out = run_channel_agent("ntfy", "say hi", max_budget="1.5")
    assert out.startswith("Error: the ntfy agent task could not run:")
    assert "no-daemon.sock" in out
    # The workspace env var (unset before the call) is unset again.
    import os

    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # The dispatch work dir is prepared under KISS_HOME.
    assert (tmp_path / "channel_work").is_dir()


def test_dispatch_uses_launcher_workspace_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The workspace env var follows the launcher's reference-counting
    # registry (shared with the channel CLIs): overlapping dispatches
    # keep it pointing at an active workspace, and the last exit
    # removes it — a pre-existing value counts as stale, exactly as in
    # kiss.agents.third_party_agents._kiss_web_launcher.
    import os

    from kiss.agents.third_party_agents._kiss_web_launcher import (
        _enter_workspace,
        _exit_workspace,
    )

    monkeypatch.setenv("KISS_CHANNEL_WORKSPACE", "stale-ws")
    _enter_workspace("other-ws")  # a concurrent dispatch is active
    try:
        out = run_channel_agent("ntfy", "say hi", workspace="my-ws")
        assert out.startswith("Error: the ntfy agent task could not run:")
        # After this dispatch exits, the concurrent one is still
        # active, so the env var points at its workspace.
        assert os.environ["KISS_CHANNEL_WORKSPACE"] == "other-ws"
    finally:
        _exit_workspace("other-ws")
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ


def test_channel_tools_reserved_against_mcp_collisions() -> None:
    from kiss.agents.sorcar.mcp_servers import _RESERVED_TOOL_NAMES

    assert {"run_channel_agent", "cron_job"} <= _RESERVED_TOOL_NAMES


def test_dispatch_uses_recorded_daemon_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inside the kiss-web daemon the cron scheduler records the
    # daemon's own UDS at boot; the dispatch must target it even when
    # KISS_SORCAR_SOCK points elsewhere.
    recorded = tmp_path / "recorded-daemon.sock"
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", str(recorded))
    assert _daemon_sock_path() == str(recorded)
    out = run_channel_agent("ntfy", "say hi")
    assert "recorded-daemon.sock" in out


def test_agent_class_resolution() -> None:
    import kiss.agents.third_party_agents.slack_agent as slack_agent

    cls = _agent_class(slack_agent)
    assert cls is not None and cls.__name__ == "SlackAgent"
    # A module defining no BaseChannelAgent subclass of its own
    # (imported classes do not count) resolves to None.
    assert _agent_class(channel_agents) is None


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
    # module as ``agent_path`` makes the daemon use the module as its
    # own tools file (its ``get_tools()`` returns the tool list).
    import kiss.agents.third_party_agents.ntfy_agent as ntfy_agent

    cmd = {"agentPath": ntfy_agent.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"toolsFile"}
    assert cmd["toolsFile"] == ntfy_agent.__file__


def test_agent_script_get_tools_list_normalizes_to_own_path(
    tmp_path: Path,
) -> None:
    script = tmp_path / "self_tools_agent.py"
    script.write_text(textwrap.dedent("""
        def _hello() -> str:
            \"\"\"Say hello.

            Returns:
                A greeting.
            \"\"\"
            return "hello"

        def get_tools() -> list:
            return [_hello]
    """))
    cmd = {"agentPath": str(script), "toolsFile": ""}
    assert apply_agent_overrides(cmd) == {"toolsFile"}
    assert cmd["toolsFile"] == str(script)


def test_agent_script_get_tools_wrong_type_still_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("def get_tools():\n    return 42\n")
    cmd = {"agentPath": str(script), "toolsFile": ""}
    with pytest.raises(AgentFileError, match="get_tools"):
        apply_agent_overrides(cmd)


def test_get_tools_and_sorcar_wiring() -> None:
    assert get_tools() == [run_channel_agent]
    # The module lives in the sorcar package and never imports from
    # kiss.agents.third_party_agents at module scope (soft plugin).
    source_text = Path(channel_agents.__file__).read_text(encoding="utf-8")
    assert "/agents/sorcar/" in channel_agents.__file__
    for line in source_text.splitlines():
        assert not line.startswith("from kiss.agents.third_party_agents")
        assert not line.startswith("import kiss.agents.third_party_agents")
    # The default Sorcar toolset registers the tool.
    agent_source = Path(channel_agents.__file__).parent / "sorcar_agent.py"
    assert "tools.append(run_channel_agent)" in agent_source.read_text(
        encoding="utf-8"
    )
    # The system prompt directs the agent to dispatch immediately.
    system_md = Path(channel_agents.__file__).parents[2] / "SYSTEM.md"
    assert "run_channel_agent" in system_md.read_text(encoding="utf-8")
