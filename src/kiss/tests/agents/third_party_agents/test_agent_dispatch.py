# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the immediate agent dispatch tool.

Everything runs against the real installed channel modules and the
real agent-script loader — no mocks or test doubles (``monkeypatch``
is used only to isolate environment variables, the working directory,
and the cron module's daemon-endpoint default between tests, and to
capture the daemon submission that a live dispatch would perform).  Branches
not exercised here, and why they need no doubles-based tests:

- ``run_agent``'s successful dispatch path submits a task to the
  kiss-web daemon and needs a live LLM endpoint (unavailable and
  non-deterministic in unit tests); the dispatch plumbing up to the
  daemon endpoint is covered via the unreachable-daemon path (and, with
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
    _daemon_endpoint_file,
    available_channels,
    make_run_agent_tool,
)
from kiss.agents.sorcar.agent_file import apply_agent_overrides, channel_workspace, load_layers
from kiss.agents.sorcar.sea_commands import load_sea
from kiss.agents.third_party_agents.auth_status import _agent_class
from kiss.core.config import kiss_home
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

# The standalone tool (no calling-task work directory): relative agent
# paths resolve against the process working directory and path-mode
# sub-tasks run in ``$KISS_HOME/agent_work``.  The closure captures
# only the work-dir string, so one instance is safe across tests.
run_agent = make_run_agent_tool("")


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate KISS_HOME, the daemon endpoint, and the workspace env var.

    ``KISS_SORCAR_LOCAL`` points at a missing endpoint file in a temp
    dir so a dispatch can never reach a real daemon that happens to be
    running on this machine, and the cron module's recorded daemon
    endpoint is reset so a scheduler started elsewhere cannot redirect
    the dispatch.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.delenv("KISS_CHANNEL_WORKSPACE", raising=False)
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    return tmp_path


@pytest.fixture()
def captured_dispatch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every ``daemon_client.run`` call at the daemon-client boundary.

    The real dispatch path runs up to that boundary; each call is recorded
    as ``{"prompt": ..., **kwargs}`` and answered with a successful
    ``TaskResult`` so the tool returns normally.
    """
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="ok", success=True, cost=0.0, tokens=0, steps=0)

    monkeypatch.setattr(daemon_client, "run", capture_run)
    return captured


def _write_helper_script(caller: Path) -> Path:
    """Create *caller* with a minimal ``helper.py`` agent script; return the script path."""
    caller.mkdir()
    script = caller / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    return script


def test_available_channels_discovery() -> None:
    channels = available_channels()
    for expected in ("slack", "telegram", "discord", "email", "ntfy"):
        assert expected in channels
    # SEAs of another kind (``a2a``, a session with peer-agent tools),
    # hidden SEAs (``oai``) and private modules are not channels.
    for other in ("a2a", "oai", "channel_cli", "backend_utils"):
        assert other not in channels
    assert channels == sorted(channels)


def test_docstring_lists_channels() -> None:
    doc = run_agent.__doc__ or ""
    assert "{channels}" not in doc
    assert "slack" in doc and "telegram" in doc


def test_unknown_agent_error() -> None:
    out = run_agent("say hi", "no_such_channel")
    assert out.startswith("Error: unknown agent")
    assert "not a path to a .py SEA file" in out
    assert "slack" in out


def test_channel_name_is_normalized() -> None:
    # Case/whitespace variants still resolve; the unreachable daemon
    # then fails the dispatch cleanly instead of "unknown agent".
    out = run_agent("say hi", "  NTFY ")
    assert "unknown agent" not in out
    assert out.startswith("Error: the ntfy agent task could not run:")


def test_empty_task_error() -> None:
    assert run_agent("   ", "slack") == ("Error: task must be a non-empty string.")


def test_bad_budget_error() -> None:
    out = run_agent("say hi", "slack", max_budget="cheap")
    assert out == "Error: max_budget must be a number, got 'cheap'."
    for bad in ("nan", "inf", "0", "-2"):
        out = run_agent("say hi", "slack", max_budget=bad)
        assert out == (f"Error: max_budget must be a positive finite number, got {bad!r}.")


def test_channel_alias_normalization() -> None:
    # The SYSTEM.md directive names channels with natural spelling;
    # case, spaces, hyphens, and underscores must all resolve.
    for alias, canonical in (
        ("Home Assistant", "homeassistant"),
        ("home_assistant", "homeassistant"),
        ("SLACK", "slack"),
    ):
        out = run_agent("say hi", alias)
        assert "unknown agent" not in out
        assert out.startswith(f"Error: the {canonical} agent task could not run:")


def test_hyphenated_alias_is_a_channel_not_a_path() -> None:
    # A hyphen is a channel-name separator, not a path marker: the
    # alias resolves to the channel even though "-" appears in it.
    out = run_agent("say hi", "home-assistant")
    assert "unknown agent" not in out
    assert out.startswith("Error: the homeassistant agent task could not run:")


def test_channel_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    out = run_agent("say hi", "ntfy", max_budget="1.5")
    assert out.startswith("Error: the ntfy agent task could not run:")
    assert "no-daemon.json" in out
    # The workspace env var (unset before the call) is unset again.
    import os

    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # Channel dispatches run in the channel agents' shared work
    # directory (the same default their poll-mode runner uses).
    assert (tmp_path / "channel_work").is_dir()
    assert not (tmp_path / "agent_work").exists()


def test_dispatch_pins_tab_scope_to_calling_work_dir(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
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

    caller = tmp_path / "caller_project"
    caller.mkdir()
    tool = make_run_agent_tool(str(caller))

    # Channel mode: executes in the shared channel_work scratch dir,
    # but the tab is scoped to the caller's project.  Every mode also
    # records the parsed call bound on the daemon (the one-hour default
    # when the tool's ``timeout`` argument is empty and the script's
    # ``settings()`` name none) while the daemon wait itself has no
    # deadline: the bound is enforced by the call joining its job
    # thread, and a sub-task the bound hands back is stopped only by
    # ``agent_job(..., "kill")`` or the end of the calling run.
    captured_dispatch.clear()
    tool("say hi", "ntfy")
    assert captured_dispatch[0]["work_dir"] == str(tmp_path / "channel_work")
    assert captured_dispatch[0]["scope_work_dir"] == str(caller)
    assert captured_dispatch[0]["timeout"] is None
    assert captured_dispatch[0]["record_timeout"] == agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS
    assert "stop_on_timeout" not in captured_dispatch[0]

    # Cron mode: the cron module's ``settings()`` name the cron work
    # dir, so the sub-task executes there, scoped to the caller; an
    # explicit ``timeout`` argument is parsed and forwarded.
    captured_dispatch.clear()
    tool("run 'echo hi' every 5 minutes", "cron", timeout="42.5")
    assert captured_dispatch[0]["work_dir"] == cron_agent.cron_work_dir()
    assert captured_dispatch[0]["scope_work_dir"] == str(caller)
    assert captured_dispatch[0]["record_timeout"] == 42.5

    # Path mode: executes in the caller's project (scope == work_dir).
    script = caller / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    captured_dispatch.clear()
    tool("say hi", str(script))
    assert captured_dispatch[0]["work_dir"] == str(caller)
    assert captured_dispatch[0]["scope_work_dir"] == str(caller)
    assert captured_dispatch[0]["record_timeout"] == agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS


def _daemon_run_command(call: dict[str, Any]) -> dict[str, Any]:
    """Return the daemon-side ``run`` command the captured dispatch *call* becomes.

    Only the wire fields the agent script's ``settings()`` can override
    are mapped (:data:`kiss.agents.sorcar.agent_file.SETTING_FIELDS`), so a test
    can apply ``apply_agent_overrides`` to exactly what the dispatcher sent.
    """
    return {
        "agentPath": call["extension_agent_path"],
        "useWorktree": call["use_worktree"],
        "autoCommit": call["auto_commit"],
        "classifyTasks": call["classify_tasks"],
        "isParallel": call["is_parallel"],
        "useWebTools": call["use_web_tools"],
        "useMemory": call["use_memory"],
        "appendToSystemPrompt": call["append_to_system_prompt"],
    }


def test_channel_and_cron_lifecycle_is_pinned_off_by_their_settings(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
) -> None:
    """Channel and cron sub-tasks run outside the project git lifecycle.

    A channel/cron sub-task executes in a scratch directory
    (``~/.kiss/channel_work`` / ``~/.kiss/cron/work``), so worktree
    setup would only copy whatever git repository happens to enclose
    that directory — a dirty repo at ``$HOME`` once stalled a gmail
    dispatch for minutes copying 65 GB before the sub-task's tab could
    even appear.  The dispatcher no longer special-cases them: it sends
    the same wire values as for any other sub-task (the persisted "Use
    worktree" / "Auto commit" settings, no classifier override), and
    the ``channel`` preset of the module's ``settings()`` pins
    worktree, auto-commit, classification and fan-out off on the
    daemon (``apply_agent_overrides``), where every script's settings
    win.  A path-mode agent script with the default ``session`` preset
    keeps the standard lifecycle on the calling project.  The real
    dispatch path is exercised up to the daemon-client boundary; only
    that boundary call is captured.
    """
    caller = tmp_path / "caller_project"
    caller.mkdir()
    tool = make_run_agent_tool(str(caller))
    isolated = IsolatedKissHome("kiss-dispatch-lifecycle-")
    try:
        # Both persisted settings on (the defaults).
        for agent, task in (("ntfy", "say hi"), ("cron", "run 'echo hi' every 5 minutes")):
            captured_dispatch.clear()
            tool(task, agent)
            sent = captured_dispatch[0]
            assert sent["use_worktree"] is True, agent
            assert sent["auto_commit"] is True, agent
            assert sent["classify_tasks"] is None, agent
            assert sent["is_parallel"] is True, agent
            cmd = _daemon_run_command(sent)
            overridden = apply_agent_overrides(cmd)
            assert {"useWorktree", "autoCommit", "classifyTasks", "isParallel"} <= overridden
            assert cmd["useWorktree"] is False, agent
            assert cmd["autoCommit"] is False, agent
            assert cmd["classifyTasks"] is False, agent
            assert cmd["isParallel"] is False, agent

        # Path mode, ``session`` preset: worktree + auto-commit follow
        # the persisted settings; classification follows the daemon's
        # configured default; the script's settings change nothing.
        script = caller / "helper.py"
        script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
        captured_dispatch.clear()
        tool("say hi", str(script))
        sent = captured_dispatch[0]
        assert sent["use_worktree"] is True
        assert sent["auto_commit"] is True
        assert sent["classify_tasks"] is None
        cmd = _daemon_run_command(sent)
        assert apply_agent_overrides(cmd) == {"model"}
        assert cmd["useWorktree"] is True and cmd["autoCommit"] is True

        # The user turned both settings off in the settings panel: a
        # path-mode sub-agent follows them like a chat-panel task would.
        isolated.write_config(is_worktree=False, auto_commit_mode=False)
        captured_dispatch.clear()
        tool("say hi", str(script))
        assert captured_dispatch[0]["use_worktree"] is False
        assert captured_dispatch[0]["auto_commit"] is False

        # Each setting is read on its own; explicit options still win.
        isolated.write_config(is_worktree=True, auto_commit_mode=False)
        captured_dispatch.clear()
        tool("say hi", str(script))
        assert captured_dispatch[0]["use_worktree"] is True
        assert captured_dispatch[0]["auto_commit"] is False
        captured_dispatch.clear()
        tool("say hi", str(script), options='{"auto_commit": true}')
        assert captured_dispatch[0]["use_worktree"] is True
        assert captured_dispatch[0]["auto_commit"] is True
    finally:
        isolated.cleanup()


def test_run_option_parse_errors(tmp_path: Path) -> None:
    """A malformed ``options`` JSON object fails before any daemon contact.

    Every key of ``options`` mirrors a keyword option of
    ``kiss.server.sorcar.run``; a value the daemon could not honour
    is reported by name with the offending text.
    """
    for name in (
        "use_worktree",
        "auto_commit",
        "use_web_tools",
        "auto_classify",
        "use_memory",
        "allow_fan_out",
    ):
        out = run_agent("say hi", "ntfy", options=f'{{"{name}": "maybe"}}')
        assert out == f"Error: {name} must be true or false, got 'maybe'."
        out = run_agent("say hi", "ntfy", options=f'{{"{name}": 1}}')
        assert out == f"Error: {name} must be true or false, got 1."
    assert run_agent("say hi", "ntfy", options='{"model_config": [1, 2]}') == (
        "Error: options['model_config'] must be a JSON dict, got list."
    )
    assert run_agent("say hi", "ntfy", options='{"chat_id": 5}') == (
        "Error: options['chat_id'] must be a JSON str, got int."
    )
    assert run_agent("say hi", "ntfy", options='{"add_to_system_prompt": true}') == (
        "Error: options['add_to_system_prompt'] must be a JSON str, got bool."
    )
    # The wire spelling is not an option key (the error names the key
    # it was renamed to); the base prompt is a script's
    # ``system_prompt()``, not an option either.
    assert run_agent("say hi", "ntfy", options='{"append_to_prompt": "x"}') == (
        "Error: options key 'append_to_prompt' was renamed to 'add_to_prompt'; use the new name."
    )
    assert run_agent("say hi", "ntfy", options='{"system_prompt": "x"}').startswith(
        "Error: options has an unknown key 'system_prompt'"
    )
    assert run_agent("say hi", "ntfy", options="[1, 2]") == (
        "Error: options must be a JSON object, got '[1, 2]'."
    )
    out = run_agent("say hi", "ntfy", options="{not json")
    assert out.startswith("Error: options must be a JSON object, got '{not json': ")
    out = run_agent("say hi", "ntfy", options='{"tools": "x.py"}')
    assert out.startswith(
        "Error: options has an unknown key 'tools'; known keys: work_dir, model, chat_id, "
    )
    out = run_agent("say hi", "ntfy", tool_profile="bogus")
    assert out.startswith("Error: tool_profile must be one of ")
    assert out.endswith("got 'bogus'.")
    # ``tool_profile`` is both an argument and an options key; the two
    # may repeat but not contradict each other.
    out = run_agent("say hi", "ntfy", tool_profile="review", options='{"tool_profile": "bash"}')
    assert out == (
        "Error: options['tool_profile'] = 'bash' contradicts the tool_profile argument "
        "'review'; pass one of them."
    )
    out = run_agent("say hi", "ntfy", options='{"tool_profile": "bogus"}')
    assert out.startswith("Error: tool_profile must be one of ")
    # Extra tools come only from the agent script's ``tools()``:
    # the tool has no tools-path arguments, and the old per-option
    # keyword arguments (and the ``model_name`` alias) are gone.
    import inspect

    params = inspect.signature(run_agent).parameters
    assert list(params) == [
        "task", "agent", "model", "tool_profile", "max_budget", "timeout", "options", "wait",
    ]
    for kwarg in ("tools", "use_worktree", "chat_id", "add_to_prompt", "model_name", "workspace"):
        with pytest.raises(TypeError):
            run_agent("say hi", "ntfy", **{kwarg: str(tmp_path / "x.py")})


def test_channel_and_cron_refuse_options_that_contradict_the_kind(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
) -> None:
    """Asking a channel/cron sub-task for a worktree is refused, never silently undone.

    Every key the ``channel`` kind sets is locked
    (``sea_settings.merge_settings``), so an explicit option that
    differs from it is a ``locked`` conflict under the one precedence
    rule; an option that agrees is forwarded as usual.
    """
    cases = (
        ('{"use_worktree": true}', "use_worktree=False (asked for True)"),
        ('{"auto_commit": "TRUE"}', "auto_commit=False (asked for True)"),
        ('{"use_worktree": "false", "auto_commit": true}', "auto_commit=False (asked for True)"),
        ('{"use_memory": true}', "use_memory=False (asked for True)"),
        ('{"allow_fan_out": true}', "allow_fan_out=False (asked for True)"),
    )
    for agent in ("ntfy", "cron"):
        for options, clash in cases:
            captured_dispatch.clear()
            out = run_agent("say hi", agent, options=options)
            assert out == f"Error: {agent}: the script locks {clash}", (agent, options)
            assert not captured_dispatch

    captured_dispatch.clear()
    out = run_agent("say hi", "ntfy", options='{"use_worktree": "false", "auto_commit": " False "}')
    assert "Error" not in out, out
    assert captured_dispatch[0]["use_worktree"] is False
    assert captured_dispatch[0]["auto_commit"] is False
    cmd = _daemon_run_command(captured_dispatch[0])
    apply_agent_overrides(cmd)
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False


def test_run_options_are_forwarded_to_daemon(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
) -> None:
    """The ``options`` JSON object reaches ``daemon_client.run`` as its keyword options.

    Keys left out forward the option's default (``None`` for the
    tri-state daemon-decides options, ``True`` for ``is_parallel``,
    ``""`` for the text options); explicit values are parsed and
    forwarded verbatim, booleans as JSON booleans or as the words
    ``"true"`` / ``"false"``, ``null`` as "not passed".  No tools path
    travels: the daemon client's ``run`` has no tools parameter, so
    the sub-task's extra tools can only come from the agent script's
    own ``tools()``.  The real dispatch path is
    exercised up to the daemon-client boundary; only that boundary
    call is captured.
    """

    caller = tmp_path / "caller_project"
    script = _write_helper_script(caller)
    tool = make_run_agent_tool(str(caller))

    # Nothing passed: the daemon's defaults decide.
    tool("say hi", str(script))
    defaults = captured_dispatch[0]
    assert defaults["chat_id"] == ""
    assert defaults["system_prompt"] == ""
    assert "tools" not in defaults
    assert "toolsFile" not in defaults
    assert "append_basic_tools" not in defaults
    assert defaults["model_config"] is None
    assert defaults["use_web_tools"] is None
    assert defaults["classify_tasks"] is None
    assert defaults["use_memory"] is None
    assert defaults["is_parallel"] is True
    assert defaults["append_to_system_prompt"] == ""
    assert defaults["append_to_prompt"] == ""
    assert defaults["tool_profile"] == ""
    assert defaults["docker_image"] == ""

    # Everything passed, in path mode: parsed and forwarded, with the
    # explicit git-lifecycle values replacing the persisted settings.
    captured_dispatch.clear()
    tool(
        "say hi",
        str(script),
        options="""{
            "chat_id": " chat-123 ",
            "model_config": {"base_url": "http://localhost:8000/v1"},
            "use_worktree": "false",
            "auto_commit": "False",
            "use_web_tools": true,
            "auto_classify": false,
            "use_memory": "true",
            "allow_fan_out": false,
            "add_to_system_prompt": "Answer in French.",
            "add_to_prompt": "Cite sources."
        }""",
        tool_profile=" review ",
    )
    sent = captured_dispatch[0]
    assert sent["chat_id"] == "chat-123"
    assert sent["system_prompt"] == ""
    assert sent["model_config"] == {"base_url": "http://localhost:8000/v1"}
    assert sent["use_worktree"] is False
    assert sent["auto_commit"] is False
    assert sent["use_web_tools"] is True
    assert sent["classify_tasks"] is False
    assert sent["use_memory"] is True
    assert sent["is_parallel"] is False
    assert sent["append_to_system_prompt"] == "Answer in French."
    assert sent["append_to_prompt"] == "Cite sources."
    assert sent["tool_profile"] == "review"
    # ``options['docker_image']`` is accepted by the parser, but
    # ``dispatch_result`` forwards only the calling task's live
    # container (``inherit_from_parent``), so an explicit value is
    # dropped; it is not asserted here until the dispatcher either
    # forwards it or stops accepting the key.

    # Path mode honours an explicit worktree request too; ``null`` and
    # ``""`` mean "not passed", so the defaults stand.
    captured_dispatch.clear()
    tool("say hi", str(script), options='{"use_worktree": true, "use_memory": null, '
                                        '"allow_fan_out": "", "chat_id": null}')
    assert captured_dispatch[0]["use_worktree"] is True
    assert captured_dispatch[0]["use_memory"] is None
    assert captured_dispatch[0]["is_parallel"] is True
    assert captured_dispatch[0]["chat_id"] == ""

    # For cron and the channel agents the ``channel`` kind locks
    # classification off: asking for it is refused, agreeing is forwarded.
    captured_dispatch.clear()
    out = tool("run 'echo hi' every 5 minutes", "cron", options='{"auto_classify": true}')
    assert out == "Error: cron: the script locks auto_classify=False (asked for True)"
    assert not captured_dispatch
    tool("say hi", "ntfy", options='{"auto_classify": "False", "use_memory": false}')
    assert captured_dispatch[0]["classify_tasks"] is False
    assert captured_dispatch[0]["use_memory"] is False


def test_dispatch_forwards_parent_identity(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
) -> None:
    """A dispatch on behalf of a calling task marks it as the parent.

    ``_dispatch`` forwards the calling agent's persisted task id and
    frontend tab id as ``daemon_client.run``'s ``parent_task_id`` /
    ``parent_tab_id``, which is what gives the sub-task the
    ``run_parallel`` sub-agent tab semantics (nested tab, nested
    history row, ``subagentDone``) instead of a top-level tab.  The
    real dispatch path is exercised up to the daemon-client boundary;
    only that boundary call is captured.  The duck-typed-caller guard
    (a ``parent_agent`` with a persisted ``last_task_id`` but no
    ``_subagent_parent_tab_id``) stays untested by design: every real
    persisting agent is a ``ChatSorcarAgent``, which always has the
    resolver, and covering it would need a fabricated stand-in object.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    caller = tmp_path / "caller_project"
    script = _write_helper_script(caller)

    # A calling agent with a persisted task row: its task id and its
    # frontend tab id ride along, so the daemon runs the sub-task as
    # that task's sub-agent.
    parent = ChatSorcarAgent("Dispatch parent")
    parent._last_task_id = "a" * 32
    parent._tab_id = "webtab-7"
    tool = make_run_agent_tool(str(caller), parent)
    tool("say hi", str(script))
    assert captured_dispatch[0]["parent_task_id"] == "a" * 32
    assert captured_dispatch[0]["parent_tab_id"] == "webtab-7"

    # The same caller identity rides along in channel mode too.
    captured_dispatch.clear()
    tool("say hi", "ntfy")
    assert captured_dispatch[0]["parent_task_id"] == "a" * 32
    assert captured_dispatch[0]["parent_tab_id"] == "webtab-7"

    # A calling agent that has not persisted a row yet (before its
    # first run) dispatches an ordinary top-level task.
    fresh = ChatSorcarAgent("Fresh parent")
    fresh._tab_id = "webtab-8"
    captured_dispatch.clear()
    make_run_agent_tool(str(caller), fresh)("say hi", str(script))
    assert captured_dispatch[0]["parent_task_id"] == ""
    assert captured_dispatch[0]["parent_tab_id"] == ""

    # Standalone use: no calling agent at all.
    captured_dispatch.clear()
    make_run_agent_tool(str(caller))("say hi", str(script))
    assert captured_dispatch[0]["parent_task_id"] == ""
    assert captured_dispatch[0]["parent_tab_id"] == ""


def test_cron_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    # "cron" (any case/spacing) routes to the built-in cron agent
    # script (``cron_agent.py``, named by its file stem like every
    # other script), not to channel lookup: the dispatch fails only on
    # the unreachable daemon and runs in the cron work directory the
    # module's ``settings()`` name.
    out = run_agent("run 'echo hi' every 5 minutes", "  Cron ")
    assert "unknown agent" not in out
    assert out.startswith("Error: the cron agent task could not run:")
    assert "no-daemon.json" in out
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
    out = run_agent("say hi", "no_such_channel")
    assert out.startswith("Error: unknown agent")
    assert "cron" in out


def test_path_mode_missing_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_agent.py"
    out = run_agent("say hi", str(missing))
    assert out.startswith("Error: agent script")
    assert "does not exist" in out


def test_path_mode_non_python_file_error(tmp_path: Path) -> None:
    not_py = tmp_path / "agent.txt"
    not_py.write_text("hello")
    out = run_agent("say hi", str(not_py))
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
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    # ``workspace`` is a channel option: a session SEA refuses it.
    out = run_agent("say hi", str(script), options='{"workspace": "ignored-ws"}')
    assert out == (
        "Error: my_researcher: options['workspace'] applies to a channel agent only; "
        "my_researcher is a session SEA"
    )
    out = run_agent("say hi", str(script))
    assert out.startswith("Error: the my_researcher agent task could not run:")
    assert "no-daemon.json" in out
    # Path mode never touches the channel workspace env var.
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # The standalone tool runs path-mode sub-tasks in agent_work.
    assert (tmp_path / "agent_work").is_dir()
    assert not (tmp_path / "channel_work").exists()


def test_default_agent_is_the_bundled_sorcar_sea(
    tmp_path: Path, captured_dispatch: list[dict[str, Any]]
) -> None:
    """``run_agent(task)`` with no ``agent`` runs ``seas/sorcar/sorcar_sea.py`` in path mode.

    The default is the installed file's absolute path (not a path
    relative to the calling work directory), so it resolves from any
    project; the sub-task runs in the calling task's work directory
    like every other path-named agent script.
    """
    from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH

    default = Path(DEFAULT_AGENT_PATH)
    assert default.is_absolute() and default.is_file()
    assert default.parts[-4:] == ("agents", "seas", "sorcar", "sorcar_sea.py")
    # The dummy SEA defines no getters: a plain Sorcar session.
    cmd = {"agentPath": DEFAULT_AGENT_PATH, "prompt": "say hi"}
    assert apply_agent_overrides(cmd) == set()
    assert cmd.pop("_runConfig") == {"sea": "sorcar", "kind": "session", "pinned": {}}
    assert cmd == {"agentPath": DEFAULT_AGENT_PATH, "prompt": "say hi"}

    caller = tmp_path / "caller_project"
    caller.mkdir()
    tool = make_run_agent_tool(str(caller))
    tool("say hi")
    assert captured_dispatch[0]["extension_agent_path"] == DEFAULT_AGENT_PATH
    assert captured_dispatch[0]["work_dir"] == str(caller)
    assert captured_dispatch[0]["prompt"] == "say hi"
    # Whitespace counts as "not given", like every other option.
    captured_dispatch.clear()
    tool("say hi", agent="   ")
    assert captured_dispatch[0]["extension_agent_path"] == DEFAULT_AGENT_PATH
    # An explicit agent still wins over the default.
    captured_dispatch.clear()
    tool("say hi", agent="ntfy")
    assert captured_dispatch[0]["extension_agent_path"].endswith("ntfy_sea.py")


def test_default_agent_unreachable_daemon_is_a_clean_error() -> None:
    # The standalone tool with no agent: path mode named after the
    # dummy SEA's file stem (without ``_sea``), failing only at the
    # unreachable daemon.
    out = run_agent("say hi")
    assert out.startswith("Error: the sorcar agent task could not run:")
    assert "no-daemon.json" in out


def test_tool_schema_requires_only_task() -> None:
    """The schema the LLM sees marks ``task`` required and every other parameter optional.

    The per-option keyword arguments of the earlier contract are gone:
    the tool has exactly the eight parameters below (the first seven
    mirror ``run_parallel``), the further ``kiss.server.sorcar.run``
    keywords travel in the ``options`` JSON object.
    """
    from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
    from kiss.core.models.model_info import model

    schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(run_agent)
    params = schema["function"]["parameters"]
    assert params["required"] == ["task"]
    assert list(params["properties"]) == [
        "task", "agent", "model", "tool_profile", "max_budget", "timeout", "options", "wait",
    ]
    agent_doc = params["properties"]["agent"]["description"]
    assert "plain Sorcar" in agent_doc
    assert "JSON object" in params["properties"]["options"]["description"]
    # The full docstring names the keys ``options`` accepts.
    for key in ("workspace", "use_worktree", "model_config", "chat_id", "inherit"):
        assert key in (run_agent.__doc__ or ""), key


def test_path_mode_detected_by_py_suffix_and_separator(
    tmp_path: Path,
) -> None:
    # ".py" suffix without a separator is path mode, not a channel.
    out = run_agent("say hi", "slack_sea.py")
    assert out.startswith("Error: agent script")
    # A separator without a ".py" suffix is path mode too — rejected
    # with the loader's .py diagnostic rather than "unknown agent".
    out = run_agent("say hi", str(tmp_path / "somedir" / "agent"))
    assert out.startswith("Error: agent script")
    assert "is not a Python (.py) file" in out


def test_relative_path_resolves_against_captured_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tool runs in the daemon process, whose CWD is unrelated to
    # the user's project: a relative agent path must resolve against
    # the CALLING task's work directory captured by the factory.
    project = tmp_path / "project"
    (project / "agents").mkdir(parents=True)
    script = project / "agents" / "reviewer.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    elsewhere = tmp_path / "daemon_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    tool = make_run_agent_tool(str(project))
    out = tool("say hi", "agents/reviewer.py")
    # The script was found (under the project, not under the CWD) and
    # the dispatch failed only on the unreachable daemon.
    assert out.startswith("Error: the reviewer agent task could not run:")
    assert "no-daemon.json" in out
    # A missing relative path names the project-anchored resolution.
    out = tool("say hi", "agents/nope.py")
    assert out.startswith("Error: agent script")
    assert str(project / "agents" / "nope.py") in out
    assert "does not exist" in out


def test_path_mode_runs_in_captured_work_dir(tmp_path: Path) -> None:
    # A path-named agent's sub-task runs in the calling task's work
    # directory — no agent_work/channel_work scratch dir is created.
    project = tmp_path / "project"
    project.mkdir()
    script = project / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    out = make_run_agent_tool(str(project))("say hi", str(script))
    assert out.startswith("Error: the helper agent task could not run:")
    assert not (tmp_path / "agent_work").exists()
    assert not (tmp_path / "channel_work").exists()


def test_standalone_relative_path_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without a captured work directory (standalone tool), a relative
    # path resolves against the process working directory.
    script = tmp_path / "local_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    monkeypatch.chdir(tmp_path)
    out = run_agent("say hi", "local_agent.py")
    assert out.startswith("Error: the local_agent agent task could not run:")


def test_dispatch_forwards_the_workspace_to_the_daemon(
    captured_dispatch: list[dict[str, Any]],
) -> None:
    # The dispatcher never touches the process-global workspace: it
    # forwards the ``workspace`` option as a wire field, and the
    # daemon's task runner holds it for the channel run's lifetime
    # (see kiss.server.task_runner).
    import os

    run_agent("say hi", "ntfy", options='{"workspace": " my-ws "}')
    assert captured_dispatch[0]["workspace"] == "my-ws"
    assert Path(captured_dispatch[0]["extension_agent_path"]).parts[-2:] == ("ntfy", "ntfy_sea.py")
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    captured_dispatch.clear()
    run_agent("say hi", "ntfy")
    assert captured_dispatch[0]["workspace"] == ""
    assert not hasattr(agent_dispatch, "WORKSPACE_WAIT_TIMEOUT_SECONDS")


def test_dispatch_uses_recorded_daemon_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inside the kiss-web daemon the cron scheduler records the
    # daemon's own endpoint file at boot; the dispatch must target it
    # even when KISS_SORCAR_LOCAL points elsewhere — in path mode too.
    recorded = tmp_path / "recorded-daemon.json"
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(recorded))
    assert _daemon_endpoint_file() == str(recorded)
    out = run_agent("say hi", "ntfy")
    assert "recorded-daemon.json" in out
    script = tmp_path / "probe_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'model': 'm'}
""")
    out = run_agent("say hi", str(script))
    assert "recorded-daemon.json" in out


def test_agent_class_resolution() -> None:
    import kiss.agents.third_party_agents.slack.slack_sea as slack_sea

    cls = _agent_class(slack_sea)
    assert cls is not None and cls.__name__ == "SlackAgent"
    # A module defining no BaseChannelAgent subclass of its own
    # (imported classes do not count) resolves to None.
    assert _agent_class(agent_dispatch) is None


def test_every_channel_module_is_dispatchable() -> None:
    import importlib

    for channel in available_channels():
        module = importlib.import_module(f"kiss.agents.third_party_agents.{channel}.{channel}_sea")
        cls = _agent_class(module)
        assert cls is not None, channel
        assert isinstance(
            getattr(cls, "channel_system_prompt", None),
            str,
        ), channel
        assert module.__file__ and Path(module.__file__).is_file(), channel
        # Every channel module is a ``channel``-preset SEA whose
        # ``tools()`` adds the channel's tools to the toolset and whose
        # ``system_prompt()`` appends the channel's guidance (the agent
        # class's ``channel_system_prompt``) to the assembled prompt.
        sea = load_sea(Path(module.__file__))
        assert sea.settings({})["kind"] == "channel", channel
        assert sea.tools([]) and all(callable(t) for t in sea.tools([])), channel
        guidance = getattr(cls, "channel_system_prompt", "")
        assert sea.system_prompt("ASSEMBLED") == (
            "ASSEMBLED\n\n" + guidance if guidance else "ASSEMBLED"
        ), channel


def test_channel_module_is_a_valid_agent_script() -> None:
    """The exact agent-script contract a channel dispatch relies on.

    Passing a channel module as ``extension_agent_path`` makes the
    daemon apply its ``settings()`` (the ``channel`` preset: no
    worktree, no auto-commit, no classifier, no fan-out, no browser,
    no memory), stage its ``tools()`` as the run's tools hook (the
    channel tools on top of the built-in toolset), append the channel
    preamble to the system-prompt suffix, and stage its
    ``system_prompt()`` as the run's system-prompt hook (the channel
    guidance).  The dispatcher sends the task text verbatim: nothing
    of the channel guidance travels in the prompt any more.
    """
    import kiss.agents.third_party_agents.ntfy.ntfy_sea as ntfy_sea
    from kiss.agents.sorcar.agent_file import CHANNEL_PREAMBLE

    cmd: dict[str, Any] = {"agentPath": ntfy_sea.__file__, "appendToSystemPrompt": "Caller suffix."}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "toolsHook", "systemPromptHook", "useWorktree", "autoCommit", "classifyTasks",
        "isParallel", "useWebTools", "useMemory", "appendToSystemPrompt", "workDir",
    }
    assert cmd["workDir"] == str(kiss_home() / "channel_work")
    assert channel_workspace(cmd, load_layers(cmd)) == "default"
    tools = cmd["toolsHook"]([])
    assert tools and all(callable(t) for t in tools)
    assert "appendBasicTools" not in cmd
    assert "toolProfile" not in cmd  # ``tools()`` keeps the built-in toolset
    assert "toolsFile" not in cmd
    for field in ("useWorktree", "autoCommit", "classifyTasks", "isParallel",
                  "useWebTools", "useMemory"):
        assert cmd[field] is False, field
    assert cmd["appendToSystemPrompt"] == (
        "Caller suffix.\n\n" + CHANNEL_PREAMBLE.format(name="ntfy")
    )
    assert cmd["systemPromptHook"]("ASSEMBLED") == (
        "ASSEMBLED\n\n" + ntfy_sea.NtfyAgent.channel_system_prompt
    )
    assert "prompt" not in cmd


def test_run_agent_tool_and_sorcar_wiring() -> None:
    tool = make_run_agent_tool("")
    assert tool.__name__ == "run_agent"
    assert "slack" in (tool.__doc__ or "")
    # The module lives in the sorcar package and never imports from
    # kiss.agents.third_party_agents at module scope (soft plugin).
    source_text = Path(agent_dispatch.__file__).read_text(encoding="utf-8")
    assert Path(agent_dispatch.__file__).parent.parts[-2:] == ("agents", "sorcar")
    for line in source_text.splitlines():
        assert not line.startswith("from kiss.agents.third_party_agents")
        assert not line.startswith("import kiss.agents.third_party_agents")
    # The default Sorcar toolset registers the tool, bound to the
    # calling task's work directory AND the calling agent itself, so
    # each dispatched sub-task's spend is folded into the calling
    # task's cost accounting.
    agent_source = Path(agent_dispatch.__file__).parent / "sorcar_agent.py"
    assert 'tools.append(make_run_agent_tool(self.work_dir or "", self))' in agent_source.read_text(
        encoding="utf-8"
    )
    # The system prompt directs the agent to dispatch immediately.
    system_md = Path(agent_dispatch.__file__).parents[2] / "SYSTEM.md"
    assert "run_agent" in system_md.read_text(encoding="utf-8")
