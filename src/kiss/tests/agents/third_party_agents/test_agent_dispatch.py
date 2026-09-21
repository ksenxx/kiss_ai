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
    for hidden in ("a2a", "oai", "channel_cli", "backend_utils"):
        assert hidden not in channels
    assert channels == sorted(channels)


def test_docstring_lists_channels() -> None:
    doc = run_agent.__doc__ or ""
    assert "{channels}" not in doc
    assert "slack" in doc and "telegram" in doc


def test_unknown_agent_error() -> None:
    out = run_agent("say hi", "no_such_channel")
    assert out.startswith("Error: unknown agent")
    assert "not a path to a .py agent script" in out
    assert "slack" in out


def test_channel_name_is_normalized() -> None:
    # Case/whitespace variants still resolve; the unreachable daemon
    # then fails the dispatch cleanly instead of "unknown agent".
    out = run_agent("say hi", "  NTFY ")
    assert "unknown agent" not in out
    assert out.startswith("Error: the ntfy agent task could not run:")


def test_empty_task_error() -> None:
    assert run_agent("   ", "slack") == (
        "Error: task must be a non-empty string."
    )


def test_bad_budget_error() -> None:
    out = run_agent("say hi", "slack", max_budget="cheap")
    assert out == "Error: max_budget must be a number, got 'cheap'."
    for bad in ("nan", "inf", "0", "-2"):
        out = run_agent("say hi", "slack", max_budget=bad)
        assert out == (
            f"Error: max_budget must be a positive finite number, "
            f"got {bad!r}."
        )


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
        assert out.startswith(
            f"Error: the {canonical} agent task could not run:"
        )


def test_hyphenated_alias_is_a_channel_not_a_path() -> None:
    # A hyphen is a channel-name separator, not a path marker: the
    # alias resolves to the channel even though "-" appears in it.
    out = run_agent("say hi", "home-assistant")
    assert "unknown agent" not in out
    assert out.startswith(
        "Error: the homeassistant agent task could not run:"
    )


def test_channel_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    out = run_agent("say hi", "ntfy", max_budget="1.5")
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
    tool("say hi", "ntfy")
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
    tool("run 'echo hi' every 5 minutes", "cron", timeout="42.5")
    assert captured[0]["work_dir"] == cron_agent.work_dir()
    assert captured[0]["scope_work_dir"] == str(caller)
    assert captured[0]["timeout"] == 42.5
    assert captured[0]["stop_on_timeout"] is True

    # Path mode: executes in the caller's project (scope == work_dir).
    script = caller / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    captured.clear()
    tool("say hi", str(script))
    assert captured[0]["work_dir"] == str(caller)
    assert captured[0]["scope_work_dir"] == str(caller)
    assert (
        captured[0]["timeout"]
        == agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS
    )
    assert captured[0]["stop_on_timeout"] is True


def test_channel_and_cron_dispatch_skip_git_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Channel and cron dispatches run outside the project git lifecycle.

    A channel/cron sub-task executes in a scratch directory
    (``~/.kiss/channel_work`` / ``~/.kiss/cron/work``), so worktree
    setup would only copy whatever git repository happens to enclose
    that directory — a dirty repo at ``$HOME`` once stalled a gmail
    dispatch for minutes copying 65 GB before the sub-task's tab could
    even appear.  The dispatch therefore pins ``use_worktree=False``
    and ``auto_commit=False``.  Classification stays ENABLED for a
    channel dispatch (``classify_tasks=None`` — the daemon default
    decides, so a simple channel task gets the lite system prompt);
    the worktree pin is safe because a verdict can only demote a
    requested worktree run, never promote a pinned-off one
    (``WorktreeSorcarAgent.run``).  Only cron — an unattended
    automation that runs repeatedly — defaults ``classify_tasks`` to ``False``.
    A path-mode agent script keeps the standard lifecycle: it operates
    on the calling project unless its own getters say otherwise.  The
    real dispatch path is exercised up to the daemon-client boundary;
    only that boundary call is captured.
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

    # Channel mode: no worktree, no auto-commit; classification
    # follows the daemon's configured default (no per-run override).
    tool("say hi", "ntfy")
    assert captured[0]["use_worktree"] is False
    assert captured[0]["auto_commit"] is False
    assert captured[0]["classify_tasks"] is None

    # Cron mode: the module getters already return False for
    # use_worktree/auto_commit, the wire fields agree with them, and
    # classification defaults off (an explicit ``classify_tasks``
    # argument can turn it on; see
    # ``test_run_options_are_forwarded_to_daemon``).
    captured.clear()
    tool("run 'echo hi' every 5 minutes", "cron")
    assert captured[0]["use_worktree"] is False
    assert captured[0]["auto_commit"] is False
    assert captured[0]["classify_tasks"] is False

    # Path mode: the standard task lifecycle (worktree + auto-commit,
    # classification following the daemon's configured default).
    script = caller / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    captured.clear()
    tool("say hi", str(script))
    assert captured[0]["use_worktree"] is True
    assert captured[0]["auto_commit"] is True
    assert captured[0]["classify_tasks"] is None


def test_run_option_parse_errors(tmp_path: Path) -> None:
    """Malformed optional arguments fail before any daemon contact.

    Every optional argument of ``run_agent`` mirrors a keyword option
    of ``kiss.server.sorcar.run``; a value the daemon could not honour
    is reported by name with the offending text.
    """
    for name in (
        "use_worktree", "auto_commit", "use_web_tools", "classify_tasks",
        "use_memory", "is_parallel", "append_basic_tools",
    ):
        out = run_agent("say hi", "ntfy", **{name: "maybe"})
        assert out == f"Error: {name} must be 'true' or 'false', got 'maybe'."
    assert run_agent("say hi", "ntfy", model_config="[1, 2]") == (
        "Error: model_config must be a JSON object, got '[1, 2]'."
    )
    out = run_agent("say hi", "ntfy", model_config="{not json")
    assert out.startswith(
        "Error: model_config must be a JSON object, got '{not json': "
    )
    missing = tmp_path / "no_such_tools.py"
    out = run_agent("say hi", "ntfy", tools=str(missing))
    assert out == f"Error: tools file '{missing}' does not exist"
    not_py = tmp_path / "tools.txt"
    not_py.write_text("")
    out = run_agent("say hi", "ntfy", tools=str(not_py))
    assert out == f"Error: tools file '{not_py}' is not a Python (.py) file"


def test_channel_and_cron_refuse_worktree_and_auto_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channel/cron git-lifecycle pin cannot be overridden.

    A channel or cron sub-task runs in a scratch directory outside any
    project, where a worktree would copy whatever repository encloses
    ``$HOME`` (see ``test_channel_and_cron_dispatch_skip_git_lifecycle``),
    so asking for one is refused; an explicit ``"false"`` agrees with
    the pin and dispatches normally.
    """
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        captured.append(kwargs)
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)

    refused = (
        "agent task always runs without a git worktree or auto-commit"
    )
    for agent in ("ntfy", "cron"):
        for kwargs in (
            {"use_worktree": "true"},
            {"auto_commit": "TRUE"},
            {"use_worktree": "false", "auto_commit": "true"},
        ):
            out = run_agent("say hi", agent, **kwargs)
            assert out.startswith(f"Error: the {agent} {refused}"), out
    assert captured == []

    run_agent("say hi", "ntfy", use_worktree="false", auto_commit=" False ")
    assert captured[0]["use_worktree"] is False
    assert captured[0]["auto_commit"] is False


def test_run_options_are_forwarded_to_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional arguments reach ``daemon_client.run`` as its keyword options.

    Empty arguments forward the option's default (``None`` for the
    tri-state daemon-decides options, ``True`` for ``is_parallel`` /
    ``append_basic_tools``); explicit values are parsed and forwarded
    verbatim.  A relative ``tools`` path resolves against the CALLING
    task's work directory (the tool runs in the daemon process, whose
    working directory is unrelated).  The real dispatch path is
    exercised up to the daemon-client boundary; only that boundary
    call is captured.
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
    script = caller / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    tools_file = caller / "extra_tools.py"
    tools_file.write_text("def get_tools():\n    return []\n")
    tool = make_run_agent_tool(str(caller))

    # Nothing passed: the daemon's defaults decide.
    tool("say hi", str(script))
    defaults = captured[0]
    assert defaults["chat_id"] == ""
    assert defaults["system_prompt"] == ""
    assert defaults["tools"] is None
    assert defaults["model_config"] is None
    assert defaults["use_web_tools"] is None
    assert defaults["use_memory"] is None
    assert defaults["is_parallel"] is True
    assert defaults["append_basic_tools"] is True
    assert defaults["append_to_system_prompt"] == ""
    assert defaults["append_to_prompt"] == ""

    # Everything passed, in path mode: parsed and forwarded, with the
    # explicit git-lifecycle values replacing the path-mode defaults.
    captured.clear()
    tool(
        "say hi", str(script),
        chat_id=" chat-123 ",
        system_prompt="You are a terse helper.",
        tools="extra_tools.py",
        model_config='{"base_url": "http://localhost:8000/v1"}',
        use_worktree="false",
        auto_commit="False",
        use_web_tools="true",
        classify_tasks="false",
        use_memory="true",
        is_parallel="false",
        append_basic_tools="false",
        append_to_system_prompt="Answer in French.",
        append_to_prompt="Cite sources.",
    )
    sent = captured[0]
    assert sent["chat_id"] == "chat-123"
    assert sent["system_prompt"] == "You are a terse helper."
    assert sent["tools"] == str(tools_file)
    assert sent["model_config"] == {"base_url": "http://localhost:8000/v1"}
    assert sent["use_worktree"] is False
    assert sent["auto_commit"] is False
    assert sent["use_web_tools"] is True
    assert sent["classify_tasks"] is False
    assert sent["use_memory"] is True
    assert sent["is_parallel"] is False
    assert sent["append_basic_tools"] is False
    assert sent["append_to_system_prompt"] == "Answer in French."
    assert sent["append_to_prompt"] == "Cite sources."

    # An absolute tools path is kept as given (resolved); path mode
    # honours an explicit worktree request too.
    captured.clear()
    tool("say hi", str(script), tools=str(tools_file), use_worktree="true")
    assert captured[0]["tools"] == str(tools_file)
    assert captured[0]["use_worktree"] is True

    # The standalone tool (no calling work dir) resolves a relative
    # tools path against the process working directory.
    monkeypatch.chdir(caller)
    captured.clear()
    run_agent("say hi", str(script), tools="extra_tools.py")
    assert captured[0]["tools"] == str(tools_file)

    # An explicit classify_tasks overrides the mode default in every
    # mode: cron's pinned-off classification and the channel/path
    # "daemon decides" default alike.
    captured.clear()
    tool("run 'echo hi' every 5 minutes", "cron", classify_tasks="true")
    assert captured[0]["classify_tasks"] is True
    captured.clear()
    tool("say hi", "ntfy", classify_tasks="False", use_memory="false")
    assert captured[0]["classify_tasks"] is False
    assert captured[0]["use_memory"] is False


def test_dispatch_forwards_parent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    from kiss.agents.sorcar import daemon_client
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    captured: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        captured.append(kwargs)
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)

    caller = tmp_path / "caller_project"
    caller.mkdir()
    script = caller / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")

    # A calling agent with a persisted task row: its task id and its
    # frontend tab id ride along, so the daemon runs the sub-task as
    # that task's sub-agent.
    parent = ChatSorcarAgent("Dispatch parent")
    parent._last_task_id = "a" * 32
    parent._tab_id = "webtab-7"
    tool = make_run_agent_tool(str(caller), parent)
    tool("say hi", str(script))
    assert captured[0]["parent_task_id"] == "a" * 32
    assert captured[0]["parent_tab_id"] == "webtab-7"

    # The same caller identity rides along in channel mode too.
    captured.clear()
    tool("say hi", "ntfy")
    assert captured[0]["parent_task_id"] == "a" * 32
    assert captured[0]["parent_tab_id"] == "webtab-7"

    # A calling agent that has not persisted a row yet (before its
    # first run) dispatches an ordinary top-level task.
    fresh = ChatSorcarAgent("Fresh parent")
    fresh._tab_id = "webtab-8"
    captured.clear()
    make_run_agent_tool(str(caller), fresh)("say hi", str(script))
    assert captured[0]["parent_task_id"] == ""
    assert captured[0]["parent_tab_id"] == ""

    # Standalone tools-file use: no calling agent at all.
    captured.clear()
    make_run_agent_tool(str(caller))("say hi", str(script))
    assert captured[0]["parent_task_id"] == ""
    assert captured[0]["parent_tab_id"] == ""


def test_cron_dispatch_unreachable_daemon_is_a_clean_error(
    tmp_path: Path,
) -> None:
    # "cron" (any case/spacing) routes to the built-in cron agent
    # script, not to channel lookup: the dispatch fails only on the
    # unreachable daemon and runs in the cron work directory.
    out = run_agent("run 'echo hi' every 5 minutes", "  Cron ")
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
    script.write_text("def model() -> str:\n    return 'm'\n")
    out = run_agent("say hi", str(script), workspace="ignored-ws")
    assert out.startswith(
        "Error: the my_researcher agent task could not run:"
    )
    assert "no-daemon.sock" in out
    # Path mode never touches the channel workspace env var.
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ
    # The standalone tool runs path-mode sub-tasks in agent_work.
    assert (tmp_path / "agent_work").is_dir()
    assert not (tmp_path / "channel_work").exists()


def test_default_agent_is_the_bundled_dummy_sea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent(task)`` with no ``agent`` runs ``seas/dummy_sea.py`` in path mode.

    The default is the installed file's absolute path (not a path
    relative to the calling work directory), so it resolves from any
    project; the sub-task runs in the calling task's work directory
    like every other path-named agent script.
    """
    from kiss.agents.sorcar import daemon_client
    from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH

    default = Path(DEFAULT_AGENT_PATH)
    assert default.is_absolute() and default.is_file()
    assert default.parts[-3:] == ("agents", "seas", "dummy_sea.py")
    # The dummy SEA defines no getters: a plain Sorcar session.
    cmd = {"agentPath": DEFAULT_AGENT_PATH, "prompt": "say hi"}
    assert apply_agent_overrides(cmd) == set()
    assert cmd == {"agentPath": DEFAULT_AGENT_PATH, "prompt": "say hi"}

    captured: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="ok", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    caller = tmp_path / "caller_project"
    caller.mkdir()
    tool = make_run_agent_tool(str(caller))
    tool("say hi")
    assert captured[0]["extension_agent_path"] == DEFAULT_AGENT_PATH
    assert captured[0]["work_dir"] == str(caller)
    assert captured[0]["prompt"] == "say hi"
    # Whitespace counts as "not given", like every other option.
    captured.clear()
    tool("say hi", agent="   ")
    assert captured[0]["extension_agent_path"] == DEFAULT_AGENT_PATH
    # An explicit agent still wins over the default.
    captured.clear()
    tool("say hi", agent="ntfy")
    assert captured[0]["extension_agent_path"].endswith("ntfy_sea.py")


def test_default_agent_unreachable_daemon_is_a_clean_error() -> None:
    # The standalone tool with no agent: path mode named after the
    # dummy SEA's file stem, failing only at the unreachable daemon.
    out = run_agent("say hi")
    assert out.startswith("Error: the dummy_sea agent task could not run:")
    assert "no-daemon.sock" in out


def test_tool_schema_requires_only_task() -> None:
    """The schema the LLM sees marks ``task`` required and ``agent`` optional."""
    from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
    from kiss.core.models.model_info import model

    schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(run_agent)
    params = schema["function"]["parameters"]
    assert params["required"] == ["task"]
    assert list(params["properties"])[:2] == ["task", "agent"]
    assert "dummy_sea.py" in params["properties"]["agent"]["description"]


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tool runs in the daemon process, whose CWD is unrelated to
    # the user's project: a relative agent path must resolve against
    # the CALLING task's work directory captured by the factory.
    project = tmp_path / "project"
    (project / "agents").mkdir(parents=True)
    script = project / "agents" / "reviewer.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    elsewhere = tmp_path / "daemon_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    tool = make_run_agent_tool(str(project))
    out = tool("say hi", "agents/reviewer.py")
    # The script was found (under the project, not under the CWD) and
    # the dispatch failed only on the unreachable daemon.
    assert out.startswith("Error: the reviewer agent task could not run:")
    assert "no-daemon.sock" in out
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
    script.write_text("def model() -> str:\n    return 'm'\n")
    out = make_run_agent_tool(str(project))("say hi", str(script))
    assert out.startswith("Error: the helper agent task could not run:")
    assert not (tmp_path / "agent_work").exists()
    assert not (tmp_path / "channel_work").exists()


def test_standalone_relative_path_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a captured work directory (standalone tool), a relative
    # path resolves against the process working directory.
    script = tmp_path / "local_agent.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    monkeypatch.chdir(tmp_path)
    out = run_agent("say hi", "local_agent.py")
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
        out = run_agent("say hi", "ntfy", workspace="my-ws")
        assert out.startswith("Error: workspace 'my-ws' could not be activated")
        # The concurrent dispatch is still active; its workspace was
        # never overwritten.
        assert os.environ["KISS_CHANNEL_WORKSPACE"] == "other-ws"
        # A dispatch SHARING the active workspace proceeds normally
        # (and fails only at the unreachable daemon socket).
        out = run_agent("say hi", "ntfy", workspace="other-ws")
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
    out = run_agent("say hi", "ntfy")
    assert "recorded-daemon.sock" in out
    script = tmp_path / "probe_agent.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    out = run_agent("say hi", str(script))
    assert "recorded-daemon.sock" in out


def test_agent_class_resolution() -> None:
    import kiss.agents.third_party_agents.slack_sea as slack_sea

    cls = _agent_class(slack_sea)
    assert cls is not None and cls.__name__ == "SlackAgent"
    # A module defining no BaseChannelAgent subclass of its own
    # (imported classes do not count) resolves to None.
    assert _agent_class(agent_dispatch) is None


def test_every_channel_module_is_dispatchable() -> None:
    import importlib

    for channel in available_channels():
        module = importlib.import_module(
            f"kiss.agents.third_party_agents.{channel}_sea"
        )
        cls = _agent_class(module)
        assert cls is not None, channel
        assert isinstance(
            getattr(cls, "channel_system_prompt", None), str,
        ), channel
        assert module.__file__ and Path(module.__file__).is_file(), channel
        assert callable(getattr(module, "tools", None)), channel


def test_channel_module_is_a_valid_agent_script() -> None:
    # The exact contract the dispatch relies on: passing a channel
    # module as ``extension_agent_path`` makes the daemon use the module as its
    # own tools file (its ``tools()`` returns the tool list).
    import kiss.agents.third_party_agents.ntfy_sea as ntfy_sea

    cmd = {"agentPath": ntfy_sea.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"toolsFile"}
    assert cmd["toolsFile"] == ntfy_sea.__file__


def test_get_tools_and_sorcar_wiring() -> None:
    tools = get_tools()
    assert len(tools) == 1
    assert tools[0].__name__ == "run_agent"
    assert "slack" in (tools[0].__doc__ or "")
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
    assert (
        'tools.append(make_run_agent_tool(self.work_dir or "", self))'
        in agent_source.read_text(encoding="utf-8")
    )
    # The system prompt directs the agent to dispatch immediately.
    system_md = Path(agent_dispatch.__file__).parents[2] / "SYSTEM.md"
    assert "run_agent" in system_md.read_text(encoding="utf-8")
