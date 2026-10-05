# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``/ask`` SEA and its wiring.

Four surfaces are pinned here, and only these — the tests use no
mocks, just the real registry, the real slash-command resolver, the
real daemon-side loader and the real dispatch code:

1. The ``ask_sea`` module itself: ``system_prompt`` MUST return the
   bundled SYSTEM_LITE ablation prompt (``_ask_system_lite.md``),
   ``add_to_system_prompt`` MUST start with the no-internet and
   answer-quickly directives and carry the answering playbook,
   ``add_to_tools`` MUST expose the single ``task_context`` tool, and
   ``settings()`` MUST be a ``worker`` with the ``none`` tool profile
   (so there is no built-in tool besides ``finish``), and ``prompt()``
   MUST append the fixed sentence carrying the ``{task_id}`` placeholder.
2. The slash-command resolver ``slash_command_task`` MUST recognise
   ``/ask <question>`` and hand back the question verbatim with the
   registered ``ask_sea.py`` path: the daemon runs the SEA directly on
   it (no relay directive, no nested sub-agent).
3. The daemon-side loader ``apply_agent_overrides`` MUST apply the
   settings to the wire fields, substitute ``{task_id}`` in what
   ``prompt()`` returns with the command's ``parentTaskId`` (empty
   string when absent) and append the playbook AFTER any caller text
   on the system-prompt suffix.
4. The dispatch layer ``_dispatch`` MUST thread the calling task's
   ``last_task_id`` to the daemon as ``parent_task_id`` and pass the
   caller's ``append_to_prompt`` through verbatim: the substitution is
   the daemon's job now, so no dispatch-side rewrite touches the text.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.ask import ask_sea
from kiss.agents.sorcar import agent_dispatch, sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions
from kiss.agents.sorcar.agent_file import apply_agent_overrides
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.core.brand import BRAND, render_brand
from kiss.core.config import kiss_home
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters

# The placeholder the daemon substitutes with the calling task's id.
_PLACEHOLDER = "{task_id}"

# The exact prompt suffix both dispatch paths use.
_EXPECTED_ADD_TO_PROMPT = (
    "The question above is about the task with id {task_id}. "
    "Call task_context with that task id, then answer the question."
)
_EXPECTED_ADD_TO_SYSTEM_PROMPT = ask_sea.add_to_system_prompt()
_EXPECTED_SUFFIX_START = (
    "**MUST FOLLOW: You MUST NOT USE internet or internet search "
    "at any point. You must answer quickly because the user is waiting.**"
)
_EXPECTED_SETTINGS = {
    "kind": "worker",
    "use_worktree": False,
    "auto_commit": False,
    "auto_classify": False,
    "allow_fan_out": False,
    "use_web_tools": False,
    "use_memory": False,
    "tool_profile": "none",
    "locked": ["tool_profile"],
}
"""``resolve_settings`` output: ``settings()`` plus the ``worker`` preset.

``system_prompt()`` is a getter the daemon applies to ``systemPrompt``
(``apply_agent_overrides``), so its text is not a settings key.
"""
_ASK_PATH = str(Path(ask_sea.__file__).resolve())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Reset the SEA registry between tests so refresh_registry is honest.

    Also removes any ``SEAS.md`` a test wrote into the session-wide
    ``$KISS_HOME`` so it cannot shadow bundled commands for later tests.
    """
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. The ask_sea module
# ---------------------------------------------------------------------------


def test_system_prompt_returns_system_lite_md() -> None:
    """system_prompt MUST return SYSTEM_LITE.md with only the brand placeholders filled."""
    text = ask_sea.system_prompt()
    expected = render_brand(ask_sea._SYSTEM_LITE_PATH.read_text(encoding="utf-8"))
    assert text == expected
    assert "{{IDENTITY}}" not in text
    assert BRAND["identity"] in text
    # SYSTEM_LITE.md is the ablation prompt: it MUST contain the
    # ``<identity>`` opening tag the ablation file starts with; a
    # blank / accidentally-empty file would silently satisfy equality
    # above.
    assert "<identity>" in text


def test_add_to_system_prompt_returns_fixed_suffix() -> None:
    """add_to_system_prompt MUST open with the two fixed directives
    and carry the answering playbook.

    The no-internet directive comes first, then the answer-quickly
    sentence (the user typed ``/ask`` into a live task and is waiting
    on the reply); the playbook names the two-call recipe
    (``task_context`` then ``finish``), the answer style (two or
    three plain sentences, one ``<p>``) and the pitfalls seen in
    earlier runs (raw DB reads, editing files).
    """
    text = ask_sea.add_to_system_prompt()
    assert text.startswith(_EXPECTED_SUFFIX_START)
    assert "exactly two tools: `task_context` and `finish`" in text
    assert text.index("task_context(task_id)") < text.index("Call `finish`")
    assert "Two or three sentences" in text and "<p>…</p>" in text
    assert "history.db" in text and "read-only" in text
    assert "task_overview" not in text and "task_transcript" not in text
    assert ask_sea.ADD_TO_PROMPT == _EXPECTED_ADD_TO_PROMPT
    # The earlier contract's name is gone: the daemon reads only
    # ``add_to_system_prompt()``, so the old name would be dead code.
    assert not hasattr(ask_sea, "append_to_system_prompt")


def test_settings_follow_the_contract() -> None:
    """``settings()`` MUST be a tool-less worker; ``prompt()`` MUST name the task.

    ``worker`` pins worktree, auto-commit, classifier, fan-out, browser
    and memory off; ``tool_profile: "none"`` keeps even the built-in
    toolset out so the answerer cannot run commands or touch files.
    ``prompt(question)`` is the question followed by the fixed sentence
    with ``{task_id}`` still a placeholder (the daemon fills it from
    ``parentTaskId``).  The resolved settings add the preset's defaults
    and nothing else: the getters are applied by the daemon.
    """
    assert ask_sea.settings() == {
        "kind": "worker", "tool_profile": "none", "locked": ["tool_profile"],
    }
    assert resolve_settings(vars(ask_sea)) == _EXPECTED_SETTINGS
    assert ask_sea.prompt("why?") == "why?\n\n" + _EXPECTED_ADD_TO_PROMPT
    assert _PLACEHOLDER in ask_sea.prompt("why?")


def test_add_to_tools_is_task_context_alone_and_legacy_getters_are_gone() -> None:
    """``add_to_tools`` MUST be ``task_context`` alone; no per-field getter remains.

    The ``none`` tool profile (not the removed ``tools()`` getter) is
    what removes the built-in toolset.  Nothing reads the old per-field
    getters any more, so a SEA defining one would ship dead code that
    silently does nothing; none may exist.
    """
    assert [t.__name__ for t in ask_sea.add_to_tools()] == ["task_context"]
    assert_no_removed_getters(ask_sea)
    assert not hasattr(ask_sea, "APPEND_TO_PROMPT")


def test_system_lite_is_bundled_next_to_the_module() -> None:
    """The prompt MUST ship inside the package, not under ``papers/``.

    ``papers/`` is excluded from sdist/wheel per ``pyproject.toml``,
    and its ``ablation/prompts/SYSTEM_LITE.md`` is a frozen record of
    the ablation run (literal identity sentence, no brand
    placeholder), so ``system_prompt()`` must read a copy that lives
    next to ``ask_sea.py`` and carries ``{{IDENTITY}}`` for
    ``render_brand``.
    """
    path = ask_sea._SYSTEM_LITE_PATH
    assert path.parent == Path(ask_sea.__file__).resolve().parent
    assert path.is_file()
    assert "{{IDENTITY}}" in path.read_text(encoding="utf-8")


def test_ask_sea_lives_in_seas_package() -> None:
    """The SEA file MUST be discoverable by the slash-command registry.

    The registry scans the bundled ``seas`` package folder; if the
    file were somewhere else the /ask command would silently disappear.
    """
    module_path = Path(ask_sea.__file__).resolve()
    assert module_path.parent.name == "ask"
    assert module_path.parents[1].name == "seas"
    assert module_path.name == "ask_sea.py"
    sea_commands.refresh_registry()
    registered = sea_commands.get_command("ask")
    assert registered is not None
    assert registered.resolve() == module_path


def test_ask_command_is_registered_by_default() -> None:
    """Refreshing the registry MUST expose /ask as a known command.

    A regression here breaks the whole slash-command flow: no
    registry entry, no resolution, no run.
    """
    commands = sea_commands.refresh_registry()
    assert "ask" in commands


# ---------------------------------------------------------------------------
# 2. The slash-command resolver
# ---------------------------------------------------------------------------


def test_slash_ask_resolves_to_the_question_and_the_bundled_sea() -> None:
    """``/ask <question>`` MUST resolve to the question verbatim and ``ask_sea.py``.

    The daemon runs the SEA directly on that text (its ``settings()``
    supply the prompt suffix and the ``none`` tool profile), so the
    resolver hands back nothing but the user's words and the path: no
    directive, no ``run_agent`` arguments, no placeholder.
    """
    sea_commands.refresh_registry()
    hit = sea_commands.slash_command_task("/ask why did the last step fail?")
    assert hit is not None
    task_text, sea_path = hit
    assert sea_path.name == "ask_sea.py"
    assert sea_path.resolve() == Path(_ASK_PATH)
    assert task_text == "why did the last step fail?"
    assert _PLACEHOLDER not in task_text
    # The settings the daemon will apply to that very run.
    assert sea_commands.sea_settings(sea_path) == _EXPECTED_SETTINGS


def test_slash_resolver_treats_unrelated_commands_the_same_way(tmp_path: Path) -> None:
    """A non-``/ask`` slash command MUST resolve to its own text and path.

    Regression guard: nothing of the ask flow (its prompt suffix, its
    playbook) leaks into a sibling SEA's resolution.
    """
    folder = tmp_path / "user-seas"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notify").mkdir()
    (folder / "notify" / "notify_sea.py").write_text("# stub\n", encoding="utf-8")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(str(folder) + "\n", encoding="utf-8")
    sea_commands.refresh_registry()

    hit = sea_commands.slash_command_task("/notify hello")
    assert hit is not None
    task_text, sea_path = hit
    assert task_text == "hello"
    assert sea_path == folder / "notify" / "notify_sea.py"
    assert sea_commands.sea_settings(sea_path) == {"kind": "session"}


def test_slash_resolver_honours_a_user_sea_shadowing_ask(tmp_path: Path) -> None:
    """A ``SEAS.md`` folder that shadows ``/ask`` MUST win the resolution.

    The bundled ``seas/ask`` has the lowest registry precedence, so a
    user SEA named ``ask`` wins; it defines none of the ask settings,
    so the daemon would run it as a plain session on the question.
    """
    shadow = tmp_path / "user-seas" / "ask"
    shadow.mkdir(parents=True)
    (shadow / "ask_sea.py").write_text("# stub\n", encoding="utf-8")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(str(shadow.parent) + "\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert sea_commands.get_command("ask") == shadow / "ask_sea.py"
    hit = sea_commands.slash_command_task("/ask what happened?")
    assert hit is not None
    task_text, sea_path = hit
    assert sea_path == shadow / "ask_sea.py"
    assert task_text == "what happened?"
    assert sea_commands.sea_settings(sea_path) == {"kind": "session"}


def test_slash_resolver_rejects_bare_ask_and_answers_help_from_description() -> None:
    """``/ask`` with no trailing text MUST NOT resolve; ``/ask help`` is the description.

    Same contract as every other slash command: an empty task text
    runs nothing, and ``help`` (in any case) is answered from
    ``description()`` without a run.
    """
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/ask") is None
    assert sea_commands.slash_command_task("/ask ") is None
    assert sea_commands.slash_command_task("/ask help") is None
    assert sea_commands.slash_command_task("/ask HELP") is None
    assert sea_commands.help_text_if_command("/ask help") == ask_sea.description()


# ---------------------------------------------------------------------------
# 3. The daemon-side loader: settings on the wire, {task_id} substitution
# ---------------------------------------------------------------------------


def test_apply_agent_overrides_applies_the_ask_settings_to_the_wire() -> None:
    """The daemon loader MUST wire the ask settings and getters onto the cmd.

    This exercises the real ``apply_agent_overrides`` path —
    :meth:`TaskRunner._run_task_inner` calls it just before the run —
    so ``system_prompt()`` lands on ``systemPrompt``, the ``worker``
    preset on ``useWorktree`` / ``autoCommit`` / ``classifyTasks`` /
    ``isParallel`` / ``useWebTools`` / ``useMemory``, the ``none`` profile
    on ``toolProfile`` (the daemon derives "no built-in tools" from it:
    nothing stages ``appendBasicTools`` any more) and ``add_to_tools()``
    on the daemon-side ``tools`` field.
    """
    cmd: dict[str, Any] = {"agentPath": _ASK_PATH, "prompt": "why did the run fail?"}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "systemPrompt", "appendToSystemPrompt", "prompt", "toolProfile", "tools",
        "useWorktree", "autoCommit", "classifyTasks", "isParallel", "useWebTools", "useMemory",
    }
    assert cmd["systemPrompt"] == ask_sea.system_prompt()
    assert cmd["appendToSystemPrompt"] == _EXPECTED_ADD_TO_SYSTEM_PROMPT
    assert cmd["toolProfile"] == "none"
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False
    assert cmd["classifyTasks"] is False
    assert cmd["isParallel"] is False
    assert cmd["useWebTools"] is False
    assert cmd["useMemory"] is False
    assert [tool.__name__ for tool in cmd["tools"]] == ["task_context"]
    assert all(callable(tool) for tool in cmd["tools"])
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd
    # The question opens the prompt; ``prompt()`` appends the task framing.
    assert cmd["prompt"].startswith("why did the run fail?\n\n")


def test_apply_agent_overrides_substitutes_task_id_with_the_parent_task_id() -> None:
    """``{task_id}`` in ``prompt()``'s result MUST become the command's ``parentTaskId``.

    Both ``/ask`` paths dispatch the answering run as a sub-agent of
    the task the question is about, so ``parentTaskId`` IS the id the
    answerer must pass to ``task_context``.
    """
    cmd: dict[str, Any] = {
        "agentPath": _ASK_PATH,
        "prompt": "why did the last step fail?",
        "parentTaskId": "task-abc-123",
    }
    apply_agent_overrides(cmd)
    assert _PLACEHOLDER not in cmd["prompt"]
    assert cmd["prompt"] == (
        "why did the last step fail?\n\n"
        + _EXPECTED_ADD_TO_PROMPT.replace(_PLACEHOLDER, "task-abc-123")
    )
    assert cmd["parentTaskId"] == "task-abc-123"


def test_apply_agent_overrides_substitutes_empty_when_no_parent_task_id() -> None:
    """A missing or non-string ``parentTaskId`` MUST still strip the placeholder.

    Leaving the literal ``{task_id}`` in place would confuse the
    answering agent; substituting with an empty string gives an
    obviously-empty task id that surfaces the bug loudly.
    """
    cmds: list[dict[str, Any]] = [
        {"agentPath": _ASK_PATH, "prompt": "q"},
        {"agentPath": _ASK_PATH, "prompt": "q", "parentTaskId": None},
        {"agentPath": _ASK_PATH, "prompt": "q", "parentTaskId": 42},
    ]
    for cmd in cmds:
        apply_agent_overrides(cmd)
        assert _PLACEHOLDER not in cmd["prompt"]
        # Every other character of the sentence is preserved.
        assert cmd["prompt"].endswith(
            "The question above is about the task with id . Call task_context "
            "with that task id, then answer the question."
        )


def test_apply_agent_overrides_keeps_the_callers_prompt_suffix_and_appends_the_system_one() -> None:
    """The caller's ``appendToPrompt`` is not the SEA's to touch; the system suffix is additive.

    ``prompt()`` shapes the prompt body; the caller's ``appendToPrompt``
    stays as sent, and ``add_to_system_prompt()`` is appended after the
    caller's system-prompt suffix (``CALLER\\n\\nTEXT``) instead of
    replacing it.
    """
    cmd: dict[str, Any] = {
        "agentPath": _ASK_PATH,
        "prompt": "q",
        "parentTaskId": "task-xyz",
        "appendToPrompt": "stale caller suffix",
        "appendToSystemPrompt": "caller system text",
    }
    apply_agent_overrides(cmd)
    assert cmd["appendToPrompt"] == "stale caller suffix"
    assert cmd["prompt"] == "q\n\n" + _EXPECTED_ADD_TO_PROMPT.replace(_PLACEHOLDER, "task-xyz")
    assert cmd["appendToSystemPrompt"] == (
        "caller system text\n\n" + _EXPECTED_ADD_TO_SYSTEM_PROMPT
    )


def test_apply_agent_overrides_leaves_a_callers_placeholder_alone_without_prompt_getter(
    tmp_path: Path,
) -> None:
    """The substitution is a property of ``prompt()``'s result, not of the wire fields.

    A script that defines no ``prompt`` leaves the task text and the
    caller's ``appendToPrompt`` untouched: a literal ``{task_id}`` in
    the caller's own text reaches the run unchanged.
    """
    other = tmp_path / "other_sea.py"
    other.write_text("# stub\n", encoding="utf-8")
    cmd: dict[str, Any] = {
        "agentPath": str(other),
        "prompt": "literal {task_id} in the task",
        "parentTaskId": "task-xyz",
        "appendToPrompt": "literal {task_id} stays here",
    }
    assert apply_agent_overrides(cmd) == set()
    assert cmd["prompt"] == "literal {task_id} in the task"
    assert cmd["appendToPrompt"] == "literal {task_id} stays here"


def test_apply_agent_overrides_substitutes_for_any_sea_defining_prompt(
    tmp_path: Path,
) -> None:
    """``{task_id}`` substitution is general: every SEA's ``prompt()`` result gets it.

    The earlier dispatch-side rewrite was special-cased on the file
    name ``ask_sea.py``; the daemon-side one is part of the ``prompt``
    getter, so a user SEA under any name (including a look-alike such
    as ``my_ask_sea.py``) gets the same treatment.
    """
    for name in ("notify_sea.py", "my_ask_sea.py", "test_ask_sea.py"):
        script = tmp_path / name
        script.write_text(
            "def prompt(task):\n"
            "    return task + ' Report on task {task_id}.'\n",
            encoding="utf-8",
        )
        cmd: dict[str, Any] = {"agentPath": str(script), "prompt": "q", "parentTaskId": "t-1"}
        assert apply_agent_overrides(cmd) == {"prompt"}
        assert cmd["prompt"] == "q Report on task t-1."


def test_add_to_prompt_is_not_a_setting(tmp_path: Path) -> None:
    """A script declaring ``add_to_prompt`` in ``settings()`` MUST fail as an unknown key."""
    script = tmp_path / "old_sea.py"
    script.write_text(
        "def settings():\n    return {'add_to_prompt': 'Report on task {task_id}.'}\n",
        encoding="utf-8",
    )
    with pytest.raises(sea_commands.SeaError, match=r"has an unknown key 'add_to_prompt'"):
        sea_commands.sea_settings(script)


# ---------------------------------------------------------------------------
# 4. The dispatch layer threads the parent id and leaves the text alone
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal stand-in for a ChatSorcarAgent with a persisted task id.

    :func:`_persisted_task_id` reads ``last_task_id``; nothing else on
    the agent is touched before ``daemon_client.run`` fires.  A
    stopper exception on ``daemon_client.run`` lets the test capture
    the outbound arguments without running the daemon.
    """

    def __init__(self, task_id: str) -> None:
        self.last_task_id = task_id


class _DispatchCaptured(BaseException):
    """Raised from the daemon stub to stop dispatch and carry kwargs.

    Inherits :class:`BaseException` (not :class:`Exception`) so it is
    NOT caught by the generic ``except Exception`` in
    :func:`dispatch_result` that turns any daemon failure into an
    "Error:" string — the test needs the exception to propagate up so
    it can read the captured kwargs.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("captured")
        self.kwargs = kwargs


def _install_daemon_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``daemon_client.run`` with a capture that raises."""
    from kiss.agents.sorcar import daemon_client

    def _fake_run(prompt: str, **kwargs: Any) -> str:
        kwargs["prompt"] = prompt
        raise _DispatchCaptured(kwargs)

    monkeypatch.setattr(daemon_client, "run", _fake_run)


def _run_dispatch(
    agent_path: str, options: RunOptions, parent_task_id: str,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> dict[str, Any]:
    """Drive :func:`dispatch_result` and return the captured kwargs."""
    _install_daemon_capture(monkeypatch)
    parent = _StubAgent(parent_task_id)
    try:
        agent_dispatch.dispatch_result(
            name="ask",
            prompt="why did the last step fail?",
            agent_path=agent_path,
            work_dir=str(tmp_path / "wd"),
            model_name="",
            budget=None,
            timeout=1.0,
            parent_agent=parent,
            scope_work_dir="",
            options=options,
            settings=sea_commands.sea_settings(Path(agent_path)),
        )
    except _DispatchCaptured as captured:
        return captured.kwargs
    raise AssertionError("daemon_client.run was not invoked")


def test_dispatch_threads_the_parent_task_id_and_passes_the_text_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``_dispatch`` MUST send the caller's id as ``parent_task_id`` and not rewrite text.

    The daemon substitutes ``{task_id}`` from that ``parentTaskId``
    (section 3), so the dispatcher no longer touches
    ``append_to_prompt``: a caller's text with the placeholder, a
    caller's text without it, and the empty default all reach the
    daemon verbatim, as does ``append_to_system_prompt``.
    """
    for text in ("literal {task_id} stays here", "no placeholder at all", ""):
        options = dataclasses.replace(
            RunOptions(),
            add_to_prompt=text,
            add_to_system_prompt="caller system text",
        )
        captured = _run_dispatch(
            _ASK_PATH, options, parent_task_id="task-abc-123",
            monkeypatch=monkeypatch, tmp_path=tmp_path,
        )
        assert captured["append_to_prompt"] == text
        assert captured["append_to_system_prompt"] == "caller system text"
        assert captured["parent_task_id"] == "task-abc-123"
        assert captured["extension_agent_path"] == _ASK_PATH
        assert captured["prompt"] == "why did the last step fail?"


def test_dispatch_with_an_empty_parent_task_id_sends_an_empty_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A caller without a persisted row MUST dispatch a top-level task (empty parent)."""
    captured = _run_dispatch(
        _ASK_PATH, RunOptions(), parent_task_id="",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    assert captured["parent_task_id"] == ""
    assert captured["parent_tab_id"] == ""
    assert captured["append_to_prompt"] == ""


def test_ask_sea_is_not_advertised_as_a_channel() -> None:
    """``available_channels()`` MUST NOT list ``ask``.

    ``/ask`` is a slash-command-only SEA (no ``BaseChannelAgent``
    subclass), so listing it as a channel would break the
    "every channel module is dispatchable" invariant checked in
    ``test_every_channel_module_is_dispatchable``.
    """
    from kiss.agents.sorcar.agent_dispatch import available_channels

    assert "ask" not in available_channels()
