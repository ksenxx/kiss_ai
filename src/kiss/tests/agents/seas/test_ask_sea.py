# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``/ask`` SEA and its wiring.

Three surfaces are pinned here, and only these — the tests use no
mocks, just the real registry, the real daemon-side loader and the
real dispatch code (``/ask <question>`` resolving to ``ask_sea.py`` is
pinned for every public SEA by ``test_bundled_seas.py``):

1. The ``AskSea`` class itself: ``system_prompt`` MUST replace the
   assembled prompt with the bundled SYSTEM_LITE ablation prompt
   (``_ask_system_lite.md``) followed by the no-internet and
   answer-quickly directives and the answering playbook, ``tools`` MUST
   add the single ``task_context`` tool, the class MUST be a
   ``WorkerSea`` with the ``none`` tool profile (so there is no built-in
   tool besides ``finish``), and ``prompt()`` MUST append the fixed
   sentence carrying the ``{task_id}`` placeholder.
2. The daemon-side loader ``apply_sea`` MUST apply the
   settings to the wire fields, substitute ``{task_id}`` in what
   ``prompt()`` returns with the command's ``parentTaskId`` (empty
   string when absent) and stage the ``system_prompt`` and ``tools``
   methods as the daemon-side ``systemPromptHook`` / ``toolsHook``.
3. The dispatch layer ``dispatch_result`` MUST thread the calling task's
   ``last_task_id`` to the daemon as ``parentTaskId`` and pass the
   caller's ``add_to_prompt`` through verbatim: the substitution is
   the daemon's job now, so no dispatch-side rewrite touches the text.
   The ``run`` command is read off a local daemon stand-in
   (``RecordingDaemon``) the real ``daemon_client.run`` connects to.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.ask import ask_sea
from kiss.agents.seas.ask.ask_sea import AskSea
from kiss.agents.sorcar import agent_dispatch, daemon_client, sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.core.brand import BRAND, render_brand
from kiss.core.tool_verdict import ALLOW
from kiss.tests.agents.seas.sea_contract import assert_worker_contract
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

# The placeholder the daemon substitutes with the calling task's id.
_PLACEHOLDER = "{task_id}"

# The exact prompt suffix both dispatch paths use.
_EXPECTED_ADD_TO_PROMPT = (
    "The question above is about the task with id {task_id}. "
    "Call task_context with that task id, then answer the question."
)
_EXPECTED_SYSTEM_PROMPT = AskSea().system_prompt("ASSEMBLED PROMPT")
_EXPECTED_LITE = render_brand(ask_sea._SYSTEM_LITE_PATH.read_text(encoding="utf-8"))
_EXPECTED_PLAYBOOK = _EXPECTED_SYSTEM_PROMPT[len(_EXPECTED_LITE) + 2 :]
_EXPECTED_SUFFIX_START = (
    "**MUST FOLLOW: You MUST NOT USE internet or internet search "
    "at any point. You must answer quickly because the user is waiting.**"
)
_ASK_PATH = str(Path(ask_sea.__file__).resolve())
_EMPTY_SEA = "from kiss.agents.seas.base.base_sea import BaseSea\n\nclass Sea(BaseSea):\n    pass\n"
"""A SEA that defines nothing: the run is a plain session on the task text."""

# Every test starts and ends with an empty SEA registry (conftest ``reset_registry``).
pytestmark = pytest.mark.usefixtures("reset_registry")


# ---------------------------------------------------------------------------
# 1. The ask_sea module
# ---------------------------------------------------------------------------


def test_system_prompt_opens_with_system_lite_md() -> None:
    """system_prompt MUST replace the assembled prompt with SYSTEM_LITE.md, brand filled.

    The assembled prompt the daemon hands in is dropped (the default
    Sorcar prompt would drown the answering rules); the result starts
    with ``_ask_system_lite.md`` with only the brand placeholders filled.
    """
    text = AskSea().system_prompt("ASSEMBLED PROMPT")
    assert "ASSEMBLED PROMPT" not in text
    assert text.startswith(_EXPECTED_LITE + "\n\n")
    assert "{{IDENTITY}}" not in text
    assert BRAND["identity"] in text
    # SYSTEM_LITE.md is the ablation prompt: it MUST contain the
    # ``<identity>`` opening tag the ablation file starts with; a
    # blank / accidentally-empty file would silently satisfy the prefix
    # check above.
    assert "<identity>" in _EXPECTED_LITE


def test_system_prompt_ends_with_the_fixed_playbook() -> None:
    """After SYSTEM_LITE, system_prompt MUST carry the two fixed directives and the playbook.

    The no-internet directive comes first, then the answer-quickly
    sentence (the user typed ``/ask`` into a live task and is waiting
    on the reply); the playbook names the two-call recipe
    (``task_context`` then ``finish``), the answer style (two or
    three plain sentences, one ``<p>``) and the pitfalls seen in
    earlier runs (raw DB reads, editing files).
    """
    text = _EXPECTED_PLAYBOOK
    assert text.startswith(_EXPECTED_SUFFIX_START)
    assert "exactly two tools: `task_context` and `finish`" in text
    assert text.index("task_context(task_id)") < text.index("Call `finish`")
    assert "Two or three sentences" in text and "<p>…</p>" in text
    assert "history.db" in text and "read-only" in text
    assert "task_overview" not in text and "task_transcript" not in text
    assert ask_sea.ADD_TO_PROMPT == _EXPECTED_ADD_TO_PROMPT


def test_settings_follow_the_contract() -> None:
    """``settings()`` MUST be a tool-less worker; ``prompt()`` MUST name the task.

    ``WorkerSea`` pins worktree, auto-commit, classifier, browser and
    memory off; ``tool_profile: "none"`` keeps even the built-in
    toolset out so the answerer cannot run commands or touch files.
    ``prompt(question)`` is the question followed by the fixed sentence
    with ``{task_id}`` still a placeholder (the daemon fills it from
    ``parentTaskId``).  The resolved settings add the base class's
    defaults and nothing else: the other methods are applied by the daemon.
    """
    sea = AskSea()
    assert_worker_contract(ask_sea, sea, {"tool_profile": "none", "locked": ["tool_profile"]})
    assert sea.prompt("why?") == "why?\n\n" + _EXPECTED_ADD_TO_PROMPT
    assert _PLACEHOLDER in sea.prompt("why?")
    assert sea_commands.base_prompt([sea], "why?", "t-9") == (
        "why?\n\n" + _EXPECTED_ADD_TO_PROMPT.replace(_PLACEHOLDER, "t-9")
    )


def test_tools_adds_task_context_alone() -> None:
    """``tools`` MUST add ``task_context`` alone.

    The ``none`` tool profile (not the removed ``tools()`` getter) is
    what removes the built-in toolset; ``tools`` keeps whatever the run
    hands it and appends.  The prompt suffix is ``ADD_TO_PROMPT``; its
    old name ``APPEND_TO_PROMPT`` must not linger beside it (the removed
    getters are checked by ``test_settings_follow_the_contract``).
    """
    assert [t.__name__ for t in AskSea().tools([])] == ["task_context"]
    assert [t.__name__ for t in AskSea().tools([print])] == ["print", "task_context"]
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


# ---------------------------------------------------------------------------
# 2. The daemon-side loader: settings on the wire, {task_id} substitution
# ---------------------------------------------------------------------------


def test_apply_sea_applies_the_ask_settings_to_the_wire() -> None:
    """The daemon loader MUST wire the ask settings and methods onto the cmd.

    This exercises the real ``apply_sea`` path —
    :meth:`TaskRunner._run_task_inner` calls it just before the run —
    so the ``WorkerSea`` defaults land on ``classifyTasks`` /
    ``useWebTools`` / ``useMemory``,
    the ``none`` profile on ``toolProfile`` (the daemon derives "no
    built-in tools" from it: nothing stages ``appendBasicTools`` any
    more), ``system_prompt`` on the daemon-side ``systemPromptHook``
    (which replaces the assembled prompt) and ``tools`` on the
    daemon-side ``toolsHook`` (which appends ``task_context``).  No
    ``systemPrompt`` / ``appendToSystemPrompt`` / ``tools`` field is
    written any more.  The four hook fields are written on every run
    (identities where the SEA overrides nothing), so the returned set
    lists only the settings and the rewritten ``prompt``.
    """
    cmd: dict[str, Any] = {"seaPath": _ASK_PATH, "prompt": "why did the run fail?"}
    overridden = apply_sea(cmd)
    assert overridden == {
        "prompt",
        "toolProfile",
        "classifyTasks",
        "useWebTools",
        "useMemory",
    }
    assert cmd["systemPromptHook"]("ASSEMBLED PROMPT") == _EXPECTED_SYSTEM_PROMPT
    assert cmd["toolProfile"] == "none"
    assert cmd["classifyTasks"] is False
    assert cmd["useWebTools"] is False
    assert cmd["useMemory"] is False
    assert [tool.__name__ for tool in cmd["toolsHook"]([])] == ["task_context"]
    assert all(callable(tool) for tool in cmd["toolsHook"]([]))
    for field in ("systemPrompt", "appendToSystemPrompt", "tools", "appendBasicTools", "toolsFile"):
        assert field not in cmd, field
    # The SEA overrides neither call hook: the staged ones are identities.
    assert cmd["llmCallHook"]([{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]
    assert cmd["toolCallHook"]("Bash", {"command": "ls"}) == ALLOW
    # The question opens the prompt; ``prompt()`` appends the task framing.
    assert cmd["prompt"].startswith("why did the run fail?\n\n")


def test_apply_sea_substitutes_empty_when_no_parent_task_id() -> None:
    """A missing or non-string ``parentTaskId`` MUST still strip the placeholder.

    Leaving the literal ``{task_id}`` in place would confuse the
    answering agent; substituting with an empty string gives an
    obviously-empty task id that surfaces the bug loudly.
    """
    cmds: list[dict[str, Any]] = [
        {"seaPath": _ASK_PATH, "prompt": "q"},
        {"seaPath": _ASK_PATH, "prompt": "q", "parentTaskId": None},
        {"seaPath": _ASK_PATH, "prompt": "q", "parentTaskId": 42},
    ]
    for cmd in cmds:
        apply_sea(cmd)
        assert _PLACEHOLDER not in cmd["prompt"]
        # Every other character of the sentence is preserved.
        assert cmd["prompt"].endswith(
            "The question above is about the task with id . Call task_context "
            "with that task id, then answer the question."
        )


def test_apply_sea_keeps_the_callers_suffixes_and_stages_the_hook() -> None:
    """The caller's ``appendToPrompt`` / ``appendToSystemPrompt`` are not the SEA's to touch.

    ``prompt()`` shapes the prompt body, with ``{task_id}`` replaced by
    the command's ``parentTaskId`` (both ``/ask`` paths dispatch the
    answerer as a sub-agent of the task the question is about, so that
    IS the id it must pass to ``task_context``); both caller suffixes
    stay as sent on the wire.  ``system_prompt`` is staged as the hook
    the run applies to its assembled prompt (default + caller suffix):
    for ``/ask`` the hook replaces that text with SYSTEM_LITE + playbook.
    """
    cmd: dict[str, Any] = {
        "seaPath": _ASK_PATH,
        "prompt": "q",
        "parentTaskId": "task-xyz",
        "appendToPrompt": "stale caller suffix",
        "appendToSystemPrompt": "caller system text",
    }
    apply_sea(cmd)
    assert cmd["appendToPrompt"] == "stale caller suffix"
    assert cmd["prompt"] == "q\n\n" + _EXPECTED_ADD_TO_PROMPT.replace(_PLACEHOLDER, "task-xyz")
    assert _PLACEHOLDER not in cmd["prompt"]
    assert cmd["parentTaskId"] == "task-xyz"
    assert cmd["appendToSystemPrompt"] == "caller system text"
    assert cmd["systemPromptHook"]("DEFAULT\n\ncaller system text") == _EXPECTED_SYSTEM_PROMPT


def test_apply_sea_fills_a_callers_placeholder_without_prompt_method(
    tmp_path: Path,
) -> None:
    """``{task_id}`` is filled in wherever it stands in the prompt, not only in a method's text.

    A SEA whose class defines no ``prompt`` runs the task text through
    the identity ``prompt()`` of :class:`BaseSea` only; a ``{task_id}``
    in the caller's own text still gets the parent task's id.  The
    caller's ``appendToPrompt`` (a wire field, not part of the prompt
    chain) is left alone.
    """
    other = tmp_path / "other_sea.py"
    other.write_text(_EMPTY_SEA, encoding="utf-8")
    cmd: dict[str, Any] = {
        "seaPath": str(other),
        "prompt": "literal {task_id} in the task",
        "parentTaskId": "task-xyz",
        "appendToPrompt": "literal {task_id} stays here",
    }
    assert apply_sea(cmd) == {"prompt"}
    assert cmd["prompt"] == "literal task-xyz in the task"
    assert cmd["appendToPrompt"] == "literal {task_id} stays here"
    # Without a parent task the placeholder becomes the empty string.
    cmd = {"seaPath": str(other), "prompt": "literal {task_id} in the task"}
    assert apply_sea(cmd) == {"prompt"}
    assert cmd["prompt"] == "literal  in the task"


def test_apply_sea_substitutes_for_any_sea_defining_prompt(
    tmp_path: Path,
) -> None:
    """``{task_id}`` substitution is general: every SEA's ``prompt()`` result gets it.

    The earlier dispatch-side rewrite was special-cased on the file
    name ``ask_sea.py``; the daemon-side one is part of the ``prompt``
    method, so a user SEA under any name (including a look-alike such
    as ``my_ask_sea.py``) gets the same treatment.
    """
    for name in ("notify_sea.py", "my_ask_sea.py", "test_ask_sea.py"):
        script = tmp_path / name
        script.write_text(
            """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def prompt(self, task):
        return task + ' Report on task {task_id}.'
""",
            encoding="utf-8",
        )
        cmd: dict[str, Any] = {"seaPath": str(script), "prompt": "q", "parentTaskId": "t-1"}
        assert apply_sea(cmd) == {"prompt"}
        assert cmd["prompt"] == "q Report on task t-1."


# ---------------------------------------------------------------------------
# 3. The dispatch layer threads the parent id and leaves the text alone
# ---------------------------------------------------------------------------


def _parent_with_task_id(task_id: str) -> ChatSorcarAgent:
    """A ``ChatSorcarAgent`` carrying the persisted row id a real ``run`` leaves behind.

    ``_last_task_id`` is the per-run attribute ``run`` assigns (and
    ``last_task_id`` reads under the agent's lock); setting it directly
    keeps the dispatch tests independent of a model endpoint.
    """
    parent = ChatSorcarAgent("ask-parent")
    parent._last_task_id = task_id
    return parent


def _run_dispatch(
    sea_path: str,
    options: RunOptions,
    parent_task_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive :func:`dispatch_result` through the real ``daemon_client.run``.

    The daemon is a local stand-in reached through ``KISS_SORCAR_LOCAL``;
    the ``run`` command it received is returned.
    """
    daemon = RecordingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        result = agent_dispatch.dispatch_result(
            name="ask",
            prompt="why did the last step fail?",
            sea_path=sea_path,
            work_dir=str(tmp_path / "wd"),
            model_name="",
            budget=None,
            timeout=1.0,
            parent_agent=_parent_with_task_id(parent_task_id),
            scope_work_dir="",
            options=options,
            settings=sea_commands.sea_settings(Path(sea_path)),
        )
    finally:
        daemon.close()
    assert isinstance(result, daemon_client.TaskResult), result
    assert result.text == "ok"
    (command,) = daemon.run_commands
    return command


def test_dispatch_threads_the_parent_task_id_and_passes_the_text_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``dispatch_result`` MUST send the caller's id as ``parentTaskId`` and not rewrite text.

    The daemon substitutes ``{task_id}`` from that ``parentTaskId``
    (section 2), so the dispatcher no longer touches
    ``add_to_prompt``: a caller's text with the placeholder, a
    caller's text without it, and the empty default all reach the
    daemon verbatim, as does ``add_to_system_prompt``.
    """
    for text in ("literal {task_id} stays here", "no placeholder at all", ""):
        options = dataclasses.replace(
            RunOptions(),
            add_to_prompt=text,
            add_to_system_prompt="caller system text",
        )
        command = _run_dispatch(
            _ASK_PATH,
            options,
            parent_task_id="task-abc-123",
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )
        assert command["appendToPrompt"] == text
        assert command["appendToSystemPrompt"] == "caller system text"
        assert command["parentTaskId"] == "task-abc-123"
        assert command["seaPath"] == _ASK_PATH
        assert command["prompt"] == "why did the last step fail?"


def test_dispatch_with_an_empty_parent_task_id_sends_an_empty_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A caller without a persisted row MUST dispatch a top-level task (empty parent)."""
    command = _run_dispatch(
        _ASK_PATH,
        RunOptions(),
        parent_task_id="",
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    assert command["parentTaskId"] == ""
    assert command["parentTabId"] == ""
    assert command["appendToPrompt"] == ""


def test_ask_sea_is_not_advertised_as_a_channel() -> None:
    """``available_channels()`` MUST NOT list ``ask``.

    ``/ask`` is a slash-command-only SEA (no ``BaseChannelAgent``
    subclass), so listing it as a channel would break the
    "every channel module is dispatchable" invariant checked in
    ``test_every_channel_module_is_dispatchable``.
    """
    assert "ask" not in agent_dispatch.available_channels()
