# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``/ask`` SEA and its wiring.

Three surfaces are pinned here, and only these — the tests use no
mocks, just the real registry, the real rewriter and the real
dispatch code:

1. The ``ask_sea`` module itself: ``system_prompt`` MUST return the
   bytes of ``papers/kisssorcar/ablation/prompts/SYSTEM_LITE.md``,
   ``is_parallel`` and ``use_web_tools`` MUST return ``False``.
2. The command rewriter ``rewrite_prompt_if_command`` MUST recognise
   ``/ask <question>`` and emit a directive that instructs the outer
   LLM to call ``run_agent`` with the fixed ``append_to_prompt`` (with
   the ``<task_id>`` placeholder still intact) and
   ``append_to_system_prompt`` this command carries.
3. The dispatch layer ``_dispatch_reserved`` MUST substitute the
   literal ``<task_id>`` in ``options.append_to_prompt`` with the
   calling task's ``last_task_id`` before the daemon round trip,
   and MUST leave the substitution untouched for any other agent
   path (so an unrelated SEA whose ``append_to_prompt`` happens to
   contain the literal string is not mutated).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions
from kiss.agents.third_party_agents import ask_sea

# The literal placeholder the /ask flow substitutes at dispatch time.
_PLACEHOLDER = "<task_id>"

# The exact strings the task description dictates.
_EXPECTED_APPEND_TO_PROMPT = (
    "Read the events of the task <task_id> from ~/.kiss/sorcar.db "
    "and answer the user question above."
)
_EXPECTED_APPEND_TO_SYSTEM_PROMPT = (
    "**MUST FOLLOW: You MUST NOT USE internet or internet search "
    "at any point."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Reset the SEA registry between tests so refresh_registry is honest."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()


# ---------------------------------------------------------------------------
# 1. The ask_sea module
# ---------------------------------------------------------------------------


def test_system_prompt_returns_system_lite_md() -> None:
    """system_prompt MUST return the bytes of SYSTEM_LITE.md verbatim."""
    text = ask_sea.system_prompt()
    expected = ask_sea._SYSTEM_LITE_PATH.read_text(encoding="utf-8")
    assert text == expected
    # SYSTEM_LITE.md is the ablation prompt: it MUST contain the
    # ``<identity>`` opening tag the ablation file starts with; a
    # blank / accidentally-empty file would silently satisfy equality
    # above.
    assert "<identity>" in text


def test_is_parallel_returns_false() -> None:
    """is_parallel MUST be False so the Q&A run does not fan out."""
    assert ask_sea.is_parallel() is False


def test_use_web_tools_returns_false() -> None:
    """use_web_tools MUST be False so the Q&A run is offline."""
    assert ask_sea.use_web_tools() is False


def test_bundled_system_lite_is_byte_identical() -> None:
    """The wheel-fallback copy MUST match the repo copy byte for byte.

    The bundled ``_ask_system_lite.md`` next to ``ask_sea.py`` is
    what ``system_prompt()`` returns from a wheel install (``papers/``
    is excluded from sdist/wheel per ``pyproject.toml``).  If it
    drifts from the repo file, wheel users get a stale ablation
    prompt while source users get the updated one — a silent split
    the ablation study cannot tolerate.
    """
    repo_bytes = ask_sea._SYSTEM_LITE_PATH.read_bytes()
    bundled_bytes = ask_sea._BUNDLED_SYSTEM_LITE_PATH.read_bytes()
    assert bundled_bytes == repo_bytes


def test_system_prompt_falls_back_to_bundled_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A missing ``papers/`` MUST NOT break ``system_prompt()``.

    Simulates a wheel install by pointing ``_SYSTEM_LITE_PATH`` at a
    non-existent file: the function MUST fall back to
    ``_BUNDLED_SYSTEM_LITE_PATH`` (identical content, verified
    above) and return the same text a repo install would.
    """
    missing = tmp_path / "does-not-exist.md"
    monkeypatch.setattr(ask_sea, "_SYSTEM_LITE_PATH", missing)
    text = ask_sea.system_prompt()
    assert text == ask_sea._BUNDLED_SYSTEM_LITE_PATH.read_text(
        encoding="utf-8",
    )


def test_ask_sea_lives_in_third_party_agents_package() -> None:
    """The SEA file MUST be discoverable by the slash-command registry.

    The registry scans the third-party package folder; if the file
    were somewhere else the /ask command would silently disappear.
    """
    module_path = Path(ask_sea.__file__).resolve()
    assert module_path.parent.name == "third_party_agents"
    assert module_path.name == "ask_sea.py"


def test_ask_command_is_registered_by_default() -> None:
    """Refreshing the registry MUST expose /ask as a known command.

    A regression here breaks the whole slash-command flow: no
    registry entry, no rewrite, no dispatch.
    """
    commands = sea_commands.refresh_registry()
    assert "ask" in commands


# ---------------------------------------------------------------------------
# 2. The rewriter
# ---------------------------------------------------------------------------


def test_rewriter_emits_ask_directive_with_fixed_arguments() -> None:
    """``/ask <question>`` MUST rewrite to a run_agent directive.

    The directive MUST reference the resolved ``ask_sea.py`` path,
    carry the exact ``append_to_prompt`` and ``append_to_system_prompt``
    strings the task description dictates (with ``<task_id>`` still a
    literal placeholder — dispatch substitutes it later), and end
    with the user's question verbatim.
    """
    sea_commands.refresh_registry()
    hit = sea_commands.rewrite_prompt_if_command(
        "/ask why did the last step fail?"
    )
    assert hit is not None
    rewritten, sea_path = hit
    assert sea_path.name == "ask_sea.py"
    assert f'agent = "{sea_path}"' in rewritten
    assert f'append_to_prompt = "{_EXPECTED_APPEND_TO_PROMPT}"' in rewritten
    assert (
        f'append_to_system_prompt = "{_EXPECTED_APPEND_TO_SYSTEM_PROMPT}"'
        in rewritten
    )
    assert rewritten.endswith("why did the last step fail?")
    # The placeholder MUST reach dispatch intact — the rewriter has no
    # access to the calling task's id yet.
    assert _PLACEHOLDER in rewritten


def test_rewriter_leaves_unrelated_slash_commands_alone(
    tmp_path: Path,
) -> None:
    """A non-``/ask`` slash command MUST NOT carry the ask arguments.

    Regression guard: the /ask branch is opt-in on the command name
    only; a sibling SEA must still get the generic directive.
    """
    folder = tmp_path / "user-seas"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notify_sea.py").write_text("# stub\n", encoding="utf-8")
    from kiss.core.config import kiss_home

    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(str(folder) + "\n", encoding="utf-8")
    sea_commands.refresh_registry()

    hit = sea_commands.rewrite_prompt_if_command("/notify hello")
    assert hit is not None
    rewritten, _ = hit
    assert "append_to_prompt" not in rewritten
    assert "append_to_system_prompt" not in rewritten
    assert _EXPECTED_APPEND_TO_SYSTEM_PROMPT not in rewritten


def test_rewriter_rejects_bare_ask_without_question() -> None:
    """``/ask`` with no trailing text MUST NOT rewrite.

    Same contract as every other slash command: an empty task text
    would be rejected downstream by ``run_agent``.
    """
    sea_commands.refresh_registry()
    assert sea_commands.rewrite_prompt_if_command("/ask") is None
    assert sea_commands.rewrite_prompt_if_command("/ask ") is None


# ---------------------------------------------------------------------------
# 3. The dispatch-time <task_id> substitution
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
    :func:`_dispatch_reserved` that turns any daemon failure into an
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
    """Drive :func:`_dispatch_reserved` and return the captured kwargs."""
    _install_daemon_capture(monkeypatch)
    parent = _StubAgent(parent_task_id)
    try:
        agent_dispatch._dispatch_reserved(
            name="ask",
            prompt="why did the last step fail?",
            agent_path=agent_path,
            work_dir=str(tmp_path / "wd"),
            model_name="",
            budget=None,
            timeout=1.0,
            parent_agent=parent,
            scope_work_dir="",
            git_lifecycle=False,
            classify=True,
            parent_reviewer=False,
            options=options,
        )
    except _DispatchCaptured as captured:
        return captured.kwargs
    raise AssertionError("daemon_client.run was not invoked")


def test_dispatch_substitutes_task_id_placeholder_for_ask_sea(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``<task_id>`` in append_to_prompt MUST be replaced for ask_sea.

    The substitution uses the calling agent's persisted
    ``last_task_id``: that is the ``task_id`` of the task that is
    running when the /ask dispatch happens.
    """
    ask_path = str(Path(ask_sea.__file__).resolve())
    options = dataclasses.replace(
        RunOptions(),
        append_to_prompt=_EXPECTED_APPEND_TO_PROMPT,
        append_to_system_prompt=_EXPECTED_APPEND_TO_SYSTEM_PROMPT,
    )
    captured = _run_dispatch(
        ask_path, options, parent_task_id="task-abc-123",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    # The literal placeholder is gone; the calling task's id is in.
    assert _PLACEHOLDER not in captured["append_to_prompt"]
    assert "task-abc-123" in captured["append_to_prompt"]
    # The append_to_system_prompt reaches the daemon untouched.
    assert (
        captured["append_to_system_prompt"] == _EXPECTED_APPEND_TO_SYSTEM_PROMPT
    )
    # Parent identity is threaded through so the answering run is a
    # sub-agent of the calling task (and its events show up in the
    # right chat webview).
    assert captured["parent_task_id"] == "task-abc-123"


def test_dispatch_substitutes_even_when_parent_task_id_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An empty parent id MUST still strip the placeholder for ask_sea.

    Leaving the literal ``<task_id>`` in place would confuse the
    answering agent; substituting with an empty string gives an
    obviously-empty task id that surfaces the bug loudly.
    """
    ask_path = str(Path(ask_sea.__file__).resolve())
    options = dataclasses.replace(
        RunOptions(), append_to_prompt=_EXPECTED_APPEND_TO_PROMPT,
    )
    captured = _run_dispatch(
        ask_path, options, parent_task_id="",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    assert _PLACEHOLDER not in captured["append_to_prompt"]
    # Every other character of the sentence is preserved.
    assert captured["append_to_prompt"].startswith(
        "Read the events of the task  from"
    )


def test_dispatch_does_not_touch_placeholder_for_other_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A non-``ask_sea.py`` dispatch MUST leave ``<task_id>`` intact.

    Guards against widening the substitution beyond the /ask flow —
    an unrelated agent that happens to use the literal string
    ``<task_id>`` in its own append_to_prompt must reach the daemon
    unchanged.
    """
    other_path = str(tmp_path / "other_sea.py")
    Path(other_path).write_text("# stub\n", encoding="utf-8")
    options = dataclasses.replace(
        RunOptions(),
        append_to_prompt="literal <task_id> stays here",
    )
    captured = _run_dispatch(
        other_path, options, parent_task_id="task-xyz",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    assert captured["append_to_prompt"] == "literal <task_id> stays here"


@pytest.mark.parametrize(
    "look_alike_name",
    ["test_ask_sea.py", "not_ask_sea.py", "my_ask_sea.py"],
)
def test_dispatch_ignores_suffix_lookalikes(
    look_alike_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A path whose stem ENDS with ``ask_sea.py`` is not ``ask_sea.py``.

    Regression for a bug where ``agent_path.endswith("ask_sea.py")``
    matched ``test_ask_sea.py`` (and any other ``*ask_sea.py``), so a
    dispatch of an unrelated file whose name happens to end that way
    had its ``append_to_prompt`` rewritten.  The guard now compares
    ``Path(agent_path).name`` against ``"ask_sea.py"`` exactly.
    """
    look_alike = tmp_path / look_alike_name
    look_alike.write_text("# stub\n", encoding="utf-8")
    options = dataclasses.replace(
        RunOptions(),
        append_to_prompt="literal <task_id> stays here",
    )
    captured = _run_dispatch(
        str(look_alike), options, parent_task_id="task-xyz",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    assert captured["append_to_prompt"] == "literal <task_id> stays here"


def test_ask_sea_is_not_advertised_as_a_channel() -> None:
    """``available_channels()`` MUST NOT list ``ask``.

    ``/ask`` is a slash-command-only SEA (no ``BaseChannelAgent``
    subclass), so listing it as a channel would break the
    "every channel module is dispatchable" invariant checked in
    ``test_every_channel_module_is_dispatchable``.
    """
    from kiss.agents.sorcar.agent_dispatch import available_channels

    assert "ask" not in available_channels()


def test_dispatch_leaves_ask_sea_append_alone_when_no_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The guard MUST also gate on the placeholder being present.

    A caller that overrides ``append_to_prompt`` to a string without
    ``<task_id>`` (an unusual but legal use of the ask agent as a
    plain read-only Q&A) must pass through unchanged.
    """
    ask_path = str(Path(ask_sea.__file__).resolve())
    options = dataclasses.replace(
        RunOptions(), append_to_prompt="no placeholder at all",
    )
    captured = _run_dispatch(
        ask_path, options, parent_task_id="task-abc",
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )
    assert captured["append_to_prompt"] == "no placeholder at all"


# ---------------------------------------------------------------------------
# 4. The ask_sea overrides seen by apply_agent_overrides
# ---------------------------------------------------------------------------


def test_apply_agent_overrides_reads_ask_sea_getters(tmp_path: Path) -> None:
    """The daemon loader MUST wire the ask_sea getters onto the cmd.

    This exercises the real ``apply_agent_overrides`` path —
    :meth:`TaskRunner._run_task_inner` calls it just before the run —
    so the three getters ``system_prompt``, ``is_parallel``,
    ``use_web_tools`` reach the ``systemPrompt`` / ``useParallel`` /
    ``webTools`` wire fields correctly.
    """
    from kiss.server.agent_file import apply_agent_overrides

    ask_path = str(Path(ask_sea.__file__).resolve())
    cmd: dict[str, Any] = {
        "agentPath": ask_path,
        "prompt": "why did the run fail?",
    }
    overridden = apply_agent_overrides(cmd)
    assert "systemPrompt" in overridden
    assert "useParallel" in overridden
    assert "webTools" in overridden
    assert cmd["systemPrompt"] == ask_sea.system_prompt()
    assert cmd["useParallel"] is False
    assert cmd["webTools"] is False
