# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the live ``/ask`` side-channel dispatch.

The user's task description names one specific interaction: while a
task is running in a tab, the user types ``/ask <question>`` in the
same chat webview.  The framework MUST NOT hand the message to the
outer agent's ``pending_user_messages`` queue (which is drained only
at the top of the next model step, so a blocked outer tool call would
delay the answer indefinitely).  Instead the daemon MUST dispatch
``ask_sea`` on a background worker with the running task's id already
substituted into ``append_to_prompt`` and ``append_to_system_prompt``
set verbatim, so the answering session runs independently and the
answer arrives in the same tab.

The tests exercise:

* ``_split_ask_command`` — the parser that decides which typed
  messages are ``/ask`` (word-boundary strict, empty-question
  rejecting).
* ``_cmd_append_user_message`` — an ``/ask`` prompt on a live tab
  MUST NOT touch ``pending_user_messages`` and MUST fire
  ``_dispatch_ask_side_channel`` with the owner task id, tab id,
  chat id, and the trimmed question.  A NON-``/ask`` prompt MUST
  still queue.
* ``_dispatch_ask_side_channel`` — the background worker calls
  ``daemon_client.run`` with the ask_sea path, the OWNER's task id
  substituted into ``append_to_prompt``, the fixed
  ``append_to_system_prompt`` (read from ``ask_sea``),
  ``parent_task_id`` / ``parent_tab_id`` / ``chat_id`` propagated,
  and ``use_worktree`` / ``auto_commit`` disabled (this is a
  read-only Q&A run).
* ``_broadcast_ask_answer`` — once ``daemon_client.run`` returns
  (or raises) the worker broadcasts ONE ``ask_answer`` event carrying
  the question, the answer text, the success flag, the asking tab id
  and the owner task id, so the running task's transcript shows the
  answer after the answering sub-agent's tab is gone.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.ask import ask_sea
from kiss.agents.sorcar import daemon_client, sea_commands
from kiss.agents.sorcar.agent_file import apply_agent_overrides
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.commands import _split_ask_command
from kiss.server.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AgentStub:
    """State-stub carrying a ``last_task_id`` for ``_owner_task_id``.

    ``_owner_task_id`` reads ``state.agent.last_task_id``; nothing
    else on the agent object is touched by the code paths under
    test, so this two-attribute stub is enough.
    """

    def __init__(self, task_id: str) -> None:
        self.last_task_id = task_id


def _clear_registry() -> None:
    agent_state.agent_states.clear()


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Real :class:`VSCodeServer` whose broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = _capture  # type: ignore[assignment]
    return server, events


def _register_running_task(
    task_id: str, tab_id: str, chat_id: str = "chat-xyz",
) -> AgentState:
    """Register a state with a live task and an attached agent stub."""
    st = AgentState(
        task_id, tab_id=tab_id, chat_id=chat_id, server_owned=True,
    )
    st.is_task_active = True
    # ``_owner_task_id`` reads only ``last_task_id`` off the agent; the
    # stub is not a WorktreeSorcarAgent and the annotation says so.
    st.agent = _AgentStub(task_id)  # type: ignore[assignment]
    agent_state.register(st)
    return st


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    _clear_registry()
    sea_commands._reset_for_tests()
    yield
    _clear_registry()
    sea_commands._reset_for_tests()


# ---------------------------------------------------------------------------
# _split_ask_command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("/ask why did the last step fail?", "why did the last step fail?"),
        ("/ask  multiple   spaces   inside  ",
         "multiple   spaces   inside"),
        ("/ask\twith\ttabs", "with\ttabs"),
        ("/ask\nnewline start", "newline start"),
    ],
)
def test_split_ask_command_accepts_word_boundary(
    prompt: str, expected: str,
) -> None:
    """``/ask`` MUST be followed by whitespace and a non-empty tail."""
    assert _split_ask_command(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "/ask",
        "/ask ",
        "/ask   \t\n",
        "/askme why did this fail",  # glued suffix — not /ask
        "/askreally?",
        "hey /ask no leading",
        " /ask leading whitespace",  # leading space — not a slash cmd
        "?ask alternate prefix",
    ],
)
def test_split_ask_command_rejects_non_matches(prompt: str) -> None:
    """Non-``/ask`` (or empty-question) prompts MUST return ``None``."""
    assert _split_ask_command(prompt) is None


# ---------------------------------------------------------------------------
# _cmd_append_user_message routing
# ---------------------------------------------------------------------------


def _install_dispatch_capture(
    server: VSCodeServer,
) -> list[dict[str, Any]]:
    """Replace ``_dispatch_ask_side_channel`` with a recorder.

    Verifies the interception in ``_cmd_append_user_message`` without
    firing an actual daemon dispatch (the socket/thread side is
    exercised separately by the dispatcher tests).
    """
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    server._dispatch_ask_side_channel = _capture  # type: ignore[assignment]
    return calls


def test_ask_message_bypasses_pending_queue_and_dispatches() -> None:
    """``/ask`` on a live tab MUST fire the side channel, not queue.

    ``pending_user_messages`` MUST be empty afterwards — the whole
    point of the side channel is that the outer agent (blocked in a
    tool call) never sees the question.  The dispatch call MUST
    carry the OWNER's task id, tab id, chat id, and the stripped
    question.
    """
    server, events = _make_server()
    _register_running_task("task-abc", "tab-1", chat_id="chat-1")
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-1",
        "prompt": "/ask why did the last step fail?",
    })

    st = agent_state.find_by_tab("tab-1")
    assert st is not None
    assert st.pending_user_messages == []
    assert st.unattributed_prompt_echoes == []
    assert calls == [{
        "tab_id": "tab-1",
        "owner_task_id": "task-abc",
        "owner_agent": st.agent,
        "chat_id": "chat-1",
        "question": "why did the last step fail?",
    }]
    # The user's typed line is echoed into the tab so it shows up in
    # the transcript above where the answer will land.
    prompt_echoes = [e for e in events if e.get("type") == "prompt"]
    assert prompt_echoes == [{
        "type": "prompt",
        "text": "/ask why did the last step fail?",
        "tabId": "tab-1",
        "taskId": "task-abc",
    }]


def test_ask_help_answers_with_the_description_without_dispatch() -> None:
    """``/ask help`` on a live tab is answered with the ask SEA's ``description()``.

    Like every ``/xxx help``, it never launches the answering agent:
    the echo and ONE ``ask_answer`` carrying the bundled ``ask`` SEA's
    description (stamped with the owner task) are broadcast, and the
    steering queue stays empty.  ``/ask help me`` is a real question
    and is dispatched as before.
    """
    from kiss.agents.seas.ask import ask_sea

    server, events = _make_server()
    _register_running_task("task-abc", "tab-1", chat_id="chat-1")
    calls = _install_dispatch_capture(server)
    sea_commands.refresh_registry()

    server._cmd_append_user_message({"tabId": "tab-1", "prompt": "/ask HELP"})

    st = agent_state.find_by_tab("tab-1")
    assert st is not None
    assert st.pending_user_messages == []
    assert calls == []
    assert [e for e in events if e.get("type") == "prompt"] == [{
        "type": "prompt", "text": "/ask HELP", "tabId": "tab-1", "taskId": "task-abc",
    }]
    assert [e for e in events if e.get("type") == "ask_answer"] == [{
        "type": "ask_answer",
        "question": "HELP",
        "text": ask_sea.AskSea().description(),
        "success": True,
        "tabId": "tab-1",
        "taskId": "task-abc",
    }]

    # ``/ask help me`` is a real question: dispatched as before.
    server._cmd_append_user_message({"tabId": "tab-1", "prompt": "/ask help me"})
    assert [c["question"] for c in calls] == ["help me"]


def test_ask_check_answers_with_the_dry_run_report_without_dispatch() -> None:
    """``/ask check`` on a live tab is the daemon's dry-run report, not a dispatch.

    ``check`` is reserved like ``help`` (``sea_commands.RESERVED_SUBCOMMANDS``):
    the report of ``sea_commands.sea_check`` is broadcast as the
    ``ask_answer`` and no answering agent is launched.
    """
    from kiss.agents.seas.ask import ask_sea

    server, events = _make_server()
    _register_running_task("task-abc", "tab-1", chat_id="chat-1")
    calls = _install_dispatch_capture(server)
    sea_commands.refresh_registry()

    server._cmd_append_user_message({"tabId": "tab-1", "prompt": "/ask check"})

    assert calls == []
    (answer,) = [e for e in events if e.get("type") == "ask_answer"]
    assert answer["success"] is True and answer["question"] == "check"
    assert answer["text"] == sea_commands.sea_check("ask", Path(ask_sea.__file__))
    assert answer["text"].startswith("/ask: " + ask_sea.AskSea().description())
    assert "tools added: task_context" in answer["text"]


def test_non_ask_message_still_queues_and_does_not_dispatch() -> None:
    """A plain follow-up MUST take the original queue path.

    Regression guard: the /ask interception is opt-in on the prefix
    only; every other message must reach ``pending_user_messages``
    exactly as before (that queue is what the drain hook reads).
    """
    server, _ = _make_server()
    st = _register_running_task("task-xyz", "tab-2", chat_id="chat-2")
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-2", "prompt": "plain follow-up",
    })

    assert st.pending_user_messages == ["plain follow-up"]
    assert calls == []


def test_ask_message_dropped_when_no_live_task() -> None:
    """No dispatch when the tab has no running task.

    Same policy as a plain follow-up: an idle tab has nothing to
    ask about.
    """
    server, _ = _make_server()
    st = AgentState(
        "task-idle", tab_id="tab-idle", chat_id="chat-idle",
        server_owned=True,
    )
    agent_state.register(st)
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-idle", "prompt": "/ask anything?",
    })

    assert st.pending_user_messages == []
    assert calls == []


def test_ask_message_with_leading_whitespace_is_not_intercepted() -> None:
    """``  /ask q`` (leading whitespace) MUST fall through to the queue.

    The SEA slash-command parser (``_split_slash_command``) rejects
    leading whitespace so ``  /ask`` is not a slash command; the
    side-channel interception MUST match that rule exactly, otherwise
    a typo (extra spaces) would silently reroute a follow-up message
    away from the running agent.
    """
    server, _ = _make_server()
    st = _register_running_task("task-lead", "tab-lead")
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-lead", "prompt": "   /ask why did this fail?",
    })

    assert st.pending_user_messages == ["   /ask why did this fail?"]
    assert calls == []


def test_ask_message_without_allocated_owner_task_id_falls_through() -> None:
    """No dispatch until the owner task has a persisted ``task_history`` row.

    Between ``run()`` entry and ``_add_task`` the running agent has
    no ``last_task_id``.  Dispatching the /ask side channel with an
    empty owner id would ship an ``append_to_prompt`` naming a
    blank task, and the answering agent would have nothing to read
    from ``~/.kiss/history.db``.  The safe behaviour is to fall
    through to the queue path (which stamps
    ``unattributed_prompt_echoes`` for the drain hook to persist
    once the task row exists) and skip the side channel.
    """
    server, _ = _make_server()
    st = AgentState(
        "task-pre", tab_id="tab-pre", chat_id="chat-pre",
        server_owned=True,
    )
    st.is_task_active = True
    # No ``agent`` attached — ``_owner_task_id`` returns ``""``, the
    # exact pre-allocation window the fix protects.
    agent_state.register(st)
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-pre", "prompt": "/ask why did this fail?",
    })

    # Side channel skipped; the /ask line queued as a steering
    # message and staged on the unattributed echo list so the drain
    # hook records it against whichever task consumes it.
    assert st.pending_user_messages == ["/ask why did this fail?"]
    assert st.unattributed_prompt_echoes == ["/ask why did this fail?"]
    assert calls == []


def test_ask_message_with_empty_question_falls_through_to_queue() -> None:
    """A bare ``/ask`` MUST NOT dispatch (no question to answer).

    It also MUST NOT vanish silently: the original queue path fires,
    exactly like any other non-blank message, so the user sees
    normal steering-message behaviour instead of a mystery no-op.
    """
    server, _ = _make_server()
    st = _register_running_task("task-q", "tab-q")
    calls = _install_dispatch_capture(server)

    server._cmd_append_user_message({
        "tabId": "tab-q", "prompt": "/ask   ",
    })

    # The original (unstripped) prompt reaches the queue: the /ask
    # parser only decides whether to intercept — trimming is not its
    # job.
    assert st.pending_user_messages == ["/ask   "]
    assert calls == []


# ---------------------------------------------------------------------------
# _dispatch_ask_side_channel
# ---------------------------------------------------------------------------


def _install_daemon_run_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Replace ``daemon_client.run`` with a recorder.

    Returns the mutable list every fake dispatch appends its kwargs
    to.  The recorder returns immediately with a canned successful
    :class:`daemon_client.TaskResult` — the answer the worker relays
    into the owner task's transcript.
    """
    calls: list[dict[str, Any]] = []

    def _fake_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        kwargs["prompt"] = prompt
        calls.append(kwargs)
        return daemon_client.TaskResult(
            text="<p>Step 3 failed because the file was missing.</p>",
            success=True,
            cost=0.01,
            tokens=10,
            steps=2,
        )

    monkeypatch.setattr(daemon_client, "run", _fake_run)
    return calls


def _wait_for(condition: Any, timeout: float = 5.0) -> None:
    """Poll *condition* until true or *timeout* seconds elapse."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for side-channel dispatch")


def test_side_channel_calls_daemon_run_with_correct_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker MUST call ``daemon_client.run`` with the pinned args.

    - ``prompt`` is the user's question (verbatim).
    - ``extension_agent_path`` is the resolved ``ask_sea.py`` path.
    - ``parent_task_id`` is the OWNER's task id: the daemon applies the
      ask SEA's methods itself — its ``prompt(task)`` (``{task_id}`` ->
      the owner id), ``system_prompt`` and ``tools`` as the run's hooks —
      so the side channel passes NO ``append_to_prompt`` /
      ``append_to_system_prompt`` of its own.
    - ``parent_tab_id`` / ``chat_id`` reach the daemon so the sub-agent
      tab lands in the running task's tab.
    - ``use_worktree`` / ``auto_commit`` are False (read-only Q&A).
    """
    server, _ = _make_server()
    calls = _install_daemon_run_capture(monkeypatch)

    server._dispatch_ask_side_channel(
        tab_id="tab-1",
        owner_task_id="task-abc",
        owner_agent=None,
        chat_id="chat-xyz",
        question="why did the last step fail?",
    )
    _wait_for(lambda: len(calls) == 1)

    kwargs = calls[0]
    assert kwargs["prompt"] == "why did the last step fail?"
    assert kwargs["extension_agent_path"] == str(Path(ask_sea.__file__))
    assert "append_to_prompt" not in kwargs
    assert "append_to_system_prompt" not in kwargs
    assert kwargs["parent_task_id"] == "task-abc"
    assert kwargs["parent_tab_id"] == "tab-1"
    assert kwargs["chat_id"] == "chat-xyz"
    assert kwargs["use_worktree"] is False
    assert kwargs["auto_commit"] is False
    assert kwargs["side_channel"] is True
    # What the daemon applies from the dispatched script for that owner:
    cmd: dict[str, Any] = {
        "agentPath": kwargs["extension_agent_path"],
        "parentTaskId": kwargs["parent_task_id"],
        "prompt": kwargs["prompt"],
    }
    apply_agent_overrides(cmd)
    assert cmd["prompt"] == (
        "why did the last step fail?\n\n"
        "The question above is about the task with id task-abc. "
        "Call task_context with that task id, then answer the question."
    )
    assert "appendToPrompt" not in cmd and "appendToSystemPrompt" not in cmd
    hooked = cmd["systemPromptHook"]("<the assembled prompt>")
    assert hooked == ask_sea.AskSea().system_prompt("<the assembled prompt>")
    assert "<the assembled prompt>" not in hooked
    assert (
        "**MUST FOLLOW: You MUST NOT USE internet or internet search at any point. "
        "You must answer quickly because the user is waiting.**"
    ) in hooked
    assert cmd["toolProfile"] == "none"
    assert [t.__name__ for t in cmd["toolsHook"]([])] == ["task_context"]
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False


def test_side_channel_ask_sea_path_resolves_to_bundled_seas_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The dispatched agent path MUST be the bundled ``ask_sea.py``.

    The bundled ``seas/`` folder has the lowest registry precedence, so
    a ``SEAS.md`` folder shipping ``ask/ask_sea.py`` shadows the ``/ask``
    chat command.  The side channel relies on the bundled module's
    ``prompt(task)`` (naming the owner task) and
    ``add_to_system_prompt()``, so it must dispatch that file even
    while the command is shadowed.
    """
    from kiss.core.config import kiss_home

    shadow = tmp_path / "user-seas" / "ask"
    shadow.mkdir(parents=True)
    (shadow / "ask_sea.py").write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return "shadow"
""", encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    seas_md = kiss_home() / "SEAS.md"
    seas_md.write_text(str(shadow.parent) + "\n", encoding="utf-8")
    try:
        sea_commands.refresh_registry()
        assert sea_commands.get_command("ask") == shadow / "ask_sea.py"

        server, _ = _make_server()
        calls = _install_daemon_run_capture(monkeypatch)

        server._dispatch_ask_side_channel(
            tab_id="tab-1", owner_task_id="task-1", owner_agent=None,
            chat_id="chat-1", question="q",
        )
        _wait_for(lambda: len(calls) == 1)
    finally:
        seas_md.unlink()

    dispatched = Path(calls[0]["extension_agent_path"]).resolve()
    assert dispatched == Path(ask_sea.__file__).resolve()


def test_side_channel_survives_daemon_run_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing dispatch MUST be swallowed (logged, not raised).

    A background daemon thread that propagates an exception would
    print an unhandled-exception traceback into the daemon log and
    then die; the interactive session must keep running either way.
    """
    server, events = _make_server()

    def _boom(prompt: str, **kwargs: Any) -> None:
        raise RuntimeError("simulated daemon failure")

    monkeypatch.setattr(daemon_client, "run", _boom)
    # The worker MUST NOT raise back into any calling thread — this
    # call would surface an unhandled thread exception in the test
    # runner if it did.  Instead the user gets a FAILED answer panel
    # naming the error, so they are not left waiting.
    server._dispatch_ask_side_channel(
        tab_id="tab-1", owner_task_id="task-1", owner_agent=None,
        chat_id="chat-1", question="q",
    )
    _wait_for(lambda: any(e["type"] == "ask_answer" for e in events))
    answers = [e for e in events if e["type"] == "ask_answer"]
    assert len(answers) == 1
    assert answers[0]["success"] is False
    assert "simulated daemon failure" in answers[0]["text"]
    assert answers[0]["question"] == "q"
    assert answers[0]["tabId"] == "tab-1"
    assert answers[0]["taskId"] == "task-1"


# ---------------------------------------------------------------------------
# _broadcast_ask_answer
# ---------------------------------------------------------------------------


def test_side_channel_broadcasts_answer_into_owner_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished dispatch MUST emit exactly one ``ask_answer`` event.

    The answering sub-agent's nested tab is closed by the frontend
    the moment it ends, so this event is the ONLY way the user sees
    the reply.  It carries the question (so the panel is
    self-contained on replay), the answer text and success flag from
    the :class:`TaskResult`, the asking tab id and the OWNER task id
    (the stamp that makes the printer record + persist it under the
    running task).
    """
    server, events = _make_server()
    _install_daemon_run_capture(monkeypatch)

    server._dispatch_ask_side_channel(
        tab_id="tab-1",
        owner_task_id="task-abc",
        owner_agent=None,
        chat_id="chat-xyz",
        question="why did step 3 fail?",
    )
    _wait_for(lambda: any(e["type"] == "ask_answer" for e in events))

    answers = [e for e in events if e["type"] == "ask_answer"]
    assert len(answers) == 1
    assert answers[0] == {
        "type": "ask_answer",
        "question": "why did step 3 fail?",
        "text": "<p>Step 3 failed because the file was missing.</p>",
        "success": True,
        "tabId": "tab-1",
        "taskId": "task-abc",
    }


def test_broadcast_ask_answer_without_owner_task_omits_task_id() -> None:
    """No owner task id → no ``taskId`` stamp (transient, still shown).

    The running task had not allocated its row at dispatch time; the
    answer is still rendered live in the asking tab, it just cannot
    be filed under a task for replay.
    """
    server, events = _make_server()
    server._broadcast_ask_answer(
        tab_id="tab-9",
        owner_task_id="",
        question="q",
        text="a",
        success=True,
    )
    assert events == [
        {
            "type": "ask_answer",
            "question": "q",
            "text": "a",
            "success": True,
            "tabId": "tab-9",
        }
    ]
