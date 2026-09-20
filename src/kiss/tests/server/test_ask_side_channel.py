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
  ``append_to_system_prompt``, ``parent_task_id`` /
  ``parent_tab_id`` / ``chat_id`` propagated, and
  ``use_worktree`` / ``auto_commit`` disabled (this is a read-only
  Q&A run).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar import daemon_client, sea_commands
from kiss.agents.third_party_agents import ask_sea
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
    st.agent = _AgentStub(task_id)
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
    from ``~/.kiss/sorcar.db``.  The safe behaviour is to fall
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
    to.  The recorder returns immediately (no result inspected by
    the caller in this side-channel path).
    """
    calls: list[dict[str, Any]] = []

    def _fake_run(prompt: str, **kwargs: Any) -> object:
        kwargs["prompt"] = prompt
        calls.append(kwargs)
        return object()

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
    - ``append_to_prompt`` embeds the OWNER's task id (substituted).
    - ``append_to_system_prompt`` is the fixed no-internet directive.
    - ``parent_task_id`` / ``parent_tab_id`` / ``chat_id`` reach the
      daemon so the sub-agent tab lands in the running task's tab.
    - ``use_worktree`` / ``auto_commit`` are False (read-only Q&A).
    """
    server, _ = _make_server()
    calls = _install_daemon_run_capture(monkeypatch)

    server._dispatch_ask_side_channel(
        tab_id="tab-1",
        owner_task_id="task-abc",
        chat_id="chat-xyz",
        question="why did the last step fail?",
    )
    _wait_for(lambda: len(calls) == 1)

    kwargs = calls[0]
    assert kwargs["prompt"] == "why did the last step fail?"
    assert kwargs["extension_agent_path"] == str(
        sea_commands.get_command("ask")
    )
    assert kwargs["append_to_prompt"] == (
        "Read the events of the task task-abc from ~/.kiss/sorcar.db "
        "and answer the user question above."
    )
    assert kwargs["append_to_system_prompt"] == (
        "**MUST FOLLOW: You MUST NOT USE internet or internet search "
        "at any point."
    )
    assert kwargs["parent_task_id"] == "task-abc"
    assert kwargs["parent_tab_id"] == "tab-1"
    assert kwargs["chat_id"] == "chat-xyz"
    assert kwargs["use_worktree"] is False
    assert kwargs["auto_commit"] is False


def test_side_channel_ask_sea_path_resolves_to_third_party_agents_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatched agent path MUST be the real ``ask_sea.py``.

    Guard against a future rename or accidental shadowing by a
    ``SEAS.md`` folder: the /ask side channel always dispatches the
    bundled ``ask_sea.py`` under ``third_party_agents``.
    """
    from pathlib import Path

    server, _ = _make_server()
    calls = _install_daemon_run_capture(monkeypatch)

    server._dispatch_ask_side_channel(
        tab_id="tab-1", owner_task_id="task-1",
        chat_id="chat-1", question="q",
    )
    _wait_for(lambda: len(calls) == 1)

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
    server, _ = _make_server()

    def _boom(prompt: str, **kwargs: Any) -> None:
        raise RuntimeError("simulated daemon failure")

    monkeypatch.setattr(daemon_client, "run", _boom)
    # No assertion needed: the worker MUST NOT raise back into any
    # calling thread — this call would surface an unhandled thread
    # exception in the test runner if it did.
    server._dispatch_ask_side_channel(
        tab_id="tab-1", owner_task_id="task-1",
        chat_id="chat-1", question="q",
    )
    # Give the daemon thread a chance to run + log.
    import time

    time.sleep(0.05)
