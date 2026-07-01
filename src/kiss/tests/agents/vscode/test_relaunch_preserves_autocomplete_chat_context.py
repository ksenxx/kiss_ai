# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: after VS Code is closed and relaunched, ghost-text
autocomplete on a restored tab must still be aware of the tab's prior
chat context — identifiers used in previous tasks of the same chat must
appear as completion candidates.

Reproduces this bug in ``VSCodeServer._cmd_complete``:

1. A prior VS Code session ran a task in tab ``T`` under
   ``chat_id = X``.  The task's persisted text contains the identifier
   ``zephyrIdentifierMarker`` (something the user might want to auto-
   complete later in the same conversation).
2. VS Code is closed (which stops the daemon).
3. VS Code is relaunched.  The webview restores its tabs from
   ``vscode.getState()`` and posts a ``ready`` message whose
   ``restoredTabs`` array names ``{tabId=T, chatId=X}``.  The extension
   replays this as a ``resumeSession`` command to the freshly-started
   daemon.
4. ``_replay_session`` handles ``resumeSession`` by recording
   ``_tab_chat_views[T] = X``.  Because no ``_RunningAgentState[T]``
   entry exists after a cold daemon restart (per the C2/C3 rule:
   loading a chat into a tab is a view operation and must not
   allocate a runtime state slot), the tab's chat association lives
   ONLY in ``_tab_chat_views``.
5. User types ``zephyr`` into the chat input.  The webview posts a
   ``complete`` command to the daemon.  Before the fix,
   ``_cmd_complete`` reads ``tab.chat_id`` from
   ``_RunningAgentState.running_agent_states.get(tab_id)`` — which
   returns ``None`` for a viewer-only tab — and passes ``chat_id=""``
   to the autocomplete pipeline.  The pipeline's
   ``_load_chat_context_text("")`` short-circuits to ``""`` and the
   ``zephyrIdentifierMarker`` from the prior task is NEVER offered as
   a completion candidate.  Autocomplete quality silently degrades
   until the user starts a new task on the tab (which finally mints
   a ``_RunningAgentState`` and populates ``tab.chat_id``).

Under the fix ``_cmd_complete`` consults ``_tab_chat_views`` as a
fallback when the tab has no runtime state (or ``tab.chat_id`` is
empty), so the resumed chat's identifiers keep flowing into ghost-text
suggestions from the very first keystroke after relaunch.

The end-to-end test drives a real :class:`VSCodeServer` through
``_handle_command`` — exactly as the extension layer would — from a
cold start, with only ``KISS_HOME`` / persistence redirected to an
isolated tmpdir.  The autocomplete worker thread is the real one; the
test waits for the ``completions`` broadcast that the worker emits.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _invalidate_chat_context_cache,
    _load_chat_context,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer

# A token that is astronomically unlikely to appear in any other file
# on disk or in any other test's persisted state — the assertion below
# relies on ``zephyrIdentifierMarker`` originating exclusively from
# the fixture task text/result we insert in step (1).
_MARKER = "zephyrIdentifierMarker"


def _redirect_persistence(tmpdir: str) -> tuple:
    """Point the persistence module at an isolated .kiss dir inside tmpdir."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _wait_for_event(
    events: list[dict[str, Any]],
    event_type: str,
    query: str,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Poll *events* for one whose type/query match, up to *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in events:
            if ev.get("type") == event_type and ev.get("query") == query:
                return ev
        time.sleep(0.01)
    return None


class TestRelaunchPreservesAutocompleteChatContext(unittest.TestCase):
    """End-to-end: after a daemon cold-start, ghost-text autocomplete on a
    restored tab must still surface identifiers from the tab's prior chat.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        # Point ``KISS_HOME`` at the tmpdir so ``_load_chat_context_text``'s
        # cache and any config lookups isolate cleanly from the real home.
        self._saved_kiss_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(Path(self._tmpdir) / ".kiss")
        self._saved_persistence = _redirect_persistence(self._tmpdir)
        # No live tab state may leak from another test in the same
        # process: cold-start semantics require an empty registry.
        _RunningAgentState.running_agent_states.clear()

    def tearDown(self) -> None:
        _restore_persistence(self._saved_persistence)
        if self._saved_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_kiss_home
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_autocomplete_uses_resumed_chat_context(self) -> None:
        # ------------------------------------------------------------------
        # 1) Simulate the state left by a prior VS Code session: a
        #    completed task in chat ``X`` whose result mentions
        #    ``_MARKER``.  This is the identifier autocomplete must
        #    still see after relaunch, without any live tab state.
        # ------------------------------------------------------------------
        prior_chat_id = "chat-relaunch-autocomplete-fixture-0000001"
        task_id, chat_id = _add_task(
            "Investigate the earlier problem",
            chat_id=prior_chat_id,
            extra={"model": "unused-in-this-test", "work_dir": self._tmpdir},
        )
        assert chat_id == prior_chat_id
        _save_task_result(
            task_id=task_id,
            result=(
                "I introduced a helper called "
                f"{_MARKER} that resolves the issue."
            ),
        )
        _flush_chat_events()
        # Bust any cached empty-context text a previous test in this
        # process may have stashed for the fixture chat id (belt-and-
        # suspenders — the fixture id is unique per test).
        _invalidate_chat_context_cache(prior_chat_id)

        # Sanity — persistence really carries the marker under this chat.
        ctx = _load_chat_context(prior_chat_id)
        combined = " ".join(
            f"{e.get('task', '')} {e.get('result', '')}" for e in ctx
        )
        assert _MARKER in combined, combined

        # ------------------------------------------------------------------
        # 2) Simulate VS Code relaunch by instantiating a FRESH
        #    :class:`VSCodeServer` (mirrors a daemon cold-start with
        #    empty ``_RunningAgentState.running_agent_states`` /
        #    ``_tab_chat_views``).
        # ------------------------------------------------------------------
        server = VSCodeServer()

        events: list[dict[str, Any]] = []
        events_lock = threading.Lock()

        def _capture(event: dict[str, Any]) -> None:
            with events_lock:
                events.append(dict(event))

        server.printer.broadcast = _capture  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # 3) The extension replays ``resumeSession`` for the restored
        #    tab.  After this call the tab has NO ``_RunningAgentState``
        #    entry (loading a chat into a tab is a view operation per
        #    C2/C3) — only ``_tab_chat_views[tab_id] = prior_chat_id``.
        # ------------------------------------------------------------------
        tab_id = "tab-restored-after-relaunch-autocomplete"
        server._handle_command(
            {
                "type": "resumeSession",
                "chatId": prior_chat_id,
                "tabId": tab_id,
            },
        )
        # Confirm the fixture: viewer-only tab (no registry entry) but
        # the chat-view mapping IS populated — this is the state under
        # which the bug fires.
        self.assertIsNone(
            _RunningAgentState.running_agent_states.get(tab_id),
            "resumeSession must not allocate a runtime tab state "
            "(C2/C3): pure viewers are tracked only in _tab_chat_views",
        )
        self.assertEqual(
            server._tab_chat_views.get(tab_id), prior_chat_id,
            "resumeSession must record the tab->chat association in "
            "_tab_chat_views for the follow-up route to work",
        )

        # ------------------------------------------------------------------
        # 4) User types the marker's prefix into the chat input.  The
        #    webview posts a ``complete`` command to the daemon.  The
        #    autocomplete worker is a real background thread; we
        #    poll for the ``completions`` broadcast.
        # ------------------------------------------------------------------
        prefix = _MARKER[:6]  # "zephyr"
        server._handle_command(
            {
                "type": "complete",
                "query": prefix,
                "tabId": tab_id,
                "connId": "cold-start-window-1",
            },
        )

        completions_event = _wait_for_event(events, "completions", prefix)
        self.assertIsNotNone(
            completions_event,
            f"Autocomplete never emitted a 'completions' event for "
            f"query={prefix!r}. events={[e.get('type') for e in events]}",
        )
        assert completions_event is not None  # for mypy
        texts = [
            c.get("text", "")
            for c in completions_event.get("completions", [])
        ]
        # ------------------------------------------------------------------
        # 5) The identifier persisted in the prior chat must appear as
        #    a completion candidate — proving that ``_cmd_complete``
        #    resolved the tab's chat id from ``_tab_chat_views`` and
        #    passed it through to ``_active_file_identifier_matches``.
        #    Under the bug ``chat_id=""`` is passed, the chat-context
        #    text is empty, and the marker is missing from ``texts``.
        # ------------------------------------------------------------------
        self.assertIn(
            _MARKER, texts,
            f"Ghost-text autocomplete lost its chat context after a "
            f"daemon cold-start: {_MARKER!r} from the prior chat "
            f"(chat_id={prior_chat_id!r}) is missing from the "
            f"completion candidates {texts!r}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
