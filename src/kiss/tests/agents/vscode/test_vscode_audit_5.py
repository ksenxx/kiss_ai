"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.agents.vscode`` — audit round 5.

Each test confirms a real issue by exercising the actual code with real
objects (no mocks/patches).  Tests are grouped by category:

Bugs
----
B1: ``_await_user_response`` reads ``_tab_states`` without ``_state_lock``,
    inconsistent with the locking discipline used in every other method.

B2: ``_handle_autocommit_action`` reads ``_tab_states`` without
    ``_state_lock`` when persisting the autocommit event.

B3: ``_complete_from_active_file`` returns the **longest** matching
    suffix instead of the shortest, so autocomplete over-completes
    (e.g. "self.method_name_long" beats "self.method" for partial "self.me").

B4: Incomplete comment in ``_run_task_inner``: ``"BUG-B fix: if this
    worktree tab has a pending branch from a"`` — the sentence is
    truncated mid-clause.

B5: ``_new_chat`` has no guard against being called on a tab with a
    running task — ``agent.new_chat()`` would reset the chat_id
    mid-flight.

Redundancies
------------
R1: The ``noqa: F401 (re-export for tests)`` comment on the
    ``_cleanup_merge_data``, ``_git``, ``_merge_data_dir`` imports in
    ``server.py`` is misleading: all three symbols are used directly
    in the same file (``_close_tab``, ``_generate_commit_message``).

R2: ``parse_task_tags`` is listed in ``server.py.__all__`` and imported
    there, but it is never referenced inside ``server.py`` — the only
    call-site is ``task_runner.py``, which imports it directly from
    ``tab_state``.

Inconsistencies
---------------
I1: ``_cmd_user_answer`` uses ``cmd.get("tabId")`` (default ``None``)
    while every other handler uses ``cmd.get("tabId", "")`` (default
    empty string).  Downstream code (``_stop_task``, ``_finish_merge``,
    ``_close_tab``) guards on ``if not tab_id`` which treats both the
    same, but the inconsistency is error-prone.

I2: ``_complete_seq`` starts at 0 while ``_complete_seq_latest`` starts
    at -1.  The first increment produces seq=1 which works, but the
    asymmetric initialization obscures the invariant that
    ``_complete_seq_latest`` always equals the most-recently-dispatched
    ``_complete_seq`` value (or -1 meaning "none dispatched yet").
"""

from __future__ import annotations

import inspect
import queue
import re
import threading
import unittest

from kiss.agents.vscode.autocomplete import _AutocompleteMixin
from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ===================================================================
# B1 — _await_user_response reads _tab_states without lock
# ===================================================================


class TestAwaitUserResponseLockingBug(unittest.TestCase):
    """B1: ``_await_user_response`` accesses ``self._tab_states`` without
    acquiring ``_state_lock``, violating the locking discipline used
    everywhere else.
    """

    def test_source_reads_tab_states_without_lock(self) -> None:
        """Structural: the source accesses ``_tab_states`` outside of any
        ``with self._state_lock`` block.
        """
        src = inspect.getsource(_TaskRunnerMixin._await_user_response)
        # The access pattern: direct dict.get without lock
        assert "_tab_states.get(" in src, (
            "_await_user_response should access _tab_states"
        )
        # Verify no _state_lock acquisition guards the access
        assert "with self._state_lock" not in src, (
            "BUG B1 confirmed: _await_user_response reads _tab_states "
            "without _state_lock"
        )

    def test_behavioral_read_without_lock(self) -> None:
        """Behavioral: demonstrate that ``_await_user_response`` reads
        ``_tab_states`` without ever acquiring ``_state_lock``.

        We set up a server with a tab that has a stop_event and a
        user_answer_queue, then call _await_user_response from a
        thread and verify it completes — all without any _state_lock
        acquisition from the _await_user_response path.
        """
        server, _ = _make_server()
        tab = server._get_tab("test-tab")
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        tab.user_answer_queue.put("hello")

        # Set thread-local so _await_user_response can find the tab
        server.printer._thread_local.stop_event = tab.stop_event
        server.printer._thread_local.tab_id = "test-tab"

        # Verify it reads without lock by confirming it succeeds even
        # when the lock is held by another thread
        lock_held = threading.Event()
        done = threading.Event()
        result_box: list[str] = []

        def hold_lock() -> None:
            with server._state_lock:
                lock_held.set()
                done.wait(timeout=5)

        def call_await() -> None:
            # Wait for the lock to be held
            lock_held.wait(timeout=5)
            # This should succeed even though _state_lock is held,
            # because _await_user_response doesn't acquire it
            server.printer._thread_local.stop_event = tab.stop_event
            server.printer._thread_local.tab_id = "test-tab"
            result_box.append(server._await_user_response())

        t1 = threading.Thread(target=hold_lock)
        t2 = threading.Thread(target=call_await)
        t1.start()
        t2.start()
        t2.join(timeout=5)
        done.set()
        t1.join(timeout=5)

        # The call succeeded without deadlocking — proves it doesn't
        # acquire _state_lock
        assert result_box == ["hello"], (
            f"Expected ['hello'], got {result_box}. "
            "BUG B1: _await_user_response bypassed the lock"
        )


# ===================================================================
# B2 — _handle_autocommit_action reads _tab_states without lock
# ===================================================================


class TestAutocommitActionLockingBug(unittest.TestCase):
    """B2: ``_handle_autocommit_action`` accesses ``self._tab_states``
    without ``_state_lock`` when persisting the autocommit event.
    """

    def test_source_reads_tab_states_without_lock(self) -> None:
        """Structural: after ``if ok:`` the method reads ``_tab_states``
        directly.
        """
        src = inspect.getsource(_MergeFlowMixin._handle_autocommit_action)
        # Find the unguarded access
        assert "_tab_states.get(tab_id)" in src.replace("self.", ""), (
            "_handle_autocommit_action should access _tab_states"
        )
        # Count lock acquisitions vs _tab_states accesses
        lock_blocks = list(re.finditer(r"with self\._state_lock", src))
        tab_accesses = list(re.finditer(r"self\._tab_states\.get", src))
        # There should be at least one unguarded access
        # (the one inside the `if ok:` block after repo_lock release)
        assert len(tab_accesses) > len(lock_blocks), (
            f"BUG B2 confirmed: {len(tab_accesses)} _tab_states.get() calls "
            f"but only {len(lock_blocks)} _state_lock blocks"
        )


# ===================================================================
# B3 — _complete_from_active_file returns longest suffix
# ===================================================================


class TestAutocompleteReturnsShortest(unittest.TestCase):
    """B3: ``_complete_from_active_file`` picks the candidate with the
    **longest** suffix, not the shortest.  For autocomplete, the
    shortest match is almost always preferred (most specific, least
    surprising).
    """

    def test_source_picks_longest(self) -> None:
        """Structural: the comparison uses ``> len(best)`` — longest wins."""
        src = inspect.getsource(_AutocompleteMixin._complete_from_active_file)
        assert "> len(best)" in src, (
            "BUG B3 confirmed: source picks longest suffix (> len(best))"
        )
        # A shortest-first autocomplete would use < len(best) or a
        # different selection strategy
        assert "< len(best)" not in src, (
            "If < len(best) were present, it would pick shortest"
        )

    def test_behavioral_longest_wins(self) -> None:
        """Behavioral: given candidates 'method' and 'method_name_long',
        typing 'me' should complete to 'thod' (shortest) but actually
        completes to 'thod_name_long' (longest).
        """
        server, _ = _make_server()
        content = "method\nmethod_name_long\n"
        result = server._complete_from_active_file(
            "me", snapshot_content=content,
        )
        # BUG: returns the LONGEST suffix
        assert result == "thod_name_long", (
            f"BUG B3 confirmed: got '{result}', expected 'thod' (shortest) "
            "but autocomplete returns longest suffix"
        )
        # The correct behavior would be:
        # assert result == "thod"

    def test_behavioral_with_dotted_identifiers(self) -> None:
        """Behavioral: dotted identifiers also get longest-suffix treatment."""
        server, _ = _make_server()
        content = "self.run\nself.run_task_inner\n"
        result = server._complete_from_active_file(
            "self.ru", snapshot_content=content,
        )
        # BUG: returns longest suffix from dotted chains
        assert result == "n_task_inner", (
            f"BUG B3 confirmed: got '{result}', expected 'n' (from self.run)"
        )


# ===================================================================
# B4 — Incomplete comment in _run_task_inner
# ===================================================================


class TestIncompleteComment(unittest.TestCase):
    """B4: The ``BUG-B fix`` comment in ``_run_task_inner`` ends with
    ``"from a"`` — the sentence is truncated mid-clause.
    """

    def test_comment_is_truncated(self) -> None:
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        # Find the BUG-B comment
        match = re.search(r"# BUG-B fix:.*", src)
        assert match is not None, "BUG-B comment should exist"
        comment = match.group(0)
        # The comment ends with "from a" — an incomplete sentence
        assert comment.rstrip().endswith("from a"), (
            f"BUG B4 confirmed: comment is truncated: '{comment}'"
        )


# ===================================================================
# B5 — _new_chat has no running-task guard
# ===================================================================


class TestNewChatNoRunningGuard(unittest.TestCase):
    """B5: ``_new_chat`` calls ``tab.agent.new_chat()`` without checking
    whether the tab has a running task.  If a task is in flight,
    ``new_chat()`` resets the agent's chat_id, which can corrupt the
    running task's persistence.
    """

    def test_source_has_no_guard(self) -> None:
        """Structural: ``_new_chat`` doesn't check ``is_task_active``,
        ``is_merging``, or ``task_thread.is_alive()``.
        """
        src = inspect.getsource(VSCodeServer._new_chat)
        assert "is_task_active" not in src, (
            "BUG B5 confirmed: _new_chat doesn't check is_task_active"
        )
        assert "is_merging" not in src, (
            "BUG B5 confirmed: _new_chat doesn't check is_merging"
        )
        assert "is_alive" not in src, (
            "BUG B5 confirmed: _new_chat doesn't check thread liveness"
        )

    def test_behavioral_new_chat_resets_running_tab(self) -> None:
        """Behavioral: calling _new_chat on a tab with is_task_active=True
        proceeds without error, resetting the chat_id mid-flight.
        """
        server, events = _make_server()
        tab = server._get_tab("running-tab")
        tab.agent._chat_id = "chat-123"
        tab.is_task_active = True
        tab.task_thread = threading.Thread(target=lambda: None)

        old_chat_id = tab.agent.chat_id
        assert old_chat_id == "chat-123"

        # This should be refused but isn't
        server._new_chat("running-tab")

        # The chat_id was reset despite the task being active
        assert tab.agent.chat_id != old_chat_id, (
            "BUG B5 confirmed: new_chat reset chat_id while task was active"
        )


# ===================================================================
# R1 — Misleading noqa: F401 comment on used imports
# ===================================================================


class TestMisleadingNoqaComment(unittest.TestCase):
    """R1: ``server.py`` imports ``_cleanup_merge_data``, ``_git``, and
    ``_merge_data_dir`` with a ``noqa: F401 (re-export for tests)``
    comment, but all three are used directly in the same file.
    """

    def test_imports_are_used_in_server(self) -> None:
        """Structural: the imported symbols appear in function bodies
        within server.py, not just in ``__all__``.
        """
        import kiss.agents.vscode.server as srv_mod

        src = inspect.getsource(srv_mod)

        # _cleanup_merge_data is used in _close_tab
        close_tab_src = inspect.getsource(VSCodeServer._close_tab)
        assert "_cleanup_merge_data" in close_tab_src, (
            "_cleanup_merge_data is used in _close_tab"
        )

        # _git is used in _generate_commit_message
        gen_commit_src = inspect.getsource(VSCodeServer._generate_commit_message)
        assert "_git" in gen_commit_src, (
            "_git is used in _generate_commit_message"
        )

        # _merge_data_dir is used in _close_tab
        assert "_merge_data_dir" in close_tab_src, (
            "_merge_data_dir is used in _close_tab"
        )

        # Yet the import line says "re-export for tests"
        assert "noqa: F401 (re-export for tests)" in src, (
            "REDUNDANCY R1 confirmed: the noqa comment claims re-export "
            "but the symbols are used directly"
        )


# ===================================================================
# R2 — parse_task_tags in __all__ but unused in server.py
# ===================================================================


class TestParseTaskTagsRedundantExport(unittest.TestCase):
    """R2: ``parse_task_tags`` is in ``server.py.__all__`` and imported
    there, but never called in ``server.py``.  The only call-site is
    ``task_runner.py``, which imports it directly from ``tab_state``.
    """

    def test_parse_task_tags_in_all_but_unused(self) -> None:
        import kiss.agents.vscode.server as srv_mod

        # It's in __all__
        assert "parse_task_tags" in srv_mod.__all__

        # Not used in the server module's own source (excluding imports)
        full_src = inspect.getsource(srv_mod)
        # Find lines that reference parse_task_tags in actual code
        # (not imports, not __all__ entries, not comments)
        usage_lines = [
            line
            for line in full_src.splitlines()
            if "parse_task_tags" in line
            and "__all__" not in line
            and not line.strip().startswith("#")
            and not line.strip().startswith(("from ", "import "))
            # Exclude __all__ list entries like '    "parse_task_tags",'
            and not line.strip().strip(",").strip('"').strip("'") == "parse_task_tags"
        ]
        assert not usage_lines, (
            f"REDUNDANCY R2 confirmed: parse_task_tags appears in "
            f"server.py only in imports and __all__, not in any "
            f"function body. usage_lines={usage_lines}"
        )

        # task_runner.py imports it directly from tab_state
        tr_src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        assert "parse_task_tags" in tr_src, (
            "REDUNDANCY R2 confirmed: task_runner uses parse_task_tags "
            "from its own import, not via server.py"
        )


# ===================================================================
# I1 — Inconsistent tabId default across command handlers
# ===================================================================


class TestTabIdDefaultInconsistency(unittest.TestCase):
    """I1: ``_cmd_user_answer`` uses ``cmd.get("tabId")`` (``None``
    default) while all other handlers use ``cmd.get("tabId", "")``
    (empty string).
    """

    def test_user_answer_uses_none_default(self) -> None:
        src = inspect.getsource(_CommandsMixin._cmd_user_answer)
        # Uses cmd.get("tabId") without a default
        assert re.search(r'cmd\.get\("tabId"\)', src), (
            "INCONSISTENCY I1 confirmed: _cmd_user_answer uses "
            "cmd.get('tabId') with no default"
        )

    def test_other_handlers_use_empty_string_default(self) -> None:
        """All other cmd handlers that read tabId use a default of ''."""
        handlers_with_tabid = [
            _CommandsMixin._cmd_run,
            _CommandsMixin._cmd_stop,
            _CommandsMixin._cmd_select_model,
            _CommandsMixin._cmd_close_tab,
            _CommandsMixin._cmd_new_chat,
            _CommandsMixin._cmd_resume_session,
            _CommandsMixin._cmd_merge_action,
            _CommandsMixin._cmd_complete,
            _CommandsMixin._cmd_get_adjacent_task,
            _CommandsMixin._cmd_generate_commit_message,
            _CommandsMixin._cmd_worktree_action,
            _CommandsMixin._cmd_autocommit_action,
        ]
        for handler in handlers_with_tabid:
            src = inspect.getsource(handler)
            if 'cmd.get("tabId"' not in src:
                continue
            # Check for empty-string default
            has_empty_default = bool(
                re.search(r'cmd\.get\("tabId",\s*""\)', src)
            )
            has_none_default = bool(
                re.search(r'cmd\.get\("tabId"\)', src)
                and not re.search(r'cmd\.get\("tabId",', src)
            )
            assert has_empty_default and not has_none_default, (
                f"INCONSISTENCY I1: {handler.__name__} should use "
                f'cmd.get("tabId", ""), not cmd.get("tabId")'
            )


# ===================================================================
# I2 — Asymmetric initialization of _complete_seq counters
# ===================================================================


class TestCompleteSeqInitInconsistency(unittest.TestCase):
    """I2: ``_complete_seq`` starts at 0 and ``_complete_seq_latest``
    starts at -1.  The first increment produces seq=1, so they match,
    but the asymmetric init obscures the invariant.
    """

    def test_asymmetric_initialization(self) -> None:
        server, _ = _make_server()
        assert server._complete_seq == 0, "seq starts at 0"
        assert server._complete_seq_latest == -1, "seq_latest starts at -1"
        # These differ — the invariant "latest == seq" is violated at init
        assert server._complete_seq != server._complete_seq_latest, (
            "INCONSISTENCY I2 confirmed: _complete_seq (0) != "
            "_complete_seq_latest (-1) at initialization"
        )

    def test_first_increment_restores_consistency(self) -> None:
        """After the first _cmd_complete, both counters agree."""
        server, _ = _make_server()
        # Simulate _cmd_complete's counter logic
        with server._state_lock:
            server._complete_seq += 1
            seq = server._complete_seq
            server._complete_seq_latest = seq
        assert server._complete_seq == server._complete_seq_latest == 1


if __name__ == "__main__":
    unittest.main()
