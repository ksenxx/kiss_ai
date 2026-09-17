# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regressions: a token-0 duplicate ``closeTab`` vs. a reopen.

A ``closeTab`` for a tab the registry does not list (a duplicate close
— e.g. two clients closing the same tab, the second landing after the
first already removed the registry row) still runs the backend cleanup
tail.  That tail used to be UNCONDITIONAL
(``_drop_tab_state(tab_id, None)``): a concurrent ``resumeSession``
reopen republishing the tab in the window between the duplicate
close's registry check and its drop had its freshly cleared
``frontend_closed`` re-marked and its rebound chat view / viewer
subscription torn down — the registry (and every client) showed an
open tab whose backend state was retired, matching neither serial
order of the two commands.

The fix orders the duplicate close on the registry's publication
clock: a token-0 close reads :meth:`TabRegistry.clock` and the cleanup
tail's guards (``VSCodeServer._tab_reopened_since``) stand down when a
publication newer than the reading stamped the tab OR the tab is
simply PRESENT in the registry — presence after an absent-close is
always a later republication, which covers the gap between the
``close_tab`` call and the clock read.  ``_replay_session``
additionally publishes the reopen BEFORE clearing ``frontend_closed``,
so a drop that raced ahead of the publication has its stale flag mark
overwritten by the reopen's clear instead of the other way round.

Real :class:`VSCodeServer`, real :class:`TabRegistry`, real
:class:`JsonPrinter`; the racing interleavings are forced through
seams on the server/registry objects (a wrapper that runs the racing
step inside the window), never through mocks of the code under test —
the same technique as ``test_review3_server_fixes.py`` and
``test_local_uds_interest_races.py``.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server.test_review2_stale_run_undo import _Base


class _ReopenAfterAbsentClose:
    """Seam: run the reopen right after ``close_tab`` found *tab_id* absent.

    This is the interleaving "the reopen's republication lands after
    the duplicate close's registry step but before its clock read":
    the publication can stamp a generation at or below the close's
    observation, so only the registry-PRESENCE clause of
    ``_tab_reopened_since`` saves the reopened tab.
    """

    def __init__(self, real: Any, reopen: Any, tab_id: str) -> None:
        self._real = real
        self._reopen = reopen
        self._tab_id = tab_id
        self.fired = False

    def __call__(self, tab_id: str) -> int:
        result: int = self._real(tab_id)
        if tab_id == self._tab_id and result == 0 and not self.fired:
            self.fired = True
            self._reopen()
        return result


class _ReopenBeforeTheDrop:
    """Seam: run the reopen between the close's clock read and its drop.

    Wraps ``VSCodeServer._drop_tab_state`` — the reopen republishes
    the tab AFTER the duplicate close's clock observation, so the
    ``republished_since`` clause of ``_tab_reopened_since`` stands the
    drop down.
    """

    def __init__(self, real: Any, reopen: Any, tab_id: str) -> None:
        self._real = real
        self._reopen = reopen
        self._tab_id = tab_id
        self.fired = False

    def __call__(self, tab_id: str, **kwargs: Any) -> None:
        if tab_id == self._tab_id and not self.fired:
            self.fired = True
            self._reopen()
        self._real(tab_id, **kwargs)


class _CloseInsidePublication:
    """Seam: run the duplicate close just before the reopen publishes.

    Wraps ``VSCodeServer._registry_update_tab`` — the first publication
    for *tab_id* runs the racing duplicate close first, which is
    exactly the interleaving "the duplicate close runs entirely inside
    the reopen, before its registry publication lands".
    """

    def __init__(self, real: Any, close: Any, tab_id: str) -> None:
        self._real = real
        self._close = close
        self._tab_id = tab_id
        self.fired = False

    def __call__(self, tab_id: str, **kwargs: Any) -> int:
        if tab_id == self._tab_id and not self.fired:
            self.fired = True
            self._close()
        result: int = self._real(tab_id, **kwargs)
        return result


class TestToken0DuplicateClose(_Base):
    """The duplicate-close cleanup tail must respect a reopen."""

    def _register_running_state(
        self, tab_id: str, task_id: str, chat_id: str,
    ) -> AgentState:
        """Register a busy state exactly as ``_cmd_run`` would."""
        state = AgentState(
            task_id, tab_id=tab_id, chat_id=chat_id, is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)
        return state

    def _closed_running_tab(self) -> AgentState:
        """Open tab ``t`` on chat ``c1`` with a running task, close it.

        Returns the (still busy, deferred-teardown) state: the
        registry row is gone, ``frontend_closed`` is raised, the tab's
        chat view is dropped — the exact starting point of a duplicate
        close.
        """
        self.server._registry_update_tab("t", chat_id="c1", create=True)
        state = self._register_running_state("t", "task-conc2026-t0", "c1")
        with self.server._state_lock:
            self.server._tab_chat_views["t"] = "c1"
        self.server._close_tab("t")
        self.assertFalse(self.server.tab_registry.has_tab("t"))
        self.assertTrue(state.frontend_closed)
        with self.server._state_lock:
            self.assertIs(agent_state.find_by_tab("t"), state)
        return state

    def _assert_reopen_survived(self, state: AgentState) -> None:
        """The reopened tab keeps its row, cleared flag and chat view."""
        self.assertTrue(
            self.server.tab_registry.has_tab("t"),
            "the reopen's registry row must survive the stale duplicate",
        )
        with self.server._state_lock:
            self.assertIs(agent_state.find_by_tab("t"), state)
            self.assertFalse(
                state.frontend_closed,
                "the stale duplicate close re-marked the reopened state",
            )
            self.assertEqual(
                self.server._tab_chat_views.get("t"), "c1",
                "the stale duplicate close tore down the rebound chat view",
            )

    def test_duplicate_close_stands_down_after_a_reopen(self) -> None:
        """Interleaving 1: the reopen completes between the duplicate
        close's registry step and its clock read (the publication gap
        covered by the registry-presence clause)."""
        state = self._closed_running_tab()

        seam = _ReopenAfterAbsentClose(
            self.server.tab_registry.close_tab,
            lambda: self.server._replay_session("c1", "t"),
            "t",
        )
        self.server.tab_registry.close_tab = seam  # type: ignore[method-assign]
        try:
            self.server._close_tab("t")
        finally:
            self.server.tab_registry.close_tab = seam._real  # type: ignore[method-assign]

        self.assertTrue(seam.fired, "the duplicate close never hit the seam")
        self._assert_reopen_survived(state)

    def test_duplicate_close_stands_down_after_a_late_reopen(self) -> None:
        """Interleaving 1b: the reopen completes between the duplicate
        close's clock read and its cleanup tail (the newer-publication
        clause)."""
        state = self._closed_running_tab()

        seam = _ReopenBeforeTheDrop(
            self.server._drop_tab_state,
            lambda: self.server._replay_session("c1", "t"),
            "t",
        )
        self.server._drop_tab_state = seam  # type: ignore[assignment, method-assign]
        try:
            self.server._close_tab("t")
        finally:
            self.server._drop_tab_state = seam._real  # type: ignore[method-assign]

        self.assertTrue(seam.fired, "the duplicate close never hit the seam")
        self._assert_reopen_survived(state)

    def test_duplicate_close_inside_the_reopen_loses_to_the_flag_clear(
        self,
    ) -> None:
        """Interleaving 2: the duplicate close runs entirely inside the
        reopen, before its registry publication.  The reopen's
        publication and flag-clear land last, so the tab must not end
        up published-yet-closed."""
        state = self._closed_running_tab()

        seam = _CloseInsidePublication(
            self.server._registry_update_tab,
            lambda: self.server._close_tab("t"),
            "t",
        )
        self.server._registry_update_tab = seam  # type: ignore[method-assign]
        try:
            self.server._replay_session("c1", "t")
        finally:
            self.server._registry_update_tab = seam._real  # type: ignore[method-assign]

        self.assertTrue(seam.fired, "the reopen never published the tab")
        self.assertTrue(self.server.tab_registry.has_tab("t"))
        with self.server._state_lock:
            self.assertIs(agent_state.find_by_tab("t"), state)
            self.assertFalse(
                state.frontend_closed,
                "a published tab must never point at a closed state",
            )
            self.assertEqual(self.server._tab_chat_views.get("t"), "c1")

    def test_serial_duplicate_close_still_retires_everything(self) -> None:
        """Control: with no reopen, the duplicate close stays a no-op
        on the registry and the state stays retired (deferred while
        busy, dropped once idle)."""
        state = self._closed_running_tab()
        self.server._close_tab("t")
        self.assertFalse(self.server.tab_registry.has_tab("t"))
        self.assertTrue(state.frontend_closed)
        # Task ends; the deferred disposal retires the state.
        with self.server._state_lock:
            state.is_task_active = False
        self.server._dispose_if_closed("t")
        with self.server._state_lock:
            self.assertIsNone(agent_state.find_by_tab("t"))
            self.assertNotIn("t", self.server._tab_chat_views)

    def test_subagent_tab_close_is_still_unconditional(self) -> None:
        """Control: a sub-agent viewer tab is never in the registry
        (generation 0), so its close must keep dropping the state even
        though the close now carries a clock observation."""
        state = AgentState(
            "task-conc2026-sub", tab_id="p__sub_x", chat_id="c-sub",
        )
        with self.server._state_lock:
            agent_state.register(state)
        self.server._close_tab("p__sub_x")
        with self.server._state_lock:
            self.assertIsNone(agent_state.find_by_tab("p__sub_x"))
        self.assertTrue(state.frontend_closed)

    def test_close_removing_replay_publication_wins_the_commit(self) -> None:
        """Review finding 1: a close that removes the replay's OWN
        publication owns the final ("closed") state — the replay's
        later commit must stand down instead of clearing
        ``frontend_closed`` and rebinding the chat view against a
        registry that says closed.

        Schedule (two real threads, gated at the registry seams):
        close marks the flag and pauses before its registry removal;
        the replay publishes the tab and pauses before its commit; the
        close removes that publication and finishes its cleanup tail;
        the replay's commit then runs last.
        """
        state = self._closed_running_tab()
        close_at_registry = threading.Event()
        replay_published = threading.Event()
        close_finished = threading.Event()
        errors: list[BaseException] = []
        real_close = self.server.tab_registry.close_tab
        real_update = self.server._registry_update_tab

        def gated_close(tab_id: str) -> int:
            if tab_id == "t" and threading.current_thread().name == "f1-close":
                close_at_registry.set()
                assert replay_published.wait(10)
            return real_close(tab_id)

        def gated_update(tab_id: str, **kwargs: Any) -> int:
            result = real_update(tab_id, **kwargs)
            if tab_id == "t" and threading.current_thread().name == "f1-replay":
                replay_published.set()
                assert close_finished.wait(10)
            return result

        self.server.tab_registry.close_tab = gated_close  # type: ignore[method-assign]
        self.server._registry_update_tab = gated_update  # type: ignore[method-assign]

        def close() -> None:
            try:
                self.server._close_tab("t")
            except BaseException as exc:  # pragma: no cover — fail loudly
                errors.append(exc)
            finally:
                close_finished.set()

        def replay() -> None:
            try:
                self.server._replay_session("c1", "t")
            except BaseException as exc:  # pragma: no cover — fail loudly
                errors.append(exc)

        close_thread = threading.Thread(target=close, name="f1-close")
        replay_thread = threading.Thread(target=replay, name="f1-replay")
        try:
            close_thread.start()
            self.assertTrue(close_at_registry.wait(10))
            replay_thread.start()
            close_thread.join(15)
            replay_thread.join(15)
            self.assertFalse(close_thread.is_alive())
            self.assertFalse(replay_thread.is_alive())
            self.assertEqual(errors, [])
            # The close owns the outcome (serial order replay → close):
            # registry closed AND backend closed.  The busy task defers
            # the teardown, exactly like a serial close of a busy tab —
            # so the chat-view entry may legitimately linger until the
            # deferred disposal below.
            self.assertFalse(self.server.tab_registry.has_tab("t"))
            with self.server._state_lock:
                self.assertIs(agent_state.find_by_tab("t"), state)
                self.assertTrue(
                    state.frontend_closed,
                    "the stale replay commit reopened a closed tab's state",
                )
            # Task ends; the deferred disposal must retire everything —
            # a cleared flag (the pre-fix broken state) would skip it.
            with self.server._state_lock:
                state.is_task_active = False
            self.server._dispose_if_closed("t")
            with self.server._state_lock:
                self.assertIsNone(agent_state.find_by_tab("t"))
                self.assertNotIn("t", self.server._tab_chat_views)
        finally:
            self.server.tab_registry.close_tab = real_close  # type: ignore[method-assign]
            self.server._registry_update_tab = real_update  # type: ignore[method-assign]

    def test_replay_commit_reinstalls_the_viewer_subscription(self) -> None:
        """Review finding 2: a token-0 duplicate close whose clock
        observation predates the replay's viewer attach can run its
        cleanup tail (removing the fresh printer subscription) between
        the attach and the publication.  The replay's commit must
        reinstall the subscription while its publication owns the row,
        so the reopened tab keeps receiving the running task's events.
        """
        source = self._register_running_state(
            "source-tab", "task-conc2026-live", "c1",
        )
        self.server.printer.subscribe_tab(source.task_id, source.tab_id)
        close_at_drop = threading.Event()
        allow_close_drop = threading.Event()
        close_done = threading.Event()
        errors: list[BaseException] = []
        real_drop = self.server._drop_tab_state
        real_update = self.server._registry_update_tab

        def gated_drop(tab_id: str, **kwargs: Any) -> None:
            if tab_id == "viewer" and (
                threading.current_thread().name == "f2-close"
            ):
                close_at_drop.set()
                assert allow_close_drop.wait(10)
            real_drop(tab_id, **kwargs)

        def gated_update(tab_id: str, **kwargs: Any) -> int:
            if tab_id == "viewer" and (
                threading.current_thread().name == "f2-replay"
            ):
                # The replay reached its publication: release the
                # stale close's cleanup tail and let it finish FIRST.
                allow_close_drop.set()
                assert close_done.wait(10)
            return real_update(tab_id, **kwargs)

        self.server._drop_tab_state = gated_drop  # type: ignore[assignment, method-assign]
        self.server._registry_update_tab = gated_update  # type: ignore[method-assign]

        def close() -> None:
            try:
                self.server._close_tab("viewer")
            except BaseException as exc:  # pragma: no cover — fail loudly
                errors.append(exc)
            finally:
                close_done.set()

        def replay() -> None:
            try:
                self.server._replay_session("c1", "viewer")
            except BaseException as exc:  # pragma: no cover — fail loudly
                errors.append(exc)

        close_thread = threading.Thread(target=close, name="f2-close")
        replay_thread = threading.Thread(target=replay, name="f2-replay")
        try:
            close_thread.start()
            self.assertTrue(close_at_drop.wait(10))
            replay_thread.start()
            close_thread.join(15)
            replay_thread.join(15)
            self.assertFalse(close_thread.is_alive())
            self.assertFalse(replay_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(self.server.tab_registry.has_tab("viewer"))
            with self.server._state_lock:
                self.assertEqual(
                    self.server._tab_chat_views.get("viewer"), "c1",
                )
            targets = self.server.printer._fanout_targets(source.task_id)
            self.assertIn(
                "viewer", targets,
                "the reopened viewer lost the running task's fan-out",
            )
            self.assertIn(source.tab_id, targets)
        finally:
            self.server._drop_tab_state = real_drop  # type: ignore[method-assign]
            self.server._registry_update_tab = real_update  # type: ignore[method-assign]

    def test_clock_contract(self) -> None:
        """API control: the clock reading orders a token-0 close
        against later republications, and ``_tab_reopened_since``
        treats presence as a reopen."""
        reg = self.server.tab_registry
        self.server._registry_update_tab("t", chat_id="c1", create=True)
        removal = reg.close_tab("t")
        self.assertGreater(removal, 0)
        self.assertEqual(reg.close_tab("t"), 0)
        observed = reg.clock()
        self.assertGreaterEqual(observed, removal)
        self.assertFalse(reg.republished_since("t", observed))
        self.assertFalse(self.server._tab_reopened_since("t", observed))
        self.server._registry_update_tab("t", chat_id="c1", create=True)
        self.assertTrue(reg.republished_since("t", observed))
        self.assertTrue(self.server._tab_reopened_since("t", observed))
        # Presence alone (a publication stamped at or below the
        # observation is indistinguishable from this at the guard) is
        # already a reopen.
        self.assertTrue(self.server._tab_reopened_since("t", reg.clock()))
