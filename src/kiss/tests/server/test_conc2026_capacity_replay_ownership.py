# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a replay at registry capacity cannot overwrite a completed close.

gpt-5.6-sol round-2 review, finding 1: when the tab registry is FULL,
``_replay_session``'s ``_registry_update_tab(..., create=True)``
publishes nothing (generation ``0``, no row), and the old
``_commit_replay_publication`` treated that as an UNCONDITIONAL legacy
commit — clearing ``frontend_closed`` and binding ``_tab_chat_views``
with no ownership check at all.  A token-0 ``_close_tab`` completing
between the failed publication and the commit had its cleanup
overwritten: the backend chat view was recreated for a tab absent from
the registry and no longer subscribed to its task's fan-out.

The fix gives non-registry tabs an explicit ownership token:
``_replay_session`` stamps a ROWLESS publication generation
(:meth:`TabRegistry.stamp_unregistered`) before attempting the row
publication, the capacity-path commit proceeds only while
``generation(tab_id)`` still equals that stamp, and every completed
close teardown retires the stamp
(:meth:`TabRegistry.retire_unregistered`) in the same ``_state_lock``
section as the rest of its cleanup.  A close after the stamp therefore
always wins (the tab stays fully closed), while an undisturbed
capacity replay keeps the legacy behaviour — the tab works as a
non-registry tab, INCLUDING the viewer fan-out subscription the old
zero-generation path used to skip.

Real :class:`VSCodeServer`, real :class:`TabRegistry`, real printer
subscriptions; the deterministic schedule parks the close at the same
seam the review reproduction used (a wrapper delegating to the real
``_registry_update_tab``).  No mocks.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.tab_registry import _MAX_TABS
from kiss.tests.server.test_review2_stale_run_undo import _Base


class CapacityReplayOwnershipTest(_Base):
    """Ownership of the ``publication == 0`` replay commit."""

    def _fill_registry(self) -> None:
        """Publish ``_MAX_TABS`` tabs so the next create is refused."""
        for i in range(_MAX_TABS):
            token = self.server._registry_update_tab(f"full-{i}", create=True)
            self.assertGreater(token, 0)
        self.assertFalse(self.server.tab_registry.has_tab("viewer"))

    def _register_live_source(self) -> AgentState:
        """Register a live task on chat ``capacity-chat``, tab ``source``."""
        source = AgentState(
            "capacity-task",
            tab_id="source",
            chat_id="capacity-chat",
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(source)
        self.server.printer.subscribe_tab(source.task_id, source.tab_id)
        return source

    def test_close_after_failed_publication_keeps_the_tab_closed(self) -> None:
        """The review's schedule: capacity replay vs completed close.

        ``_close_tab("viewer")`` completes right after the replay's
        registry publication returned ``0``; the replay's capacity
        commit must stand down — no chat-view recreation, no fan-out
        subscription, no registry row.
        """
        self._fill_registry()
        source = self._register_live_source()

        real_update = self.server._registry_update_tab
        close_finished = False

        def update_then_close(tab_id: str, **kwargs: Any) -> int:
            nonlocal close_finished
            publication = real_update(tab_id, **kwargs)
            if tab_id == "viewer":
                self.assertEqual(publication, 0)
                self.server._close_tab("viewer")
                close_finished = True
            return publication

        self.server._registry_update_tab = update_then_close  # type: ignore[method-assign]
        try:
            self.server._replay_session("capacity-chat", "viewer")
        finally:
            self.server._registry_update_tab = real_update  # type: ignore[method-assign]

        self.assertTrue(close_finished)
        self.assertFalse(self.server.tab_registry.has_tab("viewer"))
        self.assertNotIn(
            "viewer", self.server._tab_chat_views,
            "the capacity replay commit overwrote a completed close "
            "and recreated the backend chat view",
        )
        self.assertNotIn(
            "viewer", self.server.printer._fanout_targets(source.task_id),
            "a closed viewer tab was left (or re-)subscribed to the "
            "running task's fan-out",
        )
        # The close retired the rowless publication stamp, so a stale
        # commit can never qualify later either.
        self.assertEqual(self.server.tab_registry.generation("viewer"), 0)

    def test_undisturbed_capacity_replay_still_binds_and_subscribes(
        self,
    ) -> None:
        """No racing close: the legacy capacity behaviour is preserved.

        The tab cannot get a registry row, but its backend chat view
        is bound AND (fixing the skip in the old zero-generation path)
        the viewer is subscribed to the live task's fan-out.
        """
        self._fill_registry()
        source = self._register_live_source()
        self.server._replay_session("capacity-chat", "viewer")
        self.assertFalse(self.server.tab_registry.has_tab("viewer"))
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
            "an undisturbed capacity replay no longer binds the tab",
        )
        self.assertIn(
            "viewer", self.server.printer._fanout_targets(source.task_id),
            "an undisturbed capacity replay did not install the "
            "viewer's fan-out subscription",
        )

    def test_close_completed_before_the_replay_loses(self) -> None:
        """A close that finished BEFORE the replay began is superseded.

        Serializes as close-then-replay: the replay's stamp postdates
        the close, nobody retires it, and the commit (re)installs the
        binding and the subscription.
        """
        self._fill_registry()
        source = self._register_live_source()
        self.server._close_tab("viewer")
        self.server._replay_session("capacity-chat", "viewer")
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
        )
        self.assertIn(
            "viewer", self.server.printer._fanout_targets(source.task_id),
        )

    def test_busy_close_after_failed_publication_keeps_the_tab_closed(
        self,
    ) -> None:
        """Round-3 finding 1: the BUSY close path retires the stamp too.

        The closing tab itself runs a live task, so ``_close_tab``
        takes the busy path of ``_drop_tab_state`` — it marks
        ``frontend_closed`` and returns WITHOUT reaching
        ``_teardown_tab_resources`` (the state must survive until the
        task ends).  Pre-fix, only that teardown tail retired the
        rowless stamp, so the replay's pending capacity commit still
        qualified: it cleared ``frontend_closed`` and rebound the chat
        view, and the reopened flag then suppressed the deferred
        ``_dispose_if_closed`` forever.  Post-fix the busy path
        retires the stamp at the close's own ``_state_lock``
        linearization point, so the commit stands down and the
        deferred disposal still runs once the task ends.
        """
        self._fill_registry()
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)

        real_update = self.server._registry_update_tab
        close_returned = False

        def update_then_close(tab_id: str, **kwargs: Any) -> int:
            nonlocal close_returned
            publication = real_update(tab_id, **kwargs)
            if tab_id == "viewer":
                self.assertEqual(publication, 0)
                self.server._close_tab("viewer")
                close_returned = True
                self.assertTrue(state.frontend_closed)
                # The busy close retired the rowless stamp at its
                # linearization point, before returning.
                self.assertEqual(
                    self.server.tab_registry.generation("viewer"), 0,
                    "busy close did not retire the rowless stamp",
                )
            return publication

        self.server._registry_update_tab = update_then_close  # type: ignore[method-assign]
        try:
            self.server._replay_session("capacity-chat", "viewer")
        finally:
            self.server._registry_update_tab = real_update  # type: ignore[method-assign]

        self.assertTrue(close_returned)
        # The replay's capacity commit stood down: the tab stays
        # closed, with no chat-view recreation and no leaked stamp.
        self.assertTrue(
            state.frontend_closed,
            "the capacity replay commit overwrote a completed busy close",
        )
        self.assertNotIn("viewer", self.server._tab_chat_views)
        self.assertFalse(self.server.tab_registry.has_tab("viewer"))
        self.assertEqual(self.server.tab_registry.generation("viewer"), 0)
        # The busy state itself survives the close (the task is still
        # running)...
        self.assertIs(agent_state.find_by_tab("viewer"), state)
        # ...and the deferred disposal still runs when the task ends —
        # pre-fix the replay's cleared flag suppressed it forever.
        with self.server._state_lock:
            state.is_task_active = False
        self.server._dispose_if_closed("viewer")
        self.assertIsNone(agent_state.find_by_tab("viewer"))

    def test_replay_stamped_after_a_busy_close_supersedes_it(self) -> None:
        """A replay whose stamp postdates the busy close still wins.

        The retire-at-close fix must not make a busy close permanent:
        serialized as close-then-replay, the replay's fresh stamp is
        never retired, so its capacity commit reopens the backend view
        (``frontend_closed`` cleared, chat view bound) exactly like the
        immediate-close variant ``test_close_completed_before_the_replay_loses``.
        """
        self._fill_registry()
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)
        self.server._close_tab("viewer")
        self.assertTrue(state.frontend_closed)
        self.server._replay_session("capacity-chat", "viewer")
        self.assertFalse(
            state.frontend_closed,
            "a replay stamped after the busy close must supersede it",
        )
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
        )
        # Cleanup: end the task and dispose nothing (tab reopened).
        with self.server._state_lock:
            state.is_task_active = False
        self.server._dispose_if_closed("viewer")
        self.assertIs(agent_state.find_by_tab("viewer"), state)
        with self.server._state_lock:
            agent_state.unregister(state.task_id, state)

    def test_deferred_disposal_stands_down_for_a_replay_after_its_claim(
        self,
    ) -> None:
        """Round-4 finding 1: a claimed deferred disposal vs a reopen.

        The task of a closed-while-busy tab ends;
        ``_dispose_if_closed`` claims the state and stalls before its
        teardown (the reviewer's schedule).  A capacity replay then
        stamps a FRESH rowless generation — strictly after the claim's
        clock observation — and commits a reopen.  Pre-fix the resumed
        teardown had no ownership token: it unregistered the reopened
        state and deleted the fresh stamp and chat view.  Post-fix the
        teardown stands down at its reopen guards and the replay wins.
        """
        self._fill_registry()
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)

        self.server._close_tab("viewer")
        self.assertTrue(state.frontend_closed)
        with self.server._state_lock:
            state.is_task_active = False

        claimed = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []
        real_teardown = self.server._teardown_tab_resources

        def gated_teardown(
            tab_id: str,
            state: AgentState | None,
            removal_token: int | None = None,
        ) -> None:
            if tab_id == "viewer":
                claimed.set()
                self.assertTrue(release.wait(10))
            real_teardown(tab_id, state, removal_token)

        self.server._teardown_tab_resources = gated_teardown  # type: ignore[method-assign]

        def dispose() -> None:
            try:
                self.server._dispose_if_closed("viewer")
            except BaseException as exc:  # pragma: no cover — fails test
                errors.append(exc)

        thread = threading.Thread(target=dispose)
        try:
            thread.start()
            self.assertTrue(claimed.wait(10))
            # The replay starts strictly after the explicit close AND
            # after the disposal's claim: it must win.
            self.server._replay_session("capacity-chat", "viewer")
            self.assertFalse(state.frontend_closed)
            self.assertEqual(
                self.server._tab_chat_views.get("viewer"), "capacity-chat",
            )
            self.assertGreater(
                self.server.tab_registry.generation("viewer"), 0,
            )

            release.set()
            thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

            # The stale teardown stood down: the reopen survives.
            self.assertIs(
                agent_state.find_by_tab("viewer"), state,
                "the claimed deferred disposal unregistered a state "
                "reopened after its claim",
            )
            self.assertFalse(state.frontend_closed)
            self.assertFalse(state.is_merging)
            self.assertEqual(
                self.server._tab_chat_views.get("viewer"), "capacity-chat",
                "the stale teardown deleted the fresh replay's chat view",
            )
            self.assertGreater(
                self.server.tab_registry.generation("viewer"), 0,
                "the stale teardown retired the fresh replay's stamp",
            )
        finally:
            release.set()
            thread.join(15)
            self.server._teardown_tab_resources = real_teardown  # type: ignore[method-assign]
            with self.server._state_lock:
                if agent_state.find_by_tab("viewer") is state:
                    agent_state.unregister(state.task_id, state)

    def test_deferred_disposal_retires_a_stamp_from_before_its_claim(
        self,
    ) -> None:
        """A replay stamped BEFORE the deferred claim loses to it.

        Mirrors the immediate busy close's discipline: the claim's
        linearization point retires a pre-existing rowless stamp
        atomically with its clock observation
        (:meth:`TabRegistry.retire_unregistered_and_observe`), so the
        replay's late capacity commit stands down and the tab stays
        fully closed.
        """
        self._fill_registry()
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)
        self.server._close_tab("viewer")
        with self.server._state_lock:
            state.is_task_active = False

        # The replay's first step lands before the disposal claims.
        fallback = self.server.tab_registry.stamp_unregistered("viewer")
        self.assertGreater(fallback, 0)

        self.server._dispose_if_closed("viewer")
        self.assertIsNone(agent_state.find_by_tab("viewer"))
        self.assertEqual(
            self.server.tab_registry.generation("viewer"), 0,
            "the deferred claim did not retire the pre-claim stamp",
        )

        # The replay's commit resumes after the disposal: it must
        # stand down (its stamp is gone) and leave the tab closed.
        self.server._commit_replay_publication(
            "viewer", "capacity-chat", 0, None,
            fallback_publication=fallback,
        )
        self.assertNotIn("viewer", self.server._tab_chat_views)
        self.assertEqual(self.server.tab_registry.generation("viewer"), 0)

    def test_deferred_disposal_stands_down_when_the_tab_regained_a_row(
        self,
    ) -> None:
        """A registry ROW published after the close parks the disposal.

        A row for a ``frontend_closed`` tab can only come from a later
        republication (the close removed the previous row), so the
        deferred disposal must not claim or tear anything down; the
        reopen's commit then clears the flag.
        """
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)
        self.server._close_tab("viewer")
        with self.server._state_lock:
            state.is_task_active = False

        # A (non-capacity) reopen republishes the row, but its commit
        # has not run yet when the disposal fires.
        row = self.server._registry_update_tab(
            "viewer", chat_id="capacity-chat", create=True,
        )
        self.assertGreater(row, 0)

        self.server._dispose_if_closed("viewer")
        self.assertIs(
            agent_state.find_by_tab("viewer"), state,
            "the deferred disposal tore down a tab that regained a row",
        )
        self.assertFalse(state.is_merging)
        self.assertTrue(self.server.tab_registry.has_tab("viewer"))

        self.server._commit_replay_publication(
            "viewer", "capacity-chat", row, None,
        )
        self.assertFalse(state.frontend_closed)
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
        )
        with self.server._state_lock:
            agent_state.unregister(state.task_id, state)

    def test_commit_without_any_publication_stands_down(self) -> None:
        """``publication == 0`` with no rowless stamp commits nothing.

        Reachable when the tab gained a registry row between the stamp
        attempt (which then returns ``0``) and an ``update_tab`` that
        still failed; the newest publisher owns the tab, so the stale
        commit must not touch the backend state.
        """
        self.server._commit_replay_publication(
            "orphan-tab", "orphan-chat", 0, None,
        )
        self.assertNotIn("orphan-tab", self.server._tab_chat_views)


class DeferredTeardownLinearizationTest(_Base):
    """Round-6 finding 1: teardown vs rowless stamps is linearizable.

    The deferred disposal's claim
    (:meth:`TabRegistry.retire_unregistered_and_observe`) was already
    atomic, but the teardown TAIL was not: it re-checked ownership
    through two separate registry reads and then mutated state across
    two ``_state_lock`` sections, while ``stamp_unregistered`` takes
    only the registry lock.  A rowless capacity replay stamped after a
    complete (false) reopen check but before the following
    unregister / stamp retirement was erased by the stale teardown
    (gpt-5.6-sol round-6 review, finding 1).

    The fix makes :meth:`TabRegistry.finalize_removal` the teardown's
    single linearization point — ownership re-check and rowless stamp
    retirement under ONE registry lock acquisition, at the head of ONE
    ``_state_lock`` section holding every destructive step.  A stamp
    therefore lands strictly before the decision (the whole teardown
    stands down; the replay's commit wins with the state intact) or
    strictly after it (the stamp survives untouched and the commit
    replays into the fully torn-down tab — the serial "closed, then
    reopened" order).  These tests park a real deferred disposal at
    both sides of that point with a delegating wrapper (the reviewer's
    reproduction seam); no mocks.
    """

    def _fill_registry(self) -> None:
        """Publish ``_MAX_TABS`` tabs so the next create is refused."""
        for i in range(_MAX_TABS):
            token = self.server._registry_update_tab(f"full-{i}", create=True)
            self.assertGreater(token, 0)
        self.assertFalse(self.server.tab_registry.has_tab("viewer"))

    def _closed_idle_viewer_state(self) -> AgentState:
        """Register tab ``viewer``, close it busy, then let it go idle."""
        state = AgentState(
            "viewer-task",
            tab_id="viewer",
            chat_id="capacity-chat",
            server_owned=True,
            is_task_active=True,
        )
        with self.server._state_lock:
            agent_state.register(state)
        self.server._close_tab("viewer")
        with self.server._state_lock:
            state.is_task_active = False
        return state

    def _dispose_gated_at_finalize(
        self, *, park_before: bool,
    ) -> tuple[threading.Thread, threading.Event, threading.Event,
               list[BaseException]]:
        """Run ``_dispose_if_closed('viewer')`` parked at the decision.

        Wraps the registry's ``finalize_removal`` so the disposal
        thread parks immediately before (*park_before* true) or
        immediately after (false) the teardown's atomic ownership
        decision, then waits for the test to release it.

        Returns:
            ``(thread, parked, release, errors)`` — the started
            disposal thread, the event set when it reached the gate,
            the event releasing it, and its collected exceptions.
        """
        registry = self.server.tab_registry
        real_finalize = registry.finalize_removal
        parked = threading.Event()
        release = threading.Event()
        latched = False

        def gated_finalize(tab_id: str, token: int) -> bool:
            nonlocal latched
            if tab_id != "viewer" or latched:
                return real_finalize(tab_id, token)
            latched = True
            if park_before:
                parked.set()
                assert release.wait(15)
                return real_finalize(tab_id, token)
            result = real_finalize(tab_id, token)
            parked.set()
            assert release.wait(15)
            return result

        registry.finalize_removal = gated_finalize  # type: ignore[method-assign]
        self.addCleanup(
            setattr, registry, "finalize_removal", real_finalize,
        )
        errors: list[BaseException] = []

        def dispose() -> None:
            try:
                self.server._dispose_if_closed("viewer")
            except BaseException as exc:  # pragma: no cover — test guard
                errors.append(exc)

        thread = threading.Thread(target=dispose)
        thread.start()
        return thread, parked, release, errors

    def test_stamp_before_the_decision_parks_the_whole_teardown(
        self,
    ) -> None:
        """A replay stamped after the claim, before the decision, wins.

        The stamp lands between the deferred claim and the teardown's
        atomic ownership decision.  The decision must observe the
        newer generation and stand the ENTIRE teardown down: the state
        stays registered, the stamp survives, and the replay's pending
        capacity commit reopens the tab.
        """
        self._fill_registry()
        state = self._closed_idle_viewer_state()
        thread, parked, release, errors = self._dispose_gated_at_finalize(
            park_before=True,
        )
        try:
            self.assertTrue(parked.wait(15))
            fallback = self.server.tab_registry.stamp_unregistered("viewer")
            self.assertGreater(fallback, 0)
        finally:
            release.set()
            thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

        self.assertIs(
            agent_state.find_by_tab("viewer"), state,
            "the stale deferred teardown unregistered a state whose "
            "tab was reopened before its ownership decision",
        )
        self.assertFalse(state.is_merging)
        self.assertEqual(
            self.server.tab_registry.generation("viewer"), fallback,
            "the stale deferred teardown erased the fresh replay stamp",
        )
        self.server._commit_replay_publication(
            "viewer", "capacity-chat", 0, None,
            fallback_publication=fallback,
        )
        self.assertFalse(state.frontend_closed)
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
        )
        with self.server._state_lock:
            agent_state.unregister(state.task_id, state)

    def test_stamp_after_the_decision_survives_the_teardown(self) -> None:
        """The round-6 headline schedule: stamp right after the check.

        The stamp lands immediately after the teardown's complete
        (owning) ownership decision, before the state unregister and
        the printer/chat-view cleanup — the seam where the old split
        guard erased it.  The stamp must SURVIVE: the teardown
        completes (the serial "closed" state), and the replay's commit
        then reopens the tab with coherent view ownership, exactly as
        a replay into a long-closed tab would.
        """
        self._fill_registry()
        self._closed_idle_viewer_state()
        thread, parked, release, errors = self._dispose_gated_at_finalize(
            park_before=False,
        )
        try:
            self.assertTrue(parked.wait(15))
            fallback = self.server.tab_registry.stamp_unregistered("viewer")
            self.assertGreater(fallback, 0)
        finally:
            release.set()
            thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

        self.assertIsNone(agent_state.find_by_tab("viewer"))
        self.assertEqual(
            self.server.tab_registry.generation("viewer"), fallback,
            "stale deferred teardown retired a rowless replay stamped "
            "after its complete ownership decision",
        )
        self.server._commit_replay_publication(
            "viewer", "capacity-chat", 0, None,
            fallback_publication=fallback,
        )
        self.assertEqual(
            self.server._tab_chat_views.get("viewer"), "capacity-chat",
            "the surviving replay stamp did not qualify its commit",
        )

    def test_reopened_since_and_finalize_removal_semantics(self) -> None:
        """Functional contract of the two new atomic registry steps."""
        registry = self.server.tab_registry

        # reopened_since: newer generation OR row presence, one call.
        self.assertFalse(registry.reopened_since("ghost", 0))
        stamp = registry.stamp_unregistered("ghost")
        self.assertGreater(stamp, 0)
        self.assertTrue(registry.reopened_since("ghost", stamp - 1))
        self.assertFalse(registry.reopened_since("ghost", stamp))
        row = self.server._registry_update_tab("rowed", create=True)
        self.assertGreater(row, 0)
        self.assertTrue(
            registry.reopened_since("rowed", row),
            "row presence must count as reopened even at its own token",
        )

        # finalize_removal: owns and retires only when nothing is newer.
        self.assertFalse(
            registry.finalize_removal("ghost", stamp - 1),
            "a newer stamp must park the teardown",
        )
        self.assertEqual(registry.generation("ghost"), stamp)
        self.assertTrue(registry.finalize_removal("ghost", stamp))
        self.assertEqual(registry.generation("ghost"), 0)
        self.assertFalse(
            registry.finalize_removal("rowed", row + 1),
            "a registry row must park the teardown",
        )
        self.assertTrue(registry.has_tab("rowed"))
        self.assertEqual(registry.generation("rowed"), row)
        # Blank ids never own anything but must not raise.
        self.assertTrue(registry.finalize_removal("", 0))

    def test_rowless_stamp_and_retire_are_row_safe(self) -> None:
        """The rowless token never disturbs a registered tab's token.

        ``stamp_unregistered`` refuses to overstamp a row's publication
        (that would invalidate its owner's pending generation-qualified
        undo), and ``retire_unregistered`` leaves a row's token alone;
        both are no-ops for a blank id.
        """
        registry = self.server.tab_registry
        self.assertEqual(registry.stamp_unregistered(""), 0)
        registry.retire_unregistered("")  # must not raise

        row_token = self.server._registry_update_tab("rowed", create=True)
        self.assertGreater(row_token, 0)
        self.assertEqual(registry.stamp_unregistered("rowed"), 0)
        registry.retire_unregistered("rowed")
        self.assertEqual(registry.generation("rowed"), row_token)

        stamp = registry.stamp_unregistered("rowless")
        self.assertGreater(stamp, 0)
        self.assertEqual(registry.generation("rowless"), stamp)
        registry.retire_unregistered("rowless")
        self.assertEqual(registry.generation("rowless"), 0)

        # The atomic retire-and-observe used by the deferred disposal
        # obeys the same row-safety rules: a blank id retires nothing,
        # a row's token survives, and a rowless stamp is retired in the
        # same locked step that draws the clock observation.
        blank_token = registry.retire_unregistered_and_observe("")
        self.assertGreaterEqual(blank_token, row_token)
        registry.retire_unregistered_and_observe("rowed")
        self.assertEqual(registry.generation("rowed"), row_token)
        stamp2 = registry.stamp_unregistered("rowless")
        self.assertGreater(stamp2, 0)
        observed = registry.retire_unregistered_and_observe("rowless")
        self.assertEqual(registry.generation("rowless"), 0)
        self.assertGreaterEqual(observed, stamp2)
        later = registry.stamp_unregistered("rowless")
        self.assertGreater(
            later, observed,
            "a publication after the observation must outrank it",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
