# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Round-3 regressions for the gpt-5.6-sol review findings in
``tmp/review3-server.md``.

1. **Atomic generation capture** — ``_registry_update_tab`` used to
   re-read ``TabRegistry.generation`` AFTER releasing the registry
   lock (and after a blocking ``tabs_state`` broadcast), so it could
   return a LATER publisher's token; ``_cmd_run``'s conditional undo
   then deleted that later publication.  ``update_tab`` now returns
   its own stamp from the locked update.

2. **Post-task commit claim leaks** — the claim was published under
   ``_state_lock`` but recorded for release only afterwards
   (``held_claim = repo``); an injected ``KeyboardInterrupt`` in that
   gap (or one skipping the ``finally``) stranded the claim until
   daemon restart.  Claims are now appended to a caller-armed holder
   BEFORE publication, releases are identity-conditional, and a claim
   whose owner thread died heals on the next check.

3. **Removal-generation wiring** — explicit-close and
   chat-displacement cleanup tails (``_drop_tab_state`` /
   ``_teardown_tab_resources``) were identity-blind: a reopen landing
   between the registry removal and the tail's state drop had its NEW
   backend state and chat mapping destroyed while its registry row
   survived.  Removals now carry a clock token and the tails stand
   down when ``TabRegistry.republished_since`` reports a newer
   publication.

Real :class:`VSCodeServer`, real :class:`TabRegistry`, real threads;
schedules are driven deterministically (a wrapper parking inside the
real ``tabs_state`` broadcast; ``sys.settrace`` raising at an exact
publication line — the same bytecode-boundary class the production
``PyThreadState_SetAsyncExc`` stop can hit).
"""

from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.tests.server.test_review2_stale_run_undo import _Base


class TestAtomicGenerationCapture(_Base):
    """Finding 1: the returned token is the caller's OWN stamp."""

    def test_update_tab_returns_generation_from_its_own_critical_section(
        self,
    ) -> None:
        """Sequential publications get strictly increasing tokens, and
        a stale token can no longer remove the newer publication."""
        reg = self.server.tab_registry
        _, _, g1 = reg.update_tab("t", chat_id="c1", create=True)
        _, _, g2 = reg.update_tab("t", chat_id="c1")
        self.assertGreater(g1, 0)
        self.assertGreater(g2, g1)
        self.assertFalse(reg.close_tab_if_generation("t", g1))
        self.assertTrue(reg.has_tab("t"))
        self.assertTrue(reg.close_tab_if_generation("t", g2))

    def test_parked_broadcast_does_not_leak_a_later_publishers_token(
        self,
    ) -> None:
        """The reviewer's schedule: publication A parks inside its
        ``tabs_state`` broadcast, publication B lands, A resumes.  A
        must return ITS token — pre-fix the post-hoc lookup returned
        B's, and A's compensating undo then deleted B's tab."""
        parked = threading.Event()
        release = threading.Event()
        real_broadcast = self.server._broadcast_tabs_state
        first_call = threading.Event()

        def parking_broadcast() -> None:
            if not first_call.is_set():
                first_call.set()
                parked.set()
                release.wait(timeout=30)
            real_broadcast()

        self.server._broadcast_tabs_state = parking_broadcast  # type: ignore[method-assign]
        try:
            result: list[int] = []

            def publish_a() -> None:
                result.append(
                    self.server._registry_update_tab(
                        "t", chat_id="chat-a", create=True,
                    )
                )

            thread = threading.Thread(target=publish_a, daemon=True)
            thread.start()
            self.assertTrue(parked.wait(timeout=10), "A never broadcast")
            gen_b = self.server._registry_update_tab("t", chat_id="chat-a")
            release.set()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        finally:
            release.set()
            self.server._broadcast_tabs_state = real_broadcast  # type: ignore[method-assign]

        self.assertTrue(result)
        gen_a = result[0]
        self.assertGreater(gen_b, gen_a, "A returned a later publisher's token")
        # A's stale undo must no-op; B's publication survives.
        self.assertFalse(
            self.server.tab_registry.close_tab_if_generation("t", gen_a)
        )
        self.assertTrue(self.server.tab_registry.has_tab("t"))

    def test_close_if_generation_rejects_a_nonpositive_token(self) -> None:
        """An adopted/loaded row reads generation 0; a captured 0 must
        never 'match' and remove it."""
        reg = self.server.tab_registry
        self.assertTrue(
            reg.merge_if_empty(
                [{"tabId": "legacy", "chatId": "c-legacy"}]
            )
        )
        self.assertEqual(reg.generation("legacy"), 0)
        self.assertFalse(reg.close_tab_if_generation("legacy", 0))
        self.assertTrue(reg.has_tab("legacy"))


class TestClaimLeakHardening(_Base):
    """Finding 2: no strandable gap, and stranded claims heal."""

    def test_holder_is_armed_before_publication(self) -> None:
        """Inject the stop at the exact publication bytecode: whatever
        the interleaving, a published claim is always release-armed, so
        the caller's ``finally`` (or the healing below) clears it."""
        repo = Path(self.work_dir)
        target = None
        lines, start = inspect.getsourcelines(VSCodeServer._claim_main_tree)
        for i, line in enumerate(lines):
            if "self._main_tree_claims[key] = claim" in line:
                target = start + i
                break
        self.assertIsNotNone(target)

        holder: list[Any] = []
        escaped: list[BaseException] = []

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            if (
                event == "line"
                and frame.f_lineno == target
                and frame.f_code.co_name == "_claim_main_tree"
            ):
                raise KeyboardInterrupt("injected at publication")
            return tracer

        try:
            sys.settrace(tracer)
            try:
                with self.server._state_lock:
                    self.server._claim_main_tree(
                        repo, "post-task commit", holder=holder,
                    )
            finally:
                sys.settrace(None)
        except KeyboardInterrupt as exc:
            escaped.append(exc)
        finally:
            # The caller's release discipline: everything in the
            # holder is released, identity-conditionally.
            with self.server._state_lock:
                for claim in holder:
                    self.server._release_main_tree_claim(claim)
        self.assertTrue(escaped, "the injected stop never escaped")
        # The claim was armed before (or never) published — either
        # way nothing is stranded.
        with self.server._state_lock:
            self.assertIsNone(self.server._main_tree_claim_reason(repo))

    def test_dead_owner_claim_heals_for_admission_and_claimants(self) -> None:
        """A claim stranded by a skipped ``finally`` (owner thread
        gone) must not refuse admission or new claimants forever —
        pre-fix it wedged the repository until daemon restart."""
        repo = Path(self.work_dir)

        def claim_and_die() -> None:
            with self.server._state_lock:
                self.assertTrue(
                    self.server._claim_main_tree(repo, "post-task commit")
                )

        thread = threading.Thread(target=claim_and_die)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        # The admission check heals the dead-owner claim...
        with self.server._state_lock:
            self.assertIsNone(self.server._main_tree_claim_reason(repo))
        # ...and a new claimant can publish.
        holder: list[Any] = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(repo, "discard", holder=holder)
            )
            self.assertEqual(
                self.server._main_tree_claim_reason(repo), "discard",
            )
            for claim in holder:
                self.server._release_main_tree_claim(claim)

    def test_stale_release_cannot_pop_a_successors_claim(self) -> None:
        """Release is conditional on the claim object's identity: an
        old cleanup releasing after a successor healed-and-claimed must
        leave the successor's claim in place."""
        repo = Path(self.work_dir)
        stranded: list[Any] = []

        def claim_and_die() -> None:
            with self.server._state_lock:
                self.assertTrue(
                    self.server._claim_main_tree(
                        repo, "post-task commit", holder=stranded,
                    )
                )

        thread = threading.Thread(target=claim_and_die)
        thread.start()
        thread.join(timeout=10)

        successor: list[Any] = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    repo, "manual commit", holder=successor,
                )
            )
            # The stranded owner's late release must be a no-op now.
            for claim in stranded:
                self.server._release_main_tree_claim(claim)
            self.assertEqual(
                self.server._main_tree_claim_reason(repo), "manual commit",
            )
            for claim in successor:
                self.server._release_main_tree_claim(claim)
            self.assertIsNone(self.server._main_tree_claim_reason(repo))


class TestRemovalGenerationWiring(_Base):
    """Finding 3: cleanup tails stand down after a legitimate reopen."""

    def _register_state(self, tab_id: str, task_id: str) -> AgentState:
        state = AgentState(task_id, tab_id=tab_id, chat_id="chat-" + tab_id)
        with self.server._state_lock:
            agent_state.register(state)
        return state

    def test_close_tail_stands_down_after_a_reopen(self) -> None:
        """The reviewer's explicit-close schedule: the close removes
        the row and parks; a resume republishes the tab and rebinds its
        chat view; the parked tail resumes.  The reopened publication's
        backend state and chat mapping must survive."""
        self.server._registry_update_tab("t", chat_id="chat-1", create=True)
        state = self._register_state("t", "task-review3-close")
        with self.server._state_lock:
            self.server._tab_chat_views["t"] = "chat-1"

        # The close's registry removal (the tail is parked after it).
        token = self.server.tab_registry.close_tab("t")
        self.assertTrue(token)
        # The concurrent reopen: republish + rebind.
        self.server._registry_update_tab("t", chat_id="chat-2", create=True)
        with self.server._state_lock:
            self.server._tab_chat_views["t"] = "chat-2"
            state.frontend_closed = False

        # The parked cleanup tail resumes — it must stand down.
        self.server._drop_tab_state("t", removal_token=token)

        self.assertTrue(self.server.tab_registry.has_tab("t"))
        with self.server._state_lock:
            self.assertEqual(self.server._tab_chat_views.get("t"), "chat-2")
            self.assertIs(agent_state.find_by_tab("t"), state)
            self.assertFalse(state.frontend_closed)

    def test_close_tail_still_drops_without_a_reopen(self) -> None:
        """Serial control: no reopen — the tail retires everything."""
        self.server._registry_update_tab("t", chat_id="chat-1", create=True)
        state = self._register_state("t", "task-review3-serial")
        with self.server._state_lock:
            self.server._tab_chat_views["t"] = "chat-1"
        token = self.server.tab_registry.close_tab("t")
        self.assertTrue(token)
        self.server._drop_tab_state("t", removal_token=token)
        with self.server._state_lock:
            self.assertNotIn("t", self.server._tab_chat_views)
            self.assertIsNone(agent_state.find_by_tab("t"))
        self.assertFalse(self.server.tab_registry.has_tab("t"))
        del state

    def test_displacement_tail_stands_down_after_a_reopen(self) -> None:
        """The reviewer's chat-displacement schedule: a takeover
        displaces ``old`` and parks before its returned-id cleanup; a
        later publication reopens ``old`` on another chat; the stale
        cleanup resumes and must not touch the reopened tab."""
        self.server._registry_update_tab("old", chat_id="chat-x", create=True)
        state = self._register_state("old", "task-review3-displace")
        with self.server._state_lock:
            self.server._tab_chat_views["old"] = "chat-x"

        # The takeover's atomic displacement (tail parked after it).
        changed, displaced, _ = self.server.tab_registry.update_tab(
            "new", chat_id="chat-x", create=True,
        )
        self.assertTrue(changed)
        self.assertEqual([tab for tab, _ in displaced], ["old"])
        _, removal_token = displaced[0]

        # The reopen of ``old`` on chat D.
        self.server._registry_update_tab("old", chat_id="chat-d", create=True)
        with self.server._state_lock:
            self.server._tab_chat_views["old"] = "chat-d"

        # The stale displacement cleanup resumes — it must stand down.
        self.server._prune_local_uds_tab("old")
        self.server._drop_tab_state("old", removal_token=removal_token)

        self.assertTrue(self.server.tab_registry.has_tab("old"))
        with self.server._state_lock:
            self.assertEqual(self.server._tab_chat_views.get("old"), "chat-d")
            self.assertIs(agent_state.find_by_tab("old"), state)

    def test_full_close_path_converges_with_a_racing_reopen(self) -> None:
        """End-to-end ``_close_tab`` vs reopen, every interleaving the
        seam allows: park the close between the registry removal and
        the tail, reopen, resume — registry and backend state must
        agree with the serial close-then-reopen outcome."""
        self.server._registry_update_tab("t", chat_id="chat-1", create=True)
        self._register_state("t", "task-review3-full")
        with self.server._state_lock:
            self.server._tab_chat_views["t"] = "chat-1"

        parked = threading.Event()
        release = threading.Event()
        real_prune = self.server._prune_local_uds_tab

        def parking_prune(tab_id: str) -> None:
            parked.set()
            release.wait(timeout=30)
            real_prune(tab_id)

        self.server._prune_local_uds_tab = parking_prune  # type: ignore[method-assign]
        try:
            closer = threading.Thread(
                target=self.server._close_tab, args=("t",), daemon=True,
            )
            closer.start()
            self.assertTrue(parked.wait(timeout=10), "close never parked")
            # The reopen (serially AFTER the close's removal).
            self.server._registry_update_tab(
                "t", chat_id="chat-2", create=True,
            )
            with self.server._state_lock:
                self.server._tab_chat_views["t"] = "chat-2"
                reopened = agent_state.find_by_tab("t")
                if reopened is not None:
                    reopened.frontend_closed = False
            release.set()
            closer.join(timeout=10)
            self.assertFalse(closer.is_alive())
        finally:
            release.set()
            self.server._prune_local_uds_tab = real_prune  # type: ignore[method-assign]

        # Serial outcome of close-then-reopen: tab open on chat-2 with
        # its chat mapping intact.
        self.assertTrue(self.server.tab_registry.has_tab("t"))
        with self.server._state_lock:
            self.assertEqual(self.server._tab_chat_views.get("t"), "chat-2")
