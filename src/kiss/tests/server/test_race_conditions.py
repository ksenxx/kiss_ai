# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for race conditions in sorcar/ and vscode/.

These tests deliberately interleave concurrent threads to reproduce
specific race conditions identified in
``kiss.agents.sorcar.persistence`` and
``kiss.server.autocomplete``.

Each test inserts a small ``random.uniform(0, 0.05)`` sleep at the
suspected racing statement, then runs the involved threads many times
to make the failure deterministic.
"""

from __future__ import annotations

import random
import threading
import time


class TestAutocompleteWorkerDoubleSpawn:
    """Race 2: ``_AutocompleteMixin._ensure_complete_worker`` spawns
    duplicate worker threads when called from multiple threads.

    Two callers both observe ``self._complete_worker is None``, both
    create their own ``queue.Queue`` and worker thread, and the
    second thread overwrites the first thread's published references
    on ``self`` — leaving one worker reading from an orphaned queue
    forever (zombie thread) and producers enqueuing into a queue that
    a different worker drains.
    """

    def test_double_spawn(self) -> None:
        import queue as queue_mod

        from kiss.server.autocomplete import _AutocompleteMixin

        instance = _AutocompleteMixin()
        instance._state_lock = threading.RLock()  # type: ignore[attr-defined]
        instance._complete_queue = None  # type: ignore[attr-defined]
        instance._complete_worker = None  # type: ignore[attr-defined]

        def fake_loop(self_ref: object) -> None:
            q = getattr(self_ref, "_complete_queue", None)
            if q is None:
                return
            while True:
                try:
                    item = q.get(timeout=0.5)
                except queue_mod.Empty:
                    return
                if item is None:
                    return

        original_loop = _AutocompleteMixin._complete_worker_loop
        _AutocompleteMixin._complete_worker_loop = fake_loop  # type: ignore[assignment]

        try:
            original_q_init = queue_mod.Queue.__init__

            def slow_init(self: object, *args: object, **kwargs: object) -> None:
                time.sleep(random.uniform(0.01, 0.03))
                original_q_init(self, *args, **kwargs)  # type: ignore[arg-type]

            queue_mod.Queue.__init__ = slow_init  # type: ignore[method-assign]

            try:
                threads = []
                for _ in range(16):
                    t = threading.Thread(target=instance._ensure_complete_worker)
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5)
            finally:
                queue_mod.Queue.__init__ = original_q_init  # type: ignore[method-assign]

            alive_workers = []
            for t in threading.enumerate():
                target = getattr(t, "_target", None)
                if target is None:
                    continue
                if not t.daemon or not t.is_alive():
                    continue
                if not t.name.startswith("Thread-"):
                    continue
                if getattr(target, "__name__", "") == "fake_loop":
                    alive_workers.append(t)
            assert len(alive_workers) <= 1, (
                f"double-spawn race: {len(alive_workers)} workers alive"
            )
            assert instance._complete_queue is not None
            assert instance._complete_worker is not None
            assert instance._complete_worker.is_alive()
        finally:
            _AutocompleteMixin._complete_worker_loop = (  # type: ignore[method-assign]
                original_loop
            )
            if instance._complete_queue is not None:
                try:
                    instance._complete_queue.put_nowait(None)
                except Exception:
                    pass
