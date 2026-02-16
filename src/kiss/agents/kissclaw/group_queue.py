"""Per-group message queue with concurrency control for KISSClaw."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_RETRY_SECONDS = 5.0


@dataclass
class _GroupState:
    active: bool = False
    pending_messages: bool = False
    pending_tasks: list[tuple[str, Callable[[], bool]]] = field(default_factory=list)
    retry_count: int = 0


class GroupQueue:
    """Per-group queue with global concurrency limiting.

    Work items are dispatched to background threads so that
    ``enqueue_message_check`` / ``enqueue_task`` return immediately.
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        self._max_concurrent = max_concurrent
        self._groups: dict[str, _GroupState] = {}
        self._active_count = 0
        self._waiting: list[str] = []
        self._process_messages_fn: Callable[[str], bool] | None = None
        self._shutting_down = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _get_group(self, jid: str) -> _GroupState:
        if jid not in self._groups:
            self._groups[jid] = _GroupState()
        return self._groups[jid]

    def set_process_messages_fn(self, fn: Callable[[str], bool]) -> None:
        self._process_messages_fn = fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enqueue_message_check(self, group_jid: str) -> None:
        if self._shutting_down:
            return
        with self._lock:
            state = self._get_group(group_jid)
            if state.active:
                state.pending_messages = True
                return
            if self._active_count >= self._max_concurrent:
                state.pending_messages = True
                if group_jid not in self._waiting:
                    self._waiting.append(group_jid)
                return
            # Acquire slot
            state.active = True
            state.pending_messages = False
            self._active_count += 1
        threading.Thread(
            target=self._worker_messages, args=(group_jid,), daemon=True
        ).start()

    def enqueue_task(
        self, group_jid: str, task_id: str, fn: Callable[[], bool]
    ) -> None:
        if self._shutting_down:
            return
        with self._lock:
            state = self._get_group(group_jid)
            # Reject duplicates
            if any(t[0] == task_id for t in state.pending_tasks):
                return
            if state.active:
                state.pending_tasks.append((task_id, fn))
                return
            if self._active_count >= self._max_concurrent:
                state.pending_tasks.append((task_id, fn))
                if group_jid not in self._waiting:
                    self._waiting.append(group_jid)
                return
            # Acquire slot
            state.active = True
            self._active_count += 1
        threading.Thread(
            target=self._worker_task, args=(group_jid, task_id, fn), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _worker_messages(self, group_jid: str) -> None:
        try:
            if self._process_messages_fn:
                success = self._process_messages_fn(group_jid)
                if success:
                    with self._lock:
                        self._get_group(group_jid).retry_count = 0
                else:
                    self._schedule_retry(group_jid)
        except Exception:
            logger.exception("Error processing messages for %s", group_jid)
            self._schedule_retry(group_jid)
        finally:
            self._release_and_drain(group_jid)

    def _worker_task(
        self, group_jid: str, task_id: str, fn: Callable[[], bool]
    ) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Error running task %s for %s", task_id, group_jid)
        finally:
            self._release_and_drain(group_jid)

    # ------------------------------------------------------------------
    # Drain / release
    # ------------------------------------------------------------------
    def _release_and_drain(self, group_jid: str) -> None:
        """Release the slot and start the next piece of work if any."""
        next_action: tuple | None = None
        with self._lock:
            state = self._get_group(group_jid)
            state.active = False
            self._active_count -= 1

            if self._shutting_down:
                return

            # This group has more work → re-acquire slot immediately
            if state.pending_tasks:
                task_id, fn = state.pending_tasks.pop(0)
                state.active = True
                self._active_count += 1
                next_action = ("task", group_jid, task_id, fn)
            elif state.pending_messages:
                state.active = True
                state.pending_messages = False
                self._active_count += 1
                next_action = ("messages", group_jid)
            else:
                # Nothing pending for this group → drain the global waiting list
                next_action = self._pop_waiting_locked()

        if next_action is not None:
            self._dispatch(next_action)

    def _pop_waiting_locked(self) -> tuple | None:
        """Pop the next waiting group (must hold ``_lock``)."""
        while self._waiting and self._active_count < self._max_concurrent:
            next_jid = self._waiting.pop(0)
            state = self._get_group(next_jid)
            if state.pending_tasks:
                task_id, fn = state.pending_tasks.pop(0)
                state.active = True
                self._active_count += 1
                return ("task", next_jid, task_id, fn)
            elif state.pending_messages:
                state.active = True
                state.pending_messages = False
                self._active_count += 1
                return ("messages", next_jid)
        return None

    def _dispatch(self, action: tuple) -> None:
        kind = action[0]
        if kind == "task":
            _, jid, task_id, fn = action
            threading.Thread(
                target=self._worker_task, args=(jid, task_id, fn), daemon=True
            ).start()
        elif kind == "messages":
            _, jid = action
            threading.Thread(
                target=self._worker_messages, args=(jid,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------
    def _schedule_retry(self, group_jid: str) -> None:
        with self._lock:
            state = self._get_group(group_jid)
            state.retry_count += 1
            if state.retry_count > MAX_RETRIES:
                logger.error("Max retries exceeded for %s", group_jid)
                state.retry_count = 0
                return
            delay = BASE_RETRY_SECONDS * (2 ** (state.retry_count - 1))

        def retry() -> None:
            time.sleep(delay)
            if not self._shutting_down:
                self.enqueue_message_check(group_jid)

        threading.Thread(target=retry, daemon=True).start()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._shutting_down = True

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    def is_group_active(self, group_jid: str) -> bool:
        with self._lock:
            state = self._groups.get(group_jid)
            return state.active if state else False
