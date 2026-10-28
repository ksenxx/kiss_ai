# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Keep a provider's prompt cache warm while a long tool call runs.

Anthropic's prompt cache lives 5 minutes, measured from the START of the
request that last wrote or read it, and every read renews it for free.
A tool call that runs longer than that (a test suite, a build, a
sub-agent fan-out) lets the cache expire, and the next step re-writes
the whole context at 1.25x the input price instead of reading it at
0.1x (0.025x on Claude Fable 5.1).  The 72 h audit of 2026-09-20
attributed 42 % of cache-miss spend to such gaps.

:class:`PromptCacheKeepAlive` runs a daemon thread beside the tool call
that asks the model to re-read its cached prefix every
:data:`KEEP_ALIVE_INTERVAL_SECONDS` (via ``Model.keep_prompt_cache_warm``,
a no-op for providers that need none).  A ping costs one cache read of
the context plus a few output tokens; a miss costs the cache write of the
whole context, so pings are capped per model by :func:`keep_alive_max_pings`:
a little under ``(cache_write_price - cache_read_price) / cache_read_price``
pings cost less than the one rewrite they avert, and
:data:`KEEP_ALIVE_MAX_SECONDS` bounds the wall clock.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from kiss.core import stop_signal
from kiss.core.models.model_info import _lookup_model_info

logger = logging.getLogger(__name__)

KEEP_ALIVE_INTERVAL_SECONDS = 240.0
"""Seconds between cache touches: one minute inside the 5-minute TTL."""
KEEP_ALIVE_MAX_PINGS = 10
"""Pings per tool call for a model whose prices are unknown."""
KEEP_ALIVE_MAX_SECONDS = 4 * 3600.0
"""Wall-clock bound of the pings of one tool call, whatever the prices."""
PING_PRICE_MARGIN = 0.8
"""Fraction of the break-even ping count kept, so the pings' own output
tokens and the final rewrite still come in under one cold miss."""
LONG_TOOL_TIMEOUT_SECONDS = 300.0
"""A tool call whose ``timeout``/``timeout_seconds`` argument reaches this
may outlive the cache and gets a keep-alive."""
FAN_OUT_TOOLS = frozenset({"run_parallel", "run_commands_parallel", "run_agent"})
"""Tools that block on other agents or command batches for an unbounded time."""
WAITING_TOOLS = frozenset({"ask_user_question"})
"""Tools that wait on a person, however short their timeout."""


def keep_alive_max_pings(model_name: str) -> int:
    """Return how many keep-alive pings one tool call may send on *model_name*.

    A ping re-reads the cached context at the cache-read price; letting
    the cache expire rewrites it at the cache-write price (the input
    price on a model that charges no write), so the break-even count
    is ``(write - read) / read`` and :data:`PING_PRICE_MARGIN` of it is
    kept: about 40 pings (2.7 hours) on claude-fable-5-1, 9 on a model
    reading at 0.1x.  The result is at least 1 and never more than
    :data:`KEEP_ALIVE_MAX_SECONDS` worth of intervals.

    Args:
        model_name: The model, with or without its provider prefix.

    Returns:
        The ping cap; :data:`KEEP_ALIVE_MAX_PINGS` when the model's
        prices are unknown or it has no cache-read price.
    """
    info = _lookup_model_info(model_name)
    wall_clock_cap = int(KEEP_ALIVE_MAX_SECONDS / KEEP_ALIVE_INTERVAL_SECONDS)
    if info is None or not info.cache_read_price_per_1M:
        return min(KEEP_ALIVE_MAX_PINGS, wall_clock_cap)
    read = info.cache_read_price_per_1M
    write = info.cache_write_price_per_1M or info.input_price_per_1M
    by_price = int(PING_PRICE_MARGIN * (write - read) / read)
    return max(1, min(by_price, wall_clock_cap))


def is_long_running_call(name: str, args: dict[str, Any]) -> bool:
    """Return whether the tool call *name(**args)* may outlive the prompt cache.

    Args:
        name: The tool name.
        args: The tool's keyword arguments as the model supplied them.

    Returns:
        ``True`` for fan-out tools, for the tools that wait on a person
        (:data:`WAITING_TOOLS`, whatever their timeout), for a
        ``bash_job`` or ``agent_job`` wait (whatever its timeout: the
        job is another process or agent), and for calls whose
        ``timeout`` or ``timeout_seconds`` argument is at least
        :data:`LONG_TOOL_TIMEOUT_SECONDS`.
    """
    if name in FAN_OUT_TOOLS or name in WAITING_TOOLS:
        return True
    if name in ("bash_job", "agent_job") and args.get("action") == "wait":
        return True
    for key in ("timeout_seconds", "timeout"):
        try:
            if float(args.get(key)) >= LONG_TOOL_TIMEOUT_SECONDS:  # type: ignore[arg-type]
                return True
        except (TypeError, ValueError):
            continue
    return False


class PromptCacheKeepAlive:
    """Context manager that pings the model's prompt cache while its body runs.

    Usage::

        with PromptCacheKeepAlive(model, function_map, tools_schema, touched_at) as keep_alive:
            run_the_long_tool()
        for response in keep_alive.responses:
            account_for(response)   # every ping is a billed request

    The first ping is due :data:`KEEP_ALIVE_INTERVAL_SECONDS` after
    *cache_touched_at* (the start of the request that last touched the
    cache), later ones the same interval after the previous ping.  The
    thread stops as soon as the body finishes, after a ping fails, when
    the model reports that it needs no keep-alive (``None``), or after
    *max_pings*.  ``__exit__`` joins the thread, so the body's caller may
    wait for a ping already in flight (bounded by the model's ping
    timeout, 30 s for Anthropic with retries off; typically 1-3 s), which
    keeps the model object single-threaded again once the body is left.
    Responses are only appended by the thread and read by the caller
    after :meth:`stop` has joined it, so no lock is needed.

    A task Stop is a ``KeyboardInterrupt`` injected into the task thread
    (``PyThreadState_SetAsyncExc``), delivered at the next check point:
    that can be the entry of ``__enter__``'s ``start()`` return or of
    ``__exit__`` itself, before ``_stop`` is set, and the thread would
    then ping (and bill) for up to *max_pings* intervals after the task
    ended.  :meth:`stop` is therefore idempotent and the caller repeats
    it from a ``finally`` of its own, and the thread gives up by itself
    once the stop event of the thread that created it is set.
    """

    def __init__(
        self,
        model: Any,
        function_map: dict[str, Any],
        tools_schema: list[dict[str, Any]] | None,
        cache_touched_at: float,
        max_pings: int | None = None,
    ) -> None:
        self.responses: list[Any] = []
        self.last_ping_at: float | None = None
        self._model = model
        self._function_map = function_map
        self._tools_schema = tools_schema
        self._cache_touched_at = cache_touched_at
        # Read at call time so tests can shorten the module defaults.
        self._interval = KEEP_ALIVE_INTERVAL_SECONDS
        self._max_pings = KEEP_ALIVE_MAX_PINGS if max_pings is None else max_pings
        self._stop = threading.Event()
        self._owner_stop = stop_signal.get_thread_stop_event()
        self._thread = threading.Thread(
            target=self._run, name="prompt-cache-keepalive", daemon=True
        )

    def __enter__(self) -> PromptCacheKeepAlive:
        try:
            self._thread.start()
        except BaseException:
            # ``__exit__`` never runs for an ``__enter__`` that raised,
            # and the Stop may have landed after the thread was spawned.
            self._stop.set()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop the ping thread and wait for a ping already in flight.

        Idempotent and safe before the thread started: ``__exit__``
        calls it, and the body's caller calls it again from a
        ``finally`` in case a Stop landed on ``__exit__``'s entry.
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        next_ping_at = self._cache_touched_at + self._interval
        for _ in range(self._max_pings):
            if self._stop.wait(max(0.0, next_ping_at - time.time())):
                return
            if self._owner_stop is not None and self._owner_stop.is_set():
                return
            started = time.time()
            try:
                response = self._model.keep_prompt_cache_warm(
                    self._function_map, self._tools_schema
                )
            except Exception as e:
                logger.warning("Prompt-cache keep-alive ping failed; giving up: %s", e)
                return
            if response is None:
                return
            self.responses.append(response)
            self.last_ping_at = started
            next_ping_at = started + self._interval
