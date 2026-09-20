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
whole context, so pings are capped at :data:`KEEP_ALIVE_MAX_PINGS`, past
which letting the cache go cold is cheaper.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

KEEP_ALIVE_INTERVAL_SECONDS = 240.0
"""Seconds between cache touches: one minute inside the 5-minute TTL."""
KEEP_ALIVE_MAX_PINGS = 10
"""Pings per tool call before the cache is allowed to expire.  At Fable
5.1 prices a miss on a C-token context costs C x $12.25/M and a ping
C x $0.25/M plus output, so ~40 minutes of pings match one miss on the
0.1x models and are far cheaper on Fable."""
LONG_TOOL_TIMEOUT_SECONDS = 300.0
"""A tool call whose ``timeout``/``timeout_seconds`` argument reaches this
may outlive the cache and gets a keep-alive."""
FAN_OUT_TOOLS = frozenset({"run_parallel", "run_commands_parallel", "run_agent"})
"""Tools that block on other agents or command batches for an unbounded time."""


def is_long_running_call(name: str, args: dict[str, Any]) -> bool:
    """Return whether the tool call *name(**args)* may outlive the prompt cache.

    Args:
        name: The tool name.
        args: The tool's keyword arguments as the model supplied them.

    Returns:
        ``True`` for fan-out tools and for calls whose ``timeout`` or
        ``timeout_seconds`` argument is at least
        :data:`LONG_TOOL_TIMEOUT_SECONDS`.
    """
    if name in FAN_OUT_TOOLS:
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
    after ``__exit__`` has joined it, so no lock is needed.
    """

    def __init__(
        self,
        model: Any,
        function_map: dict[str, Any],
        tools_schema: list[dict[str, Any]] | None,
        cache_touched_at: float,
        interval: float | None = None,
        max_pings: int | None = None,
    ) -> None:
        self.responses: list[Any] = []
        self.last_ping_at: float | None = None
        self._model = model
        self._function_map = function_map
        self._tools_schema = tools_schema
        self._cache_touched_at = cache_touched_at
        # Read at call time so tests can shorten the module defaults.
        self._interval = KEEP_ALIVE_INTERVAL_SECONDS if interval is None else interval
        self._max_pings = KEEP_ALIVE_MAX_PINGS if max_pings is None else max_pings
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="prompt-cache-keepalive", daemon=True
        )

    def __enter__(self) -> PromptCacheKeepAlive:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        next_ping_at = self._cache_touched_at + self._interval
        for _ in range(self._max_pings):
            if self._stop.wait(max(0.0, next_ping_at - time.time())):
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
