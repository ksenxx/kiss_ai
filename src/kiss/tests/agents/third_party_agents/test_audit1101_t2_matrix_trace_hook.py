"""The Matrix stop tests hand back the trace hook they found installed.

Each of the three tests installs a ``sys.settrace`` callback that raises the
injected ``KeyboardInterrupt``; raising from a trace function clears the
thread's trace function, so the test must restore the hook it replaced in
its ``finally``.  Ending with ``sys.settrace(None)`` instead silently
switched off a tracing-based coverage collector or debugger for the rest
of the pytest thread.  The sentinel here stands for that collector.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest

from kiss.tests.agents.third_party_agents import test_matrix_send_retry_wave2 as wave2
from kiss.tests.agents.third_party_agents import test_matrix_wave4 as wave4


def _sentinel_tracer(frame: Any, event: str, arg: Any) -> None:
    """A global trace function that traces nothing; only its identity matters."""
    return None


def _test_name(test: Callable[[], None]) -> str:
    """Parametrize id: the stop test's function name."""
    return test.__name__


@pytest.mark.parametrize(
    "stop_test",
    [
        wave2.test_stopped_run_cancels_the_coroutine,
        wave4.test_stop_while_scheduling_closes_the_coroutine_unstarted,
        wave4.test_stop_after_the_coroutine_started_cancels_it,
    ],
    ids=_test_name,
)
def test_stop_test_restores_the_trace_hook_it_found(stop_test: Callable[[], None]) -> None:
    """After the stop test passes, the hook installed before it is back in place."""
    previous = sys.gettrace()
    sys.settrace(_sentinel_tracer)
    try:
        stop_test()
        assert sys.gettrace() is _sentinel_tracer
    finally:
        sys.settrace(previous)
