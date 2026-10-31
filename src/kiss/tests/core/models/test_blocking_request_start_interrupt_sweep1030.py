# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a forced stop landing while the request's worker starts tears the request down.

``_blocking_request`` runs a non-streamed request on a worker thread and,
in a ``finally``, shuts the pending request down when this thread is
interrupted (the runner's forced stop injects ``KeyboardInterrupt`` 1 s
and 6 s after Stop).  ``worker.start()`` sat before that ``try``, so an
interrupt landing inside ``start()`` left the worker sending the request
on an orphan thread until the client's 30-minute timeout, its reply never
billed.  Moving ``start()`` into the ``try`` alone does not close the
window: an interrupt that lands before the new thread reports itself
started leaves ``is_alive()`` False although the thread runs, so the
teardown keys on the worker not having answered yet.

The interrupt is the runner's own mechanism
(``tool_interrupt.inject_async_exception``).  A ``sys.setprofile`` hook on
the requesting thread only picks the moment, the return of the C call
that spawns the worker's OS thread inside ``Thread.start``: the new
thread cannot run Python code (and so cannot report itself started)
before this thread lets go of the GIL.  Nothing is replaced.
"""

from __future__ import annotations

import sys
import threading
import time
from types import FrameType
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.core.models.openai_sse_harness import ScriptedOpenAIServer
from kiss.tests.core.models.test_f1_blocking_request_stop import (
    _UNWIND_WITHIN,
    _WORKER_THREAD,
    _SilentThenPromptGateway,
    _worker_threads,
)

_MODEL = "gpt-start-interrupt-under-test"
# The C function ``Thread.start`` spawns the OS thread with (3.13+, 3.12).
_SPAWN_FUNCTIONS = {"start_joinable_thread", "start_new_thread"}


class _InterruptAtWorkerSpawn:
    """Profile hook: raise a forced stop in this thread as ``Thread.start`` spawns the worker."""

    def __init__(self) -> None:
        self.fired = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        """Inject ``KeyboardInterrupt`` on the return of the request worker's spawn call."""
        if (
            event != "c_return"
            or frame.f_code is not threading.Thread.start.__code__
            or getattr(arg, "__name__", "") not in _SPAWN_FUNCTIONS
            or frame.f_locals["self"].name != _WORKER_THREAD
        ):
            return
        sys.setprofile(None)
        self.fired.set()
        inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def _generate_interrupted_at_worker_spawn(model: Any) -> _InterruptAtWorkerSpawn:
    """Run one stoppable non-streamed turn whose forced stop lands inside ``Thread.start``.

    Args:
        model: An initialized model whose endpoint never answers.

    Returns:
        The hook, to check that it fired.
    """
    hook = _InterruptAtWorkerSpawn()
    stop_signal.set_thread_stop_event(threading.Event())
    try:
        with pytest.raises(KeyboardInterrupt):
            sys.setprofile(hook)
            try:
                model.generate()
            finally:
                sys.setprofile(None)
    finally:
        stop_signal.set_thread_stop_event(None)
    return hook


@pytest.mark.parametrize(
    "model_class", [OpenAICompatibleModel, OpenAICompatibleModel2], ids=["chat", "responses"]
)
def test_a_forced_stop_inside_worker_start_tears_the_request_down(model_class: type) -> None:
    gateway = _SilentThenPromptGateway()
    try:
        with ScriptedOpenAIServer(gateway) as server:
            model = model_class(_MODEL, base_url=server.base_url, api_key="sk-test")
            model.initialize("hi")
            hook = _generate_interrupted_at_worker_spawn(model)
            assert hook.fired.is_set(), "the forced stop never landed inside Thread.start"
            assert model.client is None, "the interrupted request's client was not torn down"
            deadline = time.monotonic() + _UNWIND_WITHIN
            while _worker_threads() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _worker_threads(), "the request runs on as an orphan worker thread"
    finally:
        gateway.release.set()
