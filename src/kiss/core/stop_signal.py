# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Per-thread stop signal, visible to blocking call sites below the agent.

A task (or a parallel sub-agent) runs on exactly one thread, and
``VSCodeServer._stop_task`` sets that thread's :class:`threading.Event`
when the user presses Stop.  The cooperative reader,
``JsonPrinter._check_stop``, only looks at the flag when the agent
*emits* something — a print or a streamed token.  A model request that
goes quiet therefore keeps the whole task alive until the provider's
stall timeout expires (180 s for Anthropic), which is what made the Stop
button look dead for three minutes in task ``709ebce3`` (post-mortem in
``reports/stop_button_delay_2026-08-05.html``).

This module publishes the running thread's stop event process-wide so
code *below* the agent — model streams and other blocking waits that own
a watchdog thread — can observe it and abort at once.
``JsonPrinter._thread_local.stop_event`` is a property over this storage,
so binding a stop event to a thread publishes it here automatically and
there is exactly one source of truth.
"""

import threading

_state = threading.local()


def set_thread_stop_event(event: threading.Event | None) -> None:
    """Publish *event* as the stop event of the calling thread.

    Args:
        event: The event a stop request will set, or ``None`` to clear
            the binding when the thread is no longer running a task.
    """
    _state.event = event


def get_thread_stop_event() -> threading.Event | None:
    """Return the calling thread's stop event.

    Returns:
        The event bound by :func:`set_thread_stop_event`, or ``None``
        when this thread is not running a stoppable task.
    """
    return getattr(_state, "event", None)


def stop_requested() -> bool:
    """Return whether a stop has been requested for the calling thread.

    Returns:
        ``True`` when this thread has a stop event and it is set.
    """
    event = get_thread_stop_event()
    return event is not None and event.is_set()
