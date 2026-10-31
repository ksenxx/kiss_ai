# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A cancel landing while a timeout stop is pending does not restart the grace (S4-7).

``daemon_client.run(stop_on_timeout=True, cancel=event)`` sends one
``stop`` when the timeout passes and keeps reading for
``_STOP_CONFIRM_GRACE_SECONDS`` until the task confirms it is dead.
The cancel branch used to test ``cancelled`` instead of ``stopping``,
so a cancel set during that grace sent a second ``stop`` and moved the
deadline a whole grace further: an ``agent_job`` kill of a job that had
already timed out waited up to twice the grace.  Now the cancel is
only recorded; the pending stop and its deadline stay as they are.
"""

import threading
import time

import pytest

from kiss.agents.sorcar import daemon_client
from kiss.tests.agents.sorcar.test_dispatch_stop_cascade import _RecordingDaemon

GRACE = 1.0
TIMEOUT = 0.3
CANCEL_AFTER_STOP = 0.5


def test_cancel_during_the_timeout_grace_sends_no_second_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One ``stop``, one grace: the call ends at timeout + grace, as CancelledError."""
    monkeypatch.setattr(daemon_client, "_STOP_CONFIRM_GRACE_SECONDS", GRACE)
    daemon = _RecordingDaemon(mode="silent")
    cancel = threading.Event()

    def cancel_after_the_stop() -> None:
        if daemon.wait_for_command("stop"):
            time.sleep(CANCEL_AFTER_STOP)
            cancel.set()

    try:
        threading.Thread(target=cancel_after_the_stop, daemon=True).start()
        begin = time.monotonic()
        with pytest.raises(daemon_client.CancelledError) as caught:
            daemon_client.run(
                "ignores stops", endpoint_file=daemon.endpoint_file, timeout=TIMEOUT,
                stop_on_timeout=True, cancel=cancel,
            )
        elapsed = time.monotonic() - begin
        assert daemon.disconnected.wait(10)
        stops = [c for c in daemon.commands if c.get("type") == "stop"]
        assert len(stops) == 1, stops
        # Unconfirmed: the task may still be running, so its tab stays open.
        assert not any(c.get("type") == "closeTab" for c in daemon.commands)
        assert cancel.is_set()
        assert caught.value.confirmed is False
        # A second grace would end at TIMEOUT + CANCEL_AFTER_STOP + GRACE.
        assert elapsed < TIMEOUT + CANCEL_AFTER_STOP + GRACE - 0.1, elapsed
    finally:
        daemon.close()
