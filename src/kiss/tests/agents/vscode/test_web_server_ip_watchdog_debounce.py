# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the IP-change watchdog debounce.

These tests reproduce the spurious-restart bug in
:meth:`kiss.server.web_server.RemoteAccessServer._watchdog`
and verify the debounce fix.

Background
----------
Before the fix, the watchdog branch that detects local-IP changes
restarted the kiss web server on **any** single tick where
``_get_local_ips()`` returned a value different from
``self._last_ips``.  The discovery helper wraps both its UDP probe
and its ``getaddrinfo`` lookup in bare ``try/except: pass`` blocks,
so any transient WiFi roam, DHCP renewal, VPN flap, or post-sleep
DNS hiccup briefly returns ``frozenset()`` — which the old code
interpreted as a real change and used to (1) close the WebSocket
server (forcing a daemon restart) and (2) poison ``_last_ips`` so
subsequent comparisons saw the wrong baseline.

The fix:

1. Treat an empty :func:`_get_local_ips` result as "discovery
   failed, skip this tick" rather than as a network change.
2. Require :data:`_IP_CHANGE_DEBOUNCE_TICKS` consecutive ticks with
   the *same* new non-empty set before acting.
3. Defer the ``_last_ips`` update until the candidate is confirmed,
   so a transient flake cannot corrupt the baseline.
4. When the baseline is empty (initial discovery failed at startup),
   adopt the first non-empty result without restarting.

The tests below drive ``_watchdog`` directly with
``TUNNEL_CHECK_INTERVAL`` set to ``0`` and ``_get_local_ips``
monkey-patched to return controlled sequences.
"""

from __future__ import annotations

import asyncio
import socket
import tempfile
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as ws_mod
from kiss.server.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer


def _find_free_port() -> int:
    """Find an available TCP port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class _IpSequence:
    """Callable that returns successive frozensets from a fixed list.

    Behaviour after the scripted list is exhausted depends on
    ``cycle``: when ``False`` (the default), the *last* entry is
    repeated forever — convenient for tests that want a transient
    flake followed by a stable steady state.  When ``True`` the
    whole list cycles indefinitely — used by the oscillation test
    where we need the A↔B alternation to continue past the scripted
    prefix.
    """

    def __init__(
        self,
        values: list[frozenset[str]],
        *,
        cycle: bool = False,
    ) -> None:
        self.values = values
        self.calls = 0
        self.cycle = cycle

    def __call__(self) -> frozenset[str]:
        if self.cycle:
            idx = self.calls % len(self.values)
        else:
            idx = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return self.values[idx]


class _IpWatchdogTestBase(IsolatedAsyncioTestCase):
    """Common setup/teardown for IP-watchdog debounce tests.

    Each test starts a real :class:`RemoteAccessServer` bound to a
    free port on ``127.0.0.1`` (so all the underlying TLS / WebSocket
    plumbing exercises the same code path the daemon does in
    production), then cancels the auto-started ``_watchdog_task`` and
    drives ``_watchdog()`` manually with a tight tick interval and a
    scripted ``_get_local_ips`` sequence.
    """

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

        # Stop the watchdog that ``start_async`` started; the tests
        # drive their own copy with a controlled tick interval and
        # patched ``_get_local_ips``.
        if self.server._watchdog_task is not None:
            self.server._watchdog_task.cancel()
            try:
                await self.server._watchdog_task
            except asyncio.CancelledError:
                pass
            self.server._watchdog_task = None

        self._orig_interval = ws_mod.TUNNEL_CHECK_INTERVAL
        ws_mod.TUNNEL_CHECK_INTERVAL = 0
        self._orig_get_local_ips = ws_mod._get_local_ips

    async def asyncTearDown(self) -> None:
        ws_mod.TUNNEL_CHECK_INTERVAL = self._orig_interval
        ws_mod._get_local_ips = self._orig_get_local_ips  # type: ignore[assignment]
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _run_watchdog_briefly(self, seconds: float = 0.3) -> bool:
        """Run the watchdog for ``seconds`` and report whether it restarted.

        Returns ``True`` iff the watchdog returned from its main
        ``while True:`` loop within the time window — which only
        happens when the IP-change branch confirms a real change and
        calls ``self._ws_server.close(); return``.  Returns ``False``
        when the watchdog was still ticking at the deadline (it is
        then cancelled here so test teardown is clean).  Any other
        exception raised by the coroutine is propagated.
        """
        task = asyncio.create_task(self.server._watchdog())
        await asyncio.sleep(seconds)
        if task.done():
            exc = task.exception()
            if exc is not None:
                raise exc
            return True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return False


class TestSpuriousRestartFromTransientEmpty(_IpWatchdogTestBase):
    """The classic flake: a single tick where ``_get_local_ips`` is empty.

    Before the fix this caused an immediate spurious restart on
    every LAN-only deployment whenever the host's network stack
    hiccuped (DHCP renewal, VPN flap, post-sleep DNS).
    """

    async def test_single_empty_tick_does_not_restart(self) -> None:
        """A single ``frozenset()`` reading must NOT close the server."""
        real_ips = frozenset({"192.168.1.42"})
        self.server._last_ips = real_ips
        # Sequence: real → empty (flake) → real → real → real → real ...
        # The repeated trailing real reads also verify that the
        # debounce state machine recovers cleanly after the flake.
        seq = _IpSequence([
            real_ips, frozenset(), real_ips, real_ips, real_ips, real_ips,
        ])
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        diag = (
            f"seq.calls={seq.calls}, "
            f"_last_ips={self.server._last_ips}, "
            f"_pending={self.server._pending_ip_change}, "
            f"count={self.server._pending_ip_change_count}"
        )
        self.assertFalse(
            restarted,
            f"Watchdog spuriously restarted after a single empty "
            f"IP reading; {diag}",
        )
        # The baseline must not have been poisoned by the flake.
        self.assertEqual(self.server._last_ips, real_ips)
        self.assertIsNone(self.server._pending_ip_change)
        self.assertEqual(self.server._pending_ip_change_count, 0)


class TestSpuriousRestartFromSingleDivergentTick(_IpWatchdogTestBase):
    """Single divergent (but non-empty) tick must not trigger a restart.

    Mimics the case where ``getaddrinfo`` briefly resolves to a
    different interface (e.g. a Docker bridge that appeared for one
    poll) before stabilising back to the real LAN address.
    """

    async def test_single_divergent_tick_does_not_restart(self) -> None:
        """One off-set reading must not cross the debounce threshold."""
        real_ips = frozenset({"192.168.1.42"})
        bogus_ips = frozenset({"172.17.0.1"})
        self.server._last_ips = real_ips
        seq = _IpSequence([
            real_ips, bogus_ips, real_ips, real_ips, real_ips, real_ips,
        ])
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        self.assertFalse(
            restarted,
            "Watchdog spuriously restarted after a single divergent tick",
        )
        self.assertEqual(self.server._last_ips, real_ips)
        self.assertIsNone(self.server._pending_ip_change)
        self.assertEqual(self.server._pending_ip_change_count, 0)


class TestSustainedIpChangeStillRestarts(_IpWatchdogTestBase):
    """A real, sustained IP change must still trigger a restart."""

    async def test_sustained_change_restarts(self) -> None:
        """Two-or-more consecutive ticks of the new IP set restart."""
        old_ips = frozenset({"192.168.1.42"})
        new_ips = frozenset({"10.0.0.7"})
        self.server._last_ips = old_ips
        seq = _IpSequence([new_ips, new_ips, new_ips, new_ips, new_ips])
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        self.assertTrue(
            restarted,
            "Watchdog failed to restart after a sustained IP change",
        )
        # On confirmed change the baseline must advance to the new set.
        self.assertEqual(self.server._last_ips, new_ips)


class TestTunnelModeDoesNotRestart(_IpWatchdogTestBase):
    """In tunnel mode the watchdog only logs; it never restarts."""

    async def test_tunnel_mode_no_restart_on_sustained_change(self) -> None:
        """Even with a confirmed change, ``use_tunnel=True`` keeps running."""
        old_ips = frozenset({"192.168.1.42"})
        new_ips = frozenset({"10.0.0.7"})
        self.server._last_ips = old_ips
        self.server.use_tunnel = True
        seq = _IpSequence([new_ips, new_ips, new_ips, new_ips, new_ips])
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        self.assertFalse(
            restarted,
            "Watchdog restarted in tunnel mode, but cloudflared "
            "should handle the edge-reconnection itself",
        )
        # Baseline still advances so the log line records the change.
        self.assertEqual(self.server._last_ips, new_ips)


class TestEmptyBaselineRecoversWithoutRestart(_IpWatchdogTestBase):
    """A server that started before the network came up.

    The constructor seeds ``_last_ips`` from
    :func:`_get_local_ips`, which can return ``frozenset()`` when
    invoked too early (e.g. inside an OS-level launchd login script
    that runs before Wi-Fi associates).  The first successful poll
    must adopt the discovered set as the new baseline without firing
    a restart — there is nothing to migrate from.
    """

    async def test_empty_baseline_adopts_without_restart(self) -> None:
        """First non-empty reading replaces the empty baseline silently."""
        real_ips = frozenset({"192.168.1.42"})
        self.server._last_ips = frozenset()
        seq = _IpSequence([real_ips, real_ips, real_ips, real_ips, real_ips])
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        self.assertFalse(
            restarted,
            "Watchdog restarted while adopting the first non-empty "
            "baseline; this would force a needless restart on every "
            "boot where the network is slow to come up",
        )
        self.assertEqual(self.server._last_ips, real_ips)


class TestFlappingSetsNeverConfirm(_IpWatchdogTestBase):
    """If two different new sets alternate, neither must reach quorum.

    The debounce requires :data:`_IP_CHANGE_DEBOUNCE_TICKS`
    *consecutive* ticks of the *same* candidate.  An alternating
    A, B, A, B, ... pattern (e.g. a dual-NIC laptop where the
    default route ping-pongs between interfaces) must never confirm.
    """

    async def test_alternating_candidates_never_confirm(self) -> None:
        """A↔B oscillation never crosses the debounce threshold."""
        base_ips = frozenset({"192.168.1.42"})
        cand_a = frozenset({"10.0.0.7"})
        cand_b = frozenset({"172.20.0.5"})
        self.server._last_ips = base_ips
        seq = _IpSequence([cand_a, cand_b], cycle=True)
        ws_mod._get_local_ips = seq  # type: ignore[assignment]

        restarted = await self._run_watchdog_briefly()

        self.assertFalse(
            restarted,
            "Watchdog confirmed a change despite oscillating "
            "candidates that never repeated consecutively",
        )
        # The baseline must NOT advance to either alternating
        # candidate, because neither ever reached quorum.
        self.assertEqual(self.server._last_ips, base_ips)
