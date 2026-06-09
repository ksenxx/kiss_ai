# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests covering the remaining spurious-restart paths.

These tests reproduce and verify the fix for four residual paths in
:mod:`kiss.agents.vscode.web_server` that earlier code reviews flagged
as still capable of triggering an unnecessary server / cloudflared
restart, even after the IP-watchdog debounce was hardened:

1.  ``_probe_tunnel_ready`` returned ``False`` for *any* error
    (connection refused, parse error, schema change, slow CPU).  The
    watchdog conflated that with confirmed ``readyConnections == 0``
    and burned a fresh ``*.trycloudflare.com`` URL every ~10 minutes
    whenever the metrics loopback socket flaked.  The fix makes the
    helper return ``None`` on "unknown" and the watchdog skips that
    tick instead of counting it.

2.  ``_get_local_ips`` returned every address ``getaddrinfo`` saw,
    including link-local 169.254.0.0/16 (auto-assigned when DHCP
    drops out) and IPv4-mapped IPv6 (``::ffff:1.2.3.4``).  Either
    one could flap in and out across watchdog ticks and look like a
    real IP change.  The fix filters both classes out.

3.  ``_IP_CHANGE_DEBOUNCE_TICKS`` was 2 — a 60 s window is not
    enough to ride out a VPN-connect or Ethernet↔WiFi handover that
    briefly holds a new consistent IP set.  The fix raises it to 4
    (≈ 120 s) and these tests pin the new boundary.

4.  When the metrics probe was *chronically* unhealthy the watchdog
    would force-restart cloudflared, wait the startup grace, observe
    unhealthy again, force-restart again — rotating the public URL
    every ~10 minutes forever.  The fix introduces a per-restart
    exponential cool-down (60 s → 120 s → ... cap 1 h) that stops
    the loop until either the tunnel recovers or wall-clock time
    catches up.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
import unittest
from unittest import IsolatedAsyncioTestCase

import kiss.agents.vscode.web_server as ws_mod
from kiss.agents.vscode.web_server import (
    _IP_CHANGE_DEBOUNCE_TICKS,
    _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL,
    _TUNNEL_FORCE_RESTART_COOLDOWN_MAX,
    _TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY,
    _TUNNEL_STARTUP_GRACE,
    _TUNNEL_UNHEALTHY_LIMIT_QUICK,
    RemoteAccessServer,
    _get_local_ips,
)


class TestUnreachableMetricsDoesNotCountAsUnhealthy(IsolatedAsyncioTestCase):
    """The "metrics endpoint unreachable" path must NOT count as unhealthy.

    Reproduces the chief remaining quick-tunnel URL-rotation cause: a
    momentarily-unreachable ``/ready`` endpoint (post-sleep socket
    rebind, slow CPU after wake, transient loopback glitch) used to
    increment ``_tunnel_unhealthy_ticks`` exactly like a confirmed
    deregistration.  After the fix, an ``UNKNOWN`` probe result is a
    no-op for the unhealthy counter and the public URL is preserved.
    """

    async def asyncSetUp(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.server = RemoteAccessServer(use_tunnel=False)
        self.server._loop = self._loop
        self.server._tunnel_proc = self._proc
        self.server._tunnel_metrics_port = 1
        # Past startup grace so the metrics check runs every tick.
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.server._tunnel_unhealthy_ticks = 0
        # Simulate "endpoint unreachable / unknown".
        self._orig_probe = ws_mod._probe_tunnel_ready
        ws_mod._probe_tunnel_ready = lambda _port: None  # type: ignore[assignment]

    async def asyncTearDown(self) -> None:
        ws_mod._probe_tunnel_ready = self._orig_probe  # type: ignore[assignment]
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.server._tunnel_proc = None

    async def test_unknown_state_is_a_noop(self) -> None:
        """Many consecutive ``UNKNOWN`` probes must not trigger any restart."""
        for _ in range(_TUNNEL_UNHEALTHY_LIMIT_QUICK + 5):
            await self.server._check_and_restart_tunnel()
        self.assertEqual(self.server._tunnel_unhealthy_ticks, 0)
        # The original subprocess is still alive — no force-kill happened.
        self.assertIsNotNone(self.server._tunnel_proc)
        assert self.server._tunnel_proc is not None
        self.assertIsNone(self.server._tunnel_proc.poll())
        # And no cool-down state was advanced.
        self.assertEqual(self.server._tunnel_force_restart_count, 0)
        self.assertEqual(
            self.server._tunnel_force_restart_next_allowed, 0.0,
        )


class TestConfirmedZeroStillRestarts(IsolatedAsyncioTestCase):
    """A confirmed ``readyConnections == 0`` must still drive a force-restart.

    The tri-valued probe is only a no-op for ``None`` (unknown).  A
    deterministic ``False`` (the canonical "edge deregistered"
    signal) must continue to advance the unhealthy counter and, on
    reaching the limit, force-restart cloudflared.
    """

    async def asyncSetUp(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.server = RemoteAccessServer(use_tunnel=False)
        self.server._loop = self._loop
        self.server._tunnel_proc = self._proc
        self.server._tunnel_metrics_port = 1
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.server._tunnel_unhealthy_ticks = 0
        self._orig_probe = ws_mod._probe_tunnel_ready
        ws_mod._probe_tunnel_ready = lambda _port: False  # type: ignore[assignment]

        self.restart_calls: list[float] = []

        async def fake_restart() -> None:
            self.restart_calls.append(time.monotonic())

        self.server._restart_tunnel_url = fake_restart  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        ws_mod._probe_tunnel_ready = self._orig_probe  # type: ignore[assignment]
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    async def test_confirmed_zero_increments_and_restarts(self) -> None:
        """``False`` (confirmed) increments ticks; at the limit, restart fires."""
        for _ in range(_TUNNEL_UNHEALTHY_LIMIT_QUICK):
            await self.server._check_and_restart_tunnel()
        # On the limit tick, the original proc was force-killed and
        # ``_restart_tunnel_url`` was invoked exactly once.
        self.assertEqual(len(self.restart_calls), 1)
        self.assertIsNotNone(self._proc.poll())
        # First force-restart bumps the cool-down ladder to rung 1.
        self.assertEqual(self.server._tunnel_force_restart_count, 1)
        self.assertGreater(
            self.server._tunnel_force_restart_next_allowed,
            time.monotonic(),
        )


class TestLinkLocalAndMappedIpsFiltered(unittest.TestCase):
    """``_get_local_ips`` must drop link-local 169.254/16 and ::ffff: mappings.

    These flap in and out across DHCP renewals / dual-stack
    address-family oscillations and would otherwise look like real
    IP changes to the watchdog.
    """

    def setUp(self) -> None:
        self._orig_getaddrinfo = socket.getaddrinfo
        self._orig_socket = socket.socket

        class _NoUdpSocket:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise OSError("UDP discovery disabled in test")

        # Disable the UDP-connect helper so only the patched
        # ``getaddrinfo`` results feed the function.  ``_get_local_ips``
        # already swallows ``OSError`` from the UDP probe.
        setattr(socket, "socket", _NoUdpSocket)  # noqa: B010

    def tearDown(self) -> None:
        setattr(socket, "socket", self._orig_socket)  # noqa: B010
        setattr(socket, "getaddrinfo", self._orig_getaddrinfo)  # noqa: B010

    def test_link_local_excluded(self) -> None:
        """A link-local address must not appear in the result."""

        def fake_getaddrinfo(
            _host: str, _port: object, _family: object,
        ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "",
                 ("192.168.1.42", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "",
                 ("169.254.10.5", 0)),
            ]

        setattr(socket, "getaddrinfo", fake_getaddrinfo)  # noqa: B010
        ips = _get_local_ips()
        self.assertIn("192.168.1.42", ips)
        self.assertNotIn("169.254.10.5", ips)

    def test_ipv4_mapped_ipv6_excluded(self) -> None:
        """An ``::ffff:1.2.3.4``-form address must not appear in the result."""

        def fake_getaddrinfo(
            _host: str, _port: object, _family: object,
        ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "",
                 ("10.0.0.7", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "",
                 ("::ffff:10.0.0.7", 0)),
            ]

        setattr(socket, "getaddrinfo", fake_getaddrinfo)  # noqa: B010
        ips = _get_local_ips()
        self.assertIn("10.0.0.7", ips)
        self.assertNotIn("::ffff:10.0.0.7", ips)

    def test_only_link_local_yields_empty_set(self) -> None:
        """If every discovered address is link-local, the result is empty.

        An empty set is the canonical "no information" signal the
        watchdog already treats as no-op; this is the *right*
        behaviour during a captive-portal / DHCP-pending window so
        the watchdog does not see a "change" event.
        """

        def fake_getaddrinfo(
            _host: str, _port: object, _family: object,
        ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "",
                 ("169.254.10.5", 0)),
            ]

        setattr(socket, "getaddrinfo", fake_getaddrinfo)  # noqa: B010
        ips = _get_local_ips()
        self.assertEqual(ips, frozenset())


class TestIpDebounceTickConstant(unittest.TestCase):
    """The debounce constant must be at least 4 ticks (≈ 120 s)."""

    def test_debounce_is_at_least_four_ticks(self) -> None:
        """A VPN-connect / Ethernet↔WiFi handover that briefly holds a
        new consistent IP set for ≥ 60 s should *not* restart the
        server.  Earlier code used 2 ticks (60 s); the fix tightens
        this to at least 4."""
        self.assertGreaterEqual(_IP_CHANGE_DEBOUNCE_TICKS, 4)


class TestChronicMetricsFlakeCoolsDown(IsolatedAsyncioTestCase):
    """A chronically-flaky ``/ready`` endpoint must not loop force-restarts.

    Reproduces the worst remaining failure mode: a cloudflared whose
    metrics endpoint *successfully* returns ``readyConnections == 0``
    on every probe (so the result is ``False``, not ``None``).
    Without a cool-down, the watchdog would force-restart cloudflared,
    spawn a fresh quick-tunnel URL, observe it as unhealthy again,
    force-restart again — rotating ``*.trycloudflare.com`` URLs
    every ~10 minutes forever.  After the fix, the first force-restart
    starts a 60 s cool-down that doubles on each subsequent attempt
    so wall-clock-bounded test runs see at most one URL rotation.
    """

    async def asyncSetUp(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._procs: list[subprocess.Popen[str]] = []
        first_proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._procs.append(first_proc)
        self.server = RemoteAccessServer(use_tunnel=False)
        self.server._loop = self._loop
        self.server._tunnel_proc = first_proc
        self.server._tunnel_metrics_port = 1
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.server._tunnel_unhealthy_ticks = 0
        self._orig_probe = ws_mod._probe_tunnel_ready
        ws_mod._probe_tunnel_ready = lambda _port: False  # type: ignore[assignment]

        self.restart_calls: list[float] = []

        async def fake_restart() -> None:
            """Simulate a successful respawn that is *also* immediately unhealthy."""
            self.restart_calls.append(time.monotonic())
            new_proc = subprocess.Popen(
                ["sleep", "30"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._procs.append(new_proc)
            self.server._tunnel_proc = new_proc
            self.server._tunnel_metrics_port = 1
            # Skip the startup grace so the next probe counts.
            self.server._tunnel_started_at = (
                time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
            )

        self.server._restart_tunnel_url = fake_restart  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        ws_mod._probe_tunnel_ready = self._orig_probe  # type: ignore[assignment]
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.server._tunnel_proc = None

    async def test_force_restart_loop_is_bounded_by_cooldown(self) -> None:
        """Many ticks of confirmed-unhealthy must restart at most once.

        Real wall-clock time barely advances during this test, so the
        60 s initial cool-down covers the entire run.  Without the
        cool-down the watchdog would restart cloudflared exactly
        ``ticks // _TUNNEL_UNHEALTHY_LIMIT_QUICK`` times — at least
        dozens in the loop below.  With the cool-down it restarts once
        and then defers every subsequent force-restart.
        """
        # Drive the equivalent of ~50 cycles worth of unhealthy ticks.
        for _ in range(_TUNNEL_UNHEALTHY_LIMIT_QUICK * 50):
            await self.server._check_and_restart_tunnel()
        self.assertEqual(
            len(self.restart_calls), 1,
            f"Expected exactly one force-restart within the initial "
            f"cool-down window; got {len(self.restart_calls)}.  Without "
            f"the cool-down a chronically-unhealthy metrics endpoint "
            f"rotates `*.trycloudflare.com` URLs every ~10 minutes "
            f"forever.",
        )
        # The cool-down counter advanced and the next-allowed window
        # is in the future.
        self.assertEqual(self.server._tunnel_force_restart_count, 1)
        self.assertGreater(
            self.server._tunnel_force_restart_next_allowed,
            time.monotonic(),
        )

    async def test_cooldown_resets_after_sustained_healthy(self) -> None:
        """A long healthy streak after a force-restart resets the ladder.

        After a force-restart the cool-down counter advances.  Once
        the replacement cloudflared has been *continuously* healthy
        for at least :data:`_TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY`
        seconds, an unrelated future flake should not inherit the
        elevated cool-down — the counter must reset to 0.
        """
        # Pre-load the post-force-restart state.
        self.server._tunnel_force_restart_count = 3
        self.server._tunnel_force_restart_next_allowed = (
            time.monotonic() - 1
        )
        # Make the replacement appear "started long ago and healthy
        # ever since" — long enough to satisfy the reset threshold.
        self.server._tunnel_started_at = (
            time.monotonic()
            - _TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY
            - 5
        )
        ws_mod._probe_tunnel_ready = lambda _port: True  # type: ignore[assignment]

        await self.server._check_and_restart_tunnel()

        self.assertEqual(self.server._tunnel_force_restart_count, 0)
        self.assertEqual(
            self.server._tunnel_force_restart_next_allowed, 0.0,
        )

    async def test_cooldown_schedule_is_exponential(self) -> None:
        """Each consecutive force-restart doubles the cool-down.

        Drives the cool-down arithmetic in isolation: pre-seed
        ``_tunnel_force_restart_count`` and ``_tunnel_unhealthy_ticks``
        at the limit, clear the cool-down (so this tick can fire),
        run one ``_check_and_restart_tunnel``, and check the new
        ``_tunnel_force_restart_next_allowed`` matches the expected
        exponential delay.
        """
        for previous_count, expected_delay in (
            (0, _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL),
            (1, _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL * 2),
            (2, _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL * 4),
            (10, _TUNNEL_FORCE_RESTART_COOLDOWN_MAX),  # capped
        ):
            with self.subTest(previous_count=previous_count):
                self.server._tunnel_force_restart_count = previous_count
                self.server._tunnel_force_restart_next_allowed = 0.0
                self.server._tunnel_unhealthy_ticks = (
                    _TUNNEL_UNHEALTHY_LIMIT_QUICK - 1
                )
                # Need a live proc the limit-tick can force-kill; the
                # fake_restart in setUp recreates one each time.
                if (
                    self.server._tunnel_proc is None
                    or self.server._tunnel_proc.poll() is not None
                ):
                    new_proc = subprocess.Popen(
                        ["sleep", "30"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                    self._procs.append(new_proc)
                    self.server._tunnel_proc = new_proc
                self.server._tunnel_started_at = (
                    time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
                )

                before = time.monotonic()
                await self.server._check_and_restart_tunnel()
                after = time.monotonic()

                self.assertEqual(
                    self.server._tunnel_force_restart_count,
                    previous_count + 1,
                )
                self.assertGreaterEqual(
                    self.server._tunnel_force_restart_next_allowed,
                    before + expected_delay - 0.5,
                )
                self.assertLessEqual(
                    self.server._tunnel_force_restart_next_allowed,
                    after + expected_delay + 0.5,
                )


if __name__ == "__main__":
    unittest.main()
