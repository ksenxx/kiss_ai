# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: cloudflared must escape the systemd service cgroup.

Production incident this file guards against: ``kiss-web`` runs as the
``kiss-web.service`` systemd user unit, whose default
``KillMode=control-group`` makes ``systemctl --user restart kiss-web``
SIGTERM **every** process in the service's control group — including
the spawned ``cloudflared``.  ``start_new_session=True`` creates a new
process *session* but cannot leave the *cgroup*, so the quick-tunnel
died on every install-triggered restart, the next daemon logged
"cloudflared pidfile points to dead pid; ignoring", and a brand-new
``*.trycloudflare.com`` hostname was minted each time — breaking every
previously shared/bookmarked address (``DNS_PROBE_FINISHED_NXDOMAIN``).

The fix launches cloudflared through ``systemd-run --user --scope
--collect`` when (and only when) the daemon itself runs inside a
systemd service cgroup: the payload is exec'd in-place into its own
transient ``run-*.scope`` unit outside the service cgroup, where the
service restart cannot reach it, so pidfile adoption finally works.

Branch-coverage notes:

* ``_current_cgroup()``'s ``OSError`` branch (no ``/proc`` — macOS,
  BSD) is unreachable on the Linux hosts these tests run on without
  faking the filesystem; the equivalent behavior is exercised through
  ``_cloudflared_launch_prefix(cgroup="")`` instead.
* Exactly one of the two ``shutil.which("systemd-run")`` outcomes is
  reachable per machine; ``test_prefix_for_service_cgroup`` asserts
  the outcome matching the real machine state.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server import web_server as ws
from kiss.server.web_server import RemoteAccessServer

_SERVICE_CGROUP = (
    "0::/user.slice/user-1001.slice/user@1001.service"
    "/app.slice/kiss-web.service\n"
)
_SESSION_CGROUP = "0::/user.slice/user-1001.slice/session-42.scope\n"
# A transient scope *under* the user manager: contains the substring
# "user@1001.service" but its own unit ends in ".scope", so it must
# NOT trigger the prefix (regression guard for substring matching).
_TRANSIENT_SCOPE_CGROUP = (
    "0::/user.slice/user-1001.slice/user@1001.service"
    "/app.slice/run-r0123456789abcdef.scope\n"
)

_V1_SERVICE_CGROUP = (
    "12:memory:/user.slice/user-1001.slice/user@1001.service"
    "/app.slice/kiss-web.service\n"
    "5:cpu,cpuacct:/user.slice/user-1001.slice/user@1001.service"
    "/app.slice/kiss-web.service\n"
    "1:name=systemd:/user.slice/user-1001.slice/user@1001.service"
    "/app.slice/kiss-web.service\n"
)

_FAKE_CLOUDFLARED = "#!/bin/sh\nexec sleep 300\n"

# /proc-based cgroup inspection only exists on Linux; the pure string
# tests above cover the helper's branches everywhere else.
_HAS_PROC = Path("/proc/self/cgroup").exists()


def _read_cgroup(pid: int) -> str:
    """Return the cgroup path of *pid* from ``/proc/PID/cgroup``."""
    return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip()


class TestLaunchPrefixSelection(unittest.TestCase):
    """``_cloudflared_launch_prefix`` fires only inside a service cgroup."""

    def test_prefix_for_service_cgroup(self) -> None:
        """A ``*.service`` leaf yields the systemd-run prefix (if installed)."""
        prefix = ws._cloudflared_launch_prefix(cgroup=_SERVICE_CGROUP)
        if shutil.which("systemd-run") is None:
            self.assertEqual(prefix, [])
        else:
            self.assertEqual(prefix, list(ws._SYSTEMD_RUN_SCOPE_PREFIX))

    def test_no_prefix_for_session_scope(self) -> None:
        """An interactive session scope must not get the prefix."""
        self.assertEqual(
            ws._cloudflared_launch_prefix(cgroup=_SESSION_CGROUP), [],
        )

    def test_no_prefix_for_transient_scope_under_user_manager(self) -> None:
        """``user@UID.service`` mid-path must not count as a service."""
        self.assertEqual(
            ws._cloudflared_launch_prefix(cgroup=_TRANSIENT_SCOPE_CGROUP),
            [],
        )

    def test_no_prefix_for_empty_cgroup(self) -> None:
        """Empty content (non-Linux ``_current_cgroup``) yields no prefix."""
        self.assertEqual(ws._cloudflared_launch_prefix(cgroup=""), [])

    def test_prefix_for_cgroup_v1_multiline(self) -> None:
        """cgroup v1's ``N:controller:/path`` lines are parsed correctly."""
        prefix = ws._cloudflared_launch_prefix(cgroup=_V1_SERVICE_CGROUP)
        if shutil.which("systemd-run") is None:
            self.assertEqual(prefix, [])
        else:
            self.assertEqual(prefix, list(ws._SYSTEMD_RUN_SCOPE_PREFIX))

    def test_current_cgroup_matches_proc_contents(self) -> None:
        """``_current_cgroup`` returns the raw cgroup lines (or '')."""
        text = ws._current_cgroup()
        if not Path("/proc/self/cgroup").exists():
            self.assertEqual(text, "")
            return
        self.assertTrue(text)
        for line in text.strip().splitlines():
            # Both cgroup v2 ("0::/path") and v1 ("N:controller:/path")
            # lines have at least two colon separators.
            self.assertGreaterEqual(line.count(":"), 2, line)

    def test_default_argument_reads_real_cgroup(self) -> None:
        """``cgroup=None`` inspects this very process's cgroup."""
        self.assertEqual(
            ws._cloudflared_launch_prefix(),
            ws._cloudflared_launch_prefix(cgroup=ws._current_cgroup()),
        )


class _SpawnHarness(unittest.TestCase):
    """Shared PATH/pidfile sandbox for the ``_spawn_cloudflared`` tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        fake = tmp / "cloudflared"
        fake.write_text(_FAKE_CLOUDFLARED, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self._old_path = os.environ.get("PATH", "")
        # Keep system dirs so /bin/sh, sleep, systemd-run and the
        # fallback-test's ``false`` all resolve; the fake dir shadows
        # any real cloudflared.
        os.environ["PATH"] = f"{tmp}{os.pathsep}{self._old_path}"
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = tmp / "cloudflared.pid"
        self.server = RemoteAccessServer(use_tunnel=False)

    def tearDown(self) -> None:
        proc = self.server._tunnel_proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc is not None and proc.stderr is not None:
            proc.stderr.close()
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        os.environ["PATH"] = self._old_path
        self._tmp.cleanup()

    def _assert_spawned_and_recorded(self) -> subprocess.Popen[str]:
        """Assert a live tunnel proc whose pid matches the pidfile."""
        proc = self.server._tunnel_proc
        assert proc is not None, "no tunnel process was recorded"
        self.assertIsNone(proc.poll(), "fake cloudflared is not running")
        data = ws._load_cloudflared_pidfile()
        assert data is not None, "pidfile was not written"
        self.assertEqual(data["pid"], proc.pid)
        self.assertEqual(
            data["metrics_port"], self.server._tunnel_metrics_port,
        )
        return proc


class TestSpawnEscapesServiceCgroup(_SpawnHarness):
    """The real spawn path places cloudflared outside our cgroup."""

    def test_scope_prefix_moves_payload_out_of_our_cgroup(self) -> None:
        """systemd-run --scope runs the payload in a separate scope unit.

        Simulates being inside ``kiss-web.service`` by feeding the
        prefix computed for a service cgroup; asserts the payload's pid
        equals the ``Popen`` pid (pidfile adoption relies on this) and
        its cgroup differs from ours (a ``systemctl restart`` of our
        unit could not kill it).
        """
        prefix = ws._cloudflared_launch_prefix(cgroup=_SERVICE_CGROUP)
        if not prefix:
            self.skipTest("systemd-run not installed on this machine")
        probe = subprocess.run(
            [*prefix, "true"], capture_output=True, timeout=30,
        )
        if probe.returncode != 0:
            self.skipTest(
                "systemd-run --user --scope does not work here: "
                + probe.stderr.decode(errors="replace"),
            )
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            launch_prefix=prefix,
        )
        proc = self._assert_spawned_and_recorded()
        payload_cgroup = _read_cgroup(proc.pid)
        our_cgroup = _read_cgroup(os.getpid())
        self.assertNotEqual(
            payload_cgroup, our_cgroup,
            "cloudflared stayed in the spawner's cgroup — a systemd "
            "service restart would still kill it and rotate the "
            "public tunnel URL",
        )
        self.assertIn(".scope", payload_cgroup)
        proc.terminate()
        self.assertEqual(proc.wait(timeout=10), -15)

    def test_plain_spawn_without_prefix(self) -> None:
        """``launch_prefix=[]`` spawns cloudflared directly (non-systemd)."""
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            launch_prefix=[],
        )
        self._assert_spawned_and_recorded()


class TestSpawnPrefixFallbacks(_SpawnHarness):
    """A broken cgroup-escape prefix must never cost us the tunnel."""

    def test_prefix_binary_missing_falls_back_to_plain_spawn(self) -> None:
        """FileNotFoundError on the prefix binary drops the prefix."""
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            launch_prefix=["/nonexistent/kiss-no-such-systemd-run"],
        )
        proc = self._assert_spawned_and_recorded()
        if _HAS_PROC:
            # The fallback spawn is a direct child in OUR cgroup.
            self.assertEqual(
                _read_cgroup(proc.pid), _read_cgroup(os.getpid()),
            )

    def test_prefix_immediate_exit_falls_back_to_plain_spawn(self) -> None:
        """A prefix that exits at once (broken user manager) is dropped."""
        # ``false`` swallows the cloudflared argv and exits 1 within the
        # fail-fast window — exactly how a broken ``systemd-run`` fails.
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            launch_prefix=["false"],
        )
        self._assert_spawned_and_recorded()

    def test_missing_cloudflared_without_prefix_still_raises(self) -> None:
        """With no prefix, a missing cloudflared propagates as before."""
        os.environ["PATH"] = "/nonexistent-empty-dir"
        with self.assertRaises(FileNotFoundError):
            self.server._spawn_cloudflared(
                ["--url", "https://localhost:1", "--no-tls-verify"],
                launch_prefix=[],
            )

    def test_missing_cloudflared_after_prefixed_failures_raises(self) -> None:
        """Prefixed exits followed by a missing cloudflared re-raise cleanly.

        The failed prefixed proc retained in ``last_proc`` must be
        reaped before the ``FileNotFoundError`` propagates; the
        ``-W error::ResourceWarning`` verification run catches any
        stderr pipe left to the garbage collector here.
        """
        false_bin = shutil.which("false")
        assert false_bin is not None
        os.environ["PATH"] = "/nonexistent-empty-dir"
        with self.assertRaises(FileNotFoundError):
            self.server._spawn_cloudflared(
                ["--url", "https://localhost:1", "--no-tls-verify"],
                launch_prefix=[false_bin],
            )


class TestSpawnRetryOnImmediateExit(_SpawnHarness):
    """Immediate exits without a prefix still use the port-retry path."""

    def test_all_attempts_fail_records_last_proc(self) -> None:
        """A cloudflared that always dies at once exhausts the retries."""
        fake = Path(self._tmp.name) / "cloudflared"
        fake.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
        started = time.monotonic()
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            retries=2,
            launch_prefix=[],
        )
        self.assertLess(time.monotonic() - started, 30)
        proc = self.server._tunnel_proc
        assert proc is not None
        self.assertEqual(proc.returncode, 7)
        if proc.stderr is not None:
            proc.stderr.close()

    def test_retries_one_still_falls_back_from_broken_prefix(self) -> None:
        """Prefix failures must not consume the single retry attempt."""
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            retries=1,
            launch_prefix=["false"],
        )
        self._assert_spawned_and_recorded()

    def test_single_failure_keeps_prefix_on_next_attempt(self) -> None:
        """One immediate exit (e.g. port TOCTOU) must not lose the escape.

        The fake cloudflared fails exactly once, then stays up.  With a
        working ``systemd-run --scope`` prefix the second — prefixed —
        attempt must succeed OUTSIDE our cgroup; dropping the prefix
        after a single failure would silently reintroduce the
        URL-rotation-on-restart bug for that daemon lifetime.
        """
        prefix = ws._cloudflared_launch_prefix(cgroup=_SERVICE_CGROUP)
        if not prefix or not _HAS_PROC:
            self.skipTest("systemd-run not installed on this machine")
        probe = subprocess.run(
            [*prefix, "true"], capture_output=True, timeout=30,
        )
        if probe.returncode != 0:
            self.skipTest("systemd-run --user --scope does not work here")
        marker = Path(self._tmp.name) / "failed-once"
        fake = Path(self._tmp.name) / "cloudflared"
        fake.write_text(
            "#!/bin/sh\n"
            f'if [ ! -e "{marker}" ]; then touch "{marker}"; exit 1; fi\n'
            "exec sleep 300\n",
            encoding="utf-8",
        )
        self.server._spawn_cloudflared(
            ["--url", "https://localhost:1", "--no-tls-verify"],
            launch_prefix=prefix,
        )
        proc = self._assert_spawned_and_recorded()
        self.assertNotEqual(
            _read_cgroup(proc.pid), _read_cgroup(os.getpid()),
            "the cgroup-escape prefix was dropped after a single "
            "immediate exit — a one-off metrics-port TOCTOU would "
            "permanently reintroduce tunnel-URL rotation",
        )


class _ShimHarness(unittest.TestCase):
    """A live stderr=PIPE child standing in for cloudflared."""

    def setUp(self) -> None:
        self.child: subprocess.Popen[str] = subprocess.Popen(
            ["sleep", "300"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.shim: subprocess.Popen[bytes] | None = None

    def tearDown(self) -> None:
        for proc in (self.shim, self.child):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        if self.child.stderr is not None:
            self.child.stderr.close()

    def _assert_live_shim(self) -> subprocess.Popen[bytes]:
        assert self.shim is not None, "no drain shim was spawned"
        self.assertIsNone(self.shim.poll(), "drain shim is not running")
        return self.shim


class TestDrainShimEscapesServiceCgroup(_ShimHarness):
    """The stderr drain shim must escape the service cgroup too.

    If the shim stays in ``kiss-web.service``, ``systemctl restart``
    kills it, the pipe's last read end closes, and the escaped
    cloudflared dies of SIGPIPE on its next stderr write — rotating
    the URL through the back door.
    """

    def test_shim_lands_outside_our_cgroup(self) -> None:
        """A prefixed shim runs in its own transient scope."""
        prefix = ws._cloudflared_launch_prefix(cgroup=_SERVICE_CGROUP)
        if not prefix or not _HAS_PROC:
            self.skipTest("systemd-run not installed on this machine")
        probe = subprocess.run(
            [*prefix, "true"], capture_output=True, timeout=30,
        )
        if probe.returncode != 0:
            self.skipTest("systemd-run --user --scope does not work here")
        self.shim = RemoteAccessServer._spawn_stderr_drain_shim(
            self.child, launch_prefix=prefix,
        )
        shim = self._assert_live_shim()
        self.assertNotEqual(
            _read_cgroup(shim.pid), _read_cgroup(os.getpid()),
            "the drain shim stayed in the spawner's cgroup — a systemd "
            "service restart would kill it and SIGPIPE the escaped "
            "cloudflared",
        )

    def test_prefixed_shim_immediate_exit_falls_back_to_plain_cat(
        self,
    ) -> None:
        """A prefix that dies at once must still leave a live drainer."""
        self.shim = RemoteAccessServer._spawn_stderr_drain_shim(
            self.child, launch_prefix=["false"],
        )
        self._assert_live_shim()

    def test_missing_prefix_binary_falls_back_to_plain_cat(self) -> None:
        """OSError on the prefix binary must still leave a live drainer."""
        self.shim = RemoteAccessServer._spawn_stderr_drain_shim(
            self.child,
            launch_prefix=["/nonexistent/kiss-no-such-systemd-run"],
        )
        self._assert_live_shim()

    def test_plain_shim_without_prefix(self) -> None:
        """``launch_prefix=[]`` spawns the plain cat directly."""
        self.shim = RemoteAccessServer._spawn_stderr_drain_shim(
            self.child, launch_prefix=[],
        )
        self._assert_live_shim()

    def test_no_stderr_pipe_returns_none(self) -> None:
        """A child without a stderr pipe yields no shim."""
        plain = subprocess.Popen(
            ["sleep", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            self.assertIsNone(
                RemoteAccessServer._spawn_stderr_drain_shim(
                    plain, launch_prefix=[],
                ),
            )
        finally:
            plain.kill()
            plain.wait()


if __name__ == "__main__":
    unittest.main()
