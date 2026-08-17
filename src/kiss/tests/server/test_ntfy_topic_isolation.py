# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: tests must never pollute the production ntfy topic.

The web server advertises its Cloudflare quick-tunnel URL by POSTing to
a machine-stable ntfy.sh topic.  Historically ``_get_machine_topic()``
derived that topic from hostname+MAC only, so a test running with an
isolated temporary ``KISS_HOME`` resolved to the *same* topic as the
production daemon, and ``_post_url_if_changed()`` had no per-server
endpoint injection — any test driving the tunnel-restart path made a
real HTTP POST to the real https://ntfy.sh on the production topic,
polluting it with fixture URLs.

These tests pin both fixes:

* topic isolation — a non-default ``KISS_HOME`` salts the identity
  hash, so isolated processes can never compute the production topic,
  while the default-home derivation (and the stored ``ntfy_topic``
  file precedence) stays byte-identical for existing installs;
* endpoint injection — ``RemoteAccessServer(ntfy_base_url=...)`` routes
  the post to a local emulator end-to-end through the real
  tunnel-restart path with a fake ``cloudflared``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.core.vscode_config import save_config
from kiss.server.web_server import RemoteAccessServer, _get_machine_topic
from kiss.tests.server._ntfy_emulator import NtfyServerContext


def _production_default_home_topic() -> str:
    """Compute the topic a default-home production daemon derives.

    Mirrors the documented derivation formula for the default
    ``~/.kiss`` home: SHA-256 of ``{hostname}:{MAC}``, truncated.
    """
    identity = f"{platform.node()}:{uuid.getnode()}"
    return "kiss-" + hashlib.sha256(identity.encode()).hexdigest()[:32]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class TestTopicIsolationPerKissHome(unittest.TestCase):
    """Non-default KISS_HOMEs must never yield the production topic."""

    def setUp(self) -> None:
        self._old_kiss_home = os.environ.get("KISS_HOME")
        self._tmp_homes: list[str] = []

    def tearDown(self) -> None:
        if self._old_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._old_kiss_home
        for d in self._tmp_homes:
            shutil.rmtree(d, ignore_errors=True)

    def _topic_under_temp_home(self) -> str:
        home = tempfile.mkdtemp(prefix="kiss-ntfy-home-")
        self._tmp_homes.append(home)
        os.environ["KISS_HOME"] = home
        return _get_machine_topic()

    def test_isolated_homes_never_compute_production_topic(self) -> None:
        """Two temp KISS_HOMEs differ from the default-home topic and
        from each other."""
        production_topic = _production_default_home_topic()
        topic_a = self._topic_under_temp_home()
        topic_b = self._topic_under_temp_home()
        self.assertNotEqual(topic_a, production_topic)
        self.assertNotEqual(topic_b, production_topic)
        self.assertNotEqual(topic_a, topic_b)

    def test_stored_topic_file_wins(self) -> None:
        """A pre-existing ``KISS_HOME/ntfy_topic`` file overrides the
        derivation, so persisted topics survive the salting change."""
        home = tempfile.mkdtemp(prefix="kiss-ntfy-home-")
        self._tmp_homes.append(home)
        (Path(home) / "ntfy_topic").write_text(
            "kiss-stored-topic-abc123\n", encoding="utf-8",
        )
        os.environ["KISS_HOME"] = home
        self.assertEqual(_get_machine_topic(), "kiss-stored-topic-abc123")

    def test_default_home_derivation_formula_unchanged(self) -> None:
        """With KISS_HOME unset and a faked ``$HOME``, the default-home
        branch still derives the exact historical hostname+MAC hash, so
        every existing production install keeps its topic."""
        fake_home = tempfile.mkdtemp(prefix="kiss-ntfy-fakehome-")
        self._tmp_homes.append(fake_home)
        (Path(fake_home) / ".kiss").mkdir()
        old_home = os.environ.get("HOME")
        os.environ.pop("KISS_HOME", None)
        os.environ["HOME"] = fake_home
        try:
            topic = _get_machine_topic()
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
        self.assertEqual(topic, _production_default_home_topic())


class TestTunnelRestartPostsToInjectedEndpoint(IsolatedAsyncioTestCase):
    """End-to-end leak reproduction through the real restart path.

    A server with an isolated KISS_HOME, tunneling enabled, a fake
    ``cloudflared`` on PATH, and a local ntfy emulator injected must
    deliver its URL post to the emulator — on a topic different from
    the production machine topic.  Before the fix this test failed at
    construction (no ``ntfy_base_url`` parameter existed): the post
    went to the real https://ntfy.sh on the production topic.
    """

    async def asyncSetUp(self) -> None:
        self._kiss_home = tempfile.mkdtemp(prefix="kiss-ntfy-e2e-home-")
        self._old_kiss_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = self._kiss_home
        save_config({"remote_password": "test-pw"})
        self.ntfy = NtfyServerContext()
        self._tmpdir = tempfile.mkdtemp(prefix="kiss-ntfy-e2e-bin-")
        self._old_path = os.environ.get("PATH", "")
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            use_tunnel=False,
            ntfy_base_url=self.ntfy.base_url,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        if self.server._tunnel_proc is not None:
            self.server._tunnel_proc.terminate()
            try:
                self.server._tunnel_proc.wait(timeout=5)
            except Exception:
                self.server._tunnel_proc.kill()
            self.server._tunnel_proc = None
        await self.server.stop_async()
        self.ntfy.stop()
        if self._old_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._old_kiss_home
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._kiss_home, ignore_errors=True)

    async def test_post_lands_on_emulator_not_production_topic(self) -> None:
        """The restart-path post reaches the injected emulator on the
        isolated (non-production) topic."""
        cf = os.path.join(self._tmpdir, "cloudflared")
        with open(cf, "w") as f:
            f.write(
                "#!/bin/bash\n"
                'echo "INF https://isolation-e2e.trycloudflare.com" >&2\n'
                "sleep 60\n"
            )
        os.chmod(cf, 0o755)
        os.environ["PATH"] = self._tmpdir + ":" + self._old_path

        dead: subprocess.Popen[str] = subprocess.Popen(["true"], text=True)
        dead.wait()
        self.server._tunnel_proc = dead
        self.server.use_tunnel = True

        await self.server._check_and_restart_tunnel()

        self.assertEqual(
            self.server._active_url,
            "https://isolation-e2e.trycloudflare.com",
        )
        self.assertEqual(len(self.ntfy.posts), 1)
        topic, body, _headers = self.ntfy.posts[0]
        self.assertEqual(body, "https://isolation-e2e.trycloudflare.com")
        self.assertEqual(topic, _get_machine_topic())
        self.assertNotEqual(topic, _production_default_home_topic())


if __name__ == "__main__":
    unittest.main()
