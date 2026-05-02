"""Integration tests for named-tunnel URL configuration.

A Cloudflare named tunnel uses a token (a JWT containing AccountTag,
TunnelID and TunnelSecret) plus a public hostname configured in the
Cloudflare Zero Trust dashboard.  The token does not contain the
hostname, and ``cloudflared tunnel run --token <T>`` does not echo it
to stderr in a parseable form.  The metrics ``/quicktunnel`` endpoint
also returns nothing for named tunnels.  Therefore the only way to
expose a fixed URL for a named tunnel is for the user to supply it
out-of-band: via ``CLOUDFLARE_TUNNEL_URL`` env var, the ``tunnel_url``
key in ``~/.kiss/config.json``, or the ``tunnel_url`` constructor
argument to :class:`RemoteAccessServer`.

These tests verify that pathway end-to-end:

1. Constructor accepts and stores ``tunnel_url``.
2. ``_resolve_tunnel_settings`` reads from env var, then config.
3. ``_start_named_tunnel`` returns the configured URL once the
   subprocess logs a "registered" connection line, instead of the
   placeholder sentinel string.
4. ``_start_named_tunnel`` returns ``None`` if the subprocess exits
   before any registered-connection line appears.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server as ws
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _resolve_tunnel_settings,
)


def _write_fake_cloudflared(tmpdir: Path, stderr_lines: list[str],
                            exit_code: int = 0) -> Path:
    """Write a fake ``cloudflared`` shell script to *tmpdir*.

    The script writes *stderr_lines* to stderr (one per line) and exits
    with *exit_code*.  Used to drive ``_start_named_tunnel`` without a
    real cloudflared binary.
    """
    body = "#!/bin/sh\n"
    for line in stderr_lines:
        body += f"printf '%s\\n' {json.dumps(line)} >&2\n"
    body += f"exit {exit_code}\n"
    script = tmpdir / "cloudflared"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


class TestNamedTunnelUrlConstructor(unittest.TestCase):
    """The RemoteAccessServer constructor must accept tunnel_url."""

    def test_constructor_accepts_tunnel_url(self) -> None:
        """tunnel_url is stored on the instance."""
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy-token",
            tunnel_url="https://kiss.example.com",
        )
        self.assertEqual(srv.tunnel_url, "https://kiss.example.com")

    def test_default_tunnel_url_is_none(self) -> None:
        """Without tunnel_url, the attribute defaults to None."""
        srv = RemoteAccessServer(use_tunnel=False)
        self.assertIsNone(srv.tunnel_url)


class TestStartNamedTunnelUsesConfiguredUrl(unittest.TestCase):
    """_start_named_tunnel must return self.tunnel_url when set."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)
        # Prepend tmpdir to PATH so subprocess.Popen finds our fake.
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self._tmpdir}{os.pathsep}{self._old_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        self._tmp.cleanup()

    def test_returns_configured_url_after_connection_registered(self) -> None:
        """When a 'Connection registered' line appears, return tunnel_url."""
        _write_fake_cloudflared(
            self._tmpdir,
            [
                "INF Starting tunnel tunnelID=abc",
                "INF Registered tunnel connection connIndex=0 ip=1.2.3.4",
            ],
        )
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy-token",
            tunnel_url="https://kiss.example.com",
        )
        try:
            url = srv._start_named_tunnel()
            self.assertEqual(url, "https://kiss.example.com")
        finally:
            srv._stop_tunnel()

    def test_returns_none_when_subprocess_exits_without_registration(
        self,
    ) -> None:
        """Subprocess that dies before any 'registered' line returns None."""
        _write_fake_cloudflared(
            self._tmpdir,
            ["ERR failed to authenticate"],
            exit_code=1,
        )
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy-token",
            tunnel_url="https://kiss.example.com",
        )
        try:
            url = srv._start_named_tunnel()
            self.assertIsNone(url)
        finally:
            srv._stop_tunnel()

    def test_without_tunnel_url_returns_sentinel(self) -> None:
        """Existing behavior preserved when tunnel_url is not configured."""
        _write_fake_cloudflared(
            self._tmpdir,
            [
                "INF Starting tunnel tunnelID=abc",
                "INF Registered tunnel connection connIndex=0",
            ],
        )
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy-token",
        )
        try:
            url = srv._start_named_tunnel()
            self.assertIsNotNone(url)
            assert url is not None
            self.assertIn("named tunnel running", url)
        finally:
            srv._stop_tunnel()


class TestResolveTunnelSettings(unittest.TestCase):
    """_resolve_tunnel_settings reads env var first, config as fallback."""

    def setUp(self) -> None:
        # Snapshot env vars and config loader so we can restore.
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in ("CLOUDFLARE_TUNNEL_TOKEN", "CLOUDFLARE_TUNNEL_URL")
        }
        self._saved_loader = ws.load_config

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        ws.load_config = self._saved_loader  # type: ignore[assignment]

    def test_env_var_overrides_config(self) -> None:
        """CLOUDFLARE_TUNNEL_URL env var wins over config file value."""
        os.environ["CLOUDFLARE_TUNNEL_TOKEN"] = "from-env"
        os.environ["CLOUDFLARE_TUNNEL_URL"] = "https://from-env"
        ws.load_config = lambda: {  # type: ignore[assignment]
            "tunnel_token": "from-cfg",
            "tunnel_url": "https://from-cfg",
        }
        token, url = _resolve_tunnel_settings()
        self.assertEqual(token, "from-env")
        self.assertEqual(url, "https://from-env")

    def test_falls_back_to_config_when_env_unset(self) -> None:
        """When env var is absent, the config file value is used."""
        ws.load_config = lambda: {  # type: ignore[assignment]
            "tunnel_token": "tok-cfg",
            "tunnel_url": "https://cfg-only",
        }
        token, url = _resolve_tunnel_settings()
        self.assertEqual(token, "tok-cfg")
        self.assertEqual(url, "https://cfg-only")

    def test_returns_none_when_neither_set(self) -> None:
        """Both env vars unset and config empty returns (None, None)."""
        ws.load_config = lambda: {}  # type: ignore[assignment]
        token, url = _resolve_tunnel_settings()
        self.assertIsNone(token)
        self.assertIsNone(url)


if __name__ == "__main__":
    unittest.main()
