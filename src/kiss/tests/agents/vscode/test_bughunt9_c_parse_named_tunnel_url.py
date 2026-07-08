# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: banner/doc hosts must never be parsed as tunnel URLs.

``cloudflared`` prints update notices and documentation links to stderr
(e.g. ``https://developers.cloudflare.com/...`` and
``https://github.com/cloudflare/cloudflared``).  The named-tunnel URL
parser used to return ANY non-local ``https?://`` host, so those banner
hosts leaked into ``~/.kiss/remote-url.json`` and the ntfy message board
as the public tunnel URL.
"""

from __future__ import annotations

import unittest

from kiss.agents.vscode.web_server import _parse_named_tunnel_url


class TestParseNamedTunnelUrlRejectsBannerHosts(unittest.TestCase):
    """_parse_named_tunnel_url must skip cloudflared banner/doc hosts."""

    def test_update_notice_line_returns_none(self) -> None:
        """The developers.cloudflare.com update notice is not a tunnel URL."""
        line = (
            "2024-01-01T00:00:00Z INF Update available: see "
            "https://developers.cloudflare.com/cloudflare-one/connections/"
        )
        self.assertIsNone(_parse_named_tunnel_url(line, None))

    def test_github_repo_line_returns_none(self) -> None:
        """The github.com repo link banner is not a tunnel URL."""
        line = (
            "INF Please report issues at "
            "https://github.com/cloudflare/cloudflared/issues"
        )
        self.assertIsNone(_parse_named_tunnel_url(line, None))

    def test_cloudflare_com_line_returns_none(self) -> None:
        """www.cloudflare.com / cloudflare.com banner hosts are rejected."""
        self.assertIsNone(
            _parse_named_tunnel_url("INF See https://www.cloudflare.com/", None),
        )
        self.assertIsNone(
            _parse_named_tunnel_url("INF See https://cloudflare.com/tos", None),
        )

    def test_real_hostname_still_returned(self) -> None:
        """A genuine public hostname line still yields the URL."""
        self.assertEqual(
            _parse_named_tunnel_url(
                "INF https://myhost.trycloudflare.com registered", None,
            ),
            "https://myhost.trycloudflare.com",
        )

    def test_registered_line_returns_configured_url(self) -> None:
        """A registered-connection line returns the configured URL."""
        self.assertEqual(
            _parse_named_tunnel_url(
                "INF Registered tunnel connection connIndex=0",
                "https://kiss.example.com",
            ),
            "https://kiss.example.com",
        )

    def test_localhost_still_rejected(self) -> None:
        """Local URLs remain rejected (pre-existing behavior)."""
        self.assertIsNone(
            _parse_named_tunnel_url("INF http://127.0.0.1:20241/metrics", None),
        )


if __name__ == "__main__":
    unittest.main()
