# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): OAuth error redirects hang ``sorcar mcp auth``.

When the user denies authorization, the OAuth 2.1 authorization server
redirects to the callback with ``?error=access_denied`` (and optionally
``error_description``) instead of ``?code=...`` — RFC 6749 §4.1.2.1.
``_OAuthCallbackServer.do_GET`` only recognized redirects carrying a
``code`` parameter, so a denial was answered with **404** (the user's
browser showed nothing useful) and the done event was never set:
``sorcar mcp auth`` then sat for the full 300-second ``AUTH_TIMEOUT``
before reporting a generic timeout, hiding the real reason.

The redirect must be acknowledged with a page, and ``wait()`` must fail
*immediately* with the server-supplied error.
"""

from __future__ import annotations

import time
import urllib.request

import pytest

from kiss.ui.cli.mcp_cli import _OAuthCallbackServer


def test_error_redirect_fails_fast_with_reason() -> None:
    """A denial redirect is acknowledged and surfaces its error at once."""
    server = _OAuthCallbackServer()
    try:
        url = (
            f"http://localhost:{server.port}/callback"
            "?error=access_denied&error_description=User+denied+access"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            # The browser must get a real page, not a 404.
            assert resp.status == 200
        start = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            server.wait(timeout=5)
        elapsed = time.monotonic() - start
        # Fail fast — not by exhausting the timeout.
        assert elapsed < 2, f"wait() blocked for {elapsed:.1f}s on a denial"
        assert "access_denied" in str(excinfo.value)
    finally:
        server.close()


def test_error_probe_never_clobbers_captured_code() -> None:
    """A success redirect followed by an error request still succeeds."""
    server = _OAuthCallbackServer()
    try:
        with urllib.request.urlopen(
            f"http://localhost:{server.port}/callback?code=ok42&state=s1",
            timeout=10,
        ) as resp:
            assert resp.status == 200
        with urllib.request.urlopen(
            f"http://localhost:{server.port}/callback?error=access_denied",
            timeout=10,
        ):
            pass
        code, state = server.wait(timeout=10)
        assert code == "ok42"
        assert state == "s1"
    finally:
        server.close()
