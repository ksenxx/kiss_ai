# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local Slack OAuth emulation shared by the Slack sign-in tests.

``SlackOAuthState`` answers Slack's ``oauth.v2.access`` endpoint the way
Slack does for a PKCE public client with a localhost redirect: the code
exchange nests a rotating user token under ``authed_user``; the refresh
grant answers with top-level fields.  ``SlackApiServer`` is a real
loopback HTTP server exposing that endpoint plus a minimal Web API.
``sign_in`` drives ``authenticate_slack()`` / ``finish_slack_auth()``
end to end by replaying the browser's redirect to the loopback port.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlsplit

import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server

CLIENT_ID = "kiss-test-slack-client"


class SlackOAuthState:
    """Token-endpoint behaviour and a record of every grant it received."""

    def __init__(self, expires_in: int = 43200) -> None:
        self.expires_in = expires_in
        self.forms: list[dict[str, str]] = []
        self.issued = 0
        self.refresh_token = ""
        self.challenge = ""
        # When True the next issued access token ends in ``-invalid``, so
        # the emulated ``auth.test`` rejects it.
        self.invalid_next = False
        # When True the code exchange answers without a user token.
        self.omit_user_token = False

    def _issue(self) -> tuple[str, str]:
        self.issued += 1
        suffix = "-invalid" if self.invalid_next else ""
        self.invalid_next = False
        self.refresh_token = f"xoxe-1-refresh-{self.issued}"
        return f"xoxp-access-{self.issued}{suffix}", self.refresh_token

    def answer(self, body: str) -> dict[str, Any]:
        """Answer one ``oauth.v2.access`` POST.

        Args:
            body: The url-encoded request body.

        Returns:
            Slack's JSON answer (``ok: false`` on any protocol error).
        """
        form = {k: v[0] for k, v in parse_qs(body).items()}
        self.forms.append(form)
        if "client_secret" in form or form.get("client_id") != CLIENT_ID:
            return {"ok": False, "error": "invalid_client"}
        if form.get("grant_type") == "refresh_token":
            if form.get("refresh_token") != self.refresh_token:
                return {"ok": False, "error": "invalid_refresh_token"}
            access, refresh = self._issue()
            return {
                "ok": True,
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": self.expires_in,
                "token_type": "user",
            }
        verifier = form.get("code_verifier", "")
        digest = hashlib.sha256(verifier.encode()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != self.challenge:
            return {"ok": False, "error": "invalid_code_verifier"}
        if form.get("code") != "test-code":
            return {"ok": False, "error": "invalid_code"}
        access, refresh = self._issue()
        if self.omit_user_token:
            return {"ok": True, "app_id": "A1", "authed_user": {"id": "U1"}}
        return {
            "ok": True,
            "app_id": "A1",
            "authed_user": {
                "id": "U1",
                "scope": "chat:write,channels:read",
                "access_token": access,
                "token_type": "user",
                "refresh_token": refresh,
                "expires_in": self.expires_in,
            },
            "team": {"id": "T1", "name": "KISS"},
        }


class _Handler(BaseHTTPRequestHandler):
    server: Any

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Serve the token endpoint and the Web API."""
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
        auth = self.headers.get("Authorization", "")
        self.server.requests.append((self.path, auth))
        if self.path == "/api/oauth.v2.access":
            data = self.server.oauth.answer(body)
        elif auth.startswith("Bearer xoxp-access-") and not auth.endswith("-invalid"):
            data = {"ok": True, "user_id": "U1", "user": "alice", "team": "KISS", "url": ""}
        else:
            data = {"ok": False, "error": "invalid_auth"}
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


class SlackApiServer(ThreadedHTTPServer):
    """Loopback Slack emulator recording ``(path, Authorization)`` pairs."""

    def __init__(self, oauth: SlackOAuthState) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.oauth = oauth
        self.requests: list[tuple[str, str]] = []
        self.base_url = f"http://127.0.0.1:{self.server_address[1]}"
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut the server down."""
        stop_http_server(self, self._thread)


def start_sign_in(tools: dict[str, Any], oauth: SlackOAuthState) -> dict[str, Any]:
    """Call ``authenticate_slack()`` and remember the PKCE challenge.

    Args:
        tools: The agent's tools by name.
        oauth: The emulated token endpoint (receives the challenge).

    Returns:
        The decoded ``consent_required`` answer.
    """
    started: dict[str, Any] = json.loads(tools["authenticate_slack"]())
    query = parse_qs(urlsplit(started["verification_uri"]).query)
    oauth.challenge = query["code_challenge"][0]
    return started


def sign_in(tools: dict[str, Any], oauth: SlackOAuthState) -> dict[str, Any]:
    """Run the whole browser sign-in: start, approve, finish.

    Args:
        tools: The agent's tools by name.
        oauth: The emulated token endpoint.

    Returns:
        The decoded ``finish_slack_auth()`` answer.
    """
    started = start_sign_in(tools, oauth)
    state = parse_qs(urlsplit(started["verification_uri"]).query)["state"][0]
    # What the user's browser does after they click Allow.
    requests.get(
        "http://127.0.0.1:53682/callback", params={"code": "test-code", "state": state}, timeout=10
    )
    done: dict[str, Any] = json.loads(tools["finish_slack_auth"]())
    return done
