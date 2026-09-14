# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Boundary-compatible ``slack_sdk`` WebClient transport.

:class:`MuseWebClient` is a drop-in :class:`slack_sdk.WebClient` whose
token is a Muse surrogate: every Web API call is shipped to the
Muse-auth daemon, which swaps the surrogate for the real ``xoxb-``
token at the network boundary.  The override point is
``BaseClient._perform_urllib_http_request_internal`` — the innermost
transport seam (it receives a fully built ``urllib.request.Request``
and returns ``{"status", "headers", "body"}``) — so all of slack_sdk's
request building (JSON/multipart bodies, auth header, user agent) and
response handling stay intact.

``files_upload_v2``'s second step — POSTing the file bytes to the
pre-signed upload URL that ``files.getUploadURLExternal`` returned —
is ALSO routed through the boundary via the ``_upload_file`` override:
although that request carries no credential, the URL comes from a
server response, so Sentinel must confirm the destination host before
any content leaves the machine.

Kept in its own module so importing :mod:`.client` does not pull in
``slack_sdk``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import ssl as ssl_module
import urllib.request
from typing import Any

from slack_sdk import WebClient
from slack_sdk.web.file_upload_v2_result import FileUploadV2Result

from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    _boundary_call,
)

_CHARSET_RE = re.compile(r"charset=([\w.-]+)", re.IGNORECASE)


class MuseWebClient(WebClient):
    """``slack_sdk`` WebClient that executes at the Muse daemon boundary.

    Construct it with the Muse *service* name (``slack`` or
    ``slack-<workspace>``) and the surrogate token; the agent process
    never holds the real bot token.  Sentinel denials surface as
    ``{"ok": false, "error": "muse_auth_denied", ...}`` API responses,
    so ``SlackResponse.validate()`` raises a normal ``SlackApiError``
    carrying the grant instructions.
    """

    def __init__(self, muse_service: str, surrogate: str, **kwargs: Any) -> None:
        kwargs.setdefault("retry_handlers", [])
        super().__init__(token=surrogate, **kwargs)
        self._muse_service = muse_service

    def _perform_urllib_http_request_internal(
        self, url: str, req: urllib.request.Request
    ) -> dict[str, Any]:
        """Execute one built urllib request through the Muse boundary.

        Args:
            url: Complete request URL (e.g.
                ``https://slack.com/api/chat.postMessage``).
            req: The urllib request slack_sdk built (headers carry the
                surrogate bearer; data carries the JSON or multipart
                body).

        Returns:
            ``{"status": int, "headers": dict, "body": str|bytes}`` in
            the shape ``BaseClient._perform_urllib_http_request``
            expects (bytes only for binary ``application/gzip``
            downloads such as ``admin.analytics.getFile``).

        Raises:
            MuseAuthError: When the daemon cannot execute the request.
        """
        headers = {str(k): str(v) for k, v in req.header_items()}
        raw: Any = req.data or b""
        body = raw.encode() if isinstance(raw, str) else bytes(raw)
        reply = _boundary_call(
            self._muse_service,
            str(req.get_method() or "POST").upper(),
            url,
            headers,
            body,
            float(self.timeout),
        )
        if not reply.get("ok"):
            if reply.get("denied"):
                denial = json.dumps(
                    {
                        "ok": False,
                        "error": "muse_auth_denied",
                        "detail": str(reply.get("error", "")),
                    }
                )
                return {
                    "status": 403,
                    "headers": {"content-type": "application/json;charset=utf-8"},
                    "body": denial,
                }
            raise MuseAuthError(str(reply.get("error", "muse-auth boundary error")))
        resp_headers = dict(reply.get("headers", {}))
        content = base64.b64decode(reply.get("body_b64", ""))
        content_type = next(
            (v for k, v in resp_headers.items() if k.lower() == "content-type"), ""
        )
        if "application/gzip" in content_type.lower():
            return {"status": int(reply["status"]), "headers": resp_headers, "body": content}
        match = _CHARSET_RE.search(content_type)
        charset = match.group(1) if match else "utf-8"
        return {
            "status": int(reply["status"]),
            "headers": resp_headers,
            "body": content.decode(charset, errors="replace"),
        }

    def _upload_file(
        self,
        *,
        url: str,
        data: bytes,
        logger: logging.Logger,
        timeout: int,
        proxy: str | None,
        ssl: ssl_module.SSLContext | None,
    ) -> FileUploadV2Result:
        """Upload ``files_upload_v2`` content through the Muse boundary.

        The upload URL is supplied by Slack's API response, so it must
        pass Sentinel's host allowlist and write policy before the file
        content leaves the machine (the inherited implementation would
        POST it directly with ``urlopen``).  The surrogate bearer rides
        along so the daemon accepts the request; ``files.slack.com``
        ignores the swapped-in credential on the pre-signed URL.

        Args:
            url: Pre-signed upload URL from ``files.getUploadURLExternal``.
            data: The file content bytes.
            logger: Ignored (slack_sdk signature compatibility).
            timeout: Request timeout in seconds.
            proxy: Ignored; the daemon owns the egress path.
            ssl: Ignored; the daemon owns the egress path.

        Returns:
            The upload result (``files_upload_v2`` raises
            ``SlackRequestError`` itself on a non-200 status, including
            Sentinel denials surfaced as 403).

        Raises:
            MuseAuthError: When the daemon cannot execute the request.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        reply = _boundary_call(
            self._muse_service, "POST", url, headers, bytes(data), float(timeout)
        )
        if not reply.get("ok"):
            if reply.get("denied"):
                return FileUploadV2Result(status=403, body=str(reply.get("error", "")))
            raise MuseAuthError(str(reply.get("error", "muse-auth boundary error")))
        content = base64.b64decode(reply.get("body_b64", ""))
        return FileUploadV2Result(
            status=int(reply["status"]), body=content.decode("utf-8", errors="replace")
        )
