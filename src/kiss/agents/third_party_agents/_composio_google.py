# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Workspace access brokered by Composio.

Google only lets a verified OAuth app request Gmail, Drive and similar
scopes from arbitrary users, so KISS does not ship its own Google app.
It uses Composio (MIT-licensed SDK, hosted token custody) instead:

1. ``authenticate_<service>()`` calls ``connected_accounts.link`` and
   hands the user a Composio Connect Link; the user signs in to Google
   and clicks Allow on Composio's managed OAuth app.
2. ``finish_<service>_auth()`` waits for the connected account to
   become ``ACTIVE`` and remembers its ID under
   ``$KISS_HOME/third_party_agents/<service>/composio.json``.
3. API calls go through Composio's proxy (``tools.proxy``), which
   injects and refreshes the Google token server-side.  The token never
   reaches this process: Composio masks it in every API answer.

:class:`ComposioSession` gives REST backends the ``requests`` call shape
and :class:`ComposioHttp` gives ``googleapiclient`` the ``httplib2``
shape, so the Google agents keep their request code unchanged.  JSON
bodies travel as the proxy's ``body``; anything else (Drive multipart
uploads) as its base64 ``binary_body``.

Configuration: ``COMPOSIO_API_KEY`` (or the key saved by
``authenticate_<service>(api_key=...)``); optional
``KISS_COMPOSIO_USER_ID`` (default ``kiss-default``) and
``KISS_COMPOSIO_AUTH_CONFIG_<SERVICE>`` to use a specific auth config
(Google Chat requires one: Composio has no managed Chat app).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import requests
from requests.structures import CaseInsensitiveDict

from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.core.browser_handoff import BROWSER_TAB, open_for_user
from kiss.core.config import kiss_home

# KISS service name -> Composio toolkit slug.
TOOLKITS: dict[str, str] = {
    "gmail": "gmail",
    "google_calendar": "googlecalendar",
    "google_drive": "googledrive",
    "google_docs": "googledocs",
    "google_sheets": "googlesheets",
    "googlechat": "google_chat",
}

# How long finish_<service>_auth() waits for the connection to activate
# before answering "pending".
_FINISH_WAIT_SECONDS = 5.0
# Connected-account states the SDK keeps polling (INACTIVE can recover).
_WAITABLE_STATUSES = ("INITIATED", "INITIALIZING", "INACTIVE")
# Headers the proxy must not receive: Composio injects Authorization,
# and the body/transport headers are rebuilt by the proxy itself.
_DROPPED_HEADERS = {"authorization", "content-length", "host", "accept-encoding", "user-agent"}


def service_dir(service: str) -> Path:
    """Return a Google service's state directory, honoring ``KISS_HOME``.

    Args:
        service: KISS service name (e.g. ``"google_drive"``).

    Returns:
        ``$KISS_HOME/third_party_agents/<service>``.
    """
    return kiss_home() / "third_party_agents" / service


def _state_path(service: str) -> Path:
    """Return the file remembering *service*'s Composio connection."""
    return service_dir(service) / "composio.json"


def _api_key_path() -> Path:
    """Return the file holding a Composio API key saved by an auth tool."""
    return service_dir("google") / "composio_api_key.json"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file, returning ``{}`` when missing or invalid."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def composio_api_key() -> str:
    """Return the Composio project API key.

    Returns:
        ``$COMPOSIO_API_KEY`` when set, else the key saved with
        :func:`save_api_key`, else ``""``.
    """
    env = os.environ.get("COMPOSIO_API_KEY", "").strip()
    return env or str(_read_json(_api_key_path()).get("api_key") or "")


def save_api_key(api_key: str) -> None:
    """Persist a Composio API key (0600) for all Google agents.

    Args:
        api_key: The Composio project API key.
    """
    write_private_file(_api_key_path(), json.dumps({"api_key": api_key.strip()}))


def composio_user_id() -> str:
    """Return the Composio user ID the connections are grouped under."""
    return os.environ.get("KISS_COMPOSIO_USER_ID", "").strip() or "kiss-default"


def connected_account_id(service: str) -> str:
    """Return *service*'s active Composio connected-account ID, or ``""``."""
    return str(_read_json(_state_path(service)).get("connected_account_id") or "")


def _client() -> Any:
    """Build a Composio SDK client from the configured API key.

    Raises:
        RuntimeError: When no API key is configured.
    """
    api_key = composio_api_key()
    if not api_key:
        raise RuntimeError(
            "No Composio API key. Get a project API key (ak_...) at "
            "https://dashboard.composio.dev/~/project/settings/api-keys "
            "(Settings → Project Settings → API Keys) and set COMPOSIO_API_KEY, "
            "or pass it as api_key=... . Note: old consumer keys (ck_...) no "
            "longer work; use a project key starting with ak_."
        )
    from composio import Composio

    return Composio(api_key=api_key)


def _auth_config_id(client: Any, service: str) -> str:
    """Return the auth config to connect *service* through.

    ``KISS_COMPOSIO_AUTH_CONFIG_<SERVICE>`` wins; otherwise the newest
    auth config of the toolkit is reused, and a Composio-managed one is
    created when the project has none.

    Args:
        client: Composio SDK client.
        service: KISS service name.

    Returns:
        The auth config ID.
    """
    override = os.environ.get(f"KISS_COMPOSIO_AUTH_CONFIG_{service.upper()}", "").strip()
    if override:
        return override
    toolkit = TOOLKITS[service]
    items = list(client.auth_configs.list(toolkit_slug=toolkit).items)
    if items:
        return str(max(items, key=lambda item: str(item.created_at or "")).id)
    created = client.auth_configs.create(toolkit, {"type": "use_composio_managed_auth"})
    return str(created.id)


def start_connect(service: str, label: str) -> dict[str, Any]:
    """Create a Composio Connect Link for *service* and hand it to the user.

    Args:
        service: KISS service name.
        label: Human-readable service label.

    Returns:
        A JSON-ready ``consent_required`` answer with the link, or
        ``{"ok": False, "error": ...}``.
    """
    try:
        client = _client()
        request = client.connected_accounts.link(
            composio_user_id(), _auth_config_id(client, service), allow_multiple=True
        )
    except Exception as e:
        return {"ok": False, "error": f"Could not start the {label} connection: {e}"}
    state = _read_json(_state_path(service))
    superseded = str(state.get("pending_id") or "")
    if superseded and superseded != str(request.id):
        # The user may have approved that earlier link: do not leave an
        # active account behind that KISS no longer remembers.
        _delete_quietly(client, superseded)
    state["pending_id"] = str(request.id)
    write_private_file(_state_path(service), json.dumps(state))
    url = str(request.redirect_url or "")
    opened_in = open_for_user(url)
    if opened_in == BROWSER_TAB:
        step_one = (
            "The Google sign-in page is already open in the Browser tab that every "
            "KISS surface has just switched to, so the user is looking at it: do NOT "
            "ask them to open a URL. 1) Call ask_user_question() telling the user to "
            "sign in to Google in that Browser tab, click Allow, and reply here when "
            "done (valid for about 10 minutes); only if they cannot see the page, "
            f"give them {url} to open themselves."
        )
    else:
        step_one = (
            "1) Call ask_user_question() with this exact URL for the user to open in "
            f"their OWN browser if no window appeared: {url} (valid for about 10 "
            "minutes)."
        )
    return {
        "ok": True,
        "status": "consent_required",
        "verification_uri": url,
        "opened_in": opened_in,
        "browser_opened": bool(opened_in),
        "instructions": (
            f"Connect {label} through Composio: the USER signs in to Google and "
            "clicks Allow; you only tell them what to do. Do NOT open the page in "
            "your built-in browser and never ask for the user's Google password or "
            f"2FA code. {step_one} 2) Then call finish_{service}_auth(); if it "
            "returns 'pending', wait a few seconds and call it again."
        ),
    }


def finish_connect(service: str, label: str) -> dict[str, Any]:
    """Wait briefly for the pending connection to become ``ACTIVE``.

    On success the new connected account replaces the previous one,
    which is deleted at Composio.

    Args:
        service: KISS service name.
        label: Human-readable service label.

    Returns:
        ``{"ok": True, ...}`` once active, ``{"ok": False, "status":
        "pending", ...}`` while the user has not finished, or an error.
    """
    state = _read_json(_state_path(service))
    pending = str(state.get("pending_id") or "")
    if not pending:
        return {
            "ok": False,
            "error": f"no sign-in in progress; call authenticate_{service}() first",
        }
    try:
        client = _client()
        account = client.connected_accounts.get(pending)
        status = str(account.status)
        if status in _WAITABLE_STATUSES:
            try:
                account = client.connected_accounts.wait_for_connection(
                    pending, timeout=_FINISH_WAIT_SECONDS
                )
            except Exception as e:
                if "Timeout" in type(e).__name__:
                    return {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not approved yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                raise
            status = str(account.status)
        # Re-read: another authenticate_<service>() call may have started
        # a newer sign-in while this one waited; that one now owns the state.
        state = _read_json(_state_path(service))
        if str(state.get("connected_account_id") or "") == pending:
            # An overlapping finish already recorded this account.
            return {"ok": True, "message": f"{label} connected through Composio."}
        if str(state.get("pending_id") or "") != pending:
            _delete_quietly(client, pending)
            return {
                "ok": False,
                "error": f"a newer {label} sign-in was started; finish that one instead",
            }
        if status != "ACTIVE":
            state.pop("pending_id", None)
            write_private_file(_state_path(service), json.dumps(state))
            return {"ok": False, "error": f"{label} connection ended in state {status}"}
        previous = str(state.get("connected_account_id") or "")
        if previous and previous != pending:
            _delete_quietly(client, previous)
    except Exception as e:
        return {"ok": False, "error": f"{label} connection failed: {e}"}
    write_private_file(_state_path(service), json.dumps({"connected_account_id": pending}))
    return {"ok": True, "message": f"{label} connected through Composio."}


def _delete_quietly(client: Any, account_id: str) -> None:
    """Delete a Composio connected account, ignoring failures."""
    try:
        client.connected_accounts.delete(account_id)
    except Exception:
        pass


def clear_connection(service: str) -> None:
    """Forget *service*'s connection and delete it at Composio.

    Args:
        service: KISS service name.
    """
    state = _read_json(_state_path(service))
    accounts = [str(state.get(k) or "") for k in ("connected_account_id", "pending_id")]
    if any(accounts) and composio_api_key():
        try:
            client = _client()
            for account in filter(None, accounts):
                _delete_quietly(client, account)
        except Exception:
            pass
    _state_path(service).unlink(missing_ok=True)


class ComposioResponse:
    """The subset of :class:`requests.Response` the Google backends use."""

    def __init__(self, status: int, headers: dict[str, str], content: bytes, url: str) -> None:
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers)
        self.content = content
        self.url = url
        self.reason = requests.status_codes._codes.get(status, ("",))[0].upper()  # type: ignore[attr-defined]

    @property
    def ok(self) -> bool:
        """Return True for a status below 400."""
        return self.status_code < 400

    @property
    def text(self) -> str:
        """Return the body decoded as UTF-8."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Return the decoded JSON body."""
        return json.loads(self.content or b"null")

    def raise_for_status(self) -> None:
        """Raise :class:`requests.HTTPError` for a 4xx/5xx status."""
        if not self.ok:
            raise requests.HTTPError(
                f"{self.status_code} Error for url: {self.url}: {self.text[:500]}", response=self  # type: ignore[arg-type]
            )


def proxy_request(
    service: str,
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> ComposioResponse:
    """Send one Google API request through Composio's proxy.

    Args:
        service: KISS service name whose connection signs the request.
        method: HTTP method.
        url: Absolute Google API URL (its query string is kept).
        params: Extra query parameters.
        body: JSON-serializable body, raw JSON text/bytes, or ``None``.
        headers: Request headers (Authorization is dropped: Composio
            injects the real one).

    Returns:
        The upstream answer as a :class:`ComposioResponse`.

    Raises:
        RuntimeError: When *service* is not connected.
    """
    from composio_client import omit

    account = connected_account_id(service)
    if not account:
        raise RuntimeError(f"{service} is not connected; call authenticate_{service}() first")
    # Original header names are kept; lookups use the lower-cased key.
    headers = {k: str(v) for k, v in (headers or {}).items()}
    lower = {k.lower(): k for k in headers}
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    for key, value in (params or {}).items():
        values = value if isinstance(value, (list, tuple)) else [value]
        query.extend((key, _query_value(v)) for v in values)
    if headers.pop(lower.pop("x-http-method-override", ""), "") == "GET":
        # googleapiclient turns an over-long GET into a form-encoded POST;
        # the proxy wants the real GET with its query back.
        method = "GET"
        query.extend(parse_qsl(_body_text(body), keep_blank_values=True))
        body = None
        headers.pop(lower.pop("content-type", ""), None)
    parameters: list[dict[str, str]] = [
        {"name": k, "type": "query", "value": v} for k, v in query
    ]
    for key, value in headers.items():
        if key.lower() not in _DROPPED_HEADERS:
            parameters.append({"name": key, "type": "header", "value": value})
    content_type = headers.get(lower.get("content-type", ""), "")
    json_body, binary_body = _split_body(body, content_type)
    endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    # The SDK's ``tools.proxy`` wrapper has no ``binary_body``; call the
    # generated client directly, without retries (a proxied write is not
    # idempotent).
    answer = _client().client.without_retries.tools.proxy(
        endpoint=endpoint,
        method=method.upper(),
        body=json_body if json_body is not None else omit,
        binary_body=binary_body if binary_body is not None else omit,
        connected_account_id=account,
        parameters=parameters or omit,
    )
    return _to_response(answer, url)


def _query_value(value: Any) -> str:
    """Render a query value the way ``requests`` does (booleans lowercase)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _body_text(body: Any) -> str:
    """Return *body* as text (``""`` for ``None``)."""
    if body is None:
        return ""
    return body.decode("utf-8") if isinstance(body, bytes) else str(body)


def _split_body(body: Any, content_type: str) -> tuple[Any, dict[str, str] | None]:
    """Split a request body into the proxy's ``body`` and ``binary_body``.

    JSON bodies (dicts, lists, or JSON text) travel as ``body``; anything
    else (a Drive multipart upload, raw media) is base64-encoded into
    ``binary_body`` with its content type.

    Args:
        body: The request body.
        content_type: The request's Content-Type header (may be empty).

    Returns:
        ``(json_body, binary_body)``; one of them is ``None``.
    """
    if body is None:
        return None, None
    if isinstance(body, (dict, list)):
        return body, None
    raw = body if isinstance(body, bytes) else str(body).encode("utf-8")
    if not raw:
        return None, None
    if "json" in content_type or not content_type:
        try:
            return json.loads(raw.decode("utf-8")), None
        except (ValueError, UnicodeDecodeError):
            pass
    return None, {
        "base64": base64.b64encode(raw).decode("ascii"),
        "content_type": content_type or "application/octet-stream",
    }


def _to_response(answer: Any, url: str) -> ComposioResponse:
    """Convert a Composio ``ToolProxyResponse`` into a response object."""
    headers = dict(answer.headers or {})
    binary = answer.binary_data
    if binary is not None:
        download = requests.get(binary.url, timeout=60)
        download.raise_for_status()
        headers.setdefault("Content-Type", binary.content_type)
        return ComposioResponse(int(answer.status), headers, download.content, url)
    data = answer.data
    if data is None:
        content = b""
    elif isinstance(data, str):
        content = data.encode()
    else:
        content = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    return ComposioResponse(int(answer.status), headers, content, url)


class ComposioSession:
    """``requests``-shaped client that sends every call through Composio.

    Args:
        service: KISS service name whose connection signs the requests.
    """

    def __init__(self, service: str) -> None:
        self.service = service

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: Any = None,  # noqa: A002 - requests keyword
        data: Any = None,
        headers: dict[str, str] | None = None,
        **_: Any,
    ) -> ComposioResponse:
        """Send one request (``requests.request`` signature subset).

        Args:
            method: HTTP method.
            url: Absolute URL.
            params: Query parameters.
            json: JSON body.
            data: Raw body (JSON text, or bytes sent as a binary body).
            headers: Request headers.

        Returns:
            The upstream answer.
        """
        body = json if json is not None else data
        return proxy_request(self.service, method, url, params, body, headers)

    def get(self, url: str, **kwargs: Any) -> ComposioResponse:
        """Send a GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> ComposioResponse:
        """Send a POST request."""
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> ComposioResponse:
        """Send a PUT request."""
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> ComposioResponse:
        """Send a PATCH request."""
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> ComposioResponse:
        """Send a DELETE request."""
        return self.request("DELETE", url, **kwargs)


class ComposioHttp:
    """``httplib2.Http``-shaped transport for ``googleapiclient``.

    Pass it as ``build(..., http=ComposioHttp(service))``.

    Args:
        service: KISS service name whose connection signs the requests.
    """

    def __init__(self, service: str) -> None:
        self.service = service

    def request(
        self,
        uri: str,
        method: str = "GET",
        body: Any = None,
        headers: dict[str, str] | None = None,
        **_: Any,
    ) -> tuple[Any, bytes]:
        """Send one request and answer the way ``httplib2`` does.

        Args:
            uri: Absolute URL with query string.
            method: HTTP method.
            body: Request body (JSON text for Google API calls).
            headers: Request headers.

        Returns:
            ``(httplib2.Response, content)``.
        """
        import httplib2  # type: ignore[import-untyped]

        answer = proxy_request(self.service, method, uri, None, body, headers)
        info = {k.lower(): v for k, v in answer.headers.items()}
        info["status"] = str(answer.status_code)
        return httplib2.Response(info), answer.content
