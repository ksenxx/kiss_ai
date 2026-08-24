# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""OpenAI-compatible API Server Agent — kiss-web as an OpenAI-style backend.

Embeds a small HTTP server that speaks the OpenAI chat API, so any
OpenAI-style frontend (Open WebUI, LibreChat, an ``openai`` SDK script)
becomes a chat surface for the kiss-web daemon.  Stores config in
``~/.kiss/third_party_agents/openai_compat/config.json``.

Endpoints:

* ``GET /v1/models`` — unauthenticated model list (single ``kiss-sorcar`` model).
* ``POST /v1/chat/completions`` — requires ``Authorization: Bearer <api_key>``.
  The last user message becomes the prompt of a task submitted through the
  public :func:`kiss.server.sorcar.run` API, with any system/developer
  messages forwarded as the task's ``system_prompt``; the daemon's result
  summary is returned as the assistant message.  ``"stream": true`` responds with a
  single ``text/event-stream`` chunk carrying the full content, then
  ``data: [DONE]``.

Conversation persistence: OpenAI clients are stateless and resend the whole
message list, so each request is mapped to a persistent daemon chat by
hashing the conversation prefix (sha256 of the canonical JSON of all
messages except the last user message) and looking the key up in
``chat_map.json`` next to the config file.  After a successful run the
daemon ``chat_id`` is stored under the full message list's key and under
the key of that list plus the assistant reply, so the client's next
request — which appends the assistant reply and a new user message — maps
back to the same daemon chat.

Usage::

    kiss-oai -t 'configure the OpenAI-compatible API server'  # authenticate
    kiss-oai --serve                # run the API server in the foreground

    agent = OpenAICompatAgent()
    agent.run(prompt_template="Report the API server status")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    start_http_server,
    stop_http_server,
)
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "18092"
_DEFAULT_BIND_HOST = "127.0.0.1"
_MAX_BODY_BYTES = 5 * 1024 * 1024
_MAX_CHAT_MAP_ENTRIES = 5000

_OPENAI_COMPAT_DIR = Path.home() / ".kiss" / "third_party_agents" / "openai_compat"
_config = ChannelConfig(_OPENAI_COMPAT_DIR, ("api_key", "port"))

_chat_map_lock = threading.Lock()


def _chat_map_path() -> Path:
    """Return the conversation→chat_id map file path (KISS_HOME-aware)."""
    return _config.path.parent / "chat_map.json"


def _content_text(content: Any) -> str:
    """Flatten an OpenAI message ``content`` field to plain text.

    Args:
        content: A string, ``None``, or a list of content parts
            (dicts with a ``text`` field, per the OpenAI API).

    Returns:
        The concatenated text of the content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _system_prompt_text(messages: list[Any]) -> str:
    """Collect system/developer message contents into one system prompt.

    Args:
        messages: OpenAI-style message dicts.

    Returns:
        The text of all ``system`` and ``developer`` messages, in
        order, joined by blank lines; ``""`` when there are none.
    """
    parts = [
        _content_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") in ("system", "developer")
    ]
    return "\n\n".join(p for p in parts if p.strip())


def _conversation_key(messages: list[Any]) -> str:
    """Return the sha256 hex key of a message list.

    Messages are canonicalized to ``{"role", "content"}`` dicts with
    flattened text content, so a client resending the same conversation
    (possibly with extra metadata fields) produces the same key.

    Args:
        messages: OpenAI-style message dicts.

    Returns:
        Hex sha256 digest of the canonical JSON of *messages*.
    """
    canonical = [
        {
            "role": str(m.get("role", "")) if isinstance(m, dict) else "",
            "content": _content_text(m.get("content")) if isinstance(m, dict) else str(m),
        }
        for m in messages
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lookup_chat_id(key: str) -> str:
    """Look up the daemon chat id stored for a conversation key.

    Args:
        key: Conversation key from :func:`_conversation_key`.

    Returns:
        The stored chat id, or ``""`` when unknown.
    """
    with _chat_map_lock:
        path = _chat_map_path()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ""
        if not isinstance(data, dict):
            return ""
        return str(data.get(key, "") or "")


def _store_chat_id(key: str, chat_id: str) -> None:
    """Persist a conversation key → daemon chat id mapping.

    The map is trimmed to its most recent :data:`_MAX_CHAT_MAP_ENTRIES`
    entries so a long-running server cannot grow it unboundedly.

    Args:
        key: Conversation key from :func:`_conversation_key`.
        chat_id: The daemon chat session id to store.
    """
    with _chat_map_lock:
        path = _chat_map_path()
        data: dict[str, str] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = {str(k): str(v) for k, v in loaded.items()}
            except (json.JSONDecodeError, OSError):
                data = {}
        data.pop(key, None)
        data[key] = chat_id
        while len(data) > _MAX_CHAT_MAP_ENTRIES:
            data.pop(next(iter(data)))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if sys.platform != "win32":
            tmp_path.chmod(0o600)
        os.replace(tmp_path, path)


def _send_json(handler: BaseHTTPRequestHandler, status: int, obj: dict[str, Any]) -> None:
    """Send a JSON response on a request handler.

    Args:
        handler: The active request handler.
        status: HTTP status code.
        obj: JSON-serializable response body.
    """
    body = json.dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler: BaseHTTPRequestHandler, status: int, message: str, err_type: str) -> None:
    """Send an OpenAI-style error response.

    Args:
        handler: The active request handler.
        status: HTTP status code.
        message: Human-readable error message.
        err_type: OpenAI error type (e.g. ``invalid_request_error``).
    """
    _send_json(
        handler,
        status,
        {"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


class OpenAICompatChannelBackend(ToolMethodBackend):
    """Channel backend embedding an OpenAI-compatible HTTP API server.

    Serves ``GET /v1/models`` and ``POST /v1/chat/completions``; each
    authenticated chat completion request is executed as a kiss-web
    daemon task via :func:`kiss.server.sorcar.run` inside the request
    handler thread.
    """

    def __init__(self) -> None:
        self._api_key: str = ""
        self._port: int = 0
        self._bind_host: str = _DEFAULT_BIND_HOST
        self._model_name: str = ""
        self._server: ThreadedHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the stored config and start the embedded API server."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No OpenAI-compatible API config found."
            return False
        self._api_key = cfg["api_key"]
        self._bind_host = cfg.get("bind_host") or _DEFAULT_BIND_HOST
        self._model_name = cfg.get("model_name", "")
        try:
            self._port = int(cfg["port"])
        except ValueError:
            self._connection_info = f"Invalid port in config: {cfg['port']!r}"
            return False
        return self._start_server()

    def _start_server(self) -> bool:
        """Start the OpenAI-compatible HTTP server on the configured address."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/v1/models":
                    _send_json(
                        self,
                        200,
                        {
                            "object": "list",
                            "data": [{"id": "kiss-sorcar", "object": "model", "owned_by": "kiss"}],
                        },
                    )
                else:
                    _send_error(self, 404, f"Unknown path: {self.path}", "invalid_request_error")

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/v1/chat/completions":
                    _send_error(self, 404, f"Unknown path: {self.path}", "invalid_request_error")
                    return
                try:
                    backend._handle_chat_completions(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    logger.error("chat/completions handler error", exc_info=True)
                    try:
                        _send_error(self, 500, "Internal server error.", "api_error")
                    except OSError:
                        pass

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        self._server, self._server_thread, error = start_http_server(
            (self._bind_host, self._port),
            Handler,
            log=logger,
            started_log=None,
            error_prefix="OpenAI-compatible API bind failed",
            error_log="Could not start OpenAI-compatible API server: %s",
        )
        if error is not None or self._server is None:
            self._connection_info = error or ""
            return False
        bound_port = self._server.server_address[1]
        self._connection_info = f"OpenAI-compatible API serving on {self._bind_host}:{bound_port}"
        logger.info("%s", self._connection_info)
        return True

    def _handle_chat_completions(self, handler: BaseHTTPRequestHandler) -> None:
        """Serve one ``POST /v1/chat/completions`` request."""
        auth = handler.headers.get("Authorization", "")
        token = auth[len("Bearer ") :].strip() if auth.startswith("Bearer ") else ""
        if not token or not hmac.compare_digest(token, self._api_key):
            _send_error(handler, 401, "Invalid or missing bearer token.", "invalid_request_error")
            return
        try:
            length = int(handler.headers.get("Content-Length"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            length = -1
        if length < 0:
            _send_error(
                handler,
                400,
                "Missing or invalid Content-Length header.",
                "invalid_request_error",
            )
            return
        if length > _MAX_BODY_BYTES:
            _send_error(handler, 413, "Request body too large.", "invalid_request_error")
            return
        body = handler.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _send_error(handler, 400, "Malformed JSON body.", "invalid_request_error")
            return
        if not isinstance(payload, dict):
            _send_error(
                handler, 400, "Request body must be a JSON object.", "invalid_request_error"
            )
            return
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            _send_error(
                handler,
                400,
                "'messages' must be a non-empty list.",
                "invalid_request_error",
            )
            return
        last_user = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user = i
                break
        if last_user < 0:
            _send_error(
                handler,
                400,
                "'messages' must include a user message.",
                "invalid_request_error",
            )
            return
        prompt = _content_text(messages[last_user].get("content"))
        if not prompt.strip():
            _send_error(handler, 400, "The last user message is empty.", "invalid_request_error")
            return

        prior = messages[:last_user] + messages[last_user + 1 :]
        chat_id = _lookup_chat_id(_conversation_key(prior))
        system_prompt = _system_prompt_text(messages)
        try:
            from kiss.server.sorcar import run as sorcar_run

            result = sorcar_run(
                prompt, chat_id=chat_id, model=self._model_name, system_prompt=system_prompt
            )
        except Exception as e:
            _send_error(handler, 502, f"kiss-web daemon unavailable: {e}", "api_error")
            return
        if not result.success:
            _send_error(handler, 502, f"kiss-web daemon task failed: {result.text}", "api_error")
            return

        text = result.text
        if result.chat_id:
            _store_chat_id(_conversation_key(messages), result.chat_id)
            with_reply = list(messages) + [{"role": "assistant", "content": text}]
            _store_chat_id(_conversation_key(with_reply), result.chat_id)

        # Label the response with the model actually forwarded to the
        # daemon; when none is configured the daemon default runs, which
        # is exposed as the stable "kiss-sorcar" alias — never the
        # client's unforwarded request value.
        model = self._model_name or "kiss-sorcar"
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if payload.get("stream"):
            self._send_stream(handler, completion_id, created, model, text)
            return
        _send_json(
            handler,
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": result.tokens,
                    "total_tokens": result.tokens,
                },
            },
        )

    def _send_stream(
        self,
        handler: BaseHTTPRequestHandler,
        completion_id: str,
        created: int,
        model: str,
        text: str,
    ) -> None:
        """Send the result as a single SSE chunk followed by ``[DONE]``."""
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: this adapter is request/response only.

        Inbound messages arrive as HTTP requests and are answered in
        the same request, so there is never a pending message queue.
        """
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Unsupported: replies are delivered as HTTP responses.

        Raises:
            RuntimeError: Always; the adapter has no outbound channel.
        """
        raise RuntimeError(
            "The OpenAI-compatible API adapter is inbound-only; "
            "replies are delivered as HTTP responses."
        )

    def disconnect(self) -> None:
        """Stop the embedded API server and release backend resources."""
        self._server, self._server_thread = stop_http_server(self._server, self._server_thread)

    def openai_compat_status(self) -> str:
        """Report the configured API server port and whether it is serving.

        Returns:
            JSON string with the configured bind host and port, and
            whether a server socket is currently bound there (checked
            with a real TCP connection when this process is not the
            one running the server).
        """
        try:
            cfg = _config.load()
            if not cfg:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "Not configured. Use authenticate_openai_compat().",
                    }
                )
            host = cfg.get("bind_host") or _DEFAULT_BIND_HOST
            if self._server is not None:
                return json.dumps(
                    {
                        "ok": True,
                        "bind_host": host,
                        "port": self._server.server_address[1],
                        "bound": True,
                    }
                )
            try:
                port = int(cfg["port"])
            except ValueError:
                return json.dumps({"ok": False, "error": f"Invalid port: {cfg['port']!r}"})
            bound = False
            if port > 0:
                probe_host = "127.0.0.1" if host == "0.0.0.0" else host
                try:
                    with socket.create_connection((probe_host, port), timeout=2.0):
                        bound = True
                except OSError:
                    bound = False
            return json.dumps({"ok": True, "bind_host": host, "port": port, "bound": bound})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class OpenAICompatAgent(BaseChannelAgent):
    """Channel agent exposing kiss-web through an OpenAI-compatible API."""

    channel_system_prompt = (
        "You are reachable through an OpenAI-compatible chat completion API "
        "(kiss-oai). Clients talk to you like an OpenAI model: each request "
        "carries the whole conversation and your task result is returned as "
        "the assistant message. There is no outbound messaging tool; answer "
        "in your final result text."
    )

    def __init__(self) -> None:
        super().__init__("OpenAI-compatible API Agent")
        self._backend = OpenAICompatChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._api_key = cfg["api_key"]
            self._backend._bind_host = cfg.get("bind_host") or _DEFAULT_BIND_HOST
            self._backend._model_name = cfg.get("model_name", "")
            try:
                self._backend._port = int(cfg["port"])
            except ValueError:
                self._backend._port = 0

    def _is_authenticated(self) -> bool:
        """Return True if the backend is configured."""
        return bool(self._backend._api_key)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_openai_compat_auth() -> str:
            """Check if the OpenAI-compatible API server is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._api_key:
                return (
                    "Not configured. Use authenticate_openai_compat() to set the "
                    "bearer api_key clients must present, the port, and the bind "
                    "host, then run `kiss-oai --serve` to start the API server."
                )
            return json.dumps(
                {
                    "ok": True,
                    "bind_host": agent._backend._bind_host,
                    "port": agent._backend._port,
                    "model_name": agent._backend._model_name,
                }
            )

        def authenticate_openai_compat(
            api_key: str,
            port: str = _DEFAULT_PORT,
            bind_host: str = _DEFAULT_BIND_HOST,
            model_name: str = "",
        ) -> str:
            """Configure the OpenAI-compatible API server.

            Args:
                api_key: Bearer token clients must send on
                    ``POST /v1/chat/completions``.
                port: TCP port for the embedded HTTP server.
                bind_host: Interface to bind (default 127.0.0.1;
                    use 0.0.0.0 to expose on the network).
                model_name: Optional kiss model name forwarded to the
                    daemon for each task (daemon default when empty).

            Returns:
                Configuration result or error message.
            """
            if not api_key.strip():
                return "api_key cannot be empty."
            try:
                port_num = int(port)
            except ValueError:
                port_num = -1
            if not 0 <= port_num <= 65535:
                return f"Invalid port: {port!r}"
            _config.save(
                {
                    "api_key": api_key.strip(),
                    "port": str(port_num),
                    "bind_host": bind_host.strip() or _DEFAULT_BIND_HOST,
                    "model_name": model_name.strip(),
                }
            )
            agent._backend._api_key = api_key.strip()
            agent._backend._port = port_num
            agent._backend._bind_host = bind_host.strip() or _DEFAULT_BIND_HOST
            agent._backend._model_name = model_name.strip()
            return json.dumps(
                {
                    "ok": True,
                    "message": "OpenAI-compatible API configured. "
                    "Start the server with: kiss-oai --serve",
                }
            )

        def clear_openai_compat_auth() -> str:
            """Clear the stored OpenAI-compatible API configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._api_key = ""
            agent._backend._port = 0
            agent._backend._bind_host = _DEFAULT_BIND_HOST
            agent._backend._model_name = ""
            return "OpenAI-compatible API configuration cleared."

        return [check_openai_compat_auth, authenticate_openai_compat, clear_openai_compat_auth]


def _serve() -> None:
    """Run the OpenAI-compatible API server in the foreground."""
    backend = OpenAICompatChannelBackend()
    if not backend.connect():
        print(backend.connection_info)
        print("Not configured. Run: kiss-oai -t 'configure the OpenAI-compatible API server'")
        sys.exit(1)
    print(f"{backend.connection_info} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        backend.disconnect()


def main() -> None:
    """Run the OpenAICompatAgent from the command line.

    With ``--serve`` the OpenAI-compatible API server runs in the
    foreground; otherwise the standard channel CLI handles the task.
    """
    if "--serve" in sys.argv:
        _serve()
        return
    channel_main(
        OpenAICompatAgent,
        "kiss-oai",
        channel_name="OpenAI-compatible API",
        make_backend=None,
    )


def get_tools() -> list:
    """Return the OpenAI-compatible API tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return OpenAICompatAgent()._get_tools()


if __name__ == "__main__":
    main()
