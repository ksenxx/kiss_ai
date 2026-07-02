# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``sorcar mcp`` management subcommand.

Implements ``sorcar mcp add/list/get/remove/auth/logout/debug``:

* ``add`` — register a server (stdio command or remote http/sse URL)
  in the user (``~/.kiss/mcp.json``) or project
  (``<work_dir>/.kiss/mcp.json``) scope.
* ``list`` — show every configured server; ``--ping`` also connects
  and reports live status and tool counts.
* ``get`` — show one server's configuration as JSON.
* ``remove`` — delete a server from every writable config file.
* ``auth`` — run the interactive OAuth 2.1 browser flow for a remote
  server (dynamic client registration + PKCE via the MCP SDK) and
  persist the tokens under ``~/.kiss/mcp_auth/``.
* ``logout`` — delete a server's stored OAuth tokens.
* ``debug`` — connect to a server and dump its capabilities, tools
  (with input schemas and permission status), resources, and prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.server
import json
import threading
import urllib.parse
import webbrowser
from typing import Any

from kiss.agents.sorcar.mcp_servers import (
    CONNECT_TIMEOUT,
    FileTokenStorage,
    MCPServerConfig,
    _enter_transport,
    _sanitize,
    build_oauth_provider,
    format_mcp_listing,
    load_mcp_permissions,
    load_mcp_servers,
    mcp_tool_permission,
    remove_mcp_server,
    save_mcp_server,
)

# How long ``sorcar mcp auth`` waits for the browser redirect.
AUTH_TIMEOUT = 300.0


def _parse_kv(pairs: list[str], sep: str) -> tuple[tuple[str, str], ...]:
    """Parse repeated ``KEY<sep>VALUE`` CLI options into tuples.

    Args:
        pairs: The raw option values (e.g. ``["FOO=bar"]``).
        sep: The key/value separator (``"="`` for env, ``":"`` for
            headers).

    Returns:
        The parsed ``(key, value)`` tuples.

    Raises:
        SystemExit: When an entry has no separator.
    """
    out: list[tuple[str, str]] = []
    for pair in pairs:
        key, found, value = pair.partition(sep)
        if not found or not key.strip():
            raise SystemExit(
                f"Invalid option {pair!r}: expected KEY{sep}VALUE"
            )
        out.append((key.strip(), value.strip()))
    return tuple(out)


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``sorcar mcp`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="sorcar mcp", description="Manage MCP servers for Sorcar.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    add = sub.add_parser(
        "add", help="Add an MCP server",
        description=(
            "Add an MCP server. Options must come before the server "
            "name; everything after the name is the stdio command "
            "(or the URL for --transport http/sse). Example: "
            "sorcar mcp add --env KEY=VAL myserver npx -y some-mcp"
        ),
    )
    add.add_argument("name", help="Unique server name")
    add.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="Command and args (stdio) or URL (http/sse)",
    )
    add.add_argument(
        "--transport", choices=("stdio", "http", "sse"), default="stdio",
        help="Server transport (default: stdio)",
    )
    add.add_argument(
        "--scope", choices=("user", "project"), default="user",
        help="Where to save the server (default: user)",
    )
    add.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment variable for stdio servers (repeatable)",
    )
    add.add_argument(
        "--header", action="append", default=[], metavar="'Key: Value'",
        help="HTTP header for http/sse servers (repeatable)",
    )

    lst = sub.add_parser("list", help="List configured MCP servers")
    lst.add_argument(
        "--ping", action="store_true",
        help="Also connect to each server and report live status",
    )

    get = sub.add_parser("get", help="Show one server's configuration")
    get.add_argument("name")

    rem = sub.add_parser("remove", help="Remove an MCP server")
    rem.add_argument("name")

    auth = sub.add_parser(
        "auth", help="Authenticate to a remote server via OAuth",
    )
    auth.add_argument("name")
    auth.add_argument(
        "--no-browser", action="store_true",
        help="Print the authorization URL instead of opening a browser",
    )

    logout = sub.add_parser("logout", help="Delete stored OAuth tokens")
    logout.add_argument("name")

    debug = sub.add_parser(
        "debug", help="Connect to a server and dump its capabilities",
    )
    debug.add_argument("name")
    return parser


def run_mcp_cli(argv: list[str], work_dir: str = ".") -> int:
    """Entry point for ``sorcar mcp ...``.

    Args:
        argv: The arguments after ``mcp`` (e.g. ``["list", "--ping"]``).
        work_dir: The project directory for project-scope config.

    Returns:
        The process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)
    handler = {
        "add": _cmd_add,
        "list": _cmd_list,
        "get": _cmd_get,
        "remove": _cmd_remove,
        "auth": _cmd_auth,
        "logout": _cmd_logout,
        "debug": _cmd_debug,
    }[args.subcommand]
    return handler(args, work_dir)


def _cmd_add(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp add``."""
    target = list(args.target)
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        print("Error: missing command (stdio) or URL (http/sse).")
        return 1
    if args.transport == "stdio":
        cfg = MCPServerConfig(
            name=args.name,
            transport="stdio",
            command=target[0],
            args=tuple(target[1:]),
            env=_parse_kv(args.env, "="),
        )
    else:
        cfg = MCPServerConfig(
            name=args.name,
            transport=args.transport,
            url=target[0],
            headers=_parse_kv(args.header, ":"),
        )
    path = save_mcp_server(cfg, args.scope, work_dir)
    print(f"Added {args.transport} MCP server {args.name!r} to {path}")
    return 0


def _cmd_list(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp list``."""
    print(format_mcp_listing(work_dir, connect=args.ping))
    return 0


def _cmd_get(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp get``."""
    cfg = load_mcp_servers(work_dir).get(args.name)
    if cfg is None:
        print(f"Unknown MCP server: {args.name}")
        return 1
    print(f"{cfg.name} ({cfg.source}):")
    print(json.dumps(cfg.to_json(), indent=2))
    return 0


def _cmd_remove(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp remove``."""
    removed = remove_mcp_server(args.name, work_dir)
    if not removed:
        print(f"Unknown MCP server: {args.name}")
        return 1
    for path in removed:
        print(f"Removed {args.name!r} from {path}")
    return 0


def _cmd_logout(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp logout``."""
    if FileTokenStorage(args.name).clear():
        print(f"Deleted stored OAuth tokens for {args.name!r}.")
        return 0
    print(f"No stored OAuth tokens for {args.name!r}.")
    return 1


class _OAuthCallbackServer:
    """Tiny local HTTP server capturing the OAuth redirect.

    Listens on ``localhost`` (random free port), records the ``code``
    and ``state`` query parameters of the first ``/callback`` request,
    and shows a "you may close this tab" page.
    """

    def __init__(self) -> None:
        self.code: str | None = None
        self.state: str | None = None
        self._done = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                code = (params.get("code") or [""])[0] or None
                if code is None:
                    # Not the OAuth redirect (e.g. a favicon probe):
                    # never clobber an already-captured code.
                    self.send_response(404)
                    self.end_headers()
                    return
                outer.code = code
                outer.state = (params.get("state") or [""])[0] or None
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Sorcar MCP authentication complete."
                    b"</h2>You may close this tab.</body></html>"
                )
                outer._done.set()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.HTTPServer(("localhost", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()

    def wait(self, timeout: float) -> tuple[str, str | None]:
        """Block until the redirect arrives (or *timeout* expires).

        Returns:
            The ``(code, state)`` pair.

        Raises:
            TimeoutError: When no redirect arrives in time.
        """
        if not self._done.wait(timeout) or not self.code:
            raise TimeoutError("Timed out waiting for the OAuth redirect.")
        return self.code, self.state

    def close(self) -> None:
        """Shut the callback server down."""
        self._server.shutdown()
        self._server.server_close()


async def _connect_once(cfg: MCPServerConfig, auth: Any) -> dict[str, Any]:
    """Connect to *cfg* once and collect everything ``debug`` prints.

    The transport and session are opened and closed in this single
    task (anyio cancel-scope requirement).

    Returns:
        Dict with ``server_info``, ``capabilities``, ``tools``,
        ``resources``, and ``prompts`` keys.
    """
    from contextlib import AsyncExitStack

    from mcp import ClientSession

    async with AsyncExitStack() as stack:
        read, write = await _enter_transport(stack, cfg, auth)
        session = await stack.enter_async_context(ClientSession(read, write))
        init = await session.initialize()
        info: dict[str, Any] = {
            "server_info": init.serverInfo.model_dump(mode="json", exclude_none=True),
            "capabilities": init.capabilities.model_dump(
                mode="json", exclude_none=True,
            ),
            "tools": [],
            "resources": [],
            "prompts": [],
        }
        with contextlib.suppress(Exception):
            info["tools"] = (await session.list_tools()).tools
        if init.capabilities.resources is not None:
            with contextlib.suppress(Exception):
                info["resources"] = (await session.list_resources()).resources
        if init.capabilities.prompts is not None:
            with contextlib.suppress(Exception):
                info["prompts"] = (await session.list_prompts()).prompts
        return info
    raise RuntimeError("unreachable")  # pragma: no cover - AsyncExitStack guard


def _cmd_auth(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp auth`` (interactive OAuth login)."""
    cfg = load_mcp_servers(work_dir).get(args.name)
    if cfg is None:
        print(f"Unknown MCP server: {args.name}")
        return 1
    if cfg.transport == "stdio":
        print(
            f"{args.name!r} is a stdio server; it runs locally and "
            "needs no OAuth login."
        )
        return 0

    callback = _OAuthCallbackServer()

    async def redirect_handler(url: str) -> None:
        print(f"\nOpen this URL to authorize Sorcar:\n  {url}\n")
        if not args.no_browser:
            webbrowser.open(url)

    async def callback_handler() -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, callback.wait, AUTH_TIMEOUT)

    auth = build_oauth_provider(
        cfg,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        redirect_port=callback.port,
    )
    try:
        info = asyncio.run(
            asyncio.wait_for(
                _connect_once(cfg, auth), CONNECT_TIMEOUT + AUTH_TIMEOUT,
            )
        )
    except Exception as exc:
        print(f"Authentication failed: {exc}")
        return 1
    finally:
        callback.close()
    server = info["server_info"]
    token_path = FileTokenStorage(cfg.name).path
    print(
        f"✓ Connected to {server.get('name', cfg.name)} "
        f"({len(info['tools'])} tools)."
    )
    if token_path.exists():
        print(f"OAuth tokens stored at {token_path}")
    else:
        print("Server did not require OAuth; no tokens were stored.")
    return 0


def _cmd_debug(args: argparse.Namespace, work_dir: str) -> int:
    """Handle ``sorcar mcp debug`` (connect and dump capabilities)."""
    cfg = load_mcp_servers(work_dir).get(args.name)
    if cfg is None:
        print(f"Unknown MCP server: {args.name}")
        return 1
    auth = build_oauth_provider(cfg) if cfg.transport in ("http", "sse") else None
    print(f"Connecting to {cfg.name} ({cfg.transport}) ...")
    try:
        info = asyncio.run(
            asyncio.wait_for(_connect_once(cfg, auth), CONNECT_TIMEOUT)
        )
    except Exception as exc:
        print(f"✗ Connection failed: {exc}")
        if cfg.transport in ("http", "sse"):
            print(f"If the server requires login, run: sorcar mcp auth {cfg.name}")
        return 1
    print(f"✓ Connected.\nServer info: {json.dumps(info['server_info'])}")
    print(f"Capabilities: {json.dumps(info['capabilities'])}")
    rules = load_mcp_permissions()
    print(f"\nTools ({len(info['tools'])}):")
    for tool in info["tools"]:
        full_name = f"{_sanitize(cfg.name)}_{_sanitize(str(tool.name))}"
        decision = mcp_tool_permission(full_name, rules) if rules else "allow"
        mark = "✓" if decision == "allow" else "✗"
        desc = " ".join(str(tool.description or "").split())
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"  {mark} {full_name} [{decision}] {desc}")
        print(f"      inputSchema: {json.dumps(tool.inputSchema)}")
    if info["resources"]:
        print(f"\nResources ({len(info['resources'])}):")
        for res in info["resources"]:
            print(f"  {res.uri} — {res.name}")
    if info["prompts"]:
        print(f"\nPrompts ({len(info['prompts'])}):")
        for prompt in info["prompts"]:
            print(f"  {prompt.name}")
    return 0
