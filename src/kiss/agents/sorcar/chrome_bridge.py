"""Bridge between KISS and the user's everyday Chrome via the KISS Sorcar Bridge extension.

Chrome ignores ``--remote-debugging-port`` on the default profile since
Chrome 136, so CDP cannot reach the browser where the user is actually
logged in.  The sanctioned path into that browser is an extension.  The
companion MV3 extension in ``chrome_extension/`` (loaded unpacked by the
user, once) connects over a loopback WebSocket to the server in this
module and executes tab/scripting commands on KISS's behalf.

Protocol (JSON text frames), a mutual challenge-response handshake:

- server -> extension: ``{"type": "challenge", "nonce": "<hex>"}`` at
  connect.
- extension -> server: ``{"type": "hello", "proof": sha256(token+":"+nonce)}``;
  a wrong proof gets ``hello_error`` and a 4001 close.  The raw token
  never crosses the wire, so a rogue local process that occupies the
  port learns nothing reusable (each nonce is single-use).
- server -> extension: ``{"type": "hello_ok", "proof": sha256(nonce+":"+token)}``;
  the extension verifies this before executing any command, so the rogue
  listener cannot drive the browser either.
- extension -> server: ``{"type": "ping"}`` keepalive; answered with
  ``{"type": "pong"}`` (WebSocket activity keeps the MV3 service worker
  alive).
- server -> extension: ``{"id": N, "cmd": "<name>", "args": {...}, "timeout_ms": T}``;
  *T* is how long the server still waits for the reply, a little less than
  its own timeout, and is the command's deadline in the extension (counted
  from its arrival), so a command that runs over is answered with an error
  while the server still listens.
- extension -> server: ``{"id": N, "ok": true, "result": ...}`` or
  ``{"id": N, "ok": false, "error": "<message>"}``.

The server binds an ephemeral loopback port.  The port and the token
(generated at server start) are written to ``bridge-config.json`` (created
mode 600) inside ``$KISS_HOME/chrome_extension/``, the directory the
bundled extension files are installed into and the one place the
unpacked extension can read them from (it re-reads the file on every
connection attempt, so a restarted daemon on a new port is found
without touching Chrome).  The socket is loopback-only.

The daemon (opt-in ``chrome_bridge`` setting) starts the bridge at boot
and takes the config file over; any other KISS process starts it lazily
on the first ``use_my_chrome`` call and leaves the file alone while the
bridge it names is still alive, so a CLI run beside the daemon never
steals the extension from it.
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from concurrent.futures import Future
from contextlib import AbstractContextManager
from itertools import count
from pathlib import Path
from typing import Any

from websockets.sync.client import connect
from websockets.sync.server import Server, ServerConnection, serve

from kiss.core.brand import BRAND, PRODUCT_NAME, render_brand
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.user_config import load_config
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # screenshots ride this socket as data URLs
CONFIG_FILE_NAME = "bridge-config.json"

_EXTENSION_SOURCE = Path(__file__).resolve().parent / "chrome_extension"
_EXTENSION_FILES = ("manifest.json", "background.js")
_PROBE_TIMEOUT_SECS = 1.0
_CONFIG_LOCK_TIMEOUT_SECS = 5.0
_REPLY_MARGIN_SECS = 1.0  # the extension's deadline runs at most this much short of ours


def extension_dir() -> Path:
    """Return the install directory of the unpacked extension.

    ``$KISS_HOME/chrome_extension`` (``~/.kiss/chrome_extension`` by
    default): a stable, user-owned path that survives package upgrades
    (Chrome keeps pointing at it) and that stays writable on a read-only
    site-packages install.  Resolved on every call so a ``KISS_HOME`` set
    after import (the test suite's conftest) is honored.

    Returns:
        Absolute path of the extension directory (may not exist yet).
    """
    return kiss_home() / "chrome_extension"


def bundled_extension_file(name: str, brand: dict[str, str] = BRAND) -> bytes:
    """Return the bundled extension file *name* as it is installed.

    The manifest names the extension after the brand (``{{PRODUCT_NAME}}
    Bridge``), so a white-label install never shows the stock product
    name in ``chrome://extensions``; the other files are copied verbatim.

    Args:
        name: One of ``_EXTENSION_FILES``.
        brand: The brand strings (the product's by default).

    Returns:
        The file content, brand placeholders filled in for the manifest.
    """
    data = (_EXTENSION_SOURCE / name).read_bytes()
    if name != "manifest.json":
        return data
    # The placeholders sit inside JSON strings: escape the brand values
    # (a quote or backslash in a product name must not break the file).
    escaped = {key: json.dumps(value)[1:-1] for key, value in brand.items()}
    return render_brand(data.decode("utf-8"), escaped).encode("utf-8")


def install_extension() -> Path:
    """Copy the bundled extension files into :func:`extension_dir`.

    Files that already match the bundled copy are left untouched; a
    changed ``background.js`` is overwritten in place, which Chrome picks
    up at the next service-worker start (a reload from
    ``chrome://extensions`` picks up a manifest change).

    Returns:
        The extension directory.
    """
    target = extension_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in _EXTENSION_FILES:
        data = bundled_extension_file(name)
        dest = target / name
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
    return target


class ChromeBridgeError(Exception):
    """Raised when the bridge cannot reach or use the user's Chrome extension."""


def start_if_enabled() -> None:
    """Start the process-wide bridge when the ``chrome_bridge`` setting is on.

    Called by the daemon at boot and after every settings save, so the
    extension is reachable from the first ``use_my_chrome`` call.  The
    daemon owns the bridge whenever it runs (``takeover``: its port and
    token replace whatever ``bridge-config.json`` named).  A start
    failure is logged, never fatal: the first tool call retries lazily.
    An already running bridge is kept (switching the setting off takes
    effect at the next daemon start).
    """
    if ChromeBridge._instance is not None or not load_config().get("chrome_bridge"):
        return
    try:
        bridge = ChromeBridge.instance(takeover=True)
    except ChromeBridgeError as e:
        logger.warning("Chrome bridge not started: %s", e)
        return
    logger.info(
        "chrome bridge serving 127.0.0.1:%d for the extension in %s",
        bridge.port,
        extension_dir(),
    )


class _Connection:
    """One authenticated extension connection, its send lock and its commands."""

    def __init__(self, ws: ServerConnection):
        self.ws = ws
        self.lock = threading.Lock()  # serializes sends
        self.pending: dict[int, Future] = {}
        self.pending_lock = threading.Lock()


class ChromeBridge:
    """Loopback WebSocket server the KISS Sorcar Bridge extension connects to.

    A process-wide singleton (see :meth:`instance`).  Commands are issued
    with :meth:`call`, which blocks the calling thread until the extension
    answers or the timeout expires.
    """

    _instance: ChromeBridge | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls, takeover: bool = False) -> ChromeBridge:
        """Return the process-wide bridge, starting its server on first use.

        The server is stopped (and its config file withdrawn) at
        interpreter exit, so a finished CLI run never leaves the extension
        pointed at a dead port.

        Args:
            takeover: Passed to :meth:`start` on first use.  The daemon
                starts with ``True``; a lazy start from a tool call keeps
                the default and defers to a live bridge of another process.

        Returns:
            The running :class:`ChromeBridge`.

        Raises:
            ChromeBridgeError: if the server cannot start (see :meth:`start`).
        """
        with cls._instance_lock:
            if cls._instance is None:
                bridge = cls()
                bridge.start(takeover=takeover)
                atexit.register(bridge.stop)
                cls._instance = bridge
            return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Stop the process-wide bridge, if one was started, and forget it.

        The daemon calls this on shutdown; the next :meth:`instance` call
        starts a fresh server.  A no-op when no bridge was started.
        """
        with cls._instance_lock:
            bridge, cls._instance = cls._instance, None
        if bridge is not None:
            bridge.stop()

    def __init__(self, port: int = 0, write_config: bool = True):
        self._port = port
        self._write_config = write_config
        self._token = secrets.token_hex(16)
        self._ids = count(1)
        self._conn: _Connection | None = None
        self._connected = threading.Event()
        # Held while _conn and _connected change together: a drop that checks
        # "is this still the live connection?" must not clear a newer one.
        self._conn_lock = threading.Lock()
        self._server: Server | None = None

    @property
    def extension_connected(self) -> bool:
        """True while an authenticated extension connection is live."""
        return self._conn is not None

    @property
    def port(self) -> int:
        """The bound loopback port (resolved after :meth:`start`)."""
        return self._port

    @property
    def token(self) -> str:
        """This server run's auth token (also written to bridge-config.json)."""
        return self._token

    def start(self, takeover: bool = False) -> None:
        """Bind the loopback socket, publish bridge-config.json, serve in a thread.

        With ``write_config`` the bundled extension files are installed
        into :func:`extension_dir` and the bound port and token written
        next to them -- unless the file already names a bridge that still
        answers (another KISS process, normally the daemon) and *takeover*
        is False: then nothing is bound and the caller is told which
        process owns the extension, since Chrome keeps one connection and
        a rewritten file would only strand it once that process exits.

        Args:
            takeover: Publish this server even over a live one (the daemon,
                which owns the extension whenever it runs).

        Raises:
            ChromeBridgeError: if the socket cannot be bound, the extension
                files or the config file cannot be written, the server
                thread cannot start, or another process's bridge owns
                the config file.
        """
        if self._write_config and not takeover:
            other = _live_bridge_port(extension_dir() / CONFIG_FILE_NAME)
            if other is not None:
                raise ChromeBridgeError(
                    f"another KISS process already serves the Chrome bridge on port {other} "
                    "(normally the kiss-web daemon, which the extension stays connected to); "
                    "run this task through that process"
                )
        try:
            self._server = serve(
                self._handle,
                "127.0.0.1",
                self._port,
                max_size=MAX_MESSAGE_BYTES,
            )
        except OSError as e:
            raise ChromeBridgeError(f"cannot listen on 127.0.0.1:{self._port}: {e}") from e
        self._port = self._server.socket.getsockname()[1]
        if self._write_config:
            try:
                _write_config(install_extension() / CONFIG_FILE_NAME, self._port, self._token)
            except OSError as e:  # an unwritable KISS_HOME, or the config lock stuck
                self._server.shutdown()
                self._server = None
                raise ChromeBridgeError(
                    f"cannot publish the bridge in {extension_dir()}: {e}"
                ) from e
        try:
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
        except RuntimeError as e:  # thread exhaustion
            # Close the socket nobody would serve and withdraw the config
            # that points the extension at it.
            self.stop()
            raise ChromeBridgeError(f"cannot start the bridge server thread: {e}") from e
        logger.info("chrome bridge listening on 127.0.0.1:%d", self._port)

    def stop(self) -> None:
        """Stop the server, close the live connection, fail in-flight commands.

        The config file is removed when it still names this server, so the
        extension reports a missing bridge instead of retrying a dead port
        (a newer bridge's file, recognised by its token, is left alone).
        """
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        conn = self._conn  # one read: the handler thread may drop it meanwhile
        if conn is not None:
            self._drop_connection(conn, "bridge server stopped")
            try:
                conn.ws.close()  # half-dead handlers keep the extension pinging otherwise
            except Exception:  # noqa: BLE001 - already gone
                logger.debug("Exception caught", exc_info=True)
        if self._write_config:
            _remove_config_if_owned(extension_dir() / CONFIG_FILE_NAME, self._token)

    def wait_for_extension(self, timeout: float) -> bool:
        """Block until an extension is connected or *timeout* seconds pass.

        A bridge started lazily by the first tool call is found by the
        extension on its next reconnect attempt (a few seconds); waiting
        here turns that first call's certain failure into a result.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True when an extension is connected.
        """
        return self._connected.wait(timeout)

    def call(self, cmd: str, args: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
        """Send a command to the extension and wait for its result.

        Args:
            cmd: Command name handled by the extension (e.g. "list_tabs",
                "evaluate").
            args: Command arguments.
            timeout: Seconds to wait for the extension's reply.

        Returns:
            The extension's ``result`` value.

        Raises:
            ChromeBridgeError: if no extension is connected, the command
                times out, or the extension reports an error.
        """
        conn = self._conn
        if conn is None:
            raise ChromeBridgeError(
                f"the {PRODUCT_NAME} Bridge extension is not connected. "
                f"Install it once: open chrome://extensions, enable Developer mode, "
                f"click 'Load unpacked' and choose {extension_dir()}. "
                "If it is installed, click its toolbar icon to wake it."
            )
        msg_id = next(self._ids)
        fut: Future = Future()
        with conn.pending_lock:
            conn.pending[msg_id] = fut
        # The deadline covers queueing behind other sends too: a caller whose
        # command never got on the wire inside its timeout fails instead of
        # stacking up behind a wedged peer.
        deadline = time.monotonic() + timeout
        if not conn.lock.acquire(timeout=max(timeout, 0.0)):
            with conn.pending_lock:
                conn.pending.pop(msg_id, None)
            raise ChromeBridgeError(
                f"the extension connection stayed busy past {timeout:.0f}s; "
                f"'{cmd}' was not sent"
            )
        # The extension's deadline runs a little short of ours (a tenth, at
        # most _REPLY_MARGIN_SECS): its "deadline exceeded" reply arrives
        # while we still wait, instead of our timeout dropping the connection
        # and every other command in flight.
        budget = max(deadline - time.monotonic(), 0.0)
        timeout_ms = round((budget - min(budget / 10, _REPLY_MARGIN_SECS)) * 1000)
        try:
            conn.ws.send(json.dumps(
                {"id": msg_id, "cmd": cmd, "args": args or {}, "timeout_ms": timeout_ms}
            ))
        except Exception as e:
            with conn.pending_lock:
                conn.pending.pop(msg_id, None)
            if self._conn is conn:
                self._drop_connection(conn, "send failed")
            raise ChromeBridgeError(f"lost the extension connection: {e}") from e
        finally:
            conn.lock.release()
        try:
            return fut.result(max(deadline - time.monotonic(), 0.0))
        except TimeoutError:
            with conn.pending_lock:
                conn.pending.pop(msg_id, None)
            # A peer that stops answering would wedge every later command
            # behind its stuck socket; drop it so the extension reconnects
            # fresh and any sender blocked in sendall() is released.
            if self._conn is conn:
                self._drop_connection(conn, "command timed out")
                try:
                    conn.ws.close()
                except Exception:  # noqa: BLE001 - already gone
                    logger.debug("Exception caught", exc_info=True)
            raise ChromeBridgeError(
                f"the extension did not answer '{cmd}' within {timeout:.0f}s"
            ) from None

    # -- connection handling (server thread) -------------------------------

    def _handle(self, ws: ServerConnection) -> None:
        # Mutual challenge-response: the raw token never crosses the wire, so
        # a rogue loopback listener occupying the port learns nothing it can
        # reuse (each nonce is single-use), and the extension verifies the
        # server's proof before executing any of its commands.
        conn = _Connection(ws)
        nonce = secrets.token_hex(16)
        try:
            ws.send(json.dumps({"type": "challenge", "nonce": nonce}))
            first = json.loads(ws.recv(timeout=10))
            expected = _proof(self._token, nonce)
            if first.get("type") != "hello" or not hmac.compare_digest(
                expected, str(first.get("proof", ""))
            ):
                try:
                    ws.send(json.dumps({"type": "hello_error", "error": "bad token"}))
                    ws.close(4001, "bad token")
                except Exception:  # noqa: BLE001 - peer already gone
                    logger.debug("Exception caught", exc_info=True)
                return
            ws.send(json.dumps({"type": "hello_ok", "proof": _proof(nonce, self._token)}))
            with self._conn_lock:
                old, self._conn = self._conn, conn
                self._connected.set()
            if old is not None:  # a second Chrome profile connected: keep the newest
                try:
                    old.ws.close(4002, "replaced by a newer connection")
                except Exception:  # noqa: BLE001 - already gone
                    logger.debug("Exception caught", exc_info=True)
            logger.info("chrome bridge extension connected")
            while True:
                self._on_message(conn, json.loads(ws.recv()))
        except Exception:  # noqa: BLE001 - any recv/parse failure ends the connection
            logger.debug("Exception caught", exc_info=True)
        finally:
            self._drop_connection(conn, "extension disconnected")

    def _on_message(self, conn: _Connection, msg: dict[str, Any]) -> None:
        if msg.get("type") == "ping":
            with conn.lock:
                conn.ws.send(json.dumps({"type": "pong"}))
            return
        msg_id = msg.get("id")
        if not isinstance(msg_id, int):
            return
        with conn.pending_lock:
            fut = conn.pending.pop(msg_id, None)
        if fut is None:
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(ChromeBridgeError(str(msg.get("error", "extension error"))))

    def _drop_connection(self, conn: _Connection, reason: str) -> None:
        with self._conn_lock:
            if self._conn is conn:
                self._conn = None
                self._connected.clear()
        with conn.pending_lock:
            pending = list(conn.pending.values())
            conn.pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(ChromeBridgeError(reason))


def _proof(key: str, message: str) -> str:
    """The handshake proof: SHA-256 of "<key>:<message>".

    The extension proves the token with ``_proof(token, nonce)``; the
    server proves itself with ``_proof(nonce, token)`` — different inputs,
    so one side's proof can never be reflected back as the other's.
    """
    return hashlib.sha256(f"{key}:{message}".encode()).hexdigest()


def _config_lock(config: Path) -> AbstractContextManager[bool]:
    """The lock serialising the publication and the withdrawal of *config*."""
    lock = config.with_name(f".{config.name}.lock")
    return exclusive_file_lock(lock, timeout=_CONFIG_LOCK_TIMEOUT_SECS)


def _write_config(config: Path, port: int, token: str) -> None:
    """Write the connection file the extension fetches at startup (mode 600).

    The staged copy has mode 600 from the start (the token must never
    sit in a world-readable file), and the write holds the config lock,
    so a predecessor's :func:`_remove_config_if_owned` cannot unlink
    this file between its ownership check and its unlink.

    Raises:
        OSError: when the file cannot be written; ``TimeoutError`` when
            the lock stays held past :data:`_CONFIG_LOCK_TIMEOUT_SECS`.
    """
    with _config_lock(config):
        atomic_write_text(config, json.dumps({"port": port, "token": token}), mode=0o600)


def _remove_config_if_owned(config: Path, token: str) -> None:
    """Delete *config* only while it still names *token*, under the config lock.

    A successor bridge (normally a restarted daemon) may have published
    its own file already; deleting that would leave the extension unable
    to find it.  A failure is logged, never raised: this runs at exit.
    """
    try:
        with _config_lock(config):
            if _read_config(config).get("token") == token:
                config.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not withdraw %s", config, exc_info=True)


def _read_config(config: Path) -> dict[str, Any]:
    """Return the parsed connection file, or ``{}`` when missing or unreadable."""
    try:
        data = json.loads(config.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _live_bridge_port(config: Path) -> int | None:
    """Return the port *config* names when a bridge still answers there.

    A bridge is recognised by the ``challenge`` frame it sends on connect,
    so a stale file whose port was reused by an unrelated service does not
    count.  The probe connection is dropped unauthenticated; the server
    side treats it as a peer that never said hello.
    """
    port = _read_config(config).get("port")
    if not isinstance(port, int):
        return None
    try:
        with connect(f"ws://127.0.0.1:{port}", open_timeout=_PROBE_TIMEOUT_SECS) as ws:
            first = json.loads(ws.recv(timeout=_PROBE_TIMEOUT_SECS))
    except Exception:  # noqa: BLE001 - nothing (or not a bridge) listens there
        return None
    return port if isinstance(first, dict) and first.get("type") == "challenge" else None
