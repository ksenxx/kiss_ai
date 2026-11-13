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
- extension -> server: ``{"type": "hello", "proof": sha256(token+":"+nonce),
  "version": "<the version of the manifest Chrome loaded>"}``;
  a wrong proof gets ``hello_error`` and a 4001 close.  The raw token
  never crosses the wire, so a rogue local process that occupies the
  port learns nothing reusable (each nonce is single-use).  A version
  other than the installed one (:func:`extension_version`) gets
  ``hello_error`` and a 4004 close: Chrome runs a copy of the extension
  it loaded before the installed one was written, and that copy, so
  refused, reloads the extension (see :func:`install_extension`).
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
import os
import secrets
import socket
import threading
import time
from concurrent.futures import Future
from contextlib import AbstractContextManager
from itertools import count
from pathlib import Path
from typing import Any

from websockets.sync.client import connect
from websockets.sync.server import Server, ServerConnection, serve

from kiss.core import tool_interrupt
from kiss.core.brand import BRAND, PRODUCT_NAME, render_brand
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import ACQUIRE_SLICE_SECONDS, LockTimeoutError, locked_call
from kiss.core.user_config import load_config
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # screenshots ride this socket as data URLs
CONFIG_FILE_NAME = "bridge-config.json"

_EXTENSION_SOURCE = Path(__file__).resolve().parent / "chrome_extension"
_PROBE_TIMEOUT_SECS = 1.0
_CONFIG_LOCK_TIMEOUT_SECS = 5.0
_REPLY_MARGIN_SECS = 1.0  # the extension's deadline runs at most this much short of ours
_PONG_TIMEOUT_SECS = 5.0  # a pong not written by then: the extension stopped reading
_SERVER_EXIT_SECS = 5.0  # a shut-down server's thread leaves its accept loop within moments


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


def bundled_extension(brand: dict[str, str] = BRAND) -> dict[str, bytes]:
    """Return the extension's files as they are installed, by name.

    The manifest names the extension after the brand (``{{PRODUCT_NAME}}
    Bridge``), so a white-label install never shows the stock product
    name in ``chrome://extensions``, and its ``version`` names the build:
    three numbers taken from a hash of both bundled files, so that any
    change to either is a new version.  The worker is the bundled one
    under a name of that build's own, ``background-<version>.js``, which
    the manifest points at (see :func:`install_extension`).

    Args:
        brand: The brand strings (the product's by default).

    Returns:
        The content of every file :func:`install_extension` writes, in
        the order it writes them: the worker, then the manifest.
    """
    worker = (_EXTENSION_SOURCE / "background.js").read_bytes()
    # The placeholders sit inside JSON strings: escape the brand values
    # (a quote or backslash in a product name must not break the file).
    escaped = {key: json.dumps(value)[1:-1] for key, value in brand.items()}
    source = (_EXTENSION_SOURCE / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(render_brand(source, escaped))
    digest = hashlib.sha256(worker + json.dumps(manifest).encode("utf-8")).digest()
    # Chrome takes up to four numbers, each at most 65535.
    build = (int.from_bytes(digest[at : at + 2], "big") for at in (0, 2, 4))
    manifest["version"] = "1." + ".".join(map(str, build))
    worker_name = f"background-{manifest['version']}.js"
    manifest["background"]["service_worker"] = worker_name
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    return {worker_name: worker, "manifest.json": text.encode("utf-8")}


def extension_version(brand: dict[str, str] = BRAND) -> str:
    """Return the version of the manifest :func:`install_extension` writes.

    The extension names it in its hello: a bridge takes only the
    extension Chrome loaded from the files that bridge installed.

    Args:
        brand: The brand strings (the product's by default).

    Returns:
        The version, ``1.<n>.<n>.<n>``.
    """
    return str(json.loads(bundled_extension(brand)["manifest.json"])["version"])


def install_extension() -> Path:
    """Install the bundled extension into :func:`extension_dir`.

    Chrome may load the extension at any moment, and it reads a worker in
    two steps, its size and about a millisecond later its bytes: a worker
    replaced in between by a longer one is cut off at the old size, and
    a worker that does not parse is gone until the extension is reloaded.
    So no worker file is ever replaced.  Every build's worker has a file
    of its own, written before the manifest that names it, and the
    manifest, which Chrome reads in one piece, is replaced atomically:
    whichever manifest Chrome reads, the worker it names is there, whole.

    Nor is a worker file removed: those of other builds (the
    ``background.js`` of an install from before this layout among them)
    stay.  A Chrome that loaded a build's manifest starts the worker it
    names from disk whenever the extension is switched on, for as long
    as that Chrome runs: without the file the extension has no worker
    ("Service worker registration failed") until it is reloaded by hand;
    with it the worker runs, is refused by the bridge and reloads the
    extension.  No install can tell whether such a Chrome is left, or
    whether another process is installing that build right now.  A
    worker is some 50 KB, and one is added only when the extension's
    code changes.

    Chrome runs the worker it loaded until the extension is reloaded or,
    the installed version being another by then, until Chrome starts
    again.  The worker reloads the extension itself when a bridge refuses
    its version (``refused`` in ``background.js``).  A worker from before
    that answer keeps running until the user reloads the extension on
    ``chrome://extensions``; the bridge refuses it and says so
    (:meth:`ChromeBridge.call`).

    Returns:
        The extension directory.
    """
    target = extension_dir()
    target.mkdir(parents=True, exist_ok=True)
    files = bundled_extension()
    for name, data in files.items():
        dest = target / name
        if not dest.exists() or dest.read_bytes() != data:
            atomic_write_text(dest, data.decode("utf-8"), create_mode=0o644)
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
    """One authenticated extension connection, its send lock and its commands.

    The locks are reentrant because task threads take them through
    :func:`~kiss.core.locked.locked_call`: a Stop (an injected
    ``KeyboardInterrupt``) landing right after a plain ``with``'s
    acquire would leave the lock held by a dead thread (CPython
    gh-74174), wedging every later command.
    """

    def __init__(self, ws: ServerConnection):
        self.ws = ws
        self.lock = threading.RLock()  # serializes sends
        self.pending: dict[int, Future] = {}
        self.pending_lock = threading.RLock()


class ChromeBridge:
    """Loopback WebSocket server the KISS Sorcar Bridge extension connects to.

    A process-wide singleton (see :meth:`instance`).  Commands are issued
    with :meth:`call`, which blocks the calling thread until the extension
    answers or the timeout expires.
    """

    _instance: ChromeBridge | None = None
    _instance_lock = threading.RLock()  # taken through locked_call: Stop-safe

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
        return locked_call(cls._instance_lock, cls._instance_locked, takeover)

    @classmethod
    def _instance_locked(cls, takeover: bool) -> ChromeBridge:
        if cls._instance is None:
            bridge = cls()
            try:
                bridge.start(takeover=takeover)
                atexit.register(bridge.stop)
                cls._instance = bridge
            except BaseException:
                # A Stop (an injected KeyboardInterrupt) landing anywhere before
                # the installation: a live server nobody owns would make every
                # later start defer to it as "another KISS process".  stop()
                # is idempotent: a failed start() has already run it.
                cls._instance = None
                bridge.stop()
                raise
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Stop the process-wide bridge, if one was started, and forget it.

        The daemon calls this on shutdown; the next :meth:`instance` call
        starts a fresh server.  A no-op when no bridge was started.
        """
        bridge = locked_call(cls._instance_lock, cls._forget_instance_locked)
        if bridge is not None:
            bridge.stop()

    @classmethod
    def _forget_instance_locked(cls) -> ChromeBridge | None:
        bridge, cls._instance = cls._instance, None
        return bridge

    def __init__(self, port: int = 0, write_config: bool = True):
        self._port = port
        self._write_config = write_config
        self._token = secrets.token_hex(16)
        self._ids = count(1)
        self._conn: _Connection | None = None
        self._connected = threading.Event()
        self._extension_version = extension_version()
        self._outdated = False  # the last extension to say hello was an older copy
        # Held while _conn and _connected change together: a drop that checks
        # "is this still the live connection?" must not clear a newer one.
        # Reentrant: task threads take it through locked_call (see _Connection).
        self._conn_lock = threading.RLock()
        self._server: Server | None = None
        self._server_thread: threading.Thread | None = None  # serves _server

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
        try:
            if self._write_config:
                self._publish(takeover)
            else:
                self._serve(self._bind())
            logger.info("chrome bridge listening on 127.0.0.1:%d", self._port)
        except BaseException:
            # A ChromeBridgeError (a socket bound by then has nobody to serve
            # it; a file written by then, when the thread could not start,
            # names it), or a Stop (an injected KeyboardInterrupt) landing
            # anywhere in here, after the config was written and the server
            # thread started included: a bridge its caller never got would
            # otherwise keep serving and own the file.
            self.stop()
            raise

    def _bind(self) -> Server:
        """Bind the loopback socket; connections queue on it until :meth:`_serve`.

        Raises:
            ChromeBridgeError: the socket cannot be bound.
        """
        try:
            server = serve(
                self._handle,
                "127.0.0.1",
                self._port,
                max_size=MAX_MESSAGE_BYTES,
            )
        except OSError as e:
            raise ChromeBridgeError(f"cannot listen on 127.0.0.1:{self._port}: {e}") from e
        self._server = server
        self._port = server.socket.getsockname()[1]
        return server

    def _serve(self, server: Server) -> None:
        """Serve the bound socket of *server* in a thread.

        Raises:
            ChromeBridgeError: the thread cannot start (thread exhaustion).
        """
        # Recorded before its start: a Stop landing in the start must not
        # leave a serving thread that stop() does not wait for.
        self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        try:
            self._server_thread.start()
        except RuntimeError as e:
            raise ChromeBridgeError(f"cannot start the bridge server thread: {e}") from e

    def _publish(self, takeover: bool) -> None:
        """Bind, write bridge-config.json and serve, unless a live bridge owns the file.

        The probe for another process's bridge, the write and the server
        thread's start are one critical section under the config lock: two
        processes starting together would otherwise both find no live
        bridge and both publish, and the one whose file lost would serve a
        port no extension ever reads.  The socket is bound after the
        probe: a start that defers to a live bridge (every
        ``use_my_chrome`` call of a CLI run beside the daemon) creates no
        server at all.  The thread starts after the write, still under
        the lock: the next starter's probe (which needs this server's
        challenge frame) waits for the lock and finds it answering, the
        extension's connection queues on the bound socket meanwhile, and
        a file that cannot be written starts no serving loop (websockets
        leaves the loop's selector, a descriptor, to the garbage
        collector).  The
        staged copy of the file has mode 600 from the start (the token
        must never sit in a world-readable file), and the lock keeps a
        predecessor's :func:`_remove_config_if_owned` from unlinking the
        file between its ownership check and its unlink.

        Raises:
            ChromeBridgeError: another process's bridge owns the file and
                *takeover* is False; the extension files or the config
                file cannot be written; the lock stays held past
                :data:`_CONFIG_LOCK_TIMEOUT_SECS`; the socket cannot be
                bound; the thread cannot start.
        """
        config = extension_dir() / CONFIG_FILE_NAME
        try:
            with _config_lock(config):
                if not takeover:
                    other = _live_bridge_port(config)
                    if other is not None:
                        raise ChromeBridgeError(
                            f"another KISS process already serves the Chrome bridge on port "
                            f"{other} (normally the kiss-web daemon, which the extension stays "
                            "connected to); run this task through that process"
                        )
                install_extension()
                server = self._bind()
                atomic_write_text(
                    config, json.dumps({"port": self._port, "token": self._token}), mode=0o600
                )
                self._serve(server)
        except OSError as e:  # an unwritable KISS_HOME, or the config lock stuck
            raise ChromeBridgeError(f"cannot publish the bridge in {extension_dir()}: {e}") from e

    def stop(self) -> None:
        """Stop the server, close the live connection, fail in-flight commands.

        The config file is removed when it still names this server, so the
        extension reports a missing bridge instead of retrying a dead port
        (a newer bridge's file, recognised by its token, is left alone).
        """
        server, thread = locked_call(self._conn_lock, self._close_listener)
        conn = self._conn  # one read: the handler thread may drop it meanwhile
        if conn is not None:
            self._drop_connection(conn, "bridge server stopped")
            # Shut down before the close, as call()'s timeout path does: a
            # close alone waits up to websockets' close_timeout (10 s) on a
            # peer that stopped reading, and blocks behind a sender stuck
            # in sendall(), which holds the protocol mutex.
            _shutdown_socket(conn.ws.socket)
            try:
                conn.ws.close()  # half-dead handlers keep the extension pinging otherwise
            except Exception:  # noqa: BLE001 - already gone
                logger.debug("Exception caught", exc_info=True)
        if self._write_config:
            _remove_config_if_owned(extension_dir() / CONFIG_FILE_NAME, self._token)
        if server is not None:
            # Last, and holding no lock: it can wait for the handlers of the
            # connections just closed, and nothing above may wait for it.
            _release_server(server, thread)

    def _close_listener(self) -> tuple[Server | None, threading.Thread | None]:
        """Stop listening and hand the server and its thread to one caller, under ``_conn_lock``.

        Under that lock a handshake still in flight (the listener's close
        does not end it) checks, before publishing its connection, that
        this bridge still runs.  Only the ``stop()`` that gets the server
        releases it, so two stops at once (the daemon's shutdown and the
        exit hook) never close its descriptors twice.  Nothing here
        waits: ``Server.shutdown()`` is that caller's, once the lock is
        free (:func:`_release_server`).  The server is forgotten after
        the close: a Stop landing in between leaves it to the next
        ``stop()``, which closes it again (harmless).
        """
        server, thread = self._server, self._server_thread
        if server is not None:
            server.socket.close()
            self._server = self._server_thread = None
        return server, thread

    def wait_for_extension(self, timeout: float) -> bool:
        """Block until an extension is connected or *timeout* seconds pass.

        A bridge started lazily by the first tool call is found by the
        extension on its next reconnect attempt (a few seconds); waiting
        here turns that first call's certain failure into a result.
        Waited in short slices, as the reply is (:func:`_result_before`):
        a task's Stop is not delivered inside one long wait.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True when an extension is connected.

        Raises:
            ToolCallInterrupted: When the calling tool call was stopped.
        """
        return tool_interrupt.wait_interruptible(self._connected, timeout)

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
        if conn is None and self._outdated:
            raise ChromeBridgeError(
                f"Chrome loaded its copy of the {PRODUCT_NAME} Bridge extension before the one "
                f"in {extension_dir()} was installed, and it is not running the new one. Open "
                "chrome://extensions, switch on Developer mode if it is off (Chrome runs the "
                "extension only with it), and click the reload button (the round arrow) on "
                "the extension's card. With Developer mode on, later updates load by "
                "themselves."
            )
        if conn is None:
            raise ChromeBridgeError(
                f"the {PRODUCT_NAME} Bridge extension is not connected. "
                f"Install it once: open chrome://extensions, enable Developer mode, "
                f"click 'Load unpacked' and choose {extension_dir()}. "
                "If it is installed, keep Developer mode on (Chrome switches the extension "
                "off without it) and click its toolbar icon to wake it."
            )
        msg_id = next(self._ids)
        fut: Future = Future()
        # The deadline covers queueing behind other sends too: a caller whose
        # command never got on the wire inside its timeout fails instead of
        # stacking up behind a wedged peer.
        deadline = time.monotonic() + timeout
        # Registered through completion inside one try, every lock taken
        # through locked_call: a Stop (an injected KeyboardInterrupt) landing
        # anywhere in between, including right after a blocking acquire
        # returned (CPython gh-74174), leaves neither the entry nor a lock
        # behind.
        try:
            locked_call(conn.pending_lock, conn.pending.__setitem__, msg_id, fut)
            try:
                locked_call(
                    conn.lock, self._send, conn, msg_id, cmd, args, deadline,
                    acquire_timeout=max(deadline - time.monotonic(), 0.0),
                )
            except LockTimeoutError:
                raise ChromeBridgeError(
                    f"the extension connection stayed busy past {timeout:.0f}s; "
                    f"'{cmd}' was not sent"
                ) from None
            try:
                return _result_before(fut, deadline)
            except TimeoutError:
                # A peer that stops answering would wedge every later command
                # behind its stuck socket; drop it so the extension reconnects
                # fresh.  The socket is shut down before the close: a close
                # alone waits up to websockets' close_timeout (10 s) for a
                # close frame this peer does not send, and blocks behind a
                # sender stuck in sendall(), which holds the protocol mutex;
                # the shutdown fails that write, gives the peer end-of-file
                # and makes the close return at once.
                if self._conn is conn:
                    self._drop_connection(conn, "command timed out")
                    _shutdown_socket(conn.ws.socket)
                    try:
                        conn.ws.close()
                    except Exception:  # noqa: BLE001 - already gone
                        logger.debug("Exception caught", exc_info=True)
                raise ChromeBridgeError(
                    f"the extension did not answer '{cmd}' within {timeout:.0f}s"
                ) from None
        finally:
            locked_call(conn.pending_lock, conn.pending.pop, msg_id, None)

    def _send(
        self, conn: _Connection, msg_id: int, cmd: str, args: dict[str, Any] | None, deadline: float
    ) -> None:
        """Write the command frame to *conn* (its send lock held) before *deadline*.

        The extension's deadline runs a little short of ours (a tenth, at
        most _REPLY_MARGIN_SECS): its "deadline exceeded" reply arrives
        while we still wait, instead of our timeout dropping the connection
        and every other command in flight.

        Raises:
            ChromeBridgeError: the frame could not be written; the
                connection is dropped.
        """
        budget = max(deadline - time.monotonic(), 0.0)
        timeout_ms = round((budget - min(budget / 10, _REPLY_MARGIN_SECS)) * 1000)
        frame = json.dumps({"id": msg_id, "cmd": cmd, "args": args or {}, "timeout_ms": timeout_ms})
        self._write(conn, frame, budget, f"'{cmd}'")

    def _write(self, conn: _Connection, frame: str, budget: float, what: str) -> None:
        """Write *frame* to *conn* (its send lock held) within *budget* seconds.

        The socket write has no deadline of its own: once a peer that
        stopped draining its socket has filled the buffers, ``sendall``
        blocks for good.  A watchdog shuts the socket down when the
        budget passes mid-write (``shutdown`` takes none of websockets'
        mutexes and fails the blocked write), and the connection is then
        dropped like any other failed send.  The watchdog is bound to
        *conn*'s socket and cuts nothing once the write has returned:
        ``cancel()`` does not stop a timer whose thread is already in
        its body, so the cut and the end of the write both go through
        the ``settled`` lock, which the write keeps once it has it.

        Raises:
            ChromeBridgeError: the frame could not be written (*what*
                names it); the connection is dropped.
        """
        cut = threading.Event()
        settled = threading.Lock()
        watchdog = threading.Timer(budget, _cut_socket, (conn.ws.socket, cut, settled))
        try:
            watchdog.start()
            conn.ws.send(frame)
        except Exception as e:
            if self._conn is conn:
                self._drop_connection(conn, "send failed")
            if cut.is_set():
                raise ChromeBridgeError(
                    f"the extension stopped reading; {what} could not be sent within "
                    f"the {budget:.0f}s left of its timeout"
                ) from None
            raise ChromeBridgeError(f"lost the extension connection: {e}") from e
        finally:
            watchdog.cancel()
            settled.acquire()  # waits out a cut under way: two calls that never block

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
            if first.get("version") != self._extension_version:
                # Chrome runs a copy of the extension it loaded before these
                # files were installed, or another process's install replaced
                # them: they are put back, and the worker, so refused, finds
                # another version on disk than its own and reloads the
                # extension.  A worker from before that answer existed runs
                # on: the user reloads the extension once.
                self._outdated = True
                if self._write_config:
                    install_extension()
                ws.send(json.dumps({"type": "hello_error", "error": "outdated extension"}))
                ws.close(4004, "outdated extension")
                return
            ws.send(json.dumps({"type": "hello_ok", "proof": _proof(nonce, self._token)}))
            with self._conn_lock:
                if self._server is None:  # stopped while this handshake was in flight
                    return  # closed on return: the extension looks for a live bridge
                old, self._conn = self._conn, conn
                self._connected.set()
                self._outdated = False
            if old is not None:  # a second Chrome profile connected: keep the newest
                # Closed off this thread: the close waits up to websockets'
                # close_timeout (10 s) for the old peer's close frame, and a
                # wedged old peer would hold this connection's reads (below)
                # for that long, so a short command on the healthy new
                # connection would time out and drop it.
                threading.Thread(target=_close_replaced, args=(old.ws,), daemon=True).start()
            logger.info("chrome bridge extension connected")
            while True:
                self._on_message(conn, json.loads(ws.recv()))
        except Exception:  # noqa: BLE001 - any recv/parse failure ends the connection
            logger.debug("Exception caught", exc_info=True)
        finally:
            self._drop_connection(conn, "extension disconnected")

    def _on_message(self, conn: _Connection, msg: dict[str, Any]) -> None:
        if msg.get("type") == "ping":
            # Bounded like a command's write: unbounded, a pong to a peer
            # that stopped reading held the send lock for good, so every
            # command failed as busy while the peer still counted as live.
            with conn.lock:
                self._write(conn, json.dumps({"type": "pong"}), _PONG_TIMEOUT_SECS, "the pong")
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
        # Also run on task threads (call()'s timeout path, stop()), so both
        # sections go through locked_call (see _Connection).
        def forget() -> None:
            if self._conn is conn:
                self._conn = None
                self._connected.clear()

        def take_pending() -> list[Future]:
            pending = list(conn.pending.values())
            conn.pending.clear()
            return pending

        locked_call(self._conn_lock, forget)
        for fut in locked_call(conn.pending_lock, take_pending):
            if not fut.done():
                fut.set_exception(ChromeBridgeError(reason))


def _result_before(fut: Future, deadline: float) -> Any:
    """Return *fut*'s result, waiting until *deadline* in short slices.

    ``Future.result(timeout)`` blocks in one C-level lock wait, where an
    injected ``KeyboardInterrupt`` (a task Stop) is not delivered: a Stop
    would wait out the whole command timeout.  Sliced like the lock
    acquisitions of :func:`~kiss.core.locked.locked_call`, the Stop lands
    within :data:`~kiss.core.locked.ACQUIRE_SLICE_SECONDS`.

    Raises:
        TimeoutError: *deadline* passed before the future completed.
    """
    while True:
        try:
            return fut.result(min(max(deadline - time.monotonic(), 0.0), ACQUIRE_SLICE_SECONDS))
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise


def _close_replaced(ws: ServerConnection) -> None:
    """Close the extension connection *ws* with code 4002, which keeps that profile down.

    The extension stays away on 4002 (it was replaced by another Chrome
    profile); any other close code makes it reconnect, and the two
    profiles would replace each other forever.  Its own handler thread is
    woken by the close and drops the connection as a disconnect.
    """
    try:
        ws.close(4002, "replaced by a newer connection")
    except Exception:  # noqa: BLE001 - already gone
        logger.debug("Exception caught", exc_info=True)


def _cut_socket(sock: socket.socket, cut: threading.Event, settled: threading.Lock) -> None:
    """Watchdog body of :meth:`ChromeBridge._write`: fail a write blocked on a full buffer.

    ``shutdown`` wakes the thread blocked in ``sendall`` with an error
    (and the connection's reader with end-of-file) without taking any
    of websockets' mutexes, which that thread holds.

    Args:
        sock: The connection's socket.
        cut: Set before the shutdown, so the sender can tell the deadline
            from any other write failure.
        settled: Held for the cut; kept by the write once it has
            returned, and nothing is cut then.
    """
    if not settled.acquire(blocking=False):
        return
    try:
        cut.set()
        _shutdown_socket(sock)
    finally:
        settled.release()


def _shutdown_socket(sock: socket.socket) -> None:
    """Shut *sock* down both ways: a blocked write fails and the peer gets end-of-file.

    Takes none of websockets' mutexes, so it works from any thread while
    a sender holds them; a socket the peer already closed is left alone.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:  # already closed by the peer
        logger.debug("Exception caught", exc_info=True)


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


def _release_server(server: Server, thread: threading.Thread | None) -> None:
    """Shut down *server*, whose listening socket is closed, and close what it leaves open.

    Called once per server (:meth:`ChromeBridge._close_listener`), holding
    no lock.  ``Server.shutdown()`` wakes the serving loop; from
    websockets 17 on it also waits for that loop and for every
    connection handler to end.  A handler takes the bridge's lock before
    it ends, and a loop that never started (the publication failed after
    the bind, or the thread could not start) would be waited for for
    good: such a server is not shut down, its closed socket serves
    nobody.

    What stays open is the loop's wake-up channel: a pipe that websockets
    15 never closes (two descriptors per server, and a start that fails
    after the bind makes one server per ``use_my_chrome`` call), a socket
    pair that websockets 17 closes in its shutdown and from its loop, so
    not for a server that had neither.  It is closed once *thread* has
    left its selector, which watches one end: closed under it, the thread
    would sleep for good; one still serving after
    :data:`_SERVER_EXIT_SECS` keeps it.
    """
    if thread is not None and thread.ident is not None:  # its loop started
        try:
            server.shutdown()
        except RuntimeError:
            # websockets 17 at interpreter exit: the thread pool it closes
            # other open connections on (a handshake in flight) is refused.
            # They end with the process.
            logger.debug("Exception caught", exc_info=True)
        thread.join(_SERVER_EXIT_SECS)
        if thread.is_alive():  # pragma: no cover - the closed socket ends its accept loop
            logger.warning("the chrome bridge server thread did not exit; its pipe stays open")
            return
    for name in ("shutdown_watcher", "shutdown_notifier"):  # absent on Windows
        end = getattr(server, name, None)
        if isinstance(end, int):
            os.close(end)
        elif end is not None:
            end.close()


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
