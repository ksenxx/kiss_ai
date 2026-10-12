# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Remote Sorcar machines: the registry and the cross-machine task history.

A *machine* is another ``kiss-web`` daemon the local daemon may query.
The registry lives in ``$KISS_HOME/machines.json`` (mode 0600, as it may
hold a remote password) and every machine's TLS certificate is pinned at
registration time under ``$KISS_HOME/machines/<name>.pem``: the daemon's
local CA when it serves one (``/ca.crt``), otherwise its leaf certificate.
Later connections verify the chain against that file only (hostname
verification is off: a daemon's auto-generated certificate names its
LAN addresses, not necessarily the public one the user typed).

A machine registered as ``user@host`` (an *SSH machine*) is reached
through an SSH tunnel instead of a public address and needs no
password: :class:`SshTunnel` reads the remote daemon's endpoint file
(``~/<brand home>/sorcar-local.json``: its loopback port and per-start
local token) with ``ssh user@host cat ...``, forwards a free local
port to that loopback port with ``ssh -N -L`` and the link
authenticates through the tunnel with the token, which the remote
daemon accepts from a loopback peer whatever its ``remote_password``.
The ``ssh`` client's own configuration applies (``~/.ssh/config``
aliases, keys, agent), or the file named by ``$KISS_SSH_CONFIG``; the
login must work non-interactively (``BatchMode``), land in a POSIX
shell (the endpoint is read with ``sh -c``, so a Windows host whose
OpenSSH logs in to PowerShell cannot be an SSH machine) and an unknown
host key is accepted on first use.

:func:`merged_history` is what the daemon's ``getHistory`` handler uses
to show the tasks of every machine in the CHATS panel: it asks each
remote daemon for the same page (same query and tag, its own rows only)
over a persistent authenticated WebSocket (:class:`MachineLink`),
stamps every row with its ``machine`` name and the machine's URL, keeps
one copy per task id (a database copied between machines yields the
same ids on both) and re-sorts the union the way the local list is
sorted: running tasks first, then newest first.  The ``/machine`` SEA
(:mod:`kiss.agents.seas.machine.machine_sea`) is the user-facing way to
edit the registry through :func:`add_machine` and :func:`remove_machine`.
"""

from __future__ import annotations

import atexit
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as _ws_connect

from kiss.agents.sorcar.local_endpoint import LOCAL_ENDPOINT_FILE, send
from kiss.core.brand import HOME_DIR
from kiss.core.config import kiss_home
from kiss.core.file_lock import lock_exclusive, unlock

logger = logging.getLogger(__name__)

MACHINES_FILE = "machines.json"
MACHINE_CERTS_DIR = "machines"
DEFAULT_PORT = 8787
HISTORY_PAGE = 50
"""Rows per ``getHistory`` page (``server._get_history``'s page size)."""
LOCAL_MACHINE = "__local__"
"""``getHistory``'s ``machine`` value asking for the daemon's own rows only."""
REQUEST_TIMEOUT = 8.0
"""Seconds one remote request may take before the machine counts as down."""
DOWN_BACKOFF = 30.0
"""Seconds a cached link waits after a failure before trying the machine again."""
SSH_TIMEOUT = 30.0
"""Seconds the two ``ssh`` logins that open an SSH machine's tunnel may take together."""
SSH_CONFIG_ENV = "KISS_SSH_CONFIG"
"""Environment variable naming an ``ssh`` config file to use instead of the user's."""
REMOTE_ENDPOINT_COMMAND = (
    f"sh -c 'cd \"${{KISS_HOME:-$HOME/{HOME_DIR}}}\" && cat {LOCAL_ENDPOINT_FILE} "
    "&& cat remote-url.json 2>/dev/null'"
)
"""Remote command printing the daemon's endpoint file, then its published URLs if any
(both under ``$KISS_HOME`` or the brand home)."""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class MachineError(Exception):
    """A registry or connection problem, with a message fit for the user."""


@dataclass
class Machine:
    """One registered remote daemon.

    Attributes:
        name: Short label shown in the CHATS panel (``[A-Za-z0-9._-]``).
        url: ``https://host:port`` of the remote ``kiss-web``; empty for
            an SSH machine, whose address is a tunnel opened on demand.
        password: Its ``remote_password``; empty for an SSH machine and
            for a daemon reached through a tunnel the user opened.
        ssh: ``user@host[:port]`` of an SSH machine, empty otherwise.
    """

    name: str
    url: str = ""
    password: str = ""
    ssh: str = ""

    def address(self) -> str:
        """Return the address the user registered (``ssh user@host`` or the URL)."""
        return f"ssh {self.ssh}" if self.ssh else self.url

    def cert_file(self) -> Path:
        """Return the path of the certificate pinned for this machine."""
        return kiss_home() / MACHINE_CERTS_DIR / f"{self.name}.pem"


def ws_url(url: str) -> str:
    """Return the WebSocket endpoint of the daemon at *url* (``wss://host:port/ws``)."""
    return "wss://" + url[len("https://"):] + "/ws"


@dataclass
class HistoryPage:
    """Rows of a history reply and the date range of the history they come from."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    date_min: float | None = None
    date_max: float | None = None


def machines_file() -> Path:
    """Return the registry path, ``$KISS_HOME/machines.json``."""
    return kiss_home() / MACHINES_FILE


def local_machine_name() -> str:
    """Return this host's short name, the label of the local daemon's rows."""
    return socket.gethostname().strip().split(".")[0] or "localhost"


def load_machines() -> list[Machine]:
    """Return the registered machines in registration order ([] when none)."""
    try:
        raw = json.loads(machines_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    machines: list[Machine] = []
    for item in raw.get("machines", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        if not item.get("url") and not item.get("ssh"):
            continue
        machines.append(Machine(
            name=str(item["name"]), url=str(item.get("url", "")),
            password=str(item.get("password", "")), ssh=str(item.get("ssh", "")),
        ))
    return machines


def machine_host(machine: Machine) -> str:
    """Return the host of *machine*'s address (the URL's hostname, or the SSH login's)."""
    text = machine.url or "ssh://" + machine.ssh
    try:
        return urlsplit(text).hostname or ""
    except ValueError:
        return ""


def is_local_machine(name_or_host: str) -> bool:
    """Return whether *name_or_host* names this machine.

    True for :data:`LOCAL_MACHINE`, ``localhost``, this host's name
    (short or fully qualified, see :func:`local_machine_name`), a
    loopback address, or an address of one of this host's interfaces
    (whatever DNS says: a VPN or container bridge address counts).
    Case-insensitive; IPv6 brackets and spelling are normalised.
    """
    text = name_or_host.strip().strip("[]").lower()
    if not text:
        return False
    names = {LOCAL_MACHINE, "localhost", local_machine_name().lower(),
             socket.gethostname().lower()}
    try:
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    if text in names:
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    # An address is this host's when a socket can be bound to it (no
    # packet is sent); the one portable way to ask the interfaces.
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.bind((str(address), 0))
            return True
    except OSError:
        pass
    # A link-local IPv6 address cannot be bound without its interface
    # (`fe80::1%en0`); the host's own name still resolves to it, scope
    # and all, so ask DNS last.
    try:
        resolved = socket.getaddrinfo(socket.gethostname(), None)
    except OSError:
        return False
    bare = ipaddress.ip_address(str(address).split("%")[0])
    return any(
        ipaddress.ip_address(str(info[4][0]).split("%")[0]) == bare for info in resolved
    )


def find_machine(name_or_host: str) -> Machine | None:
    """Return the registered machine called *name_or_host*, or the one at that host.

    A host (a name or an IP address, an IPv6 one with or without
    brackets) matches the hostname of an HTTPS machine's URL or the
    host of an SSH machine's ``user@host`` login; a name wins over a
    host.  ``None`` when no machine matches.
    """
    text = name_or_host.strip()
    host = text.strip("[]").lower()
    machines = load_machines()
    for machine in machines:
        if machine.name == text:
            return machine
    for machine in machines:
        if host and machine_host(machine).lower() == host:
            return machine
    return None


def save_machines(machines: list[Machine]) -> None:
    """Write the registry atomically (created 0600) and drop cached links."""
    path = machines_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{MACHINES_FILE}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"machines": [asdict(m) for m in machines]}, handle, indent=2)
    os.replace(tmp, path)
    close_links()


@contextmanager
def registry_lock() -> Iterator[None]:
    """Hold the cross-process lock serialising registry read-modify-writes."""
    path = machines_file().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        lock_exclusive(handle)
        try:
            yield
        finally:
            unlock(handle)


def normalize_url(url: str) -> str:
    """Return *url* as ``https://host:port``.

    Accepts ``host``, ``host:port``, ``https://host[:port][/path]`` and
    ``wss://host[:port][/ws]``; the port defaults to ``8787``.

    Raises:
        MachineError: When *url* names no host or has a malformed port.
    """
    text = url.strip()
    if "://" not in text:
        text = "https://" + text
    parts = urlsplit(text)
    try:
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise MachineError(f"'{url}' is not a machine address: {exc}") from exc
    if parts.scheme not in ("https", "wss", "http", "ws") or not host:
        raise MachineError(f"'{url}' is not a machine address; expected https://host:port")
    return f"https://{_bracketed(host)}:{port or DEFAULT_PORT}"


def _bracketed(host: str) -> str:
    """Return *host* bracketed when it is an IPv6 literal."""
    return f"[{host}]" if ":" in host else host


def is_ssh_address(address: str) -> bool:
    """Return True when *address* names an SSH login (``user@host`` or ``ssh://...``)."""
    text = address.strip()
    return text.startswith("ssh://") or ("@" in text and "://" not in text)


def normalize_ssh(address: str) -> str:
    """Return *address* as ``user@host`` or ``user@host:port``.

    Accepts ``user@host``, ``user@host:port`` and ``ssh://[user@]host[:port]``.

    Raises:
        MachineError: When *address* names no host.
    """
    text = address.strip()
    parts = urlsplit(text if text.startswith("ssh://") else "ssh://" + text)
    try:
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise MachineError(f"'{address}' is not an SSH address: {exc}") from exc
    if parts.scheme != "ssh" or not host or parts.path not in ("", "/"):
        raise MachineError(f"'{address}' is not an SSH address; expected user@host[:port]")
    login = f"{parts.username}@{_bracketed(host)}" if parts.username else _bracketed(host)
    return f"{login}:{port}" if port else login


def default_name(address: str) -> str:
    """Return the machine name implied by *address*: its host, made a valid name."""
    text = address.strip()
    if "://" not in text:
        text = ("ssh://" if is_ssh_address(text) else "https://") + text
    host = urlsplit(text).hostname or "machine"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", host).strip("._-") or "machine"


def ssh_command(ssh: str, *options: str, command: str = "") -> list[str]:
    """Return the ``ssh`` command line logging into *ssh* (``user@host[:port]``).

    *options* go before the destination, *command* (if any) after it.
    Key-based, non-interactive login only (``BatchMode``); an unknown
    host key is accepted and remembered on first use; the user's ssh
    configuration applies unless ``$KISS_SSH_CONFIG`` names another,
    except connection multiplexing (``ControlMaster``), which is off so
    that the tunnel is a process of its own that ends with the daemon.
    """
    parts = urlsplit("ssh://" + ssh)
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "LogLevel=ERROR", *options,
    ]
    config = os.environ.get(SSH_CONFIG_ENV)
    if config:
        cmd += ["-F", config]
    if parts.port:
        cmd += ["-p", str(parts.port)]
    host = parts.hostname or ""
    cmd.append(f"{parts.username}@{host}" if parts.username else host)
    return cmd + [command] if command else cmd


def pin_certificate(url: str) -> str:
    """Fetch and return the PEM to trust for the daemon at *url*.

    Prefers the daemon's local CA (``GET /ca.crt``) so the leaf
    certificate may be renewed later; when the daemon serves no CA (it
    was started with an explicit ``--certfile``), pins the leaf itself.

    Raises:
        MachineError: When the daemon cannot be reached.
    """
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(
            url + "/ca.crt", timeout=REQUEST_TIMEOUT, context=insecure,
        ) as resp:
            return str(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        pass
    except (OSError, ValueError) as exc:
        raise MachineError(f"Cannot reach {url}: {exc}") from exc
    parts = urlsplit(url)
    assert parts.hostname is not None
    try:
        return ssl.get_server_certificate(
            (parts.hostname, parts.port or DEFAULT_PORT), timeout=REQUEST_TIMEOUT,
        )
    except OSError as exc:
        raise MachineError(f"Cannot read the TLS certificate of {url}: {exc}") from exc


def add_machine(name: str, address: str, password: str = "") -> Machine:
    """Register (or re-register) *name* at *address* after a successful handshake.

    *address* is ``https://host:port`` (``host[:port]``) with the
    daemon's *password*, or ``user@host[:port]`` for an SSH machine
    (*password* unused: the tunnel authenticates with the remote's
    local token).  An empty *name* defaults to the host.  Pins the
    daemon's certificate to a staging file, authenticates once through
    it, and only then publishes the pin and the registry entry, so a
    wrong address or password changes nothing (an existing machine of
    that name keeps its pin and entry).

    Raises:
        MachineError: On a bad name or address, an unreachable daemon
            or a rejected password.
    """
    name = name or default_name(address)
    if not _NAME_RE.match(name):
        raise MachineError(
            f"'{name}' is not a valid machine name: letters, digits, '.', '_' and '-' only"
        )
    if is_ssh_address(address):
        machine = Machine(name=name, ssh=normalize_ssh(address))
    else:
        machine = Machine(name=name, url=normalize_url(address), password=password)
    cert = machine.cert_file()
    cert.parent.mkdir(parents=True, exist_ok=True)
    staged = cert.with_name(f".{cert.name}.{os.getpid()}.{secrets.token_hex(4)}.new")
    link = MachineLink(machine, cert_file=staged)
    try:
        staged.write_text(pin_certificate(link.base_url()), encoding="utf-8")
        link.request({"type": "ping"}, "pong")
    except MachineError:
        staged.unlink(missing_ok=True)
        raise
    finally:
        link.close()
    with registry_lock():
        os.replace(staged, cert)
        machines = [m for m in load_machines() if m.name != name]
        machines.append(machine)
        save_machines(machines)
    return machine


def remove_machine(name: str) -> bool:
    """Forget *name*; return False when no such machine is registered."""
    with registry_lock():
        machines = load_machines()
        kept = [m for m in machines if m.name != name]
        if len(kept) == len(machines):
            return False
        save_machines(kept)
        for suffix in (".pem", ".ssh.log"):
            (kiss_home() / MACHINE_CERTS_DIR / f"{name}{suffix}").unlink(missing_ok=True)
    return True


def machine_status(machine: Machine) -> str:
    """Return ``"reachable"`` or the reason the daemon cannot be used."""
    link = MachineLink(machine)
    try:
        link.request({"type": "ping"}, "pong")
        return "reachable"
    except MachineError as exc:
        return str(exc)
    finally:
        link.close()


def _free_port() -> int:
    """Return a loopback TCP port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


class SshTunnel:
    """An ``ssh -N -L`` forward of an SSH machine's daemon port, plus its token.

    :meth:`open` reads the remote endpoint file (the daemon's loopback
    port and local token) and its published URLs, starts ``ssh -N -L
    127.0.0.1:<free port>:127.0.0.1:<daemon port>`` and waits for the
    local port to accept connections; :attr:`url`, :attr:`token` and
    :attr:`web_url` describe the tunnel last opened (they stay readable
    after :meth:`close`, for rows fetched through it).  The ``ssh``
    process lives until :meth:`close` (or the daemon exits); its
    messages go to ``$KISS_HOME/machines/<name>.ssh.log``.
    """

    def __init__(self, machine: Machine) -> None:
        self.machine = machine
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""
        """``https://127.0.0.1:<local port>`` of the tunnel last opened (empty before)."""
        self.token = ""
        """The remote daemon's local token read when the tunnel was opened."""
        self.web_url = ""
        """Where a browser reaches the remote web UI: the daemon's public (cloudflared)
        URL, else its first LAN URL, else the tunnel's :attr:`url` (this host only)."""
        self.command: list[str] = []
        """The ``ssh -N -L ...`` command line last started."""

    def is_open(self) -> bool:
        """Return True while the ``ssh`` process is running."""
        return self.process is not None and self.process.poll() is None

    def log_file(self) -> Path:
        """Return the file the ``ssh`` process writes its messages to."""
        return kiss_home() / MACHINE_CERTS_DIR / f"{self.machine.name}.ssh.log"

    def open(self, timeout: float = SSH_TIMEOUT) -> None:
        """Start the tunnel unless it is up, within *timeout* seconds.

        Raises:
            MachineError: When ``ssh`` is missing, the login fails, no
                daemon endpoint file exists on the machine or the
                forward does not come up in time.
        """
        if self.is_open():
            return
        self.close()
        deadline = time.monotonic() + timeout
        endpoint, published = self._read_endpoint(deadline)
        listener = urlsplit(endpoint["url"])
        remote_host = _bracketed(listener.hostname or "127.0.0.1")
        remote_port = listener.port or DEFAULT_PORT
        local_port = _free_port()
        log = self.log_file()
        log.parent.mkdir(parents=True, exist_ok=True)
        self.command = ssh_command(
            self.machine.ssh, "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{local_port}:{remote_host}:{remote_port}",
        )
        with open(log, "wb") as handle:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=handle, start_new_session=True,
            )
        while True:
            if self.process.poll() is not None:
                raise self._failed(f"ssh {self.machine.ssh} failed", log.read_bytes())
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise self._failed(
                        f"ssh {self.machine.ssh} did not forward port {remote_port} "
                        f"within {timeout:g}s", log.read_bytes(),
                    ) from None
                time.sleep(0.1)
        self.url = f"https://127.0.0.1:{local_port}"
        self.token = endpoint["token"]
        lan = published.get("lan")
        self.web_url = str(
            published.get("tunnel")
            or (lan[0] if isinstance(lan, list) and lan and lan[0] else "")
            or self.url
        )

    def _read_endpoint(self, deadline: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the remote endpoint file's content and its published URLs (``{}`` if none).

        Raises:
            MachineError: When ``ssh`` is missing, the login fails or
                the endpoint file is missing or not a daemon's.
        """
        cmd = ssh_command(self.machine.ssh, command=REMOTE_ENDPOINT_COMMAND)
        try:
            run = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True,
                timeout=_remaining(deadline), check=False,
            )
        except FileNotFoundError as exc:
            raise MachineError("ssh is not installed on this machine") from exc
        except subprocess.TimeoutExpired as exc:
            raise MachineError(f"ssh {self.machine.ssh} did not answer in time") from exc
        if run.returncode == 255:
            raise self._failed(f"ssh {self.machine.ssh} failed", run.stderr)
        endpoint, published = _decode_two(run.stdout.decode("utf-8", "replace"))
        if (
            not isinstance(endpoint.get("url"), str)
            or not isinstance(endpoint.get("token"), str)
            or not endpoint["token"]
        ):
            raise self._failed(
                f"no Sorcar daemon is running on {self.machine.ssh} "
                f"(cannot read its {LOCAL_ENDPOINT_FILE})", run.stderr,
            )
        return endpoint, published

    def _failed(self, message: str, stderr: bytes) -> MachineError:
        """Return a :class:`MachineError` of *message* and ssh's last line, tunnel closed."""
        self.close()
        lines = stderr.decode("utf-8", "replace").strip().splitlines()
        return MachineError(f"{message}: {lines[-1]}" if lines else message)

    def close(self) -> None:
        """Stop the ``ssh`` process, if any (the address and token stay readable)."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None


def _decode_two(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode up to two concatenated JSON objects in *text* (``{}`` for a missing or bad one)."""
    objects: list[dict[str, Any]] = []
    rest = text.lstrip()
    while rest and len(objects) < 2:
        try:
            value, end = json.JSONDecoder().raw_decode(rest)
        except ValueError:
            break
        objects.append(value if isinstance(value, dict) else {})
        rest = rest[end:].lstrip()
    objects += [{}] * (2 - len(objects))
    return objects[0], objects[1]


class MachineLink:
    """One persistent, authenticated WebSocket to a remote daemon.

    Connects lazily on the first :meth:`request` (for an SSH machine,
    opening its :class:`SshTunnel` first), re-connects after a failure
    and serialises requests, so the daemon keeps one socket per machine
    across CHATS refreshes.  After a failure the link reports the
    machine down without trying again for :data:`DOWN_BACKOFF` seconds,
    so an offline machine does not add a connection timeout to every
    CHATS refresh and a rejected password is not retried often enough
    to trip the remote daemon's brute-force lockout.
    """

    def __init__(self, machine: Machine, cert_file: Path | None = None) -> None:
        self.machine = machine
        self.tunnel = SshTunnel(machine) if machine.ssh else None
        self._cert_file = cert_file if cert_file is not None else machine.cert_file()
        self._ws: ClientConnection | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._retry_at = 0.0
        self._last_error = ""
        self._retired = False

    def base_url(self) -> str:
        """Return the ``https://host:port`` the link talks to, opening the tunnel if needed.

        Raises:
            MachineError: When an SSH machine's tunnel cannot be opened
                (the link then backs off like after any failure).
        """
        if self.tunnel is None:
            return self.machine.url
        with self._lock:
            self._open_tunnel()
            return self.tunnel.url

    def _open_tunnel(self) -> None:
        """Open the SSH tunnel unless it is up; a failure starts the backoff."""
        assert self.tunnel is not None
        if self._retired:
            raise MachineError(f"{self.machine.name}: the machine registry changed, try again")
        if time.monotonic() < self._retry_at:
            raise MachineError(self._last_error)
        try:
            self.tunnel.open()
        except MachineError as exc:
            raise self._fail(f"{self.machine.name}: {exc}") from exc

    def _fail(self, message: str) -> MachineError:
        """Record *message* as the link's failure, start the backoff, return the error.

        An SSH machine's tunnel is closed too: the next attempt reads the
        endpoint file again, so a daemon restarted on the machine (new
        port, new token) is found without any registry change.
        """
        self._retry_at = time.monotonic() + DOWN_BACKOFF
        self._last_error = message
        if self.tunnel is not None:
            self.tunnel.close()
        return MachineError(message)

    def _connect(self, deadline: float) -> ClientConnection:
        """Open and authenticate the socket before the monotonic *deadline*.

        An SSH machine (tunnel already open) is authenticated with its
        local token, any other with the registered password.
        """
        machine = self.machine
        where = f"{machine.name} ({machine.address()})"
        if self._retired:
            raise MachineError(f"{machine.name}: the machine registry changed, try again")
        if time.monotonic() < self._retry_at:
            raise MachineError(self._last_error)
        try:
            ctx = ssl.create_default_context(cafile=str(self._cert_file))
        except (OSError, ssl.SSLError) as exc:
            raise self._fail(f"{where} has no usable pinned certificate: {exc}") from exc
        ctx.check_hostname = False
        if self.tunnel is not None:
            url, auth = self.tunnel.url, {"type": "auth", "token": self.tunnel.token}
        else:
            url, auth = machine.url, {"type": "auth", "password": machine.password}
        try:
            ws = _ws_connect(
                ws_url(url), ssl=ctx, open_timeout=_remaining(deadline),
                close_timeout=2.0, compression=None,
            )
        except (OSError, InvalidHandshake, InvalidURI, ssl.SSLError) as exc:
            raise self._fail(f"{where} is unreachable: {exc}") from exc
        try:
            send(ws, json.dumps(auth), timeout=_remaining(deadline))
            reply = json.loads(ws.recv(timeout=_remaining(deadline)))
        except (ConnectionClosed, OSError, ValueError) as exc:
            ws.close()
            raise self._fail(f"{where} did not complete the handshake: {exc}") from exc
        if not isinstance(reply, dict) or reply.get("type") != "auth_ok":
            ws.close()
            if isinstance(reply, dict) and reply.get("type") == "auth_required":
                raise self._fail(f"{where} rejected the password")
            raise self._fail(f"{where} refused the connection: {reply!r}")
        return ws

    def request(
        self, cmd: dict[str, Any], reply_type: str, timeout: float = REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* event answering it.

        The command is stamped with a fresh ``generation``; a reply that
        echoes a ``generation`` must echo this one (``history`` does),
        other frames the daemon pushes meanwhile are skipped.  *timeout*
        bounds the whole call (connecting included), however many
        frames arrive or are already buffered; only the SSH tunnel of
        an SSH machine, when it has to be (re)opened first, has its own
        :data:`SSH_TIMEOUT` budget.

        Raises:
            MachineError: When the daemon is unreachable, rejects the
                password, or does not answer within *timeout* seconds.
        """
        with self._lock:
            if self.tunnel is not None and not self.tunnel.is_open():
                self._drop()
                self._open_tunnel()
            deadline = time.monotonic() + timeout
            if self._ws is None:
                self._ws = self._connect(deadline)
            self._generation += 1
            generation = self._generation
            try:
                # Bounded like the reads: a plain ``ws.send`` has no
                # deadline, and one stuck under this lock (the daemon
                # stopped reading) would wedge every later request to
                # the machine, and the link's close at exit.
                send(
                    self._ws, json.dumps(cmd | {"generation": generation}),
                    timeout=_remaining(deadline),
                )
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"no {reply_type} within {timeout:g}s")
                    msg = json.loads(self._ws.recv(timeout=_remaining(deadline)))
                    if not isinstance(msg, dict) or msg.get("type") != reply_type:
                        continue
                    if "generation" in msg and msg["generation"] != generation:
                        continue
                    return msg
            except (ConnectionClosed, TimeoutError, ValueError, OSError) as exc:
                self._drop()
                raise self._fail(
                    f"{self.machine.name} ({self.machine.address()}) did not answer: {exc}"
                ) from exc

    def _drop(self) -> None:
        """Close and forget the socket (no error is recorded)."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                logger.debug("closing link to %s failed", self.machine.name, exc_info=True)
            self._ws = None

    def open_connection(self, timeout: float = REQUEST_TIMEOUT) -> ClientConnection:
        """Open a fresh authenticated socket to the daemon for a long-lived stream.

        A stream (a chat tab running on the machine, see
        :mod:`kiss.server.remote_tabs`) relays frames for as long as it
        lives, so it cannot take turns on the socket :meth:`request`
        shares; an SSH machine's tunnel is shared.  The caller owns the
        returned socket and closes it.

        Raises:
            MachineError: When the daemon is unreachable or rejects the
                password (the link backs off as after a failed request).
        """
        with self._lock:
            if self.tunnel is not None and not self.tunnel.is_open():
                self._drop()
                self._open_tunnel()
            return self._connect(time.monotonic() + timeout)

    def close(self) -> None:
        """Close the socket and the tunnel; the next :meth:`request` re-connects."""
        with self._lock:
            self._drop()
            if self.tunnel is not None:
                self.tunnel.close()

    def retire(self) -> None:
        """Close for good: every later :meth:`request` fails instead of re-connecting.

        Waits for a request in flight to finish.  Used when the link
        leaves the daemon-wide cache, so a caller that fetched it just
        before cannot re-open a tunnel nothing closes any more.
        """
        with self._lock:
            self._retired = True
            self._drop()
            if self.tunnel is not None:
                self.tunnel.close()


def _remaining(deadline: float) -> float:
    """Seconds left before the monotonic *deadline* (a small positive floor)."""
    return max(0.001, deadline - time.monotonic())


_links: dict[tuple[str, str, str, str], MachineLink] = {}
_links_lock = threading.Lock()


def link_for(machine: Machine) -> MachineLink:
    """Return the daemon-wide cached link to *machine*."""
    key = (machine.name, machine.url, machine.password, machine.ssh)
    with _links_lock:
        link = _links.get(key)
        if link is None:
            link = _links[key] = MachineLink(machine)
        return link


def close_links() -> None:
    """Retire and forget every cached link and tunnel (after a registry change, at exit)."""
    with _links_lock:
        links = list(_links.values())
        _links.clear()
    for link in links:
        link.retire()


atexit.register(close_links)


def fetch_history(machine: Machine, query: str, tag: str, limit: int) -> HistoryPage:
    """Return up to *limit* history rows of *machine* matching *query* and *tag*.

    Pages through the remote ``getHistory`` (``HISTORY_PAGE`` rows per
    reply), asking for the machine's own rows only (``machine:
    LOCAL_MACHINE``) so two daemons registered with each other do not
    query each other without end, and stamps every row with ``machine``
    (the name) and ``machine_url`` (the registered URL, or an SSH
    machine's :attr:`SshTunnel.web_url`).  The page's date range is
    the remote history's.

    Raises:
        MachineError: When the machine cannot be queried.
    """
    link = link_for(machine)
    page = HistoryPage()
    while len(page.rows) < limit:
        reply = link.request({
            "type": "getHistory", "query": query, "tag": tag, "offset": len(page.rows),
            "machine": LOCAL_MACHINE,
        }, "history")
        url = link.tunnel.web_url if link.tunnel is not None else machine.url
        rows = [s for s in reply.get("sessions") or [] if isinstance(s, dict)]
        for session in rows:
            session["machine"] = machine.name
            session["machine_url"] = url
        page.rows.extend(rows)
        date_range = reply.get("dateRange")
        if isinstance(date_range, dict):
            page.date_min, page.date_max = _float_or_none(date_range.get("min")), _float_or_none(
                date_range.get("max"),
            )
        if len(rows) < HISTORY_PAGE:
            break
    del page.rows[limit:]
    return page


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _fetch_quietly(machine: Machine, query: str, tag: str, limit: int) -> HistoryPage:
    try:
        return fetch_history(machine, query, tag, limit)
    except MachineError as exc:
        logger.warning("CHATS: %s", exc)
        return HistoryPage()


def merged_history(
    local: HistoryPage, machines: list[Machine], query: str, tag: str, offset: int, limit: int,
) -> HistoryPage:
    """Return page *offset*..*offset+limit* of the union of all machines' history.

    *local* must hold the local rows ``0 .. offset+limit`` (not just the
    requested page) so the union can be re-paginated; its date range is
    widened by the remote histories'.  Rows are ordered running first,
    then newest first, like the local list; rows that tie keep their
    source's order, local before remote, earlier machine first.  Of
    several copies of one task id the one that sorts first is kept.
    Because every source is itself ordered that way and the ranking of
    a copy never depends on what else is loaded, the page is the same
    whichever pages were loaded before it: a task never shows twice or
    goes missing across pages.  An unreachable machine contributes
    nothing (a warning is logged).
    """
    need = offset + limit
    with ThreadPoolExecutor(max_workers=max(1, len(machines))) as pool:
        futures = [pool.submit(_fetch_quietly, m, query, tag, need) for m in machines]
        pages = [local, *(future.result() for future in futures)]
    best: dict[str, tuple[tuple[bool, float, int, int], dict[str, Any]]] = {}
    for source, page in enumerate(pages):
        for rank, session in enumerate(page.rows):
            key = str(session.get("task_id") or session.get("id") or id(session))
            order = (*_history_order(session), -source, -rank)
            if key not in best or order > best[key][0]:
                best[key] = (order, session)
    ranked = sorted(best.values(), key=_ranked_order, reverse=True)
    mins = [p.date_min for p in pages if p.date_min is not None]
    maxs = [p.date_max for p in pages if p.date_max is not None]
    return HistoryPage(
        [session for _order, session in ranked[offset:need]],
        min(mins) if mins else None, max(maxs) if maxs else None,
    )


def _history_order(session: dict[str, Any]) -> tuple[bool, float]:
    """Sort key of the merged list: running tasks first, then newest first."""
    return bool(session.get("is_running")), float(session.get("timestamp") or 0.0)


def _ranked_order(
    item: tuple[tuple[bool, float, int, int], dict[str, Any]],
) -> tuple[bool, float, int, int]:
    """The order tuple of a ``(order, session)`` pair kept by :func:`merged_history`."""
    return item[0]
