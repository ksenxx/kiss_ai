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

A machine registered as ``user@host`` (an *SSH machine*) needs no
password: it is enrolled once over ssh and reached through an ssh
port-forward afterwards with a client certificate (mutual TLS).
:func:`enroll_over_ssh` runs one ``ssh user@host sh -c ...`` login
(:data:`ENROLL_COMMAND`) that copies this daemon's local CA
(``$KISS_HOME/tls/ca.pem``) into the remote daemon's
``tls/trusted-clients/`` directory (found under ``~/<brand home>``
whatever the remote's brand) and prints back the remote's endpoint
file (its daemon port) and its own CA.  The machine is then registered
at ``https://host:port`` (what the user sees) with that CA pinned.
Every connection goes through one ``ssh -N -L`` forward per login
(:class:`SshTunnel`, kept for the daemon's lifetime and reopened when
it dies) from a local loopback port to ``127.0.0.1:port`` on the
machine, so its daemon port need not be reachable from here (cloud
firewalls rarely open it; the ssh port is), and presents this daemon's
client certificate (``$KISS_HOME/tls/client.pem``, issued by its CA,
see :func:`client_cert_pair`), which the remote daemon admits like its
own local token whatever its remote password: the daemon sniffs the
first byte of every connection, so TLS through the forward is served
like a direct TLS connection.  The ``ssh`` client's own configuration
applies (``~/.ssh/config`` aliases, keys, agent), or the file named by
``$KISS_SSH_CONFIG``; the login must work non-interactively
(``BatchMode``), allow TCP forwarding, land in a POSIX shell (a
Windows host whose OpenSSH logs in to PowerShell cannot be an SSH
machine) and an unknown host key is accepted on first use.  The remote
daemon must understand enrollment (it has a ``trusted-clients``
directory).  The ssh login is the credential that matters: a link
arrives on the machine as a loopback peer, admitted like a local
browser whatever its certificate, so revoking a machine is removing
its ssh access there; deleting its ``trusted-clients`` file only
refuses the certificate on direct (non-loopback) connections.

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

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus, InvalidURI
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as _ws_connect

from kiss.agents.sorcar._concurrency import _race_delay, locked_call
from kiss.agents.sorcar.local_endpoint import LOCAL_ENDPOINT_FILE, send
from kiss.core import tls_certs
from kiss.core.brand import HOME_DIR
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock, lock_exclusive, unlock

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
LINK_PING_INTERVAL = 20.0
"""Seconds between the keepalive pings a link's sockets send the remote daemon."""
LINK_PING_TIMEOUT = 20.0
"""Seconds a keepalive ping may go unanswered before the socket counts as dead."""
SSH_TIMEOUT = 30.0
"""Seconds an ``ssh`` login (the enrollment, opening a port-forward) may take."""
ENROLL_TIMEOUT = 30.0
"""Seconds :func:`add_machine` waits for the enrolled machine's first handshake.

The remote daemon admits the new CA within a few seconds of the file
landing (it polls its ``trusted-clients`` directory); the first
handshakes are refused until then and tried again."""
SSH_CONFIG_ENV = "KISS_SSH_CONFIG"
"""Environment variable naming an ``ssh`` config file to use instead of the user's."""
TLS_LOCK_TIMEOUT = 10.0
"""Seconds to wait for the TLS directory's lock when issuing the client certificate."""
_TRUSTED = f"tls/{tls_certs.TRUSTED_CLIENTS_DIR}"
ENROLL_COMMAND = (
    "sh -c '"
    f'for d in "$KISS_HOME" "$HOME/{HOME_DIR}" "$HOME"/.*; do '
    'case "$d" in ""|*/.|*/..) continue;; esac; '
    f'if [ -f "$d/{LOCAL_ENDPOINT_FILE}" ]; then '
    f'[ -d "$d/{_TRUSTED}" ] || {{ '
    'echo "its daemon predates certificate enrollment: upgrade it" >&2; exit 3; }; '
    f'cat > "$d/{_TRUSTED}/.NAME.$$.tmp" && mv -f "$d/{_TRUSTED}/.NAME.$$.tmp" '
    f'"$d/{_TRUSTED}/NAME.pem" || exit 2; '
    f'cat "$d/{LOCAL_ENDPOINT_FILE}"; echo; '
    f'c=$(sed -n "s/^ *\\"ca\\": *\\"\\(.*\\)\\",*$/\\1/p" "$d/{LOCAL_ENDPOINT_FILE}"); '
    f'cat "${{c:-$d/tls/{tls_certs.CA_CERT_FILE}}}"; exit 0; '
    "fi; done; "
    f'echo "no $HOME/.*/{LOCAL_ENDPOINT_FILE}" >&2; exit 1\''
)
"""Remote command enrolling the CA on stdin as ``trusted-clients/NAME.pem``.

``NAME`` is replaced by :func:`enroll_over_ssh`.  The daemon's home is
the first directory holding an endpoint file among ``$KISS_HOME``, the
local brand's home (``~/.kiss``) and every other ``~/.*``: the remote
daemon's brand (hence its state directory) may differ from the local
one and may change when it is re-installed, and a stopped daemon
removes its endpoint file, so the home found is the running daemon's.
A daemon without a ``trusted-clients`` directory predates enrollment
(exit 3).  The CA is staged under the shell's pid and renamed so the
daemon never reads a half-written file and two enrollments at once do
not share a staging file.  Then the endpoint file (the daemon port)
and the certificate to pin are printed, separated by a newline: the
file named by the endpoint's ``ca`` (the daemon's local CA, or the
certificate it was started with), else ``tls/ca.pem``.  The login
shell only sees a single-quoted string, so the command avoids ``!``
(csh expands history even in quotes); ``.`` and ``..`` are skipped by
the ``case``."""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class MachineError(Exception):
    """A registry or connection problem, with a message fit for the user."""


@dataclass
class Machine:
    """One registered remote daemon.

    Attributes:
        name: Short label shown in the CHATS panel (``[A-Za-z0-9._-]``).
        url: ``https://host:port`` of the remote ``kiss-web``: as
            registered for a URL machine, the ssh host and the daemon
            port learnt at enrollment for an SSH machine (empty for an
            SSH machine registered before enrollment existed, which
            must be registered again).
        password: Its remote password; empty for an SSH machine
            (its client certificate is its credential).
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
    return _hostname(machine.url or "ssh://" + machine.ssh)


def machine_hosts(machine: Machine) -> set[str]:
    """Return every host *machine* is known by: its URL's and, for an SSH machine, its login's.

    The two differ when the login names an ssh config alias that
    enrollment resolved (:func:`ssh_host`) into the URL.
    """
    hosts = {machine_host(machine)}
    if machine.ssh:
        hosts.add(_hostname("ssh://" + machine.ssh))
    hosts.discard("")
    return hosts


def _hostname(url: str) -> str:
    """Return the hostname of *url* ('' when it has none or is malformed)."""
    try:
        return urlsplit(url).hostname or ""
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
    brackets) matches the hostname of a machine's URL or the host of an
    SSH machine's ``user@host`` login (:func:`machine_hosts`); a name
    wins over a host.  ``None`` when no machine matches.
    """
    text = name_or_host.strip()
    host = text.strip("[]").lower()
    machines = load_machines()
    for machine in machines:
        if machine.name == text:
            return machine
    for machine in machines:
        if host and host in {h.lower() for h in machine_hosts(machine)}:
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

    *options* go first (``ssh`` keeps the first value it sees of an
    option, so an ``-o`` among them overrides a default below), the
    destination after them, *command* (if any) last.  Key-based,
    non-interactive login only (``BatchMode``); an unknown host key is
    accepted and remembered on first use; the user's ssh configuration
    applies unless ``$KISS_SSH_CONFIG`` names another, except
    connection multiplexing (``ControlMaster``), which is off so that
    no master process outlives the login.
    """
    parts = urlsplit("ssh://" + ssh)
    cmd = [
        "ssh", *options, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "LogLevel=ERROR",
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
    (*password* unused): this daemon is then enrolled on the machine
    over ssh (:func:`enroll_over_ssh`), which yields the machine's
    URL (its host and daemon port, reached through an ssh port-forward
    from then on, see :func:`tunnel_port`) and its CA.  An empty *name*
    defaults to the host.  Pins the daemon's certificate to a staging
    file, authenticates once through it (an SSH machine admits the
    client certificate within a few seconds of enrollment, so its
    handshake is retried for up to
    :data:`ENROLL_TIMEOUT` seconds), and only then publishes the pin
    and the registry entry, so a wrong address or password changes
    nothing (an existing machine of that name keeps its pin and entry).

    Raises:
        MachineError: On a bad name or address, an unreachable daemon,
            a failed enrollment or a rejected password.
    """
    name = name or default_name(address)
    if not _NAME_RE.match(name):
        raise MachineError(
            f"'{name}' is not a valid machine name: letters, digits, '.', '_' and '-' only"
        )
    if is_ssh_address(address):
        ssh = normalize_ssh(address)
        url, pem = enroll_over_ssh(ssh)
        machine = Machine(name=name, url=url, ssh=ssh)
        deadline = time.monotonic() + ENROLL_TIMEOUT
    else:
        machine = Machine(name=name, url=normalize_url(address), password=password)
        pem = pin_certificate(machine.url)
        deadline = time.monotonic()
    cert = machine.cert_file()
    cert.parent.mkdir(parents=True, exist_ok=True)
    staged = cert.with_name(f".{cert.name}.{os.getpid()}.{secrets.token_hex(4)}.new")
    staged.write_text(pem, encoding="utf-8")
    try:
        _first_handshake(machine, staged, deadline)
    except MachineError:
        staged.unlink(missing_ok=True)
        raise
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
        (kiss_home() / MACHINE_CERTS_DIR / f"{name}.pem").unlink(missing_ok=True)
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


def _first_handshake(machine: Machine, cert_file: Path, deadline: float) -> None:
    """Ping *machine* through *cert_file* until it answers or *deadline* passes.

    A failed attempt is tried again a second later on a fresh link (a
    link backs off for :data:`DOWN_BACKOFF` seconds after a failure);
    the last error is raised once the monotonic *deadline* has passed.

    An SSH machine is pinged through a forward of these handshakes'
    own, closed on return: the login's shared forward
    (:func:`tunnel_port`) may carry the links and streams of a machine
    already registered there, which a failed registration must leave
    as they are.

    Raises:
        MachineError: The last attempt's error.
    """
    tunnel = (
        SshTunnel(machine.ssh, urlsplit(machine.url).port or DEFAULT_PORT)
        if machine.ssh else None
    )
    try:
        while True:
            link = MachineLink(machine, cert_file=cert_file, tunnel=tunnel)
            try:
                link.request({"type": "ping"}, "pong")
                return
            except MachineError:
                if time.monotonic() >= deadline:
                    raise
            finally:
                link.close()
    finally:
        if tunnel is not None:
            tunnel.close()
        time.sleep(1.0)


def client_cert_pair() -> tuple[Path, Path]:
    """Return this daemon's client certificate and key, issuing them when needed.

    ``$KISS_HOME/tls/client.pem`` and ``client-key.pem``, signed by
    the local CA (:func:`kiss.core.tls_certs.ensure_local_client_pair`,
    which generates the CA too when the daemon never ran), under the
    TLS directory's lock shared with the daemon's own certificate
    renewals.

    Raises:
        MachineError: When the lock is not free within
            :data:`TLS_LOCK_TIMEOUT` seconds.
    """
    tls_dir = kiss_home() / "tls"
    try:
        with exclusive_file_lock(tls_dir / tls_certs.TLS_LOCK_FILE, timeout=TLS_LOCK_TIMEOUT):
            return tls_certs.ensure_local_client_pair(tls_dir)
    except (TimeoutError, OSError) as exc:
        raise MachineError(f"cannot issue the client certificate in {tls_dir}: {exc}") from exc


def ssh_host(ssh: str) -> str:
    """Return the host name ``ssh`` (``user@host[:port]``) actually connects to.

    ``ssh -G`` prints the effective configuration, so an alias of the
    user's ssh config resolves to its ``HostName``; the host as typed
    when ssh cannot be asked.
    """
    typed = urlsplit("ssh://" + ssh).hostname or ""
    try:
        run = subprocess.run(
            ssh_command(ssh, "-G"), stdin=subprocess.DEVNULL, capture_output=True,
            timeout=10.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return typed
    for line in run.stdout.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition(" ")
        if key == "hostname" and value.strip():
            return value.strip()
    return typed


def enroll_over_ssh(ssh: str, timeout: float = SSH_TIMEOUT) -> tuple[str, str]:
    """Enroll this daemon's CA on the SSH machine *ssh*; return its URL and CA.

    One ``ssh`` login runs :data:`ENROLL_COMMAND` with the local CA on
    its stdin, which stores it under the daemon's
    ``tls/trusted-clients/`` as ``<this host>-<CA fingerprint>.pem``
    (:func:`kiss.core.tls_certs.ca_label`: enrolling again is
    harmless, a regenerated CA lands as a new file) and prints the
    daemon's endpoint file and its CA.

    Args:
        ssh: ``user@host[:port]``.
        timeout: Seconds the login may take.

    Returns:
        ``(url, ca_pem)``: ``https://<host>:<daemon port>``, the host
        being what ssh connects to (:func:`ssh_host`), and the PEM
        certificate to pin for it.  The URL names the machine on
        screen; connections reach the port through an ssh forward
        (:func:`tunnel_port`), so the machine need not expose it.

    Raises:
        MachineError: When ``ssh`` is missing or fails, no daemon runs
            on the machine, its daemon predates enrollment, or the
            reply is not a daemon's.
    """
    ca_cert_path = client_cert_pair()[0].with_name(tls_certs.CA_CERT_FILE)
    command = ENROLL_COMMAND.replace("NAME", tls_certs.ca_label(ca_cert_path))
    try:
        run = subprocess.run(
            ssh_command(ssh, command=command), input=ca_cert_path.read_bytes(),
            capture_output=True, timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise MachineError("ssh is not installed on this machine") from exc
    except subprocess.TimeoutExpired as exc:
        raise MachineError(f"ssh {ssh} did not answer in time") from exc
    if run.returncode != 0:
        message = {
            1: f"no Sorcar daemon is running on {ssh} (cannot find its {LOCAL_ENDPOINT_FILE})",
            2: f"cannot store this daemon's CA on {ssh}",
            3: f"cannot enroll on {ssh}",
        }.get(run.returncode, f"ssh {ssh} failed")
        raise MachineError(_with_last_line(message, run.stderr))
    marker = "-----BEGIN CERTIFICATE-----"
    endpoint_text, found, ca_rest = run.stdout.decode("utf-8", "replace").partition(marker)
    try:
        endpoint = json.loads(endpoint_text)
        port = urlsplit(endpoint["url"]).port
    except (ValueError, KeyError, TypeError):
        port = None
    if not found or not port:
        raise MachineError(f"{ssh} did not answer the enrollment as a Sorcar daemon")
    return f"https://{_bracketed(ssh_host(ssh))}:{port}", marker + ca_rest


def _with_last_line(message: str, stderr: bytes) -> str:
    """Return *message* followed by the last line of *stderr*, if any."""
    lines = stderr.decode("utf-8", "replace").strip().splitlines()
    return f"{message}: {lines[-1]}" if lines else message


class SshTunnel:
    """An ``ssh -N -L`` port-forward from a local loopback port to an SSH machine's daemon.

    :meth:`open` starts ``ssh -N -L 127.0.0.1:<local>:127.0.0.1:<remote>
    user@host`` (through :func:`ssh_command`) and waits until the local
    port accepts connections; :meth:`alive` says whether the ssh process
    still runs and the port still answers; :meth:`close` ends the
    process.  :func:`tunnel_port` keeps one tunnel per ``ssh`` login
    and re-opens a dead one on the next connection.
    """

    def __init__(self, ssh: str, remote_port: int) -> None:
        self.ssh = ssh
        self.remote_port = remote_port
        self.local_port = 0
        self.process: subprocess.Popen[bytes] | None = None
        # RLock: :func:`locked_call` needs the owner check on release.
        # Every lock here is taken on task threads too (the ``/machine``
        # SEA's tools), where the runner's Stop injects
        # ``KeyboardInterrupt`` at any check point, including the one
        # right after a ``with lock:``'s ``__enter__`` (gh-74174).
        self._lock = threading.RLock()
        self._last_line = ""

    def alive(self) -> bool:
        """Return whether the ssh process runs and its local port accepts a connection."""
        return (
            self.process is not None and self.process.poll() is None
            and _port_accepts(self.local_port)
        )

    def open(self, timeout: float = SSH_TIMEOUT) -> int:
        """Start the forward when it is not alive; return the local port.

        Raises:
            MachineError: When ``ssh`` is missing, exits (its last
                stderr line is quoted) or does not open the port within
                *timeout* seconds.
        """
        return locked_call(self._lock, self._open_locked, timeout)

    def _open_locked(self, timeout: float) -> int:
        """:meth:`open`'s body, under the lock."""
        if self.alive():
            return self.local_port
        self._kill()
        self.local_port = _free_port()
        forward = f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}"
        cmd = ssh_command(
            self.ssh, "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
            "-L", forward,
        )
        self._last_line = ""
        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise MachineError(f"ssh is not installed on this machine: {exc}") from exc
        reader = threading.Thread(
            target=self._read_stderr, args=(self.process,), daemon=True,
            name=f"ssh-tunnel-stderr {self.ssh}",
        )
        reader.start()
        deadline = time.monotonic() + timeout
        while True:
            if self.process.poll() is not None:
                reader.join(timeout=2.0)
                message = f"ssh {self.ssh} cannot forward its daemon port {self.remote_port}"
                raise MachineError(
                    f"{message}: {self._last_line}" if self._last_line else message
                )
            if _port_accepts(self.local_port):
                return self.local_port
            if time.monotonic() >= deadline:
                self._kill()
                raise MachineError(f"ssh {self.ssh} did not open the port forward in time")
            time.sleep(0.05)

    def _read_stderr(self, process: subprocess.Popen[bytes]) -> None:
        """Keep the last stderr line of *process* (and keep its pipe from filling up)."""
        assert process.stderr is not None
        for raw in process.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._last_line = line

    def close(self) -> None:
        """Kill the ssh process, if any, and forget the port."""
        locked_call(self._lock, self._kill)

    def _kill(self) -> None:
        """Kill the ssh process, if any, and forget the port (under the lock)."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None
        self.local_port = 0


def _free_port() -> int:
    """Return a loopback port that is free right now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_accepts(port: int) -> bool:
    """Return whether ``127.0.0.1:port`` accepts a TCP connection."""
    if not port:
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


_tunnels: dict[str, SshTunnel] = {}
_tunnels_lock = threading.RLock()


def _cached_tunnel(ssh: str, remote_port: int) -> SshTunnel:
    """Return *ssh*'s cached tunnel, a fresh one when its remote port changed (under the lock)."""
    tunnel = _tunnels.get(ssh)
    if tunnel is None or tunnel.remote_port != remote_port:
        if tunnel is not None:
            tunnel.close()
        tunnel = _tunnels[ssh] = SshTunnel(ssh, remote_port)
    return tunnel


def _drain(cache: dict[Any, Any]) -> list[Any]:
    """Empty *cache* and return what it held (under its lock)."""
    items = list(cache.values())
    cache.clear()
    return items


def tunnel_port(ssh: str, remote_port: int) -> int:
    """Return the local port forwarding to ``127.0.0.1:remote_port`` on *ssh*.

    One :class:`SshTunnel` per login is kept for the daemon's lifetime:
    it is reused while alive and opened again (on a fresh local port)
    once its ssh process died or its remote port changed (the machine
    was registered again after its daemon moved).

    The forward is opened outside ``_tunnels_lock`` (an ``ssh`` login
    takes seconds and must not hold up the other machines' tunnels), so
    :func:`close_tunnels` may have dropped the tunnel meanwhile: a
    forward opened on a tunnel the cache no longer holds would have no
    one to close it, so it is closed here and the caller is told to
    try again, as a retired :class:`MachineLink` tells it.

    Raises:
        MachineError: When the forward cannot be opened (see
            :meth:`SshTunnel.open`) or the registry changed while it
            was opening.
    """
    tunnel = locked_call(_tunnels_lock, _cached_tunnel, ssh, remote_port)
    # Test hook (no-op in production): widens the window between the
    # lookup and the open so the close_tunnels race is deterministic
    # under KISS_RACE_DELAY.
    _race_delay()
    port = tunnel.open()
    if locked_call(_tunnels_lock, _tunnels.get, ssh) is tunnel:
        return port
    tunnel.close()
    raise MachineError(f"ssh {ssh}: the machine registry changed, try again")


def close_tunnels() -> None:
    """Close and forget every ssh port-forward (after a registry change, at exit)."""
    tunnels: list[SshTunnel] = locked_call(_tunnels_lock, _drain, _tunnels)
    for tunnel in tunnels:
        tunnel.close()


class MachineLink:
    """One persistent, authenticated WebSocket to a remote daemon.

    Connects lazily on the first :meth:`request`, re-connects after a
    failure and serialises requests, so the daemon keeps one socket per
    machine across CHATS refreshes.  After a failure the link reports
    the machine down without trying again for :data:`DOWN_BACKOFF`
    seconds, so an offline machine does not add a connection timeout
    to every CHATS refresh and a rejected password is not retried
    often enough to trip the remote daemon's brute-force lockout.
    """

    def __init__(
        self, machine: Machine, cert_file: Path | None = None, tunnel: SshTunnel | None = None,
    ) -> None:
        """A link to *machine*, pinned to *cert_file* (its registered pin by default).

        *tunnel*, for an SSH machine, is a forward of the caller's own
        to dial instead of the login's shared one (:func:`tunnel_port`);
        the caller opens nothing and closes it when done.
        """
        self.machine = machine
        self._cert_file = cert_file if cert_file is not None else machine.cert_file()
        self._tunnel = tunnel
        self._ws: ClientConnection | None = None
        self._lock = threading.RLock()  # see SshTunnel: taken through locked_call
        self._generation = 0
        self._retry_at = 0.0
        self._last_error = ""
        self._retired = False

    def _check_available(self) -> None:
        """Raise the error a retired or backed-off link answers with, if any."""
        if self._retired:
            raise MachineError(f"{self.machine.name}: the machine registry changed, try again")
        if time.monotonic() < self._retry_at:
            raise MachineError(self._last_error)

    def _fail(self, message: str) -> MachineError:
        """Record *message* as the link's failure, start the backoff, return the error."""
        self._retry_at = time.monotonic() + DOWN_BACKOFF
        self._last_error = message
        return MachineError(message)

    def _connect(self, timeout: float, stream: bool = False) -> ClientConnection:
        """Open and authenticate the socket; the handshake has *timeout* seconds.

        The socket is TLS with the machine's pinned certificate as the
        only trust anchor.  An SSH machine is dialled at the local end
        of its ssh port-forward (the login's shared one,
        :func:`tunnel_port`, or the *tunnel* given to the constructor;
        opened here when it is not up, with the ssh login's own
        :data:`SSH_TIMEOUT`;
        *timeout* starts once the forward is up, so a slow login does
        not leave the handshake no time and back the link off) and
        authenticated by this daemon's client
        certificate (:func:`client_cert_pair`), which the remote daemon
        admits once this daemon's CA is enrolled there; any other
        machine is dialled directly and authenticated by the registered
        password sent in the ``auth`` frame.  The
        shared request socket (*stream* false) announces itself as a
        daemon link (``"link": true`` in its ``auth`` frame): the
        daemon answers its requests but leaves it out of the task-event
        fan-out, so another user's history click on the machine does
        not push a whole transcript at a socket that only wants its
        replies.  A stream socket (:meth:`open_connection`) relays a
        tab's events and stays an ordinary peer.

        No frame-size limit is applied in either case (``max_size``
        ``None``): the daemon is an authenticated, certificate-pinned
        peer, and with the library default of 1 MiB a transcript
        larger than that closed the socket with 1009 and reported the
        machine unreachable for :data:`DOWN_BACKOFF` seconds.

        The request socket's receive queue is unbounded (``max_queue``
        ``None``).  The daemon's watchdog pushes a ``heartbeat`` frame
        at every socket every 15 s, and the request socket only reads
        between requests: with the library default of 16 queued
        frames the reader paused after four idle minutes, stopped
        handling the pongs of this socket's own keepalive pings, and
        the keepalive closed the socket ("keepalive ping timeout").
        The next request then failed.  The queued frames cost a few
        bytes each and :meth:`request` drains them.  A stream socket
        reads continuously and keeps the default.
        """
        machine = self.machine
        where = f"{machine.name} ({machine.address()})"
        self._check_available()
        enroll_again = (
            f"; run /machine add {machine.name} {machine.ssh} again" if machine.ssh else ""
        )
        if not machine.url:
            raise self._fail(f"{where} was registered before certificate enrollment{enroll_again}")
        try:
            ctx = ssl.create_default_context(cafile=str(self._cert_file))
        except (OSError, ssl.SSLError) as exc:
            raise self._fail(f"{where} has no usable pinned certificate: {exc}") from exc
        url, forward_refused = machine.url, ""
        if machine.ssh:
            try:
                ctx.load_cert_chain(*map(str, client_cert_pair()))
            except (OSError, ssl.SSLError, MachineError) as exc:
                raise self._fail(f"{where}: no usable client certificate: {exc}") from exc
            remote_port = urlsplit(machine.url).port or DEFAULT_PORT
            try:
                port = (
                    self._tunnel.open() if self._tunnel is not None
                    else tunnel_port(machine.ssh, remote_port)
                )
                url = f"https://127.0.0.1:{port}"
            except MachineError as exc:
                raise self._fail(f"{where} is unreachable: {exc}") from exc
            # ssh drops a forwarded connection (EOF or reset) when
            # nothing listens on the daemon port of the machine.
            forward_refused = (
                f"{where} is unreachable: port {remote_port} on the machine "
                "refused the forwarded connection"
            )
        ctx.check_hostname = False
        auth = {"type": "auth", "link": not stream, "password": machine.password}
        deadline = time.monotonic() + timeout
        try:
            ws = _ws_connect(
                ws_url(url), ssl=ctx, open_timeout=_remaining(deadline),
                close_timeout=2.0, compression=None, max_size=None,
                max_queue=16 if stream else None,
                ping_interval=LINK_PING_INTERVAL, ping_timeout=LINK_PING_TIMEOUT,
            )
        except ssl.SSLError as exc:
            if forward_refused and isinstance(exc, ssl.SSLEOFError):
                raise self._fail(forward_refused) from exc
            raise self._fail(f"{where} refused the TLS handshake: {exc}{enroll_again}") from exc
        except InvalidStatus as exc:  # 403: no password set there and no trusted certificate
            raise self._fail(f"{where} refused the connection: {exc}{enroll_again}") from exc
        except (OSError, InvalidHandshake, InvalidURI) as exc:
            if forward_refused and isinstance(exc, ConnectionResetError):
                raise self._fail(forward_refused) from exc
            raise self._fail(f"{where} is unreachable: {exc}") from exc
        try:
            send(ws, json.dumps(auth), timeout=_remaining(deadline))
            reply = json.loads(ws.recv(timeout=_remaining(deadline)))
        except (ConnectionClosed, OSError, ValueError) as exc:
            ws.close()
            raise self._fail(f"{where} did not complete the handshake: {exc}") from exc
        if isinstance(reply, dict) and reply.get("type") == "auth_ok":
            return ws
        ws.close()
        if machine.ssh:
            raise self._fail(f"{where} does not trust this daemon's certificate{enroll_again}")
        if isinstance(reply, dict) and reply.get("type") == "auth_required":
            raise self._fail(f"{where} rejected the password")
        raise self._fail(f"{where} refused the connection: {reply!r}")

    def request(
        self, cmd: dict[str, Any], reply_type: str, timeout: float = REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* event answering it.

        The command is stamped with a fresh ``generation``; a reply that
        echoes a ``generation`` must echo this one (``history`` does),
        other frames the daemon pushes meanwhile are skipped.  *timeout*
        bounds the wait for the reply, however many frames arrive or
        are already buffered; connecting first, when the socket is not
        up, has a budget of *timeout* of its own for the handshake
        (an SSH machine's port-forward, opened before it, has
        :data:`SSH_TIMEOUT`).

        A socket kept from an earlier request may have died meanwhile
        (the daemon restarted, a network blip, its idle keepalive
        failed): the request is then sent again over a fresh socket
        before the machine counts as down, so a stale socket alone
        never starts the back-off.

        Raises:
            MachineError: When the daemon is unreachable, rejects the
                password, or does not answer within *timeout* seconds.
        """
        return locked_call(self._lock, self._request_locked, cmd, reply_type, timeout)

    def _request_locked(
        self, cmd: dict[str, Any], reply_type: str, timeout: float,
    ) -> dict[str, Any]:
        """:meth:`request`'s body, under the lock."""
        kept = self._ws is not None
        if self._ws is None:
            self._ws = self._connect(timeout)
        deadline = time.monotonic() + timeout
        try:
            return self._exchange(cmd, reply_type, deadline, timeout)
        except ConnectionClosed as exc:
            if not kept:
                self._drop()
                raise self._fail(
                    f"{self.machine.name} ({self.machine.address()}) did not answer: {exc}"
                ) from exc
            logger.info("%s: the link's socket had closed, reconnecting", self.machine.name)
            self._drop()
            self._ws = self._connect(_remaining(deadline))
            try:
                return self._exchange(cmd, reply_type, deadline, timeout)
            except ConnectionClosed as again:
                self._drop()
                raise self._fail(
                    f"{self.machine.name} ({self.machine.address()}) did not answer: {again}"
                ) from again

    def _exchange(
        self, cmd: dict[str, Any], reply_type: str, deadline: float, timeout: float,
    ) -> dict[str, Any]:
        """Send *cmd* on the open socket and return its *reply_type* answer (under the lock).

        *deadline* bounds the exchange (*timeout* only names the budget
        in the error).

        Raises:
            ConnectionClosed: When the socket closed before the answer
                came (the caller decides whether to reconnect).
            MachineError: When the daemon did not answer before
                *deadline* (the link then backs off).
        """
        assert self._ws is not None
        self._generation += 1
        generation = self._generation
        try:
            # Bounded like the reads: a plain ``ws.send`` has no
            # deadline, and one stuck under this lock (the daemon
            # stopped reading) would wedge every later request to
            # the machine, and the link's close at exit.
            # A password-authenticated link repeats the password in
            # every frame: the daemon checks it on each command.  An
            # SSH machine's link (client certificate) sends none.
            frame = cmd | {"generation": generation}
            if self.machine.password:
                frame["password"] = self.machine.password
            send(self._ws, json.dumps(frame), timeout=_remaining(deadline))
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"no {reply_type} within {timeout:g}s")
                msg = json.loads(self._ws.recv(timeout=_remaining(deadline)))
                if not isinstance(msg, dict) or msg.get("type") != reply_type:
                    continue
                if "generation" in msg and msg["generation"] != generation:
                    continue
                return msg
        except (TimeoutError, ValueError, OSError) as exc:
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
        shares.  Unlike that socket it is an ordinary peer of the daemon
        (it needs the tab's task events).  The caller owns the returned
        socket and closes it.

        Raises:
            MachineError: When the daemon is unreachable or rejects the
                password (the link backs off as after a failed request).
        """
        return locked_call(self._lock, self._connect, timeout, True)

    def close(self) -> None:
        """Close the socket; the next :meth:`request` re-connects."""
        locked_call(self._lock, self._drop)

    def retire(self) -> None:
        """Close for good: every later :meth:`request` fails instead of re-connecting.

        Waits for a request in flight to finish.  Used when the link
        leaves the daemon-wide cache, so a caller that fetched it just
        before cannot re-open a socket nothing closes any more.
        """
        locked_call(self._lock, self._retire_locked)

    def _retire_locked(self) -> None:
        """:meth:`retire`'s body, under the lock."""
        self._retired = True
        self._drop()


def _remaining(deadline: float) -> float:
    """Seconds left before the monotonic *deadline* (a small positive floor)."""
    return max(0.001, deadline - time.monotonic())


_links: dict[tuple[str, str, str, str], MachineLink] = {}
_links_lock = threading.RLock()


def _cached_link(machine: Machine) -> MachineLink:
    """Return the cached link to *machine*, made on first use (under the lock)."""
    key = (machine.name, machine.url, machine.password, machine.ssh)
    link = _links.get(key)
    if link is None:
        link = _links[key] = MachineLink(machine)
    return link


def link_for(machine: Machine) -> MachineLink:
    """Return the daemon-wide cached link to *machine*."""
    return locked_call(_links_lock, _cached_link, machine)


def close_links() -> None:
    """Retire and forget every cached link and ssh tunnel (after a registry change, at exit).

    A machine registered again gets a fresh tunnel (its daemon port may
    have changed), a removed one leaves no ssh process behind.
    """
    links: list[MachineLink] = locked_call(_links_lock, _drain, _links)
    for link in links:
        link.retire()
    close_tunnels()


atexit.register(close_links)


# The short host name each registered machine's daemon last gave of
# itself (registered name -> host), from the ``machine`` of its
# ``pong`` and the ``localMachine`` of its ``history`` replies; kept
# while the machine is down.
_hosts: dict[str, str] = {}


def note_host(machine: Machine, host: Any) -> None:
    """Record the host name *host* that *machine*'s daemon gave of itself.

    Ignored unless a non-empty string.
    """
    if isinstance(host, str) and host:
        _hosts[machine.name] = host


def machine_label(machine: Machine) -> str:
    """Return what to call *machine* on screen.

    Its daemon's own host name (:func:`note_host`) when it was
    registered under the default name, the host of its address
    (:func:`default_name`), so an address-registered machine reads
    ``gpu-box`` rather than ``10.0.0.7``; a name the user chose is
    kept, as is the name while the daemon has never answered.
    """
    if machine.name == default_name(machine.ssh or machine.url):
        return _hosts.get(machine.name) or machine.name
    return machine.name


def fetch_history(machine: Machine, query: str, tag: str, limit: int) -> HistoryPage:
    """Return up to *limit* history rows of *machine* matching *query* and *tag*.

    Pages through the remote ``getHistory`` (``HISTORY_PAGE`` rows per
    reply), asking for the machine's own rows only (``machine:
    LOCAL_MACHINE``) so two daemons registered with each other do not
    query each other without end, and stamps every row with ``machine``
    (the name) and ``machine_url`` (the machine's URL).  The page's
    date range is
    the remote history's.  The reply's ``localMachine`` is the
    machine's own host name (:func:`note_host`).

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
        note_host(machine, reply.get("localMachine"))
        rows = [s for s in reply.get("sessions") or [] if isinstance(s, dict)]
        for session in rows:
            session["machine"] = machine.name
            session["machine_url"] = machine.url
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
    """Sort key of the merged list: running tasks first, then latest completed first.

    The completion time is ``endTs`` (epoch ms) when set, else the launch
    ``timestamp`` (epoch seconds), the same key the local list sorts by.
    """
    end_ms = float(session.get("endTs") or 0)
    done = end_ms / 1000.0 if end_ms > 0 else float(session.get("timestamp") or 0.0)
    return bool(session.get("is_running")), done


def _ranked_order(
    item: tuple[tuple[bool, float, int, int], dict[str, Any]],
) -> tuple[bool, float, int, int]:
    """The order tuple of a ``(order, session)`` pair kept by :func:`merged_history`."""
    return item[0]
