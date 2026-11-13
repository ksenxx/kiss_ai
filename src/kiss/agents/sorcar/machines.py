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
Every connection goes through one ``ssh -N -L`` forward per login and
daemon port (:class:`SshTunnel`, kept while a machine is registered
there and reopened when it dies) from a local loopback port to
``127.0.0.1:port`` on the machine, so its daemon port need not be
reachable from here (cloud firewalls rarely open it; the ssh port is),
and presents this daemon's
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
import http.client
import ipaddress
import json
import logging
import operator
import os
import re
import secrets
import socket
import ssl
import subprocess
import threading
import time
import urllib.request
from collections.abc import Buffer, Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as _ws_connect

from kiss.agents.sorcar.history_columns import host_name
from kiss.agents.sorcar.local_endpoint import (
    LOCAL_ENDPOINT_FILE,
    FrameJoiner,
    call_off_thread,
    dial_off_thread,
    hang_up,
    recv_auth_reply,
    send,
)
from kiss.core import tls_certs
from kiss.core.brand import HOME_DIR
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import LockTimeoutError, _race_delay, locked_call
from kiss.core.processes import (
    IS_WINDOWS,
    kill_process_tree,
    popen_process_group,
    waitid_exited,
)
from kiss.core.utils import atomic_write_text

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
DIAL_GRACE = 5.0
"""Seconds past its handshake's deadline a link's dial may take to report.

Every step of the handshake ends by that deadline, and closing a socket
the daemon refused takes 2 more at most.  A dial silent after that is
where no timeout runs, in the lookup of the host's name above all, and
is given up on (:meth:`MachineLink._dial`)."""
DNS_TIMEOUT = 2.0
"""Seconds :func:`is_local_machine` waits for a lookup of this host's own name."""
PIN_TIMEOUT = 30.0
"""Seconds :func:`pin_certificate` may take in all, the lookup of the host's name included."""
CERT_MAX_BYTES = 1 << 20
"""The most bytes of ``/ca.crt`` :func:`pin_certificate` accepts (a CA is a few thousand)."""
CORRELATED_REPLIES = frozenset({"models", "history", "myModelsData", "machines", "historyDeleted"})
"""The reply types the daemon answers with the request's ``generation`` echoed.

A frame of one of these types that echoes none is not the answer
(:meth:`MachineLink.request`): a list the daemon pushed to every
connection unasked (``myModelsData`` for another client's edit,
``machines`` for a change of its registry), or the answer to an
earlier request still on its way, which no drain of the frames
already received can tell from the answer.  A daemon says so with
``echoesGeneration`` in its ``auth_ok``; one that does not (an older
version) is read as before, the first frame of the type being the
answer.
"""
DOWN_BACKOFF = 30.0
"""Seconds a cached link waits before trying a machine again, at most.

The wait starts at :data:`FIRST_BACKOFF` and doubles with every
failure in a row up to this, so a machine that is only briefly out
of reach (the laptop just woke, Wi-Fi is not back yet) is tried again
within seconds while one that is really down is not dialled on every
CHATS refresh.  A rejected password waits the full time from the first
failure: the remote daemon locks a client out after a few wrong
passwords in a row."""
FIRST_BACKOFF = 5.0
"""Seconds a cached link waits after a first failure (see :data:`DOWN_BACKOFF`)."""
SLEEP_GAP = 30.0
"""Seconds the wall clock may run ahead of the monotonic clock before the host counts as slept.

:func:`forget_after_sleep` compares the two clocks: the monotonic one
(``mach_absolute_time`` on macOS, ``CLOCK_MONOTONIC`` on Linux) stops
while the host sleeps and the wall clock does not, so the difference
is the time slept.  Sleeps shorter than this leave the connections
alone: the remote daemon closes a socket whose pings go unanswered
for 25 s, so a shorter sleep may well have kept them.
"""
LINK_PING_INTERVAL = 20.0
"""Seconds between the keepalive pings a link's sockets send the remote daemon."""
LINK_PING_TIMEOUT = 20.0
"""Seconds a keepalive ping may go unanswered before the socket counts as dead."""
SSH_TIMEOUT = 30.0
"""Seconds an ``ssh`` login (the enrollment, opening a port-forward) may take."""
FORWARD_UP = "kiss-forward-up"
"""What a port-forward's ``LocalCommand`` prints: ssh runs it once the local port is bound."""
FORWARD_ATTEMPTS = 3
"""Local ports :meth:`SshTunnel.open` tries when another program binds the one it picked."""
REGISTRY_LOCK_TIMEOUT = 60.0
"""Seconds :func:`registry_lock` waits for another holder before giving up.

A holder rewrites the registry and closes the cached links and ssh
tunnels; a tunnel being opened keeps its lock for up to
:data:`SSH_TIMEOUT`, so a waiter gives up only after twice that.
"""
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
    "for s in 1 0; do "
    f'for d in "$KISS_HOME" "$HOME/{HOME_DIR}" "$HOME"/.*; do '
    'case "$d" in ""|*/.|*/..) continue;; esac; '
    f'if [ -f "$d/{LOCAL_ENDPOINT_FILE}" ]; then '
    f'p=$(sed -n "s/^ *\\"pid\\": *\\([0-9]*\\),*$/\\1/p" "$d/{LOCAL_ENDPOINT_FILE}"); '
    '[ -n "$p" ] && kill -0 "$p" 2>/dev/null || continue; '
    '[ $s = 1 ] && m=$(ps -p "$p" -o etime= 2>/dev/null | awk -F"[-:]" "{ '
    'm = \\$(NF-1) + 1; if (NF > 2) m += 60 * \\$(NF-2); if (NF > 3) m += 1440 * \\$(NF-3); '
    'print m }" 2>/dev/null) && [ -n "$m" ] && '
    f'[ -n "$(find "$d/{LOCAL_ENDPOINT_FILE}" -mmin "+$m" 2>/dev/null)" ] && continue; '
    f'[ -d "$d/{_TRUSTED}" ] || {{ '
    'echo "its daemon predates certificate enrollment: upgrade it" >&2; exit 3; }; '
    f't=$(mktemp -u "$d/{_TRUSTED}/.NAME.XXXXXXXXXX") && (set -C; cat > "$t") && '
    f'mv -f "$t" "$d/{_TRUSTED}/NAME.pem" || exit 2; '
    f'cat "$d/{LOCAL_ENDPOINT_FILE}"; echo; '
    f'c=$(sed -n "s/^ *\\"ca\\": *\\"\\(.*\\)\\",*$/\\1/p" "$d/{LOCAL_ENDPOINT_FILE}"); '
    f'cat "${{c:-$d/tls/{tls_certs.CA_CERT_FILE}}}"; exit 0; '
    "fi; done; done; "
    f'echo "no $HOME/.*/{LOCAL_ENDPOINT_FILE}" >&2; exit 1\''
)
"""Remote command enrolling the CA on stdin as ``trusted-clients/NAME.pem``.

``NAME`` is replaced by :func:`enroll_over_ssh`.  The daemon's home is
the first directory holding the endpoint file of a running daemon
among ``$KISS_HOME``, the local brand's home (``~/.kiss``) and every
other ``~/.*``: the remote daemon's brand (hence its state directory)
may differ from the local one and may change when it is re-installed.
A stopped daemon removes its endpoint file, but one that crashed or
was killed leaves it behind, and so does the brand home of an earlier
install (``~/.kiss`` beside a live ``~/.s10s``): an endpoint file
whose ``pid`` is not alive (or missing: no daemon wrote it) is passed
over.  A live pid alone proves little: that of a daemon killed with
its machine is another process's after the reboot.  So the homes are
searched twice, and the first search also passes over an endpoint
file written before the process that has its pid was born, which
therefore did not write it (``ps``'s ``etime`` in whole minutes,
rounded up, against ``find -mmin``; a machine where the two give no
answer passes nothing over).  The second search, made when the first
found no home, takes the live pid alone: a wall clock stepped forward
since a daemon started (a resumed virtual machine, a board without a
clock) makes its own file look older than it.  A daemon without a
``trusted-clients`` directory
predates enrollment (exit 3).  The CA is staged and renamed so the
daemon never reads a half-written file, under a random name the
shell must create (``mktemp -u`` names it, the noclobber ``>`` opens
it with ``O_EXCL``): two enrollments at once do not share a staging
file, and nothing waiting at a name one could compute (the shell's
pid; a symlink there was written through) is written.  Then the
endpoint file (the daemon port) and the certificate to pin are
printed, separated by a newline: the
file named by the endpoint's ``ca`` (the daemon's local CA, or the
certificate it was started with), else ``tls/ca.pem``.  The login
shell only sees a single-quoted string, so the command avoids ``!``
(csh expands history even in quotes); ``.`` and ``..`` are skipped by
the ``case``."""

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")


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
    return host_name()


def load_machines() -> list[Machine]:
    """Return the registered machines in registration order ([] when none)."""
    try:
        raw = json.loads(machines_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    listed = raw.get("machines") if isinstance(raw, dict) else None
    machines: list[Machine] = []
    for item in listed if isinstance(listed, list) else ():
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

    The fully qualified name and a link-local IPv6 address are known
    through a lookup of this host's name, which a resolver may never
    answer: each lookup is waited for :data:`DNS_TIMEOUT` seconds
    (:func:`~kiss.agents.sorcar.local_endpoint.call_off_thread`), and
    one that has not answered by then, or had no thread to run on,
    matches nothing.
    """
    text = name_or_host.strip().strip("[]").lower()
    if not text:
        return False
    if text in {LOCAL_MACHINE, "localhost", local_machine_name().lower(),
                socket.gethostname().lower()}:
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        # A host name that is not the short name: only now pay for the
        # reverse lookup.
        try:
            return text == call_off_thread(socket.getfqdn, DNS_TIMEOUT).lower()
        except (OSError, RuntimeError):  # no answer in time; out of threads
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
        resolved = call_off_thread(
            partial(socket.getaddrinfo, socket.gethostname(), None), DNS_TIMEOUT,
        )
    except (OSError, RuntimeError):
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
    """Write the registry atomically (created 0600); drop the links of machines no longer in it."""
    path = machines_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = {"machines": [asdict(m) for m in machines]}
    atomic_write_text(path, json.dumps(registry, indent=2), mode=0o600)
    close_links(keep=machines)


@contextmanager
def registry_lock(timeout: float = REGISTRY_LOCK_TIMEOUT) -> Iterator[None]:
    """Hold the cross-process lock serialising registry read-modify-writes.

    Raises:
        MachineError: When another holder kept the lock for *timeout*
            seconds (the callers show it as the action's error text).
    """
    path = machines_file().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as held:
        try:
            held.enter_context(exclusive_file_lock(path, timeout=timeout))
        except TimeoutError as exc:
            raise MachineError("the machine registry is busy, try again") from exc
        yield


def normalize_url(url: str) -> str:
    """Return *url* as ``https://host:port``.

    Accepts ``host``, ``host:port``, ``https://host[:port][/path]`` and
    ``wss://host[:port][/ws]``; the port defaults to ``8787``.

    Raises:
        MachineError: When *url* is malformed (a port, a bracket) or names no host.
    """
    text = url.strip()
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
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
        MachineError: When *address* is malformed or names no host.
    """
    text = address.strip()
    try:
        parts = urlsplit(text if text.startswith("ssh://") else "ssh://" + text)
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise MachineError(f"'{address}' is not an SSH address: {exc}") from exc
    if parts.scheme != "ssh" or not host or parts.path not in ("", "/"):
        raise MachineError(f"'{address}' is not an SSH address; expected user@host[:port]")
    login = f"{parts.username}@{_bracketed(host)}" if parts.username else _bracketed(host)
    ssh = f"{login}:{port}" if port else login
    if not _hostname("ssh://" + ssh):  # as ssh_command and ssh_host read it back
        raise MachineError(f"'{address}' is not an SSH address; expected user@host[:port]")
    return ssh


def default_name(address: str) -> str:
    """Return the machine name implied by *address*: its host, made a valid name."""
    text = address.strip()
    if "://" not in text:
        text = ("ssh://" if is_ssh_address(text) else "https://") + text
    return re.sub(r"[^A-Za-z0-9._-]+", "-", _hostname(text)).strip("._-") or "machine"


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


class _BudgetedSocket(ssl.SSLSocket):
    """A TLS socket whose reads take :data:`REQUEST_TIMEOUT` seconds in all.

    A socket's timeout bounds each read, and ``http.client`` makes as
    many as the peer has pieces to send: one that kept sending a few
    bytes of status line, headers or body held :func:`pin_certificate`
    for as long as it liked.
    """

    _read_by = 0.0  # the monotonic time every read ends by, set at the first

    def recv_into(self, buffer: Buffer, nbytes: int | None = None, flags: int = 0) -> int:
        """Read as ``ssl.SSLSocket.recv_into`` does, in what is left of the budget.

        Args:
            buffer: Where the bytes go.
            nbytes: How many to read at most (None: as many as *buffer* holds).
            flags: The ``recv`` flags; a TLS socket takes none.

        Returns:
            The number of bytes read.

        Raises:
            TimeoutError: When the reads have used the budget up.
        """
        now = time.monotonic()
        self._read_by = self._read_by or now + REQUEST_TIMEOUT
        if now >= self._read_by:
            raise TimeoutError("timed out")
        self.settimeout(self._read_by - now)
        return super().recv_into(buffer, nbytes, flags)


def pin_certificate(url: str) -> str:
    """Fetch and return the PEM to trust for the daemon at *url*.

    Prefers the daemon's local CA (``GET /ca.crt``) so the leaf
    certificate may be renewed later; when the daemon serves no CA (it
    was started with an explicit ``--certfile``), pins the leaf itself.
    The answer to ``/ca.crt`` may take :data:`REQUEST_TIMEOUT` seconds
    and :data:`CERT_MAX_BYTES` bytes at most, whatever sends it, and a
    redirect is not followed: its target could answer on a socket with
    no such budget, and is not the daemon.  The whole fetch has
    :data:`PIN_TIMEOUT` seconds: the lookup of the host's name (or the
    proxy's) comes before any socket and takes no timeout, so the fetch
    runs on a thread this call waits for that long
    (:func:`~kiss.agents.sorcar.local_endpoint.call_off_thread`).

    Raises:
        MachineError: When the daemon cannot be reached, or what answers
            in its place sends no certificate within those bounds.
    """
    try:
        return call_off_thread(partial(_fetch_certificate, url), PIN_TIMEOUT)
    except (TimeoutError, RuntimeError) as exc:  # RuntimeError: out of threads
        raise MachineError(f"Cannot reach {url}: {exc}") from exc


def _fetch_certificate(url: str) -> str:
    """Fetch the PEM for *url*: :func:`pin_certificate`'s work; it tells the bounds and errors."""
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    insecure.sslsocket_class = _BudgetedSocket
    opener = urllib.request.OpenerDirector()  # the environment's proxy and TLS, nothing more
    opener.add_handler(urllib.request.ProxyHandler())
    opener.add_handler(urllib.request.HTTPSHandler(context=insecure))
    try:
        with opener.open(url + "/ca.crt", timeout=REQUEST_TIMEOUT) as resp:
            if 200 <= resp.status < 300:
                pem = resp.read(CERT_MAX_BYTES + 1)
                if len(pem) > CERT_MAX_BYTES:
                    raise MachineError(
                        f"Cannot pin {url}: its /ca.crt is larger than {CERT_MAX_BYTES} bytes",
                    )
                return str(pem.decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException) as exc:
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
    The pin is published first; a Stop or a registry-write error
    before the entry follows puts the previous pin back (or removes
    the new one, for a new name), so the entry the registry still
    holds keeps the certificate it was registered with.
    The staging file is removed however the call ends: a Stop's
    injected ``KeyboardInterrupt`` (the ``/machine add`` tool runs on
    the task thread) or a busy registry lock must not leave it behind.

    Raises:
        MachineError: On a bad name or address, an unreachable daemon,
            a failed enrollment or a rejected password.
    """
    name = name or default_name(address)
    if not _NAME_RE.fullmatch(name):
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
    try:
        # Bytes as the PEM came: a text write would give the pin CRLF line ends on
        # Windows, and the restore below (atomic_write_text) writes it back with LF.
        staged.write_bytes(pem.encode("utf-8"))
        _first_handshake(machine, staged, deadline)
        with registry_lock():
            machines = [m for m in load_machines() if m.name != name]
            machines.append(machine)
            try:
                # Whatever bytes the pin holds: a broken one (no text at
                # all) is mended by registering the machine again.
                previous: str | None = cert.read_text(encoding="utf-8", errors="surrogateescape")
            except FileNotFoundError:
                previous = None
            try:
                os.replace(staged, cert)
                save_machines(machines)
            except BaseException:
                # Unless the entry was published (the registry write
                # went through; what failed came after it), the
                # registry still names the daemon the previous pin
                # verified, which must not be left unverifiable.
                if machine not in load_machines():
                    if previous is None:
                        cert.unlink(missing_ok=True)
                    else:
                        atomic_write_text(cert, previous, errors="surrogateescape")
                raise
    finally:
        staged.unlink(missing_ok=True)  # gone already once published
    return machine


def remove_machine(name: str) -> bool:
    """Forget *name*; return False when no such machine is registered."""
    with registry_lock():
        machines = load_machines()
        kept = [m for m in machines if m.name != name]
        if len(kept) == len(machines):
            return False
        save_machines(kept)
        for machine in machines:
            if machine.name == name:
                machine.cert_file().unlink(missing_ok=True)
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
    link backs off for :data:`FIRST_BACKOFF` seconds at least after a
    failure); the last error is raised once the monotonic *deadline*
    has passed.

    An SSH machine is pinged through a forward of these handshakes'
    own, closed on return: the shared forward to its daemon
    (:func:`tunnel_port`) may carry the links and streams of a machine
    already registered there, which a failed registration must leave
    as they are.

    Raises:
        MachineError: The last attempt's error.
    """
    tunnel = SshTunnel(*_endpoint(machine)) if machine.ssh else None
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
            time.sleep(1.0)
    finally:
        if tunnel is not None:
            tunnel.close()


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
    except OSError as exc:  # the lock's ``TimeoutError`` is one
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
        proc = popen_process_group(
            ssh_command(ssh, command=command), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:  # no ssh, or one that cannot be run
        raise MachineError(f"ssh is not installed on this machine: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(ca_cert_path.read_bytes(), timeout=timeout)
    except BaseException as exc:
        # ssh's whole group, as ``SshTunnel._kill`` does: a ``ProxyCommand``
        # of the user's ssh configuration is a child of ssh that a timeout
        # (or a Stop) must take down with it, or it lives on, re-parented.
        kill_process_tree(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5.0)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise MachineError(f"ssh {ssh} did not answer in time") from exc
        raise
    if proc.returncode != 0:
        message = {
            1: f"no Sorcar daemon is running on {ssh} (cannot find its {LOCAL_ENDPOINT_FILE})",
            2: f"cannot store this daemon's CA on {ssh}",
            3: f"cannot enroll on {ssh}",
        }.get(proc.returncode, f"ssh {ssh} failed")
        raise MachineError(_with_last_line(message, stderr))
    marker = "-----BEGIN CERTIFICATE-----"
    endpoint_text, found, ca_rest = stdout.decode("utf-8", "replace").partition(marker)
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
    user@host`` (through :func:`ssh_command`) and waits until ssh says
    the local port is bound; :meth:`alive` says whether the ssh process
    still runs and the port still answers; :meth:`close` ends the
    process.  :func:`tunnel_port` keeps one tunnel per ``ssh`` login
    and daemon port and re-opens a dead one on the next connection.
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
        # Set by the ssh process's output once it said its forward is up.
        self._up = threading.Event()
        # Reads the ssh process's output to its end (:meth:`_read_output`).
        self._reader: threading.Thread | None = None

    def alive(self) -> bool:
        """Return whether the ssh process runs, said its forward is up, and the port answers.

        A process that never said so (its opening was interrupted) is
        not alive even if the port answers: another program may hold it.
        """
        return (
            self.process is not None and not _exited(self.process)
            and self._up.is_set() and _port_accepts(self.local_port)
        )

    def open(self, timeout: float = SSH_TIMEOUT) -> int:
        """Start the forward when it is not alive; return the local port.

        Raises:
            MachineError: When ``ssh`` is missing, no thread can be
                started to read its output, it exits (its last
                output line is quoted), cannot bind any of
                :data:`FORWARD_ATTEMPTS` local ports or does not open
                the forward within *timeout* seconds.
        """
        return locked_call(self._lock, self._open_locked, timeout)

    def _open_locked(self, timeout: float) -> int:
        """:meth:`open`'s body, under the lock."""
        if self.alive():
            return self.local_port
        deadline = time.monotonic() + timeout
        for _ in range(FORWARD_ATTEMPTS):
            if self._forward(deadline):
                return self.local_port
            if time.monotonic() >= deadline:
                break
        raise self._forward_error()

    def _forward(self, deadline: float) -> bool:
        """Start ssh forwarding a newly picked local port; return whether the forward is up.

        Up means ssh ran the ``LocalCommand`` that prints
        :data:`FORWARD_UP`, which it does only once the local port is
        bound.  The port answering would prove nothing: another program
        may bind it after :func:`_free_port` let it go and before ssh
        does, and would then get this daemon's connections.  ssh then
        fails to bind it and False is returned, for the caller to pick
        another port; ssh failing otherwise raises.
        """
        self._kill()
        self.local_port = _free_port()
        forward = f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}"
        # ClearAllForwardings in the user's ssh config would drop the -L
        # and still run the LocalCommand.
        cmd = ssh_command(
            self.ssh, "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ClearAllForwardings=no",
            "-o", "ServerAliveInterval=15", "-o", "PermitLocalCommand=yes",
            "-o", f"LocalCommand=echo {FORWARD_UP}", "-L", forward,
        )
        self._last_line = ""
        try:
            # In a group of its own: a ``ProxyCommand`` of the user's
            # ssh configuration is a child of ssh that ``_kill`` must
            # take down with it, or it keeps ssh's output pipe (and
            # the thread reading it) open after ssh is gone.
            self.process = popen_process_group(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise MachineError(f"ssh is not installed on this machine: {exc}") from exc
        taken = threading.Event()
        self._reader = threading.Thread(
            target=self._read_output,
            args=(self.process, f"bind [127.0.0.1]:{self.local_port}:", self._up, taken),
            daemon=True, name=f"ssh-tunnel-output {self.ssh}",
        )
        try:
            self._reader.start()
        except RuntimeError as exc:  # out of threads: nothing would read ssh's output
            self._kill()
            raise MachineError(f"ssh {self.ssh}: the port forward could not start: {exc}") from exc
        while not self._up.wait(0.05):
            if taken.is_set() or _exited(self.process):
                # Windows' ssh stays up after a failed bind: a moment to
                # exit by itself, unreaped (``_kill`` signals its group
                # before reaping it).
                grace = time.monotonic() + min(1.0, _remaining(deadline))
                while not _exited(self.process) and time.monotonic() < grace:
                    time.sleep(0.05)
                self._kill()
                if taken.is_set():
                    return False
                raise self._forward_error()
            if time.monotonic() >= deadline:
                self._kill()
                raise MachineError(f"ssh {self.ssh} did not open the port forward in time")
        return True

    def _forward_error(self) -> MachineError:
        """Return the error of a forward ssh gave up on, quoting ssh's last line."""
        message = f"ssh {self.ssh} cannot forward its daemon port {self.remote_port}"
        return MachineError(f"{message}: {self._last_line}" if self._last_line else message)

    def _read_output(
        self, process: subprocess.Popen[bytes], taken_prefix: str,
        up: threading.Event, taken: threading.Event,
    ) -> None:
        """Read *process*'s output to its end (so its pipe never fills up).

        :data:`FORWARD_UP` sets *up*; any other line is kept as the last
        line, and one starting with *taken_prefix* (ssh's report that it
        could not bind the local port) sets *taken*.
        """
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line == FORWARD_UP:
                up.set()
            elif line:
                self._last_line = line
                if line.startswith(taken_prefix):
                    taken.set()

    def close(self) -> None:
        """Kill the ssh process, if any, and forget the port."""
        locked_call(self._lock, self._kill)

    def close_unless_opening(self) -> None:
        """:meth:`close`, unless the forward is being opened right now.

        For a host back from sleep (:func:`forget_connections`): the
        ssh process kept from before the sleep sits on a connection
        the network may have dropped, and notices only after its own
        keepalives go unanswered (45 s), while every forwarded
        connection hangs; a login in progress is fresh and is left
        alone rather than waited for.
        """
        try:
            locked_call(self._lock, self._kill, acquire_timeout=0)
        except LockTimeoutError:
            pass  # being opened: the login is fresh

    def _kill(self) -> None:
        """Kill ssh's group, if any, then reap ssh; forget port and readiness (under the lock).

        The group is signalled while ssh is still ours to reap (nothing
        here reaps it earlier: :func:`_exited` leaves an exited ssh a
        zombie, which is still a member of its group), so the group id
        names ssh's group and nobody else's, and a ``ProxyCommand``'s
        helper that outlived an ssh which exited on its own goes down
        with it.  Then the reader is joined (bounded) so a close leaves
        no thread behind; a reader a Stop left unstarted (between its
        creation and its start in :meth:`_forward`) cannot be joined
        and is dropped with the process whose output it was to read.
        """
        if self.process is not None:
            if IS_WINDOWS or self.process.returncode is None:
                kill_process_tree(self.process)
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5.0)
            if self._reader is not None and self._reader.is_alive():
                self._reader.join(timeout=2.0)
            self._reader = None
            self.process = None
        self.local_port = 0
        self._up = threading.Event()


def _exited(process: subprocess.Popen[bytes]) -> bool:
    """Return whether *process* has exited, without reaping it.

    ``Popen.poll`` reaps an exited child, and a reaped group leader no
    longer reserves its group id (:func:`~kiss.core.processes.kill_process_group`),
    while a zombie is still a member of its group: :meth:`SshTunnel._kill`
    signals the group before it reaps, so the id still names ssh's group.
    Windows has no zombies to keep: the process handle ``Popen`` holds
    keeps the pid, polled or not.
    """
    if IS_WINDOWS or process.returncode is not None:
        return process.poll() is not None
    try:
        return waitid_exited(process.pid) is not None
    except ChildProcessError:  # reaped meanwhile by a ``wait`` on another thread
        return True


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


_Endpoint = tuple[str, int]
"""What an ssh forward leads to: the ssh login and the daemon port on that machine."""


def _endpoint(machine: Machine) -> _Endpoint:
    """Return the :data:`_Endpoint` the SSH *machine* is reached through."""
    return machine.ssh, urlsplit(machine.url).port or DEFAULT_PORT


_tunnels: dict[_Endpoint, SshTunnel] = {}
_tunnels_lock = threading.RLock()


def _cached_tunnel(ssh: str, remote_port: int) -> SshTunnel | None:
    """Return the cached tunnel to *remote_port* on *ssh*, made on first use (under the lock).

    ``None`` when the registry does not hold *ssh* at *remote_port*:
    read under the lock, so a dial of a registration the registry has
    moved past (a stream handshake still in flight on the link a
    re-registration retired, asking for the old daemon port) caches
    and opens no forward that :func:`close_tunnels` has already swept.
    """
    endpoint = (ssh, remote_port)
    if not any(m.ssh and _endpoint(m) == endpoint for m in load_machines()):
        return None
    tunnel = _tunnels.get(endpoint)
    if tunnel is None:
        tunnel = _tunnels[endpoint] = SshTunnel(ssh, remote_port)
    return tunnel


def _evict_tunnels(kept: set[_Endpoint]) -> list[SshTunnel]:
    """Forget and return the tunnels of every endpoint not in *kept* (under the lock)."""
    gone = [endpoint for endpoint in _tunnels if endpoint not in kept]
    return [_tunnels.pop(endpoint) for endpoint in gone]


def tunnel_port(ssh: str, remote_port: int) -> int:
    """Return the local port forwarding to ``127.0.0.1:remote_port`` on *ssh*.

    One :class:`SshTunnel` per login and daemon port is kept while the
    registry holds a machine there: it is reused while alive and opened
    again (on a fresh local port) once its ssh process died.  Two
    daemons of one login (two registered ports) have a forward each:
    the dial of one never closes the forward under the other's streams.
    A machine registered again after its daemon moved gets the forward
    to the new port, and the old one is closed with its registration
    (:func:`close_tunnels`).

    The forward is opened outside ``_tunnels_lock`` (an ``ssh`` login
    takes seconds and must not hold up the other machines' tunnels), so
    :func:`close_tunnels` may have dropped the tunnel meanwhile: a
    forward opened on a tunnel the cache no longer holds would have no
    one to close it, so it is closed here and the caller is told to
    try again, as a retired :class:`MachineLink` tells it.

    A forward the registry does not hold (*ssh* at *remote_port*) is
    refused before anything is cached or opened: a :class:`Machine`
    read before a ``/machine remove`` (a status check, a stream's dial)
    would otherwise cache and open its forward after the removal's
    :func:`close_tunnels` swept, and nothing would close it.  The
    registry is read under the cache's lock and the sweep runs after
    the registry is written, so a removal either shows in that read or
    sweeps the tunnel.

    Raises:
        MachineError: When the forward cannot be opened (see
            :meth:`SshTunnel.open`), the registry does not hold *ssh*
            at *remote_port*, or the registry changed while it was
            opening.
    """
    tunnel = locked_call(_tunnels_lock, _cached_tunnel, ssh, remote_port)
    if tunnel is not None:
        # Test hook (no-op in production): widens the window between the
        # lookup and the open so the close_tunnels race is deterministic
        # under KISS_RACE_DELAY.
        _race_delay()
        port = tunnel.open()
        if locked_call(_tunnels_lock, _tunnels.get, (ssh, remote_port)) is tunnel:
            return port
        tunnel.close()
    raise MachineError(f"ssh {ssh}: the machine registry changed, try again")


def close_tunnels(keep: Iterable[_Endpoint] = ()) -> None:
    """Close and forget the ssh port-forward to every endpoint not in *keep*.

    Called with the endpoints (:func:`_endpoint`) of the registry just
    written (a forward to a daemon still registered stays up, whatever
    else changed for its login) and with none at exit.  The forwards
    are closed outside the cache's lock: a ``close`` waits for a
    forward that is mid-login, which must not hold up the other
    machines' dials.
    """
    tunnels: list[SshTunnel] = locked_call(_tunnels_lock, _evict_tunnels, set(keep))
    for tunnel in tunnels:
        tunnel.close()


@dataclass
class LinkHealth:
    """A machine's back-off record: the last failure and when to dial again.

    Shared by every lane's link to the machine (:func:`link_for`), so
    a machine found down on one lane is reported down on the others
    without each dialling it in turn.
    """

    retry_at: float = 0.0
    last_error: str = ""
    failures: int = 0
    """Failures in a row; sets the next back-off (see :data:`DOWN_BACKOFF`)."""


class MachineLink:
    """One persistent, authenticated WebSocket to a remote daemon.

    Connects lazily on the first :meth:`request`, re-connects after a
    failure and serialises requests, so the daemon keeps one socket per
    machine across CHATS refreshes.  After a failure the link reports
    the machine down without trying again for a while
    (:data:`FIRST_BACKOFF` seconds, doubling with every failure in a
    row up to :data:`DOWN_BACKOFF`), so an offline machine does not add
    a connection timeout to every CHATS refresh, and a rejected
    password waits :data:`DOWN_BACKOFF` from the first failure so it is
    not retried often enough to trip the remote daemon's brute-force
    lockout.
    """

    def __init__(
        self, machine: Machine, cert_file: Path | None = None, tunnel: SshTunnel | None = None,
        health: LinkHealth | None = None,
    ) -> None:
        """A link to *machine*, pinned to *cert_file* (its registered pin by default).

        *tunnel*, for an SSH machine, is a forward of the caller's own
        to dial instead of the shared one (:func:`tunnel_port`);
        the caller opens nothing and closes it when done.  *health* is
        the back-off record to share with the machine's other lanes
        (:func:`link_for`); a link of its own by default.
        """
        self.machine = machine
        self._cert_file = cert_file if cert_file is not None else machine.cert_file()
        self._tunnel = tunnel
        self._ws: ClientConnection | None = None
        self._lock = threading.RLock()  # see SshTunnel: taken through locked_call
        self._generation = 0
        self._echoes = False  # the daemon's ``echoesGeneration`` (see CORRELATED_REPLIES)
        self._health = health if health is not None else LinkHealth()
        self._retired = False
        self._abandoned: ClientConnection | None = None  # the socket :meth:`abandon` shut down
        # Stream handshakes in flight (:meth:`open_connection`), one
        # token each, which run outside the lock; :meth:`retire` waits
        # for them.  Kept under a lock of their own: the request lock is
        # held for a whole exchange, which a stream must not wait behind.
        self._dialing: set[object] = set()
        self._idle = threading.Condition(threading.RLock())

    def _check_available(self) -> None:
        """Raise the error a retired or backed-off link answers with, if any.

        Drops the socket first when the host has slept since the last
        check (:func:`forget_after_sleep`): the socket is dead and so is
        the back-off a failed poll during the sleep may have left.  The
        drop takes the lock (re-entered by a caller holding it); the
        checks themselves read without it, like the machine's other
        lanes read :class:`LinkHealth`.
        """
        if forget_after_sleep():
            locked_call(self._lock, self._drop)
        if self._retired:
            raise self._retired_error()
        if time.monotonic() < self._health.retry_at:
            raise MachineError(self._health.last_error)

    def _retired_error(self) -> MachineError:
        """Return the error every request of a retired link fails with (no back-off)."""
        return MachineError(f"{self.machine.name}: the machine registry changed, try again")

    def _fail(self, message: str, backoff: float = 0.0) -> MachineError:
        """Record *message* as the link's failure, start the backoff, return the error.

        The back-off is *backoff* seconds when given, else
        :data:`FIRST_BACKOFF` doubled for every failure in a row, up to
        :data:`DOWN_BACKOFF`.  The message is stored before the back-off:
        the machine's other
        lanes read :class:`LinkHealth` without a lock, in the order
        :meth:`_check_available` reads it, so one that sees the new
        back-off never reports the previous (on a first failure, empty)
        message.
        """
        health = self._health
        health.last_error = message
        health.failures += 1
        # The exponent is capped: the ladder tops out at DOWN_BACKOFF
        # after a few steps, and a machine down for days must not
        # overflow the power.
        doublings = min(health.failures - 1, 8)
        wait = backoff or min(DOWN_BACKOFF, FIRST_BACKOFF * 2**doublings)
        health.retry_at = time.monotonic() + wait
        return MachineError(message)

    def abandon(self) -> None:
        """Shut the socket's connection down and clear the back-off, without the lock.

        For a host back from sleep (:func:`forget_connections`): the
        daemon at the far end closed the socket while its pings went
        unanswered, so a close handshake would only wait out
        ``close_timeout`` on a dead connection.  The shutdown wakes a
        request in flight on the socket with ``ConnectionClosed``: it
        fails as a sleep of the host, not as a failure of the machine
        (:meth:`_exchange` knows the socket), so the back-off cleared
        here is not armed again; the next request finds the socket
        closed and reconnects (:meth:`request`).  The socket is
        forgotten by whoever holds the lock next.
        """
        ws = self._ws
        if ws is not None:
            self._abandoned = ws
            hang_up(ws)
        self._health.failures = 0
        self._health.retry_at = 0.0

    def _connect(self, timeout: float) -> None:
        """Open, authenticate and keep the request socket (under the lock); see :meth:`_dial`.

        A :meth:`retire` that ran during the handshake could not reach
        a socket not yet kept: the socket is dropped here instead,
        and the request fails without going out.
        """
        self._check_available()
        self._dial(timeout, stream=False)
        if self._retired:
            self._drop()
            raise self._retired_error()

    def _dial(self, timeout: float, stream: bool) -> ClientConnection:
        """Open and authenticate a socket; the handshake has *timeout* seconds.

        Takes no lock: the shared request socket's connect holds the
        link's lock through :meth:`_connect`, while a stream socket
        (:meth:`open_connection`) is the caller's own and connects
        beside a request in flight or another stream's handshake.
        What is shared, the machine's :class:`LinkHealth`, is written
        as :meth:`_fail` describes.

        The socket is opened and authenticated on a thread of its own
        (:meth:`_handshake`, through
        :func:`~kiss.agents.sorcar.local_endpoint.dial_off_thread`,
        which tells what a Stop raised inside those calls used to
        do): a Stop of the calling task ends the wait at once, and the
        dial it gave up on closes its socket and records nothing.
        So does a dial that has not reported :data:`DIAL_GRACE` seconds
        after the handshake's deadline: the lookup of the machine's
        name (or the proxy's) takes no timeout, and held the request,
        and the link's lock with it, for as long as the resolver tried.
        The outcome is recorded here, on the caller's thread.

        The new socket has an owner at every instant.  The dialling
        thread hands it over in a local variable of this thread, and a
        Stop raised before the link kept it left it open with nobody
        to close it, its threads answering the daemon's pings.  So
        whatever cuts this method short hangs up the socket the dial
        reported (:func:`hang_up`), and the request socket (*stream*
        false) is kept as the link's here, not by the caller.  A
        stream's socket is its caller's once this method returns.

        The socket is TLS with the machine's pinned certificate as the
        only trust anchor.  An SSH machine is dialled at the local end
        of its ssh port-forward (the shared one to its daemon,
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

        Both sockets ask for permessage-deflate (``compression``
        ``"deflate"``, which the daemon at the other end grants): the
        frames a link carries are JSON, and a machine's CHATS page
        (``history``) or a relayed transcript (``task_events``) is
        three to six times smaller compressed, which is what a slow
        network between the two daemons notices.  An ssh forward
        does not compress on its own, and TLS hides the bytes from
        it anyway.

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
        enroll_again = (
            f"; run /machine add {machine.name} {machine.ssh} again" if machine.ssh else ""
        )
        if not machine.url:
            raise self._fail(f"{where} was registered before certificate enrollment{enroll_again}")
        try:
            ctx = ssl.create_default_context(cafile=str(self._cert_file))
        except OSError as exc:  # ssl.SSLError included
            raise self._fail(f"{where} has no usable pinned certificate: {exc}") from exc
        url, forward_refused = machine.url, ""
        if machine.ssh:
            try:
                ctx.load_cert_chain(*map(str, client_cert_pair()))
            except (OSError, MachineError) as exc:
                raise self._fail(f"{where}: no usable client certificate: {exc}") from exc
            remote_port = _endpoint(machine)[1]
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
        # ``parts``: the link joins the parts of a big frame (a
        # transcript, a picture) the daemon sends in pieces, see
        # ``local_endpoint.frame_parts``.
        auth = {"type": "auth", "link": not stream, "parts": True, "password": machine.password}
        deadline = time.monotonic() + timeout
        # No proxy for the local end of an ssh forward: the client's
        # default sends 127.0.0.1 to ``$https_proxy`` unless
        # ``no_proxy`` lists it, and the proxy has no such forward.
        connect = partial(
            _ws_connect, ws_url(url), ssl=ctx, open_timeout=_remaining(deadline),
            close_timeout=2.0, compression="deflate", max_size=None,
            max_queue=16 if stream else None,
            ping_interval=LINK_PING_INTERVAL, ping_timeout=LINK_PING_TIMEOUT,
            proxy=None if machine.ssh else True,
        )
        handed: list[tuple[ClientConnection, object]] = []
        try:
            try:
                ws = dial_off_thread(
                    partial(self._handshake, connect, json.dumps(auth), deadline, handed),
                    timeout + DIAL_GRACE,
                )
            except ssl.SSLError as exc:
                if forward_refused and isinstance(exc, ssl.SSLEOFError):
                    raise self._fail(forward_refused) from exc
                raise self._fail(
                    f"{where} refused the TLS handshake: {exc}{enroll_again}"
                ) from exc
            except InvalidStatus as exc:  # 403: no password set there and no trusted certificate
                raise self._fail(f"{where} refused the connection: {exc}{enroll_again}") from exc
            except (OSError, WebSocketException, ImportError) as exc:
                # WebSocketException: a failed handshake, or a URL or ``$https_proxy``
                # the client cannot use; ImportError: a SOCKS proxy without python-socks.
                if forward_refused and isinstance(exc, ConnectionResetError):
                    raise self._fail(forward_refused) from exc
                raise self._fail(f"{where} is unreachable: {exc}") from exc
            except RuntimeError as exc:  # out of threads: the dial's did not start, or the client's
                raise self._fail(f"{where}: the connection could not start: {exc}") from exc
            answer = handed[0][1]
            if isinstance(answer, Exception):
                raise self._fail(f"{where} did not complete the handshake: {answer}") from answer
            if isinstance(answer, dict) and answer.get("type") == "auth_ok":
                self._echoes = answer.get("echoesGeneration") is True
                # The machine answered: the ladder starts over, and the
                # back-off a handshake beside this one (a stream's)
                # may have left goes with it.
                self._health.failures = 0
                self._health.retry_at = 0.0
                if not stream:
                    self._ws = ws
                return ws
            if machine.ssh:
                raise self._fail(f"{where} does not trust this daemon's certificate{enroll_again}")
            if isinstance(answer, dict) and answer.get("type") == "auth_required":
                raise self._fail(f"{where} rejected the password", DOWN_BACKOFF)
            raise self._fail(f"{where} refused the connection: {answer!r}")
        except BaseException:
            # Cut short with the socket open, by a Stop of the calling
            # task above all (a refusal's socket is closed already).
            for opened, _ in handed:
                hang_up(opened)
            raise

    @staticmethod
    def _handshake(
        connect: Callable[[], ClientConnection], auth: str, deadline: float,
        handed: list[tuple[ClientConnection, object]],
    ) -> ClientConnection:
        """Open a socket with *connect* and send it the *auth* frame: :meth:`_dial`'s blocking part.

        Runs on the dialling thread
        (:func:`~kiss.agents.sorcar.local_endpoint.dial_off_thread`)
        and touches nothing of the link, so a dial its requester gave
        up on leaves no trace.  The socket is appended to *handed*
        with the daemon's answer to the frame, awaited until the
        monotonic *deadline*, or with the error the exchange ended
        with.  The socket is returned open for an ``auth_ok`` answer
        only and closed otherwise (here, not on the requester's
        thread, where a Stop could land in the close).

        Raises:
            Exception: What *connect* raised: the socket did not open.
        """
        ws = connect()
        answer: object = None
        try:
            send(ws, auth, timeout=_remaining(deadline))
            answer = recv_auth_reply(ws, timeout=_remaining(deadline))
        except (ConnectionClosed, OSError, ValueError, RuntimeError) as exc:
            # RuntimeError: out of threads, ``send`` could not start its writer.
            answer = exc
        finally:
            if not (isinstance(answer, dict) and answer.get("type") == "auth_ok"):
                ws.close()
        handed.append((ws, answer))
        return ws

    def request(
        self, cmd: dict[str, Any], reply_type: str, timeout: float = REQUEST_TIMEOUT,
        *, errors: bool = False,
    ) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* event answering it.

        An empty *reply_type* is for a command the daemon does not
        answer (``recordFileUsage``): ``{}`` is returned once it is sent.

        The command is stamped with a fresh ``generation``; a reply that
        echoes a ``generation`` must echo this one, and one of the
        :data:`CORRELATED_REPLIES` must echo it when the daemon said it
        does (``echoesGeneration`` in its ``auth_ok``); frames already
        buffered from before the command are discarded and other
        frames the daemon pushes meanwhile are skipped.  A
        reply the daemon sent in pieces (a ``fileContent`` of many MB:
        every frame but the last flagged ``more``) comes back joined
        into one.  With
        *errors*, an ``error`` frame answering the command (a refused
        custom-model name, say) comes back as a *reply_type* reply
        carrying its text as ``error`` instead of being skipped; a
        wrong-password refusal is a :class:`MachineError` either way.
        *timeout*
        bounds the wait for the reply, however many frames arrive or
        are already buffered; connecting first, when the socket is not
        up, has a budget of *timeout* of its own for the handshake
        (an SSH machine's port-forward, opened before it, has
        :data:`SSH_TIMEOUT`).

        A socket kept from an earlier request may have died meanwhile
        (the daemon restarted, a network blip, its idle keepalive
        failed): when that shows before the command went out, it is
        sent again over a fresh socket before the machine counts as
        down, so a stale socket alone never starts the back-off.  A
        close while waiting for the answer fails the request: the
        machine may have run the command, and a paste or a cherry-pick
        must not run twice.  One kept across a sleep of the
        host is dropped before use (:func:`forget_after_sleep`): it is
        dead, and a request on it would hang for its whole budget.

        The link's turn is waited for *timeout* seconds at most, too:
        a request queued behind a slow one on the same link (the
        remote app probe on the panels lane, say) fails as ``busy``
        when its own budget is up instead of waiting for the whole
        queue, so the thread it runs on is never held longer than its
        caller allowed for.  That failure starts no back-off: the
        machine is not down.

        Raises:
            MachineError: When the daemon is unreachable, rejects the
                password, does not answer within *timeout* seconds,
                or the link's turn did not come within them.
        """
        try:
            return locked_call(
                self._lock,
                self._request_locked,
                cmd,
                reply_type,
                timeout,
                errors,
                acquire_timeout=timeout,
            )
        except LockTimeoutError:
            raise MachineError(
                f"{self.machine.name} ({self.machine.address()}) is busy: "
                f"no {reply_type} within {timeout:g}s"
            ) from None

    def _request_locked(
        self, cmd: dict[str, Any], reply_type: str, timeout: float, errors: bool,
    ) -> dict[str, Any]:
        """:meth:`request`'s body, under the lock.

        The handshake, when the socket is not up, has
        :data:`REQUEST_TIMEOUT` at most even for a longer *timeout*: a
        host that drops packets makes a reachable daemon's handshake
        no longer, and the long budget of a slow exchange must not keep
        the request's thread in a dial that will not succeed.

        An exchange that ends in anything but its answer leaves the
        link no socket.  The errors handled here and in
        :meth:`_exchange` have dropped it.  Anything else is an
        exchange cut short, by a Stop of the calling task above all
        (``KeyboardInterrupt`` raised asynchronously in its thread),
        and its socket cannot be used again: the answer may still
        arrive and pass for the next request's, and a Stop raised
        inside the socket's own receive leaves it unusable (CPython
        3.14 raises it with the frame assembler's mutex held, where
        the next request waited for good; 3.13 with the mark of a
        receive in progress set, where the next request failed and
        backed the machine off).  The socket is forgotten and its
        connection shut down (:meth:`_discard`); no failure is
        recorded, the machine did nothing wrong.  The handler is
        around the calls of :meth:`_exchange`, not in it: the back
        edge of a loop is not always inside the exception table of a
        ``try`` around the loop (see ``local_endpoint._await``), and
        an exception raised there reaches a caller's handler only.
        """
        # A dial runs the sleep check itself (``_check_available``)
        # and opens the forward again; only a kept socket needs
        # either check here.
        if self._ws is not None and (forget_after_sleep() or self._forward_gone()):
            self._drop()
        kept = self._ws is not None
        if self._ws is None:
            self._connect(min(timeout, REQUEST_TIMEOUT))
        deadline = time.monotonic() + timeout
        try:
            try:
                return self._exchange(cmd, reply_type, deadline, timeout, errors)
            except ConnectionClosed as exc:
                if not kept:
                    self._drop()
                    raise self._fail(
                        f"{self.machine.name} ({self.machine.address()}) did not answer: {exc}"
                    ) from exc
                logger.info("%s: the link's socket had closed, reconnecting", self.machine.name)
                self._drop()
                self._connect(min(_remaining(deadline), REQUEST_TIMEOUT))
                try:
                    return self._exchange(cmd, reply_type, deadline, timeout, errors)
                except ConnectionClosed as again:
                    self._drop()
                    raise self._fail(
                        f"{self.machine.name} ({self.machine.address()}) did not answer: {again}"
                    ) from again
        except BaseException:
            self._discard()
            raise

    def _forward_gone(self) -> bool:
        """Return whether the kept socket's ssh forward is no longer the machine's live one.

        Under the lock.  The forward's ssh process exiting closes the
        local port the socket is connected to, but the socket learns
        of it on its reader thread: a request that wins the race with
        that thread sends on the dead socket, and a close after the
        send is never retried (:meth:`_exchange`), so the machine is
        reported down instead of dialled again, which re-opens the
        forward (:func:`tunnel_port`).  So the forward is asked
        instead: its process has exited or was closed, or another lane
        opened it again on a fresh local port the socket is not
        connected to.  Never for a machine dialled directly.
        """
        if not self.machine.ssh:
            return False
        tunnel = self._tunnel
        if tunnel is None:
            tunnel = locked_call(_tunnels_lock, _tunnels.get, _endpoint(self.machine))
        if tunnel is None:
            return True
        # One read: a close on another thread sets the process to None
        # at any point (the tunnel's lock is not taken here).
        process = tunnel.process
        if process is None or _exited(process):
            return True
        assert self._ws is not None
        try:
            return int(self._ws.socket.getpeername()[1]) != tunnel.local_port
        except OSError:  # closed under the request (:meth:`retire`)
            return True

    def _exchange(
        self, cmd: dict[str, Any], reply_type: str, deadline: float, timeout: float,
        errors: bool,
    ) -> dict[str, Any]:
        """Send *cmd* on the open socket and return its *reply_type* answer (under the lock).

        *deadline* bounds the exchange (*timeout* only names the budget
        in the error); *errors* is :meth:`request`'s.

        Raises:
            ConnectionClosed: When the socket was found closed before the
                command went out (the caller may send it again on a
                fresh socket).
            MachineError: When the daemon did not answer before
                *deadline*, or the socket closed after the command went
                out: the machine may have run it, so it must not be sent
                again (the link then backs off); or the link was
                retired (:meth:`retire` closed the socket): the
                registry-changed error, with no back-off; or the host
                slept (:meth:`abandon` shut the socket down): no
                back-off either.
        """
        assert self._ws is not None
        self._generation += 1
        generation = self._generation
        correlated = self._echoes and reply_type in CORRELATED_REPLIES
        sent = False
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
            # Frames the daemon pushed since the last request (a
            # ``myModelsData`` broadcast for another client's edit,
            # say) are stale: drained first, they can neither pass for
            # this request's answer nor delay it.
            self._drain_buffered(deadline)
            send(self._ws, json.dumps(frame), timeout=_send_budget(deadline))
            sent = True
            if not reply_type:
                return {}
            # A reply sent in pieces (a big file's ``fileContent``, see
            # ``web_server._reply_frames``): every piece but the last
            # carries ``more``.  Joined here, so the caller sees one
            # reply, as from a daemon that sent it whole; each piece
            # proves the daemon is answering and buys the next one the
            # whole *timeout* again.  So does each part of a big frame
            # (``local_endpoint.frame_parts``: a replayed transcript, a
            # picture), joined the same way.
            head: dict[str, Any] | None = None
            pieces: list[str] = []
            joiner = FrameJoiner()
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"no {reply_type} within {timeout:g}s")
                try:
                    text = self._ws.recv(timeout=min(_READ_SLICE, _remaining(deadline)))
                except TimeoutError:
                    continue  # the end of a slice, or of the budget (reported above)
                whole = joiner.feed(json.loads(text))
                if whole is None:
                    deadline = time.monotonic() + timeout
                    continue
                msg = whole[0]
                if not isinstance(msg, dict):
                    continue
                if msg.get("generation", generation) != generation:
                    continue  # another request's answer
                if msg.get("type") == reply_type:
                    if correlated and "generation" not in msg:
                        continue  # pushed for another client's edit, not an answer
                    if head is None and msg.get("more") is not True:
                        return msg
                    if head is None:
                        head = msg
                    pieces.append(str(msg.get("content") or ""))
                    if msg.get("more") is True:
                        deadline = time.monotonic() + timeout
                        continue
                    head["content"] = "".join(pieces)
                    del head["more"]
                    return head
                if msg.get("type") != "error":
                    continue
                # The daemon rejects the frame's password (changed on
                # the machine since the link was made): the link is
                # down, as at connect time.
                if msg.get("code") == "auth_failed":
                    self._drop()
                    raise self._fail(
                        f"{self.machine.name} ({self.machine.address()}) rejected the password",
                        DOWN_BACKOFF,
                    )
                # The daemon refused the command (a rejected custom
                # model name, say): for a caller that asked for it,
                # that is the answer, as a reply of the expected type
                # carrying the ``error``, not a timeout that would
                # count the machine as down.
                if errors:
                    return {"type": reply_type, "error": str(msg.get("text") or "error")}
        except (ConnectionClosed, ValueError, OSError, RuntimeError) as exc:
            # OSError: a reply not read in time (``TimeoutError``) among others.
            # RuntimeError: out of threads, ``send`` could not start its writer.
            if self._retired:  # retire() shut the socket down under the request
                self._drop()
                raise self._retired_error() from exc
            # A close before the command went out is the caller's to retry
            # on a fresh socket.  After it, the machine may have the command
            # (a paste, a cherry-pick, an auto-commit): sending it again
            # could run it twice.
            if isinstance(exc, ConnectionClosed) and not sent:
                raise
            abandoned = self._ws is self._abandoned
            self._drop()
            if abandoned:
                # abandon() shut the socket down under the request: the
                # host slept, which is no failure of the machine and
                # starts no back-off.
                raise MachineError(
                    f"{self.machine.name} ({self.machine.address()}): the host slept "
                    f"while waiting for {reply_type}; try again"
                ) from exc
            raise self._fail(
                f"{self.machine.name} ({self.machine.address()}) did not answer: {exc}"
            ) from exc

    def _drain_buffered(self, deadline: float) -> None:
        """Discard the frames already received on the open socket (under the lock).

        Bounded by *deadline* but never fails on it: a budget already
        spent (a reconnect that used it up, a caller's zero *timeout*)
        leaves the frames in place and lets :meth:`_exchange` report
        the accurate ``no <reply> within <N>s`` instead of blaming a
        backlog that may not exist.
        """
        assert self._ws is not None
        while time.monotonic() < deadline:
            try:
                self._ws.recv(timeout=0)
            except TimeoutError:
                return

    def _drop(self) -> None:
        """Close and forget the socket (no error is recorded).

        Forgotten whatever ends the close (:meth:`_discard`): a Stop
        raised in the closing handshake left the link a socket that
        was still open, and on CPython 3.14 one with its protocol
        mutex held, where the link's next request waited for good.
        """
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                logger.debug("closing link to %s failed", self.machine.name, exc_info=True)
            finally:
                self._discard()

    def _discard(self) -> None:
        """Forget the socket and shut its connection down, under the lock (no error is recorded).

        For a socket in a state nobody knows (:meth:`_request_locked`
        tells how a Stop leaves one), where :meth:`_drop`'s closing
        handshake would take locks of the socket that may be held
        for good; :func:`hang_up` takes none.  Nothing is left to
        shut down after a close that completed.
        """
        ws, self._ws = self._ws, None
        if ws is not None:
            hang_up(ws)

    def open_connection(self, timeout: float = REQUEST_TIMEOUT) -> ClientConnection:
        """Open a fresh authenticated socket to the daemon for a long-lived stream.

        A stream (a chat tab running on the machine, see
        :mod:`kiss.server.remote_tabs`) relays frames for as long as it
        lives, so it cannot take turns on the socket :meth:`request`
        shares.  Unlike that socket it is an ordinary peer of the daemon
        (it needs the tab's task events).  The caller owns the returned
        socket and closes it.

        The handshake runs outside the link's lock (the availability
        check takes it only to drop the shared socket after a sleep,
        and the handshake is counted under the dial lock, where
        :meth:`retire` is checked for again): a stream
        connects beside a request in flight on the shared socket, and
        two streams to the machine (a chat tab's and the machine's view
        stream, opened together by a bind) connect at the same time
        instead of one after the other.

        The handshake is counted in inside the ``try`` that counts it
        out, as a token of its own: a Stop of the calling task is
        raised where the dial lock is released, and with the count
        taken ahead of the ``try`` it stayed, for :meth:`retire` to
        wait its whole bound on.

        Raises:
            MachineError: When the daemon is unreachable or rejects the
                password (the link backs off as after a failed request).
        """
        self._check_available()
        dial = object()
        try:
            locked_call(self._idle, self._start_dial, dial)
            return self._dial(timeout, stream=True)
        finally:
            locked_call(self._idle, self._end_dial, dial)

    def _start_dial(self, dial: object) -> None:
        """Count the stream handshake *dial* in, unless the link is retired (under the dial lock).

        The check is under the lock :meth:`retire` waits under, after
        the one :meth:`_check_available` made without it: a handshake
        counted in is one the retirement waits for, and one the
        retirement has stopped waiting for never starts.
        """
        if self._retired:
            raise self._retired_error()
        self._dialing.add(dial)

    def _end_dial(self, dial: object) -> None:
        """Count *dial* out, counted in or not; wake a waiting :meth:`retire` (dial lock held)."""
        self._dialing.discard(dial)
        self._idle.notify_all()

    def close(self) -> None:
        """Close the socket; the next :meth:`request` re-connects."""
        locked_call(self._lock, self._drop)

    def retire(self) -> None:
        """Close for good: every later :meth:`request` fails instead of re-connecting.

        Used when the link leaves the daemon-wide cache, so a caller
        that fetched it just before cannot re-open a socket nothing
        closes any more.  Does not wait for a request in flight: the
        socket's connection is shut down first, outside the lock, which
        wakes the request's read with ``ConnectionClosed`` (the request
        then fails with the registry-changed error, sent or not, and
        never re-connects), and only then the lock is taken to close
        and forget the socket.  The shutdown is :func:`hang_up`'s,
        not the socket's own close: a registry change runs on a task's
        thread, and a Stop raised in a close outside the lock left the
        retired link its socket, open, and on CPython 3.14 with the
        protocol mutex held, where a request still on the link never
        ended.  A request still in its handshake, which the shutdown
        cannot reach, is waited for (:data:`REQUEST_TIMEOUT` at most:
        it fails at the end of the handshake, :meth:`_connect`, and
        never sends its command), and so is a stream handshake
        (:meth:`open_connection`, outside the lock; the ssh login's
        :data:`SSH_TIMEOUT` more): the tunnel it opens must exist
        before :func:`close_links` sweeps the tunnels, or a removed
        machine's would be opened afresh after the sweep and kept.
        """
        self._retired = True
        ws = self._ws
        if ws is not None:
            hang_up(ws)
        locked_call(self._lock, self._drop)
        locked_call(
            self._idle, self._idle.wait_for, self._dials_done, SSH_TIMEOUT + REQUEST_TIMEOUT,
        )

    def _dials_done(self) -> bool:
        """Whether no stream handshake is in flight (under the dial lock)."""
        return not self._dialing


def _remaining(deadline: float) -> float:
    """Seconds left before the monotonic *deadline* (a small positive floor)."""
    return max(0.001, deadline - time.monotonic())


#: The least a request's command write waits, whatever is left of the
#: request's budget.  The write itself takes microseconds (a frame into
#: the kernel's buffer), but the helper thread that performs it
#: (``local_endpoint.send``) may wait longer than a spent budget's
#: floor for a core on a loaded box; a write that lost that race is no
#: daemon that stopped reading, and reporting it as one closed a sound
#: socket and hid the reply's ``no <reply> within <N>s`` (full-suite
#: run of 2026-11-05, a zero *timeout*).  The request's deadline still
#: bounds the reply (:meth:`MachineLink._exchange` checks it before
#: every read); the handshake's ``auth`` frame keeps the handshake's
#: own budget, which :meth:`MachineLink._dial` promises.
_SEND_FLOOR = 1.0

#: The longest a request blocks in one read of its answer.  A Stop of
#: the calling task is raised when a blocking read returns, not while
#: it waits: read to the end of its budget in one piece, a request
#: the daemon did not answer kept a stopped task for the whole budget.
_READ_SLICE = 0.2


def _send_budget(deadline: float) -> float:
    """Seconds a command's write gets: the time to *deadline*, :data:`_SEND_FLOOR` at least.

    Args:
        deadline: The request's monotonic deadline.
    """
    return max(_SEND_FLOOR, deadline - time.monotonic())


_Registration = tuple[str, str, str, str]
"""A machine as registered (name, URL, password, ssh login): what a link
(with its lane), a fetch and a remembered page are keyed by, so a name
registered again for another daemon or with other credentials shares
none of them with the previous registration."""


def _registration(machine: Machine) -> _Registration:
    """Return *machine*'s :data:`_Registration`."""
    return (machine.name, machine.url, machine.password, machine.ssh)


_links: dict[tuple[str, str, str, str, str], MachineLink] = {}
_links_lock = threading.RLock()


def _cached_link(machine: Machine, lane: str) -> MachineLink | None:
    """Return the cached *lane* link to *machine*, made on first use (under the lock).

    A new lane's link shares the machine's back-off record with the
    lanes already cached, so a machine found down on one lane is not
    dialled again by each of the others.  ``None`` when the registry
    no longer holds *machine* as given (see :func:`link_for`).
    """
    key = (*_registration(machine), lane)
    link = _links.get(key)
    if link is None:
        if machine not in load_machines():
            return None
        health = next(
            (other._health for other_key, other in _links.items() if other_key[:4] == key[:4]),
            None,
        )
        link = _links[key] = MachineLink(machine, health=health)
    return link


def link_for(machine: Machine, lane: str = "") -> MachineLink:
    """Return the daemon-wide cached link to *machine*.

    A link answers one request at a time, so a slow exchange (a
    machine's app probe, a long spend report) holds up every other
    request behind it.  *lane* names a separate link of its own for
    such traffic (the sidebar subpanels' polls use ``"panels"``), so
    the interactive requests (a submit's stream, the settings sheet,
    the Explorer) on the default lane never wait for it.  Every lane
    is retired together by :func:`close_links`.

    A machine the registry no longer holds as given (a
    :class:`Machine` read before a ``/machine remove`` or a
    re-registration) gets a retired link of its own, which answers
    every request with "the machine registry changed, try again" as a
    link the change retired does: a cached one would open a socket
    after the change's sweep (:func:`close_links`), and nothing would
    close it.  The registry is read under the cache's lock, under which
    the sweep evicts after the registry is written, so a change either
    shows in that read or evicts the link.

    Args:
        machine: The registered machine.
        lane: The link's lane; "" for the default one.
    """
    link = locked_call(_links_lock, _cached_link, machine, lane)
    if link is None:
        link = MachineLink(machine)
        link.retire()
    return link


def _evict_links(kept: set[_Registration]) -> list[MachineLink]:
    """Forget and return the cached links of every machine not in *kept* (under the lock)."""
    gone = [key for key in _links if key[:4] not in kept]
    return [_links.pop(key) for key in gone]


def close_links(keep: Iterable[Machine] = ()) -> None:
    """Retire and forget the cached links and ssh tunnels of every machine not in *keep*.

    :func:`save_machines` passes the registry it wrote, so only a
    machine whose entry changed or vanished loses its links (an open
    stream to any other machine survives the change); the ``atexit``
    call keeps nothing.  A machine registered again gets a fresh link
    (its password may have changed) and, at another login or daemon
    port, a fresh tunnel (:func:`close_tunnels`); a removed one leaves
    no ssh process behind.  The history pages remembered for the
    registrations not kept go too: they were kept for the daemon's
    lifetime, one set per password or address a machine ever had.
    """
    kept = {_registration(m) for m in keep}
    links: list[MachineLink] = locked_call(_links_lock, _evict_links, kept)
    for link in links:
        link.retire()
    close_tunnels({_endpoint(m) for m in keep if m.ssh})
    locked_call(_last_pages_lock, _forget_pages, kept)


atexit.register(close_links)


def forget_connections() -> None:
    """Abandon every cached link's socket and ssh tunnel and clear the back-offs.

    For a host back from sleep (:func:`forget_after_sleep`): every
    socket to a remote daemon is dead (the daemon closed it when its
    pings went unanswered) and so may be the ssh forward under it, but
    nothing here can tell until a request has hung on it for its whole
    budget and backed the machine off.  The links and tunnels stay
    cached and dial afresh on their next use
    (:meth:`MachineLink.abandon`, :meth:`SshTunnel.close_unless_opening`);
    a stream through a closed forward ends and its tab reports the
    machine lost, as it would have once the stream's keepalive noticed.
    """
    links: list[MachineLink] = locked_call(_links_lock, list, _links.values())
    for link in links:
        link.abandon()
    tunnels: list[SshTunnel] = locked_call(_tunnels_lock, list, _tunnels.values())
    for tunnel in tunnels:
        tunnel.close_unless_opening()


_clock_lock = threading.RLock()  # see SshTunnel: taken through locked_call
_clocks = (time.time(), time.monotonic())
"""The wall and monotonic clocks at the last :func:`forget_after_sleep` check."""


def _slept(wall: float, mono: float) -> float:
    """Return the wall clock's lead over the monotonic one since the last check (under the lock)."""
    global _clocks
    last_wall, last_mono = _clocks
    _clocks = (wall, mono)
    return (wall - last_wall) - (mono - last_mono)


def forget_after_sleep() -> float:
    """Drop every cached connection when the host slept since the last call.

    The monotonic clock stops while the host sleeps and the wall clock
    does not, so their drift since the last call is the time slept
    (:data:`SLEEP_GAP` names how much counts).  Called before a link
    uses a cached socket and from the daemon's watchdog, so a request
    after a wake dials afresh instead of hanging on a dead socket for
    its whole budget, and nothing a poll during the sleep failed on is
    held against the machine.

    Returns:
        The seconds slept, 0.0 when the host did not sleep (or for
        less than :data:`SLEEP_GAP`).
    """
    slept = locked_call(_clock_lock, _slept, time.time(), time.monotonic())
    if slept < SLEEP_GAP:
        return 0.0
    logger.info("the host slept for %.0fs: dropping the cached links and ssh tunnels", slept)
    forget_connections()
    return slept


# The short host name each registered machine's daemon last gave of
# itself (registered name -> host), from the ``machine`` of its
# ``pong`` and the ``localMachine`` of its ``history`` replies; kept
# while the machine is down, and across restarts in HOSTS_FILE, so the
# CHATS rows synced in from a machine that is down still name it.
HOSTS_FILE = "machine_hosts.json"
HOSTS_LOCK_TIMEOUT = 10.0
"""Seconds :func:`note_host` waits for another writer of :data:`HOSTS_FILE`.

A holder reads and rewrites the small file and nothing else, so a
wait this long means a wedged holder: the name is kept in memory only
and saved by a later observation.
"""


def _saved_hosts() -> dict[str, str]:
    """Return the host names :func:`note_host` saved ({} when none or unreadable)."""
    try:
        raw = json.loads((kiss_home() / HOSTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, str)}


_hosts: dict[str, str] = _saved_hosts()
# Machines whose host name is known in memory only (its save failed): the
# next observation of the same name saves it again.
_unsaved_hosts: set[str] = set()


def note_host(machine: Machine, host: Any) -> None:
    """Record the host name *host* that *machine*'s daemon gave of itself.

    Ignored unless a non-empty string; a new name is saved to
    :data:`HOSTS_FILE` (best-effort).  The file is rewritten under a
    file lock from the names it holds plus this one, never from this
    process's whole mapping: the fan-out workers, the sidebar's machine
    rows and a ``/sync_history_and_memory`` process note names at the
    same time, and a writer putting its older mapping over a newer
    file lost the other's name for good (a later identical observation
    found it in memory and wrote nothing).  The names read back are
    taken into memory with it.
    """
    if not (isinstance(host, str) and host):
        return
    if _hosts.get(machine.name) == host and machine.name not in _unsaved_hosts:
        return
    path = kiss_home() / HOSTS_FILE
    try:
        with exclusive_file_lock(path.with_suffix(".lock"), timeout=HOSTS_LOCK_TIMEOUT):
            hosts = {**_saved_hosts(), machine.name: host}
            atomic_write_text(path, json.dumps(hosts, indent=2))
            _hosts.update(hosts)
        _unsaved_hosts.discard(machine.name)
    except OSError:
        # Known in memory, but the next identical observation retries the save.
        _hosts[machine.name] = host
        _unsaved_hosts.add(machine.name)


def machine_on_host(host: str, machines: list[Machine]) -> Machine | None:
    """Return the machine of *machines* whose daemon runs on *host*, a short host name.

    The host name its daemon gave of itself (:func:`note_host`) wins;
    otherwise the machine's name, or a host of its address
    (:func:`machine_hosts`) that is *host* or starts with ``host.``.
    Case-insensitive; ``None`` when no machine matches.
    """
    wanted = host.strip().lower()
    if not wanted:
        return None
    for machine in machines:
        if _hosts.get(machine.name, "").lower() == wanted:
            return machine
    for machine in machines:
        names = {machine.name.lower(), *(h.lower() for h in machine_hosts(machine))}
        if any(n == wanted or n.startswith(wanted + ".") for n in names):
            return machine
    return None


def label_synced_rows(sessions: list[dict[str, Any]], machines: list[Machine]) -> None:
    """Attribute the history rows a history sync copied in to the machines they ran on.

    A row of this daemon's history carries ``origin``: the host name of
    the machine its task ran on ('' for this one), which
    ``kiss.scripts.sync_db`` stamps on the rows it copies between
    databases.  The field is removed from every row; a row whose
    origin is a registered machine gets that machine's ``machine`` and
    ``machine_url``, as :func:`fetch_history` stamps the rows it brings,
    so the CHATS panel shows it as that machine's and resumes it there.
    A row whose machine is not registered stays this daemon's.  The
    host names saved by another process (a ``/sync_history_and_memory``
    run, which asks each machine its host name over ssh) are taken in
    first, so a machine whose daemon has never answered maps too.
    """
    for name, host in _saved_hosts().items():
        _hosts.setdefault(name, host)
    for session in sessions:
        machine = machine_on_host(str(session.pop("origin", "") or ""), machines)
        if machine is not None:
            session["machine"] = machine.name
            session["machine_url"] = machine.url


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


# The history page each registration of a machine last answered a view
# with ((registration, query, tag) -> the page), shown while the
# machine does not answer; of a registration's filtered views only the
# latest answered is kept (see _fetch_quietly), and a registration's
# pages go when it leaves the registry (close_links).  Read and written
# under _last_pages_lock: the fan-out workers store their pages at
# once, and one pruning the dict while another inserts into it would
# raise RuntimeError (dictionary changed size during iteration).
# Reentrant for locked_call: close_links runs on task threads too.
_last_pages: dict[tuple[str, str, str, str, str, str], HistoryPage] = {}
_last_pages_lock = threading.RLock()


def _forget_pages(kept: set[_Registration]) -> None:
    """Forget the pages remembered for every registration not in *kept* (under the lock)."""
    for key in [key for key in _last_pages if key[:4] not in kept]:
        del _last_pages[key]


HISTORY_FANOUT_WAIT = 2.0
"""Seconds a history reply waits for the registered machines' pages.

A machine that has not answered by then (its link is being dialled
again after a failure, an SSH port-forward is opening, the network to
it is slow) contributes the page it last answered with, as a machine
that does not answer does (:func:`_fetch_quietly`), and the reply
goes out: the CHATS rows of this daemon and of the machines that did
answer are not held back for it.  Its fetch runs on, and when the
page it brings differs from the one served in its place, the reply's
``late`` callback asks the pages to refresh (:func:`merged_history`).
"""

# The fetches under way, by (registration, query, tag, limit): a reply
# that comes while a machine's page is still being fetched for the same
# view waits for that fetch instead of starting another.
_FetchKey = tuple[str, str, str, str, str, str, int]
_fetches: dict[_FetchKey, Future[HistoryPage]] = {}
_fetches_lock = threading.Lock()
_fetch_pool = ThreadPoolExecutor(thread_name_prefix="chats-fan-out")


def _last_page(machine: Machine, query: str, tag: str, limit: int) -> HistoryPage:
    """Return the page *machine* last answered *query* and *tag* with, trimmed to *limit*.

    An empty page for a view the machine, as registered, has not
    answered (or not since another filtered view: see
    :func:`_fetch_quietly`), and so for a name registered again for
    another daemon (another address or login: another daemon's chats).
    """
    with _last_pages_lock:
        last = _last_pages.get((*_registration(machine), query, tag))
    if last is None:
        return HistoryPage()
    return HistoryPage(last.rows[:limit], last.date_min, last.date_max)


def _fetch_quietly(machine: Machine, query: str, tag: str, limit: int) -> HistoryPage:
    """Return *machine*'s history page, or its last one while it does not answer.

    A machine that cannot be queried (the host is down or asleep, the
    SSH tunnel timed out, the link is backing off after a failure)
    must not make its chats vanish from the CHATS panel for that
    reply: the page is remembered per registration and view and served
    again, trimmed to *limit*, while the machine does not answer (a
    warning is logged; :func:`_last_page`).  A filtered view (a *query*
    or a *tag*) is remembered too, else one whose fetch outlasts
    :data:`HISTORY_FANOUT_WAIT` never shows: the refresh its late
    callback asks for (:func:`merged_history`) would serve the empty
    fallback and fetch afresh, without end.  Only the latest filtered
    view answered is kept per registration (a page per query typed
    would grow without bound), beside the unfiltered one.

    The page of a registration that left the registry while it was on
    its way is returned but not remembered: the sweep of that
    registration's pages (:func:`close_links`) has run, and nothing
    would drop the page, or its key with the registration's password,
    for the daemon's life.  The registry is read under the cache's
    lock, under which the sweep runs after the registry is written, so
    a change either shows in that read or sweeps the page.
    """
    try:
        page = fetch_history(machine, query, tag, limit)
    except MachineError as exc:
        logger.warning("CHATS: %s", exc)
        return _last_page(machine, query, tag, limit)
    key = (*_registration(machine), query, tag)
    with _last_pages_lock:
        if machine not in load_machines():
            return page
        if query or tag:
            kept = (("", ""), key[4:])
            for other in [k for k in _last_pages if k[:4] == key[:4] and k[4:] not in kept]:
                _last_pages.pop(other, None)
        _last_pages[key] = page
    return page


def _shared_fetch(machine: Machine, query: str, tag: str, limit: int) -> Future[HistoryPage]:
    """Return the fetch of *machine*'s page for this view under way, starting one if none is.

    The fetch is shared by registration (:func:`_registration`) as
    well as view: a name registered again for another daemon, or with
    other credentials, does not join the fetch of the previous one.
    """
    key = (*_registration(machine), query, tag, limit)
    with _fetches_lock:
        future = _fetches.get(key)
        started = future is None
        if future is None:
            future = _fetch_pool.submit(_fetch_quietly, machine, query, tag, limit)
            _fetches[key] = future
    if started:
        # Outside the lock: a fetch already done (a failure cached in
        # back-off) runs the callback here and now, which takes it.
        future.add_done_callback(partial(_forget_fetch, key))
    return future


def _forget_fetch(key: _FetchKey, future: Future[HistoryPage]) -> None:
    """Drop the finished *future* from the fetches under way."""
    with _fetches_lock:
        if _fetches.get(key) is future:
            del _fetches[key]


def _landed_late(
    served: HistoryPage, late: Callable[[], None], future: Future[HistoryPage],
) -> None:
    """A page fetched after its reply went out: call *late* when it adds to what was served."""
    if future.result().rows != served.rows:
        late()


def merged_history(
    local: HistoryPage, machines: list[Machine], query: str, tag: str, offset: int, limit: int,
    late: Callable[[], None] | None = None,
) -> HistoryPage:
    """Return page *offset*..*offset+limit* of the union of all machines' history.

    *local* must hold the local rows ``0 .. offset+limit`` (not just the
    requested page) so the union can be re-paginated; its date range is
    widened by the remote histories'.  Rows are ordered running first,
    then newest first, like the local list; rows that tie keep their
    source's order, local before remote, earlier machine first.  Of
    several copies of one task id the one that sorts first is kept; it
    is of a running chat (``chat_running``) when any copy says so, as
    the copy a history sync left here is stamped by this daemon, where
    the chat does not run.
    Because every source is itself ordered that way and the ranking of
    a copy never depends on what else is loaded, the page is the same
    whichever pages were loaded before it: a task never shows twice or
    goes missing across pages.  A machine that does not answer
    contributes the page it last answered the view with, if any
    (:func:`_fetch_quietly`; a warning is logged), and so does one that
    has not answered within :data:`HISTORY_FANOUT_WAIT` seconds: its
    fetch runs on and *late*, when given, is called once the page it
    brings differs from the one served in its place, so the caller can
    have the CHATS panels refresh.
    """
    need = offset + limit
    futures = [_shared_fetch(m, query, tag, need) for m in machines]
    wait(futures, timeout=HISTORY_FANOUT_WAIT)
    pages = [local]
    for machine, future in zip(machines, futures, strict=True):
        if future.done():
            pages.append(future.result())
            continue
        served = _last_page(machine, query, tag, need)
        pages.append(served)
        if late is not None:
            future.add_done_callback(partial(_landed_late, served, late))
    best: dict[str, tuple[tuple[bool, float, int, int], dict[str, Any]]] = {}
    of_running_chat: set[str] = set()
    for source, page in enumerate(pages):
        for rank, session in enumerate(page.rows):
            key = str(session.get("task_id") or session.get("id") or id(session))
            if session.get("chat_running"):
                of_running_chat.add(key)
            order = (*_history_order(session), -source, -rank)
            if key not in best or order > best[key][0]:
                best[key] = (order, session)
    for key in of_running_chat:
        order, session = best[key]
        if not session.get("chat_running"):
            # A row of its own: the pages are kept and served again.
            best[key] = (order, {**session, "chat_running": True})
    ranked = sorted(best.values(), key=operator.itemgetter(0), reverse=True)
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
