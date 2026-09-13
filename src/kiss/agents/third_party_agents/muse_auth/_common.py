# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared paths, constants, and socket framing for Muse-style auth.

Both the daemon (vault + sentinel + network boundary) and the
agent-side client import from here; this module must not import any
other ``muse_auth`` module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home

_SERVICE_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def valid_service_name(service: Any) -> bool:
    """Return whether *service* is a safe connector service name.

    Service names become vault file names, so anything that could
    traverse directories (``..``, ``/``, absolute paths) or smuggle a
    trailing newline is rejected.  ``fullmatch`` (not ``match``) is
    used so ``"gmail\\n"`` cannot slip through the ``$`` end-anchor.

    Args:
        service: Candidate service name (any type; non-strings fail).

    Returns:
        True for lowercase ``[a-z0-9_-]`` strings of at most 64 chars.
    """
    return isinstance(service, str) and _SERVICE_NAME_RE.fullmatch(service) is not None

# Hosts each connector service is allowed to reach.  Mirrors Muse's
# per-worker credential/host ACLs: a Drive surrogate cannot be spent
# against the Gmail API.  Hostnames only (ports are ignored).
SERVICE_HOSTS: dict[str, tuple[str, ...]] = {
    "gmail": ("gmail.googleapis.com", "www.googleapis.com"),
    "google_drive": ("www.googleapis.com",),
    "google_calendar": ("www.googleapis.com",),
    "google_docs": ("docs.googleapis.com", "www.googleapis.com"),
    "google_sheets": ("sheets.googleapis.com", "www.googleapis.com"),
    "googlechat": ("chat.googleapis.com",),
    "notion": ("api.notion.com",),
    "github": ("api.github.com",),
}

SURROGATE_PREFIX = "muse-sgt."

# One JSON object per line; requests carrying request/response bodies
# are base64-encoded, so cap the frame to keep the daemon safe from
# unbounded allocations (64 MiB of base64 ~ 48 MiB of payload).
MAX_FRAME_BYTES = 64 * 1024 * 1024

READ_METHODS = ("GET", "HEAD", "OPTIONS")


def muse_auth_enabled() -> bool:
    """Return True when Muse-style auth is switched on via ``KISS_MUSE_AUTH``.

    Returns:
        True when the env var is a truthy string ("1", "true", "yes", "on").
    """
    return os.environ.get("KISS_MUSE_AUTH", "").strip().lower() in ("1", "true", "yes", "on")


def muse_auth_dir() -> Path:
    """Return the Muse-auth state directory, honoring ``KISS_HOME``.

    Returns:
        Path to ``$KISS_HOME/muse_auth``.
    """
    return kiss_home() / "muse_auth"


def socket_path() -> Path:
    """Return the daemon's Unix socket path.

    Unix socket paths are limited to ~104 bytes; when the natural
    location under ``KISS_HOME`` is too long (deep pytest temp dirs),
    fall back to a per-user, per-home path in ``/tmp`` derived from a
    hash of ``KISS_HOME`` so client and daemon always agree.

    Returns:
        Path the daemon binds and clients connect to.
    """
    natural = muse_auth_dir() / "authd.sock"
    if len(str(natural)) <= 90:
        return natural
    digest = hashlib.sha256(str(kiss_home()).encode()).hexdigest()[:12]
    return Path(f"/tmp/kiss-muse-{os.getuid()}-{digest}.sock")


def is_loopback_host(host: str) -> bool:
    """Return whether *host* is a loopback destination.

    Args:
        host: Hostname or IP literal (lowercase).

    Returns:
        True for ``localhost`` and loopback IP addresses.
    """
    if host == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def action_class(method: str) -> str:
    """Classify an HTTP method into Muse's read/write action classes.

    Args:
        method: HTTP method name (any case).

    Returns:
        ``"read"`` for GET/HEAD/OPTIONS, ``"write"`` otherwise.
    """
    return "read" if method.upper() in READ_METHODS else "write"


def send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    """Send one newline-terminated JSON frame over *sock*.

    Args:
        sock: Connected Unix socket.
        obj: JSON-serializable payload.
    """
    sock.sendall(json.dumps(obj, separators=(",", ":")).encode() + b"\n")


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Receive one newline-terminated JSON frame from *sock*.

    Args:
        sock: Connected Unix socket.

    Returns:
        The decoded JSON object.

    Raises:
        ConnectionError: When the peer closes before a full frame arrives.
        ValueError: When the frame exceeds :data:`MAX_FRAME_BYTES`.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(1 << 16)
        if not chunk:
            raise ConnectionError("muse-auth peer closed the connection mid-frame")
        total += len(chunk)
        if total > MAX_FRAME_BYTES:
            raise ValueError("muse-auth frame exceeds the 64 MiB limit")
        chunks.append(chunk)
        if chunk.endswith(b"\n"):
            return dict(json.loads(b"".join(chunks).decode()))
