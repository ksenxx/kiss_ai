# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Memory pages must be published atomically.

``MemoryDir.write`` used a plain ``Path.write_text``, which truncates the
page and then fills it incrementally.  The memory directory is explicitly
shared: parallel sub-agents (threads) and daemon-dispatched channel agents
(other processes) all read the same ``$KISS_HOME/memories`` pages via
``memory_read`` / ``memory_pull``, and ``VectorIndex.sync`` hashes the raw
bytes — so a reader in the truncate window observed an empty or half-written
page.  The fix routes the write through ``kiss.core.utils.atomic_write_text``
(stage in a sibling temp file, ``os.replace`` into position), the same
pattern every other persistent writer in kiss.core uses.

End-to-end: real files, real threads, no mocks.  The race was confirmed
against the pre-fix code by this exact reader/writer pair (torn reads
appeared within the first few hundred rewrites).
"""

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.utils import atomic_write_text, read_bytes_waiting_for_writer
from kiss.tests.conftest import IS_WINDOWS, posix_only

# Windows has no umask-derived mode bits and its chmod only toggles the
# read-only flag (S_IMODE is always 0o666 or 0o444), so the permission
# contract below is a POSIX one.  The atomic publication itself is
# exercised on every platform by the other tests in this module.
_mode_bits_only = posix_only("umask-derived and chmod-preserved mode bits")

_REWRITES = 400


def _payload(tag: str) -> str:
    """A multi-hundred-KB body whose completeness is checkable from its tail."""
    return (f"line-{tag} " * 100 + "\n") * 300 + f"END-{tag}\n"


def test_concurrent_reader_never_sees_torn_page(tmp_path: Path) -> None:
    """A reader must only ever observe a complete page, never "" or a prefix.

    One thread rewrites the same page, alternating between two large bodies
    that each end in a distinctive sentinel line; two reader threads read the
    page through the public ``MemoryDir.read`` API the whole time.  Every
    observed page text must end with one of the sentinels — an empty or
    truncated read is exactly the torn write this test guards against.
    """
    memory = MemoryDir(tmp_path / "memory")
    bodies = {"A": _payload("A"), "B": _payload("B")}
    memory.write("shared-page", bodies["A"], title="shared", summary="race probe")

    stop = threading.Event()
    anomalies: list[str] = []

    def read_loop() -> None:
        while not stop.is_set():
            raw = memory.read("shared-page").raw
            if not (raw.endswith("END-A\n") or raw.endswith("END-B\n")):
                anomalies.append(f"torn read: {len(raw)} bytes, tail={raw[-40:]!r}")
                return

    readers = [threading.Thread(target=read_loop) for _ in range(2)]
    for reader in readers:
        reader.start()
    try:
        for i in range(_REWRITES):
            if anomalies:
                break
            memory.write("shared-page", bodies["A" if i % 2 else "B"])
    finally:
        stop.set()
        for reader in readers:
            reader.join()

    assert not anomalies, anomalies[0]


def test_page_reader_gives_up_on_a_real_permission_error(tmp_path: Path) -> None:
    """The reader's wait for an in-flight replace is bounded.

    ``read_bytes_waiting_for_writer`` retries ``PermissionError`` only on
    Windows and only for a second, so a path that is genuinely denied
    still raises instead of hanging.  On POSIX a ``0o000`` file is denied
    on the first attempt; on Windows opening a directory is denied on
    every attempt, so it is the one-second deadline that gives up.
    """
    if IS_WINDOWS:
        unreadable = tmp_path
    else:
        unreadable = tmp_path / "locked.md"
        unreadable.write_text("x", encoding="utf-8")
        unreadable.chmod(0o000)
    started = time.monotonic()
    with pytest.raises(PermissionError):
        read_bytes_waiting_for_writer(unreadable)
    if IS_WINDOWS:
        assert time.monotonic() - started >= 1.0, "gave up before the deadline"


def test_atomic_write_keeps_create_and_update_semantics(tmp_path: Path) -> None:
    """The atomic write must not change ``write``'s observable behaviour.

    Covers both branches of the modified code: the create path (the root
    directory does not exist yet — the atomic writer must create it) and the
    update path (``uuid`` / ``created`` preserved, ``updated`` refreshed,
    no stray staging file left behind).
    """
    memory = MemoryDir(tmp_path / "not-yet-created" / "memory")
    created = memory.write("topic", "first body\n", title="Topic", summary="s1")
    on_disk = memory.read("topic")
    assert on_disk.raw == created.raw
    assert on_disk.body == "first body\n"
    assert on_disk.frontmatter["uuid"] == created.frontmatter["uuid"]

    updated = memory.write("topic", "second body\n")
    reread = memory.read("topic")
    assert reread.body == "second body\n"
    assert reread.frontmatter["uuid"] == created.frontmatter["uuid"]
    assert reread.frontmatter["created"] == created.frontmatter["created"]
    assert reread.frontmatter["updated"] == updated.frontmatter["updated"]
    # The staged temp file must be gone and must never surface as a page.
    assert memory.page_names() == ["topic"]
    leftovers = [p.name for p in memory.root.iterdir() if p.name != "topic.md"]
    assert leftovers == []


def _mode(path: Path) -> int:
    """Return the permission bits of *path*."""
    return stat.S_IMODE(path.stat().st_mode)


@_mode_bits_only
def test_atomic_write_keeps_page_permissions(tmp_path: Path) -> None:
    """Atomic publication must not silently change page permissions.

    ``os.replace`` publishes the STAGED inode, and ``tempfile.mkstemp``
    stages at ``0600`` — so every page used to come out ``0600``:
    fresh pages lost the umask-derived group/other bits (breaking
    group-readable memory directories) and a deliberately ``chmod``-ed
    page lost its mode on the next update.  Like ``Path.write_text``,
    a NEW page must get the umask-derived create mode and an UPDATE
    must preserve the existing file's mode.
    """
    old_umask = os.umask(0o022)
    try:
        memory = MemoryDir(tmp_path / "memory")
        memory.write("perms", "body\n", title="p", summary="s")
        page = memory.root / "perms.md"
        assert _mode(page) == 0o666 & ~0o022, oct(_mode(page))

        # A deliberate per-file mode survives an update.
        page.chmod(0o640)
        memory.write("perms", "body 2\n")
        assert _mode(page) == 0o640, oct(_mode(page))
        assert memory.read("perms").body == "body 2\n"

        # And a subsequent update keeps preserving it.
        memory.write("perms", "body 3\n")
        assert _mode(page) == 0o640, oct(_mode(page))
    finally:
        os.umask(old_umask)


@_mode_bits_only
def test_atomic_write_text_helper_permission_contract(tmp_path: Path) -> None:
    """The shared helper's create/update/explicit-mode permission rules.

    Priority order (round-2 correction): an explicit ``mode`` always
    wins, on create and update alike (the secret writers pass
    ``0o600``); an existing target keeps its current bits; a NEW target
    is created with ``create_mode`` filtered by the umask — and
    ``create_mode`` DEFAULTS to the private ``0o600`` because several
    mode-less callers store secrets (``config.json`` passwords/tokens,
    trajectories, ``MY_MODELS.json`` API keys).  Callers whose files
    are meant to be plain documents (memory pages) opt in to umask
    semantics with ``create_mode=0o666``.
    """
    old_umask = os.umask(0o027)
    try:
        # Secure default: a mode-less NEW file is private, exactly like
        # the pre-consolidation mkstemp-staged helper published.
        fresh = tmp_path / "fresh.txt"
        atomic_write_text(fresh, "x")
        assert _mode(fresh) == 0o600, oct(_mode(fresh))

        # Existing bits are preserved on update (round-1 fix retained).
        fresh.chmod(0o604)
        atomic_write_text(fresh, "y")
        assert _mode(fresh) == 0o604, oct(_mode(fresh))
        assert fresh.read_text() == "y"

        # Plain-document opt-in: create_mode is filtered by the umask
        # on creation and ignored for an existing target.
        plain = tmp_path / "plain.md"
        atomic_write_text(plain, "p", create_mode=0o666)
        assert _mode(plain) == 0o666 & ~0o027, oct(_mode(plain))
        plain.chmod(0o600)
        atomic_write_text(plain, "p2", create_mode=0o666)
        assert _mode(plain) == 0o600, "create_mode must not clobber updates"
        assert plain.read_text() == "p2"

        secret = tmp_path / "secret.env"
        atomic_write_text(secret, "s", mode=0o600)
        assert _mode(secret) == 0o600
        secret.chmod(0o644)
        atomic_write_text(secret, "s2", mode=0o600)
        assert _mode(secret) == 0o600, "explicit mode must win on update"
        assert secret.read_text() == "s2"

        # No staging leftovers on any path.
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "fresh.txt", "plain.md", "secret.env",
        ]
    finally:
        os.umask(old_umask)


@_mode_bits_only
def test_fresh_config_with_secrets_is_private(tmp_path: Path) -> None:
    """A fresh password-bearing ``config.json`` must never be group/world-readable.

    Round-2 demonstrated regression: making mode-less new files
    umask-derived meant the real ``save_config`` — which persists
    ``remote_password`` and ``tunnel_token`` — published a fresh
    ``config.json`` as ``0644`` under the common ``0022`` umask, so any
    local account could read the remote-web password (the pre-round-1
    helper published ``0600``).  ``save_config`` now forces ``0o600``:
    fresh files come out private and a previously exposed config is
    repaired on the next save.
    """
    from kiss.core import vscode_config

    root = tmp_path / "kiss-home"
    root.mkdir()
    root.chmod(0o755)
    config_path = root / "config.json"
    module_ns = cast(dict[str, Any], vars(vscode_config))
    had_dir = "CONFIG_DIR" in module_ns
    had_path = "CONFIG_PATH" in module_ns
    old_dir = module_ns.get("CONFIG_DIR")
    old_path = module_ns.get("CONFIG_PATH")
    old_umask = os.umask(0o022)
    try:
        vscode_config.CONFIG_DIR = root
        vscode_config.CONFIG_PATH = config_path
        secrets = {
            "remote_password": "regression-password",
            "tunnel_token": "regression-tunnel-token",
        }
        vscode_config.save_config(dict(secrets))
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        assert stored["remote_password"] == secrets["remote_password"]
        assert stored["tunnel_token"] == secrets["tunnel_token"]
        assert _mode(config_path) == 0o600, oct(_mode(config_path))
        assert _mode(config_path) & 0o077 == 0, "group/other bits set"

        # A config a prior release exposed as 0644 is repaired by the
        # next save instead of having its insecure mode "preserved".
        config_path.chmod(0o644)
        vscode_config.save_config({"remote_password": "rotated"})
        assert _mode(config_path) == 0o600, oct(_mode(config_path))
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        assert stored["remote_password"] == "rotated"
        assert stored["tunnel_token"] == secrets["tunnel_token"]
    finally:
        os.umask(old_umask)
        if had_dir:
            module_ns["CONFIG_DIR"] = old_dir
        else:
            module_ns.pop("CONFIG_DIR", None)
        if had_path:
            module_ns["CONFIG_PATH"] = old_path
        else:
            module_ns.pop("CONFIG_PATH", None)
