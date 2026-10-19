"""``MemoryDir.write`` and ``delete`` never leave a deleted page without its tombstone.

Two ways to get there were found and are pinned here:

* a ``write`` racing a ``delete`` of the same page (W publishes the page,
  D unlinks it and writes the tombstone, W unlinks the tombstone) -- closed
  by serialising both under the memory's ``<root>.lock`` file lock;
* a ``write`` that unlinked the tombstone before publishing the page and
  then failed to publish -- closed by publishing first.  ``delete``
  likewise writes the tombstone before it unlinks the page.

"Neither page nor tombstone" is the one state the sync with another
machine cannot tell from "never existed": it brings the old page back.
Every other failure state is "page beside a stale tombstone", which the
real merge resolves in favour of the page.
"""

import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock
from kiss.core.memoryfield.pages import MemoryDir
from kiss.scripts.merge_memory_pages import merge
from kiss.tests.conftest import is_root, posix_only

_permission_bits = pytest.mark.skipif(is_root(), reason="root ignores directory permission bits")


def _deleted_page_with_remote_copy(tmp_path: Path) -> tuple[MemoryDir, Path]:
    """A local memory whose page ``p`` was deleted after a remote copy took it."""
    local = MemoryDir(tmp_path / "local")
    local.write("p", "original body")
    shutil.copytree(local.root, tmp_path / "remote")
    time.sleep(1.1)  # the deletion must be strictly newer than the remote page
    local.delete("p")
    assert local.tombstone_path("p").exists() and not local.page_path("p").exists()
    return local, tmp_path / "remote"


@posix_only("directory permission bits")
@_permission_bits
def test_failed_publish_keeps_the_tombstone_so_the_sync_does_not_resurrect(tmp_path: Path) -> None:
    local, remote = _deleted_page_with_remote_copy(tmp_path)
    # Read-only memory root: the page cannot be staged.  Its tombstone
    # directory stays writable, so a write that cleared the tombstone first
    # would succeed at that and fail right after.
    os.chmod(local.root, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PermissionError):
            local.write("p", "replacement body")
    finally:
        os.chmod(local.root, stat.S_IRWXU)
    assert not local.page_path("p").exists()
    assert local.tombstone_path("p").exists()
    counts, _ = merge(remote, local.root, tolerance=0)
    assert counts["added"] == 0
    assert not local.page_path("p").exists(), "the sync brought the deleted page back"


@posix_only("directory permission bits")
@_permission_bits
def test_unremovable_tombstone_leaves_the_page_beside_it_and_the_page_wins(tmp_path: Path) -> None:
    local, remote = _deleted_page_with_remote_copy(tmp_path)
    tombstone = local.tombstone_path("p")
    os.chmod(tombstone.parent, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PermissionError):
            local.write("p", "second body")
        # Published before the tombstone could be touched: page beside a
        # stale tombstone, never a deleted page without one.
        assert local.read("p").body == "second body\n"
        assert tombstone.exists()
    finally:
        os.chmod(tombstone.parent, stat.S_IRWXU)
    counts, conflicts = merge(remote, local.root, tolerance=0)
    assert (counts["added"], counts["updated"], conflicts) == (0, 0, [])
    assert local.read("p").body == "second body\n"
    assert not tombstone.exists(), "the merge drops the stale tombstone"
    local.write("p", "third body")
    assert not tombstone.exists()


@posix_only("directory permission bits")
@_permission_bits
def test_failed_delete_keeps_the_page_and_the_page_wins(tmp_path: Path) -> None:
    local = MemoryDir(tmp_path / "local")
    local.write("p", "body")
    local.tombstone_path("p").parent.mkdir()
    # Read-only memory root with a writable tombstone directory: the
    # tombstone can be written, the page cannot be unlinked.
    os.chmod(local.root, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PermissionError):
            local.delete("p")
    finally:
        os.chmod(local.root, stat.S_IRWXU)
    assert local.read("p").body == "body\n"
    (tmp_path / "empty").mkdir()
    counts, _ = merge(tmp_path / "empty", local.root, tolerance=0)
    assert counts["removed"] == 0 and local.read("p").body == "body\n"
    assert not local.tombstone_path("p").exists()


def test_delete_of_a_missing_page_leaves_no_tombstone(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    memory.write("other", "body")
    with pytest.raises(FileNotFoundError):
        memory.delete("p")
    assert not memory.tombstone_path("p").exists()


def test_write_and_delete_of_one_page_are_serialised_under_the_memory_lock(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    memory.write("p", "seed")
    page_path, tombstone = memory.page_path("p"), memory.tombstone_path("p")
    barrier = threading.Barrier(3)
    threads = [
        threading.Thread(target=_write_after, args=(barrier, memory)),
        threading.Thread(target=_delete_after, args=(barrier, memory)),
    ]
    with exclusive_file_lock(Path(f"{memory.root}.lock")):
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(0.5)
        assert all(thread.is_alive() for thread in threads), "a mutation ran outside the lock"
        assert page_path.read_text().endswith("seed\n") and not tombstone.exists()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert page_path.exists() != tombstone.exists()


def test_concurrent_write_and_delete_leave_page_xor_tombstone(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    page_path, tombstone = memory.page_path("p"), memory.tombstone_path("p")
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(200):
            memory.write("p", "seed")
            barrier = threading.Barrier(2)
            threads = [
                threading.Thread(target=_write_after, args=(barrier, memory)),
                threading.Thread(target=_delete_after, args=(barrier, memory)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert page_path.exists() != tombstone.exists()
            page_path.unlink(missing_ok=True)
            tombstone.unlink(missing_ok=True)
    finally:
        sys.setswitchinterval(previous)


def _write_after(barrier: threading.Barrier, memory: MemoryDir) -> None:
    barrier.wait()
    memory.write("p", "new body")


def _delete_after(barrier: threading.Barrier, memory: MemoryDir) -> None:
    barrier.wait()
    memory.delete("p")  # the seed page is always there: the writer never removes a page
