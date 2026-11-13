# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test for the 2026-11-01 audit of :mod:`kiss.server.fs_actions` (V9 F1).

The daemon runs every Explorer action on a worker thread.  A rename or
paste checks that its destination is free and then creates it in a
separate step; two unconfirmed requests for the same name (two
surfaces' Explorers, say) could both pass the check, and the second
transfer then silently overwrote the first's entry, with a rename or
move also losing the first source.  Now the actions that change the
tree run one at a time, so the second request finds the name taken and
is refused with ``exists`` like a request that arrived later.

The files and threads are real; a profile hook on the first request's
thread only parks it between its check and its transfer, so the
interleaving is the same on every run.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.server import fs_actions
from kiss.server.fs_actions import fs_action


def _pause_before_transfer(paused: threading.Event, release: threading.Event) -> Callable[..., Any]:
    """A profile hook that parks its thread once, right before the request's transfer.

    The transfer is ``os.rename`` for a rename and ``_transfer`` for a
    paste; both come after the destination's absence check.
    """

    def hook(frame: Any, event: str, arg: Any) -> None:
        at_rename = (event == "c_call" and arg is os.rename) or (
            event == "call" and frame.f_code.co_name == "_rename_noreplace"
        )
        at_transfer = event == "call" and frame.f_code is fs_actions._transfer.__code__
        if (at_rename or at_transfer) and not paused.is_set():
            paused.set()
            release.wait(10)

    return hook


@pytest.mark.parametrize("action", ["rename", "move", "copy"])
def test_two_unconfirmed_requests_for_one_name_keep_the_first(tmp_path: Path, action: str) -> None:
    """The second request finds the name taken and is refused; the first's entry stays."""
    if action == "rename":
        first, second, target = tmp_path / "a", tmp_path / "b", tmp_path / "final"
        dest = str(target)
    else:
        for folder in ("one", "two", "dest"):
            (tmp_path / folder).mkdir()
        first, second = tmp_path / "one" / "item", tmp_path / "two" / "item"
        target, dest = tmp_path / "dest" / "item", str(tmp_path / "dest")
    first.write_text("FIRST")
    second.write_text("SECOND")
    paused, release = threading.Event(), threading.Event()
    results: dict[str, dict[str, Any]] = {}

    def request(name: str, src: Path) -> None:
        results[name] = fs_action(action, str(src), dest=dest, overwrite=False)

    def first_request() -> None:
        sys.setprofile(_pause_before_transfer(paused, release))
        try:
            request("first", first)
        finally:
            sys.setprofile(None)

    first_thread = threading.Thread(target=first_request)
    second_thread = threading.Thread(target=request, args=("second", second))
    first_thread.start()
    try:
        assert paused.wait(10), "the first request never reached its transfer"
        second_thread.start()
        # Before the fix the second request ran to completion here; now
        # it waits for the first one.
        second_thread.join(1.0)
    finally:
        release.set()
    first_thread.join(10)
    second_thread.join(10)
    assert not first_thread.is_alive() and not second_thread.is_alive()

    assert results["first"] == {"ok": True, "path": str(target)}
    assert results["second"] == {
        "error": f"A file or folder {target.name} already exists",
        "exists": True,
    }
    assert target.read_text() == "FIRST"
    assert second.read_text() == "SECOND"
    assert first.exists() == (action == "copy")


def _make_source(src: Path, kind: str) -> None:
    """Create each supported entry kind with the same source payload."""
    if kind == "directory":
        src.mkdir()
        (src / "payload").write_text("SOURCE")
    elif kind == "symlink":
        src.symlink_to("missing")
    else:
        src.write_text("SOURCE")


@pytest.mark.parametrize("action", ["rename", "move", "copy", "copy-name"])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_external_destination_survives_no_clobber(
    tmp_path: Path, action: str, kind: str,
) -> None:
    """An editor's entry created after the check is never replaced or entered."""
    dest = tmp_path / "dest"
    dest.mkdir()
    src = tmp_path / "item"
    _make_source(src, kind)
    target = dest / src.name
    if action == "rename":
        target = tmp_path / "renamed"
    elif action == "copy-name":
        target = tmp_path / "item copy"
    created = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal created
        transfer = event == "call" and frame.f_code.co_name in ("_transfer", "_rename_noreplace")
        rename = event == "c_call" and arg is os.rename
        if not created and (transfer or rename):
            created = True
            if kind == "directory":
                target.mkdir()
            else:
                target.write_text("EDITOR")

    sys.setprofile(editor)
    try:
        result = fs_action(
            "copy" if action == "copy-name" else action, str(src),
            dest=str(target if action == "rename" else tmp_path if action == "copy-name" else dest),
            overwrite=action == "copy-name",
        )
    finally:
        sys.setprofile(None)
    assert created, "the transfer hook never ran"
    assert result == {"error": f"A file or folder {target.name} already exists", "exists": True}
    if kind == "directory":
        assert list(target.iterdir()) == []
        assert (src / "payload").read_text() == "SOURCE"
    else:
        assert target.read_text() == "EDITOR"
        assert src.is_symlink() if kind == "symlink" else src.read_text() == "SOURCE"


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_overwrite_move_keeps_recreated_source(tmp_path: Path, kind: str) -> None:
    """A save at the old pathname after a successful rename is not move cleanup."""
    src = tmp_path / "item"
    dest = tmp_path / "dest"
    dest.mkdir()
    target = dest / src.name
    target.write_text("OLD TARGET")
    _make_source(src, kind)
    saved = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal saved
        if event == "return" and frame.f_code is fs_actions._rename_or_copy.__code__:
            assert not os.path.lexists(src)
            src.write_text("NEW EDITOR SAVE")
            saved = True

    sys.setprofile(editor)
    try:
        result = fs_action("move", str(src), dest=str(dest), overwrite=True)
    finally:
        sys.setprofile(None)
    assert saved, "the post-rename hook never ran"
    assert result == {"ok": True, "path": str(target)}
    assert src.read_text() == "NEW EDITOR SAVE"
    if kind == "directory":
        assert (target / "payload").read_text() == "SOURCE"
    elif kind == "symlink":
        assert target.is_symlink() and os.readlink(target) == "missing"
    else:
        assert target.read_text() == "SOURCE"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("race", ["none", "replace", "remove", "after-claim", "recreate"])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_cross_device_move_owns_source_cleanup(
    tmp_path: Path, overwrite: bool, race: str, kind: str,
) -> None:
    """The copy fallback cleans only the copied entry, not its replacement."""
    shared = Path("/dev/shm")
    if not shared.is_dir() or shared.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("requires a second writable filesystem at /dev/shm")
    src = tmp_path / "item"
    _make_source(src, kind)
    saved = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal saved
        copied = event == "return" and frame.f_code.co_name in ("_copy_file", "_copy_folder")
        linked = (
            event == "c_return" and arg is os.symlink and frame.f_code.co_name == "move"
        )
        if (
            event == "c_return" and arg is os.rename
            and frame.f_code.co_name == "_rename_or_copy" and race == "after-claim"
        ):
            src.write_text("NEW EDITOR SAVE")
        if not saved and (copied or linked):
            copied_path = Path(frame.f_locals["src"])
            if copied_path != src and not copied_path.name.startswith(".kiss-move-"):
                return
            saved = True
            if race == "recreate":
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink(missing_ok=True)
                src.write_text("NEW EDITOR SAVE")
            elif race in ("replace", "remove"):
                if os.path.lexists(src):
                    src.rename(tmp_path / "saved-original")
                if race == "replace":
                    src.write_text("NEW EDITOR SAVE")

    with tempfile.TemporaryDirectory(prefix="kiss-fs-actions-", dir=shared) as folder:
        target = Path(folder) / src.name
        if overwrite:
            target.write_text("OLD TARGET")
        sys.setprofile(editor)
        try:
            result = fs_action("move", str(src), dest=folder, overwrite=overwrite)
        finally:
            sys.setprofile(None)
        assert saved, "the copy-fallback hook never ran"
        if race in ("replace", "after-claim", "recreate"):
            assert src.read_text() == "NEW EDITOR SAVE"
        else:
            assert not os.path.lexists(src)
        assert result == {"ok": True, "path": str(target)}, result
        assert not list(tmp_path.glob(".kiss-move-*"))
        if kind == "directory":
            assert (target / "payload").read_text() == "SOURCE"
        elif kind == "symlink":
            assert target.is_symlink() and os.readlink(target) == "missing"
        else:
            assert target.read_text() == "SOURCE"


@pytest.mark.parametrize("source_stuck", [False, True])
def test_cross_device_overwrite_cleans_or_names_the_source_when_the_old_entry_stays(
    tmp_path: Path, source_stuck: bool,
) -> None:
    """An old entry that cannot be deleted does not strand the moved source unnamed.

    The replaced folder holds one that may not be written, so deleting
    it fails once the move is committed.  The source, kept as
    ``.kiss-move-*``, is deleted all the same, or named in the error
    next to the old entry when it holds such a folder too.
    """
    shared = Path("/dev/shm")
    if not shared.is_dir() or shared.stat().st_dev == tmp_path.stat().st_dev or not os.geteuid():
        pytest.skip("requires a second writable filesystem at /dev/shm and a user bound by modes")
    src = tmp_path / "item"
    _make_source(src, "directory")
    with tempfile.TemporaryDirectory(prefix="kiss-fs-actions-", dir=shared) as folder:
        target = Path(folder) / src.name
        for stuck in [target / "stuck"] + [src / "stuck"] * source_stuck:
            stuck.mkdir(parents=True)
            (stuck / "kept").write_text("KEPT")
            stuck.chmod(0o500)
        try:
            result = fs_action("move", str(src), dest=folder, overwrite=True)
            backup, = Path(folder).glob(".item.replaced-*")
            assert f"'{backup}/stuck'" in result["error"], result
            assert (target / "payload").read_text() == "SOURCE" and not os.path.lexists(src)
            retained = [str(path) for path in tmp_path.glob(".kiss-move-*/*")]
            assert retained == [path for path in retained if f"'{path}'" in result["error"]]
            assert len(retained) == source_stuck
        finally:
            for stuck in [*Path(folder).rglob("stuck"), *tmp_path.rglob("stuck")]:
                stuck.chmod(0o700)


@pytest.mark.parametrize("action", ["copy", "move", "rename"])
def test_overwrite_collision_keeps_both_entries(tmp_path: Path, action: str) -> None:
    """Rollback must not delete an editor's save to restore the previous destination.

    Nor may the publication itself: an overwrite rename that put the
    old entry aside and then renamed the source over a destination the
    editor just recreated would destroy that save and report success.
    """
    src = tmp_path / "item"
    src.write_text("SOURCE")
    if action == "rename":
        dest = target = tmp_path / "renamed"
    else:
        dest = tmp_path / "dest"
        dest.mkdir()
        target = dest / src.name
    target.write_text("OLD TARGET")
    edited = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal edited
        if edited:
            return
        if action == "rename":
            # Right after ``_replace`` renamed the old entry aside.
            at_transfer = (
                event == "c_return" and arg is os.rename
                and frame.f_code is fs_actions._replace.__code__
            )
        else:
            at_transfer = (
                event == "call" and frame.f_code.co_name in ("_transfer", "_rename_or_copy")
            )
        if at_transfer:
            edited = True
            target.write_text("EDITOR")

    sys.setprofile(editor)
    try:
        result = fs_action(action, str(src), dest=str(dest), overwrite=True)
    finally:
        sys.setprofile(None)
    assert edited and "error" in result, result
    assert target.read_text() == "EDITOR"
    assert src.read_text() == "SOURCE"
    backup, = target.parent.glob(f".{target.name}.replaced-*")
    assert backup.read_text() == "OLD TARGET"
    assert str(backup) in result["error"]


@pytest.mark.parametrize("recreate", [False, True])
@pytest.mark.parametrize("failure", ["collision", "fifo"])
def test_cross_device_failure_restores_or_retains_source(
    tmp_path: Path, recreate: bool, failure: str,
) -> None:
    """A failed copy restores its source, or retains it beside a concurrent save."""
    shared = Path("/dev/shm")
    if not shared.is_dir() or shared.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("requires a second writable filesystem at /dev/shm")
    src = tmp_path / "item"
    _make_source(src, "directory")
    if failure == "fifo":
        os.mkfifo(src / "pipe")
    edited = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal edited
        if event == "call" and frame.f_code is fs_actions._transfer.__code__:
            copied = frame.f_locals["src"]
            if not edited and copied.name.startswith(".kiss-move-"):
                edited = True
                if recreate:
                    src.write_text("NEW EDITOR SAVE")
                if failure == "collision":
                    target.mkdir()
                    (target / "sentinel").write_text("EDITOR TARGET")

    with tempfile.TemporaryDirectory(prefix="kiss-fs-actions-", dir=shared) as folder:
        target = Path(folder) / src.name
        sys.setprofile(editor)
        try:
            result = fs_action("move", str(src), dest=folder)
        finally:
            sys.setprofile(None)
        assert edited and "error" in result, result
        if recreate:
            assert src.read_text() == "NEW EDITOR SAVE"
            retained, = tmp_path.glob(".kiss-move-*")
            assert (retained / "payload").read_text() == "SOURCE"
            assert str(retained) in result["error"]
        else:
            assert (src / "payload").read_text() == "SOURCE"
            assert not list(tmp_path.glob(".kiss-move-*"))
        if failure == "collision":
            assert (target / "sentinel").read_text() == "EDITOR TARGET"
            assert sorted(p.name for p in target.iterdir()) == ["sentinel"]
        else:
            assert not target.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_cross_device_move_follows_no_link_put_where_it_keeps_the_source(
    tmp_path: Path, kind: str,
) -> None:
    """A move that has to copy copies and deletes the source it took, not what a link names.

    While it is copied the source is kept under a name of the move's own.
    That was ``entry`` in a folder ``.kiss-move-*``, used by name: an
    external writer that moved the folder away and put a link out of the
    workspace in its place had that folder's ``entry`` copied to the
    destination and then deleted.  The source now stays an entry of its
    folder, which is pinned: a link put in its place is moved as a link.
    """
    shared = Path("/dev/shm")
    if not shared.is_dir() or shared.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("requires a second writable filesystem at /dev/shm")
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    _make_source(ws / "item", kind)
    _make_source(outside / "entry", kind)
    secret = outside / "entry" / "payload" if kind == "directory" else outside / "entry"
    secret.write_text("SECRET")
    before = _shape(outside)
    swapped = False

    def writer(frame: Any, event: str, arg: Any) -> None:
        """Act once the move has put the source where it keeps it."""
        nonlocal swapped
        kept = event == "c_return" and arg is os.rename and not swapped
        if kept and frame.f_code is fs_actions._rename_or_copy.__code__:
            swapped = True
            claim, = ws.glob(".kiss-move-*")
            claim.rename(ws / "moved")
            claim.symlink_to(outside, target_is_directory=True)

    with tempfile.TemporaryDirectory(prefix="kiss-fs-actions-", dir=shared) as dest:
        sys.setprofile(writer)
        try:
            result = fs_action("move", str(ws / "item"), dest=dest)
        finally:
            sys.setprofile(None)
        assert swapped, "the external writer never acted"
        assert _shape(outside) == before, "the move changed something out of the workspace"
        assert _files(Path(dest)) == {}, "the move copied what the link names"
        assert result == {"ok": True, "path": str(Path(dest) / "item")}, result
        assert os.readlink(Path(dest) / "item") == str(outside)  # the writer's link, as a link
    assert [path.name for path in ws.iterdir()] == ["moved"]  # nothing else stays behind


@pytest.mark.parametrize("in_a_folder", [False, True])
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
def test_copy_does_not_wait_on_a_pipe_put_in_place_of_a_file(
    tmp_path: Path, in_a_folder: bool,
) -> None:
    """A file replaced by a pipe just before the copier opens it ends the copy with an error.

    Nothing ever writes to the pipe: an open that waits for a writer
    would hold the mutation lock, and every later action, for good.
    """
    src, dest = tmp_path / "item", tmp_path / "dest"
    _make_source(src, "directory" if in_a_folder else "file")
    dest.mkdir()
    victim = src / "payload" if in_a_folder else src
    result: dict[str, Any] = {}
    swapped = False

    def writer(frame: Any, event: str, arg: Any) -> None:
        nonlocal swapped
        opens = (arg is os.open and frame.f_code.co_name == "_copy_file") or (
            arg is open and frame.f_code.co_name == "copyfile"
        )
        if event == "c_call" and opens and not swapped:
            swapped = True
            victim.unlink()
            os.mkfifo(victim)

    def copy() -> None:
        sys.setprofile(writer)
        try:
            result.update(fs_action("copy", str(src), dest=str(dest)))
        finally:
            sys.setprofile(None)

    worker = threading.Thread(target=copy, daemon=True)
    worker.start()
    worker.join(30)
    stuck = worker.is_alive()
    if stuck:  # give the pipe its writer, so that a failing run leaves no thread behind
        os.close(os.open(victim, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(30)
    assert swapped and not stuck, "the copy waits for a writer of the pipe"
    assert f"`{victim}` is a named pipe" in result["error"], result  # named as the request did
    assert os.listdir(dest) == []
    after = fs_action("newFolder", str(dest), name="next")
    assert after == {"ok": True, "path": str(dest / "next")}


#: A child interpreter whose kernel answers EINVAL to a rename with flags, as it does on
#: NFS, 9p, CephFS and eCryptfs, and which then runs the tests named on its command line.
#: The answer is a seccomp filter on ``renameat2``: A = the syscall number, then its flags.
_WITHOUT_RENAME_FLAGS = """
import ctypes, platform, struct, sys
import pytest
number = {"x86_64": 316, "aarch64": 276}.get(platform.machine(), 0)
steps = [(0x20, 0, 0, 0), (0x15, 0, 3, number), (0x20, 0, 0, 48), (0x15, 1, 0, 0),
         (0x06, 0, 0, 0x00050000 | 22), (0x06, 0, 0, 0x7FFF0000)]
code = ctypes.create_string_buffer(b"".join(struct.pack("HBBI", *step) for step in steps))
class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
program = Program(len(steps), ctypes.addressof(code))
libc = ctypes.CDLL(None)
if not number or libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program)):
    sys.exit(77)
sys.exit(pytest.main([*sys.argv[1:], "-q", "-x", "-o", "addopts=", "-p", "no:cacheprovider"]))
"""


def test_actions_work_where_rename_takes_no_flags() -> None:
    """Every scenario of this module holds where the rename that refuses a target is missing.

    A file or a link is then renamed through a hard link, a folder after
    a look at the target.  A C library without ``renameat2`` (glibc
    before 2.28) takes the same path; that one line cannot be reached
    here without replacing the library.
    """
    if sys.platform != "linux":
        pytest.skip("the kernel's answer is set with a seccomp filter")
    case_rename = Path(__file__).with_name("test_fs_rename_case_sweep1030.py")
    run = subprocess.run(
        [
            sys.executable, "-c", _WITHOUT_RENAME_FLAGS, __file__,
            f"{case_rename}::test_case_only_rename_keeps_what_an_editor_saves_meanwhile",
            "-k", "not takes_no_flags",
        ],
        capture_output=True, text=True, timeout=540,
    )
    if run.returncode == 77:
        pytest.skip("no seccomp filter can be installed here")
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="requires POSIX mode enforcement")
def test_move_out_of_a_folder_that_may_not_be_written_copies(tmp_path: Path) -> None:
    """A source whose folder refuses the rename is copied, and the refusal is reported.

    Where the rename takes no flags (the test above) the hard link made
    for it is taken back when the old name cannot be removed: the entry
    must not be left under both names.
    """
    if not os.geteuid():
        pytest.skip("requires a user bound by modes")
    src, dest = tmp_path / "locked" / "item", tmp_path / "dest"
    src.parent.mkdir()
    dest.mkdir()
    src.write_text("SOURCE")
    src.parent.chmod(0o500)
    try:
        result = fs_action("move", str(src), dest=str(dest))
    finally:
        src.parent.chmod(0o700)
    assert "Permission denied" in result["error"], result
    assert (dest / "item").read_text() == src.read_text() == "SOURCE"
    assert src.stat().st_nlink == 1 and os.listdir(src.parent) == ["item"]


@pytest.mark.parametrize(
    ("menu", "name", "swap"),
    [("", "keep/sub/new/entry", "keep/sub"), ("keep/sub", "new/entry", "keep")],
)
@pytest.mark.parametrize("action", ["newFile", "newFolder"])
def test_new_entry_stays_out_of_a_folder_swapped_for_a_link(
    tmp_path: Path, action: str, menu: str, name: str, swap: str,
) -> None:
    """A folder replaced by a link out of the workspace after the checks is not entered.

    Neither a folder of the name nor one above the folder the menu was
    opened on.  The folders are pinned only where ``os`` has ``O_PATH``
    and ``dir_fd`` (Linux).  macOS and Windows keep the creation by path
    and with it this race; that branch cannot run where this test does.
    """
    if not (hasattr(os, "O_PATH") and {os.open, os.mkdir} <= os.supports_dir_fd):
        pytest.skip("the folders are pinned only where os has O_PATH and dir_fd")
    parent, outside = tmp_path / "allowed", tmp_path / "outside"
    (parent / "keep" / "sub").mkdir(parents=True)
    (outside / "sub").mkdir(parents=True)
    swapped = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal swapped
        # The request's first creating or opening call comes after its checks.
        if not swapped and event == "c_call" and arg in (os.mkdir, os.open):
            swapped = True
            (parent / swap).rename(parent / "moved")
            (parent / swap).symlink_to(outside, target_is_directory=True)

    sys.setprofile(editor)
    try:
        result = fs_action(action, str(parent / menu), name=name)
    finally:
        sys.setprofile(None)
    assert swapped and "error" in result, result
    assert not list(tmp_path.rglob("new"))


def _files(folder: Path) -> dict[str, str]:
    """The text of every file below *folder*; no link is followed."""
    return {
        path.relative_to(folder).as_posix(): path.read_text()
        for path in folder.rglob("*") if path.is_file() and not path.is_symlink()
    }


@pytest.mark.parametrize(
    ("action", "swap", "left"),
    [
        ("delete", "dir", {}),
        ("rename", "dir", {"moved/renamed": "SOURCE"}),
        ("copy", "dir", {"moved/item": "SOURCE", "dest/item": "SOURCE"}),
        ("move", "dir", {"dest/item": "SOURCE"}),
        ("copy", "dest", {"dir/item": "SOURCE", "moved/item": "SOURCE"}),
        ("move", "dest", {"moved/item": "SOURCE"}),
    ],
)
def test_a_folder_swapped_for_a_link_during_an_action_is_not_followed(
    tmp_path: Path, action: str, swap: str, left: dict[str, str],
) -> None:
    """An action works in the folders it resolved, whatever their paths name afterwards.

    Once the action has checked its paths an external writer moves the
    source's folder or the destination away and puts a link out of the
    workspace in its place.  Nothing outside is read, changed or
    created: the entry is deleted, renamed or pasted in the folder that
    was moved, and the reply names it as the request did.  Folders are
    pinned through ``/proc/self/fd`` (Linux); macOS and Windows work by
    name and keep this race, a branch that cannot run where this test does.
    """
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("folders are pinned through /proc/self/fd")
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    for folder in (ws / "dir", ws / "dest", outside):
        folder.mkdir(parents=True)
    src = ws / "dir" / "item"
    src.write_text("SOURCE")
    if swap == "dir":
        (outside / "item").write_text("OUTSIDE")
    before = _files(outside)
    swapped = False

    def writer(frame: Any, event: str, arg: Any) -> None:
        nonlocal swapped
        acts = frame.f_code.co_name in ("_remove", "_rename_noreplace", "_transfer")
        if event == "call" and acts and not swapped:
            swapped = True
            (ws / swap).rename(ws / "moved")
            (ws / swap).symlink_to(outside, target_is_directory=True)

    dest = ws / "dir" / "renamed" if action == "rename" else ws / "dest"
    sys.setprofile(writer)
    try:
        result = fs_action(action, str(src), dest=str(dest))
    finally:
        sys.setprofile(None)
    path = {"delete": src, "rename": dest}.get(action, dest / "item")
    assert swapped and result == {"ok": True, "path": str(path)}, result
    assert _files(outside) == before and _files(ws) == left


def _shape(tree: Path) -> dict[str, tuple[Any, ...]]:
    """Type, permissions, modification time (not a link's), link target and bytes of each entry."""
    shape = {}
    for path in [tree, *sorted(tree.rglob("*"))]:
        status = path.lstat()
        link = os.readlink(path) if path.is_symlink() else None
        shape[str(path.relative_to(tree))] = (
            stat.S_IFMT(status.st_mode), stat.S_IMODE(status.st_mode),
            None if link else status.st_mtime_ns, link,
            path.read_bytes() if stat.S_ISREG(status.st_mode) else None,
        )
    return shape


@pytest.mark.parametrize(
    ("when", "copied"),
    [
        ("listed", {}),
        ("opened", {"source/nested/normal.txt": "NORMAL"}),
        ("staged", {}),
        ("made", {}),
        ("file", {}),
        ("file below", {}),
        ("written", {}),
    ],
)
def test_a_copy_follows_no_link_put_below_what_it_copies(
    tmp_path: Path, when: str, copied: dict[str, str],
) -> None:
    """A copy reads and writes below the folders it was given without following a link.

    Only the folders a request names were pinned; a folder below a copied
    one was listed and then copied by name, the copy's folders and files
    were created by name.  An external writer that moves such a folder away
    and puts a link out of the workspace in its place had the link's files
    copied into the workspace (``listed``: once the folder is found in its
    parent's listing; ``opened``: once its own files are) or the workspace's
    files written through it (``staged``: once the folder the copy is
    prepared in exists; ``made``: once a folder of the copy does; ``file``:
    a link where the copy of a file is about to be created, ``file below``:
    the same in a folder of the copy; ``written``: a link in place of a
    file of the copy once it has its bytes, which then gave its target the
    times and permissions of the source).  Each folder is now opened in the
    one above it and refused if it is a link, a file is created only where
    nothing is, and what was opened is what is copied, bytes, times and
    permissions.  The reply is an error that names what it met as the
    request spelled it, and offers no destination to replace; a copy that
    met nothing is whole (``opened``, and ``written`` but for the link
    that took the file's place).  Linux only, as for the folders a request names.
    """
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("folders are pinned through /proc/self/fd")
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    for folder in (ws / "source" / "nested", ws / "dest", outside):
        folder.mkdir(parents=True)
    (ws / "source" / "nested" / "normal.txt").write_text("NORMAL")
    os.utime(ws / "source" / "nested" / "normal.txt", ns=(10**18, 10**18))  # no time of today's
    (outside / "normal.txt").write_text("OUTSIDE")
    (outside / "secret.txt").write_text("SECRET")
    before = _shape(outside)
    swapped = False

    def writer(frame: Any, event: str, arg: Any) -> None:
        """Act once, at the C call after which the copy uses what *when* names."""
        nonlocal swapped
        if swapped or not event.startswith("c_"):
            return
        returned = getattr(arg, "__name__", "") if event == "c_return" else ""
        if when in ("listed", "opened"):  # the listing has told what the entry is
            name = {"listed": "nested", "opened": "normal.txt"}[when]
            entry = getattr(getattr(arg, "__self__", None), "name", "")
            victims = [ws / "source" / "nested"] if (returned, entry) == ("is_dir", name) else []
        elif when.startswith("file"):  # the copy is about to create the file
            pattern = ".kiss-copy-*" if when == "file" else ".kiss-copy-*/entry/nested"
            victims = list((ws / "dest").glob(pattern)) if arg is open else []
        else:  # the copy has made the folder, or written the file
            done, pattern = {
                "staged": ("mkdir", ".kiss-copy-*"),
                "made": ("mkdir", ".kiss-copy-*/entry/nested"),
                "written": ("write", ".kiss-copy-*/entry/nested/normal.txt"),
            }[when]
            victims = list((ws / "dest").glob(pattern)) if returned == done else []
        if not victims:
            return
        swapped = True
        if when.startswith("file"):
            name = "entry" if when == "file" else "normal.txt"
            (victims[0] / name).symlink_to(outside / "planted.txt")
        else:
            link = outside / "normal.txt" if when == "written" else outside
            victims[0].rename(ws / "moved")
            victims[0].symlink_to(link, target_is_directory=link.is_dir())

    src = ws / "source" / "nested" / "normal.txt" if when == "file" else ws / "source"
    sys.setprofile(writer)
    try:
        result = fs_action("copy", str(src), dest=str(ws / "dest"), root=str(ws))
    finally:
        sys.setprofile(None)
    assert swapped, "the external writer never acted"
    assert _shape(outside) == before, "the copy changed something out of the workspace"
    assert _files(ws / "dest") == copied, "the copy went through the link"
    whole = bool(copied) or when == "written"
    assert ("error" in result) != whole and "exists" not in result, result  # none to replace
    assert "/proc/self/fd" not in str(result), result
    left = [path for path in (ws / "dest").glob(".kiss-copy-*") if not path.is_symlink()]
    assert not left, "the copy left the folder it was prepared in"


@pytest.mark.parametrize("closed", [False, True])
def test_a_copied_folder_is_the_one_copytree_makes(tmp_path: Path, closed: bool) -> None:
    """The copy of a folder has the entries, permissions, times and links of ``copytree``'s.

    The folders are opened one in the other instead of copied by name;
    what is copied did not change, the standard library's copy of the
    same tree being the reference.  A folder that is itself *closed* to
    writing is copied like any other, its copy as closed as it is: the
    copy is prepared in a folder of its own and published by a rename
    to another folder, which a folder its owner may not write is
    refused.  Moving such a folder stays refused whole, to a folder of
    its filesystem (as ``mv`` refuses it) and of another one: copied
    instead, its entries could not be deleted from it.
    """
    src, dest = tmp_path / "ws" / "tree", tmp_path / "ws" / "dest"
    (src / "a" / "b").mkdir(parents=True)
    (src / "empty").mkdir()
    dest.mkdir()
    (src / "a" / "file.txt").write_text("one")
    (src / "a" / "b" / "run.sh").write_text("#!/bin/sh\n")
    (src / "a" / "b" / "run.sh").chmod(0o750)
    (src / "to-dir").symlink_to("a", target_is_directory=True)
    (src / "dangling").symlink_to("nowhere")
    os.utime(src / "a" / "file.txt", ns=(1_600_000_000_123_456_789, 1_500_000_000_987_654_321))
    os.utime(src / "a", ns=(1_400_000_000_000_000_001, 1_300_000_000_000_000_002))
    (src / "a" / "b").chmod(0o555)  # its copy is filled before it is closed to writing
    trees = (src, dest / "tree", tmp_path / "reference")
    try:
        if closed:
            src.chmod(0o555)
        result = fs_action("copy", str(src), dest=str(dest), root=str(tmp_path / "ws"))
        shutil.copytree(src, trees[2], symlinks=True)
        assert result == {"ok": True, "path": str(trees[1])}
        assert _shape(trees[1]) == _shape(trees[2])
        assert os.listdir(dest) == ["tree"], "the copy left the folder it was prepared in"
        if closed and hasattr(os, "geteuid") and os.geteuid():  # root is bound by no mode
            other = tmp_path / "ws" / "other"
            other.mkdir()
            shared = Path("/dev/shm")  # another filesystem, where there is one
            with tempfile.TemporaryDirectory(dir=shared if shared.is_dir() else None) as far:
                for folder in (str(other), far):
                    moved = fs_action("move", str(src), dest=folder)
                    denied = f"move failed: [Errno 13] Permission denied: '{src}'"
                    assert moved == {"error": denied}
                    assert _shape(src) == _shape(trees[2]) and not os.listdir(folder)
                    assert sorted(os.listdir(tmp_path / "ws")) == ["dest", "other", "tree"]
    finally:
        for tree in trees:  # writable again, for the removal of tmp_path
            if tree.is_dir():
                tree.chmod(0o755)
            if (tree / "a" / "b").is_dir():
                (tree / "a" / "b").chmod(0o755)


def test_root_keeps_every_change_of_the_tree_in_the_workspace(tmp_path: Path) -> None:
    """Given *root*, an action refuses a folder that leads out of the workspace by itself.

    The daemon checks the paths of a request before it calls
    ``fs_action``; a folder swapped for a link in between is refused
    here, where the folder is resolved for good.  What that check
    accepts stays accepted: a link to a folder of the workspace is
    entered, an entry that is a link out is acted on as itself.  ``..``
    and ``/`` are no entry of a folder: none is acted on under that name.
    """
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    (ws / "dir").mkdir(parents=True)
    outside.mkdir()
    (ws / "dir" / "item").write_text("INSIDE")
    (outside / "item").write_text("OUTSIDE")
    (ws / "portal").symlink_to(outside, target_is_directory=True)
    (ws / "alias").symlink_to("dir", target_is_directory=True)
    (tmp_path / "also").symlink_to(ws, target_is_directory=True)
    root, portal, inside = str(ws), str(ws / "portal"), str(ws / "dir" / "item")
    refused = [
        fs_action("newFile", portal, name="new", root=root),
        fs_action("newFolder", portal, name="new", root=root),
        fs_action("delete", f"{portal}/item", root=root),
        fs_action("rename", f"{portal}/item", dest=f"{portal}/renamed", root=root),
        fs_action("copy", f"{portal}/item", dest=str(ws / "dir"), overwrite=True, root=root),
        fs_action("move", f"{portal}/item", dest=str(ws / "dir"), overwrite=True, root=root),
        fs_action("copy", inside, dest=portal, overwrite=True, root=root),
        fs_action("move", inside, dest=portal, overwrite=True, root=root),
    ]
    assert all("Not inside the workspace" in reply["error"] for reply in refused), refused
    assert _files(outside) == {"item": "OUTSIDE"} and _files(ws) == {"dir/item": "INSIDE"}
    made = fs_action("newFile", str(ws / "alias"), name="made", root=root)
    assert made == {"ok": True, "path": str(ws / "alias" / "made")}
    assert _files(ws) == {"dir/item": "INSIDE", "dir/made": ""}
    also = tmp_path / "also"
    assert fs_action("delete", str(also / "alias" / "made"), root=str(also)).get("ok")
    assert fs_action("delete", portal, root=root) == {"ok": True, "path": portal}
    for action in ("delete", "rename", "copy", "move"):
        above = fs_action(action, f"{root}/..", dest=str(ws / "dir"), root=root)
        assert above == {"error": f"Not found: {root}/.."}, above
    assert fs_action("copy", "/", dest=str(ws / "dir")) == {"error": "Not found: /"}
    assert _files(outside) == {"item": "OUTSIDE"} and _files(ws) == {"dir/item": "INSIDE"}
    assert sorted(os.listdir(ws)) == ["alias", "dir"] and os.listdir(ws / "dir") == ["item"]


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="requires POSIX mode enforcement")
def test_an_error_names_the_folder_as_the_request_spelled_it(tmp_path: Path) -> None:
    """No reply shows the name a folder is pinned under or a ``PosixPath(...)``, for any folder."""
    if not os.geteuid() or os.access("/", os.W_OK):
        pytest.skip("requires a user whom modes bind and who may not write to /")
    folder = tmp_path / "unlisted"
    folder.mkdir()
    (folder / "a").write_text("A")
    os.link(folder / "a", folder / "b")
    folder.chmod(0o300)
    try:
        unlisted = fs_action("rename", str(folder / "a"), dest=str(folder / "b"))
    finally:
        folder.chmod(0o700)
    assert unlisted["error"].startswith("rename failed: [Errno "), unlisted
    assert unlisted["error"].endswith(f": '{folder}'"), unlisted
    (folder / "kept").mkdir()
    folder.chmod(0o500)
    try:
        kept = fs_action("delete", str(folder / "kept"))
    finally:
        folder.chmod(0o700)
    assert kept["error"].startswith("delete failed: [Errno "), kept
    assert kept["error"].endswith(f": '{folder}/kept'"), kept
    refused = fs_action("rename", "/tmp", dest="/kiss-no-such-entry")["error"]
    assert refused.startswith("rename failed: [Errno "), refused
    assert refused.endswith("'/kiss-no-such-entry'") and "/proc/" not in refused, refused


def test_new_entry_replies_as_the_creation_by_path_did(tmp_path: Path) -> None:
    """Pinning the folders changes no reply for a name nobody is swapping.

    ``exists`` makes the page ask "Replace it?" before it shows the error,
    so a name that runs through a file or a dangling link must keep it.
    """
    parent = tmp_path / "allowed"
    (parent / "wx").mkdir(parents=True)
    (parent / "file").write_text("f")
    (parent / "dangling").symlink_to("missing")
    (parent / "up").symlink_to("..", target_is_directory=True)
    (tmp_path / "dl").symlink_to(parent / "missing")
    entries = sorted(tmp_path.rglob("*"))
    for action, blocker in ("newFile", "file"), ("newFolder", "dangling"), ("newFile", "up/dl"):
        assert fs_action(action, str(parent), name=f"{blocker}/x") == {
            "error": f"A file or folder {Path(blocker).name} already exists", "exists": True,
        }
    nested = fs_action("newFolder", str(parent), name="file/x")
    if os.name == "nt":
        assert nested == {
            "error": "A file or folder file already exists", "exists": True,
        }
    else:
        assert nested == {
            "error": f"newFolder failed: [Errno 20] Not a directory: '{parent}/file/x'",
        }
    # A name that cannot be encoded is reported at its place in the whole path.
    if os.name != "nt":
        unencodable = fs_action("newFile", str(parent), name="a\ud800")["error"]
        assert f"'\\ud800' in position {len(str(parent)) + 2}:" in unencodable, unencodable
    assert sorted(tmp_path.rglob("*")) == entries
    # A folder that may be written and searched but not read still takes new entries.
    kinds = ("File", "Folder")
    (parent / "wx").chmod(0o300)
    made = [fs_action(f"new{kind}", str(parent), name=f"wx/new/{kind}") for kind in kinds]
    (parent / "wx").chmod(0o700)
    assert made == [
        {"ok": True, "path": str(parent / "wx" / "new" / kind)} for kind in kinds
    ]
    assert (parent / "wx/new/File").is_file() and (parent / "wx/new/Folder").is_dir()
    # A name through a link to "/" keeps its "/" in the error (run only where "/" refuses it).
    if hasattr(os, "geteuid") and os.geteuid() and not os.access("/", os.W_OK):
        (tmp_path / "root").symlink_to("/", target_is_directory=True)
        name = f"{str(tmp_path)[1:]}/root/kiss-no-such-entry"
        new_file = fs_action("newFile", "/", name=name)
        assert new_file["error"].startswith("newFile failed: [Errno "), new_file
        assert new_file["error"].endswith(f": '/{name}'"), new_file
        new_folder = fs_action("newFolder", "/", name="kiss-no-such-entry/x")
        assert new_folder["error"].startswith("newFolder failed: [Errno "), new_folder
        assert new_folder["error"].endswith(": '/kiss-no-such-entry'"), new_folder
        assert not os.path.lexists("/kiss-no-such-entry")
