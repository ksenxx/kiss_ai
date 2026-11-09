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
    assert result.get("exists") is True, result
    assert "error" in result
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
        copied = event == "return" and frame.f_code.co_name in ("copy2", "copytree")
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
            if copied_path != src and not copied_path.parent.name.startswith(".kiss-move-"):
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
            if not edited and copied.parent.name.startswith(".kiss-move-"):
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
            retained, = tmp_path.glob(".kiss-move-*/entry")
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
