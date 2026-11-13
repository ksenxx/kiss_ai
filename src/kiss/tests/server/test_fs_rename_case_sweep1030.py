# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_explorer_scm_commands and is intentionally
#   shadowed by test parameters of the same name)
"""End-to-end tests: the Explorer's "Rename..." to another case of the same name.

A real daemon (:class:`~kiss.tests.server.test_explorer_scm_commands.ExplorerHarness`)
answers ``fsAction`` ``rename`` over a real wss:// connection.

On a case-insensitive file system (macOS's default APFS, Windows' NTFS,
a Linux ``casefold`` folder) ``A.txt`` finds ``a.txt``.  A rename that
changes only the case of a name must rename the entry: it is neither "a
file or folder A.txt already exists" (whose confirmed replace renamed
the source aside and then failed with "No such file or directory") nor,
by Windows' case-insensitive path equality or a casefold folder's no-op
rename of an entry onto itself, a success that renames nothing.  A
different entry whose name differs from the new one only in case is
still an existing name, never silently replaced.

Those tests need a case-insensitive temp folder (macOS, Windows, or a
``TMPDIR`` in an ext4 ``casefold`` folder) and are skipped elsewhere:
Linux's usual file systems are case-sensitive and making one that is
not takes root.  On a case-sensitive file system two names that differ
only in case are two entries, even two hard links of one file, and a
rename of one onto the other stays a replace.

The last test runs on either kind.  It calls ``fs_action`` on its own
thread, where a profile hook plays the editor that saves the entry's
name between the two steps of the case-only rename.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.server.fs_actions import fs_action
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    _request,
    harness,  # noqa: F401  (module fixture used by param name)
)


def _case_insensitive(folder: Path) -> bool:
    """Whether the file system holding *folder* finds ``CASE-PROBE`` for ``case-probe``."""
    probe = folder / "case-probe"
    probe.write_text("")
    try:
        return os.path.lexists(folder / "CASE-PROBE")
    finally:
        probe.unlink()


@pytest.fixture
def folder() -> Iterator[Path]:
    """A fresh, symlink-free temp folder (the daemon reports resolved paths)."""
    path = Path(os.path.realpath(tempfile.mkdtemp(prefix="kiss-rename-case-")))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _rename(harness: ExplorerHarness, folder: Path, name: str, new: str, **cmd: object) -> dict:
    """Rename *folder*/*name* to *folder*/*new* through the daemon; return its ``fsResult``."""
    payload = {
        "type": "fsAction", "action": "rename", "tabId": "t", "token": "r",
        "workDir": str(folder), "path": str(folder / name), "dest": str(folder / new),
        **cmd,
    }
    return _request(harness, payload, "fsResult")


def test_rename_to_its_own_name_changes_nothing(harness, folder: Path) -> None:
    (folder / "same.txt").write_text("kept\n")
    reply = _rename(harness, folder, "same.txt", "same.txt")
    assert reply.get("ok") is True, reply
    assert reply["path"] == str(folder / "same.txt")
    assert os.listdir(folder) == ["same.txt"]
    assert (folder / "same.txt").read_text() == "kept\n"


def test_case_only_rename_renames_in_place(harness, folder: Path) -> None:
    if not _case_insensitive(folder):
        pytest.skip("the temp folder's file system is case-sensitive")
    (folder / "notes.txt").write_text("my notes\n")
    (folder / "Docs").mkdir()
    (folder / "Docs" / "inside.txt").write_text("inside\n")

    renamed = _rename(harness, folder, "notes.txt", "Notes.txt")
    assert renamed.get("ok") is True, renamed
    assert "exists" not in renamed
    assert renamed["path"] == str(folder / "Notes.txt")
    folded = _rename(harness, folder, "Docs", "docs")
    assert folded.get("ok") is True, folded
    assert sorted(os.listdir(folder)) == ["Notes.txt", "docs"]
    # With ``overwrite`` (the replace the old "already exists" reply had
    # the page confirm) it renames in place too, and fails nothing.
    confirmed = _rename(harness, folder, "Notes.txt", "NOTES.txt", overwrite=True)
    assert confirmed.get("ok") is True, confirmed

    assert sorted(os.listdir(folder)) == ["NOTES.txt", "docs"]
    assert (folder / "NOTES.txt").read_text() == "my notes\n"
    assert os.listdir(folder / "docs") == ["inside.txt"]


def test_another_entry_differing_only_in_case_is_still_an_existing_name(
    harness, folder: Path,
) -> None:
    if not _case_insensitive(folder):
        pytest.skip("the temp folder's file system is case-sensitive")
    (folder / "a.txt").write_text("a\n")
    (folder / "b.txt").write_text("b\n")
    clash = _rename(harness, folder, "a.txt", "B.txt")
    assert clash["exists"] is True, clash
    assert sorted(os.listdir(folder)) == ["a.txt", "b.txt"]
    assert (folder / "b.txt").read_text() == "b\n"


def test_hard_links_differing_only_in_case_are_two_entries(harness, folder: Path) -> None:
    if _case_insensitive(folder):
        pytest.skip("two names differing only in case need a case-sensitive file system")
    (folder / "a.txt").write_text("linked\n")
    os.link(folder / "a.txt", folder / "A.txt")
    clash = _rename(harness, folder, "a.txt", "A.txt")
    assert clash["exists"] is True, clash
    assert sorted(os.listdir(folder)) == ["A.txt", "a.txt"]
    replaced = _rename(harness, folder, "a.txt", "A.txt", overwrite=True)
    assert replaced.get("ok") is True, replaced
    assert os.listdir(folder) == ["A.txt"]
    assert (folder / "A.txt").read_text() == "linked\n"


@pytest.mark.parametrize("occupied", [False, True])
def test_case_only_rename_keeps_what_an_editor_saves_meanwhile(
    folder: Path, occupied: bool,
) -> None:
    """Neither step of the rename replaces an entry saved after the first.

    Once the entry is put aside an editor saves the new name and, when
    *occupied*, the old one (one save does both on a case-insensitive
    file system): the entry then stays at its temporary name, which the
    error gives, or goes back to the old name still free.  A
    case-sensitive file system gets there with the names as hard links
    of one file, the new one removed between the same-entry check's two
    looks at it.
    """
    src, target = folder / "a.txt", folder / "A.txt"
    src.write_text("SOURCE")
    insensitive = _case_insensitive(folder)
    if not insensitive:
        os.link(src, target)
    saved = False

    def editor(frame: Any, event: str, arg: Any) -> None:
        nonlocal saved
        if not insensitive and event == "c_call" and arg is os.listdir:
            target.unlink(missing_ok=True)
        elif not saved and not os.path.lexists(src):
            saved = True
            target.write_text("EDITOR")
            if occupied:
                src.write_text("EDITOR")

    sys.setprofile(editor)
    try:
        result = fs_action("rename", str(src), dest=str(target))
    finally:
        sys.setprofile(None)
    assert saved and "error" in result, result
    assert target.read_text() == "EDITOR"
    aside = list(folder.glob(".a.txt.renaming-*"))
    if occupied or insensitive:
        assert src.read_text() == "EDITOR" and [p.read_text() for p in aside] == ["SOURCE"]
        assert str(aside[0]) in result["error"]
    else:
        assert src.read_text() == "SOURCE" and not aside and result["exists"] is True
