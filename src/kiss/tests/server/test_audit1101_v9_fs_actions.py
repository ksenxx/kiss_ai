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
import sys
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
        at_rename = event == "c_call" and arg is os.rename
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
