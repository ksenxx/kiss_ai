# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A same-size rewrite inside one filesystem timestamp tick is still a conflict.

Audit 2026-11-01 (V2 F1): the editor's version stamp was
``st_mtime_ns:st_size``.  File times are jiffy-grained on Linux (a few
milliseconds), so a save that kept the file's size and landed in the
same tick as the state the file was opened in left the stamp unchanged.
Two clients saving different equal-length texts with the same opening
stamp therefore both passed the check under ``_SAVE_FILE_LOCK``: the
second silently overwrote the first and both got ``ok: true``, against
the handler's one-winner guarantee.  The stamp is now a digest of the
file's content, so it names the bytes the editor showed, whatever the
clock did.

Live ``RemoteAccessServer`` over ``wss://`` (the shared harness of the
content-tab suites), real files.  The same-tick state is produced by
rewriting the file with equal-length bytes and giving it back the
opened ``st_mtime_ns`` (what two writes a few microseconds apart get on
their own); the concurrent trials then exercise the real race.
"""

from __future__ import annotations

import asyncio
import os

from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_save_file_command import (
    _open,
    _run_list,
    _save,
    _ws_request,
)


def test_same_size_rewrite_in_the_same_tick_is_a_conflict(harness) -> None:  # noqa: F811
    target = harness.work_dir / "same_tick.txt"
    target.write_text("aaaa\n")
    opened = _open(harness, str(target))
    st = target.stat()
    # Someone else rewrote the file with as many bytes, within the
    # same timestamp tick as the state the editor opened.
    target.write_text("bbbb\n")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    reply = _save(harness, {
        "path": str(target), "content": "cccc\n", "version": opened["version"],
    })
    assert reply["ok"] is False and reply.get("conflict") is True, reply
    assert target.read_text() == "bbbb\n"


def test_concurrent_equal_length_saves_have_one_winner(harness) -> None:  # noqa: F811
    target = harness.work_dir / "same_tick_race.txt"
    for trial in range(8):
        target.write_text("start\n")
        opened = _open(harness, str(target))

        async def _one(text: str) -> dict:
            return (await _ws_request(
                harness,
                [{
                    "type": "saveFile", "path": str(target), "content": text,
                    "version": opened["version"], "token": text.strip(),
                }],
                "fileSaved",
            ))[0]

        async def _both() -> list[dict]:
            return list(await asyncio.gather(_one("alpha\n"), _one("other\n")))

        replies = _run_list(harness, _both())
        winners = [r for r in replies if r["ok"]]
        assert len(winners) == 1, (trial, replies)
        assert target.read_text() == winners[0]["token"] + "\n"
        loser = next(r for r in replies if not r["ok"])
        assert loser.get("conflict") is True, loser
