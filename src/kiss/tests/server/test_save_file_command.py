# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_content_tab_file_links and is intentionally
#   shadowed by test parameters of the same name)
"""Protocol tests for the remote webapp's ``saveFile`` command.

An editable content tab sends ``saveFile`` (the Monaco editor's full
text) and :meth:`RemoteAccessServer._handle_save_file` writes it back
to the file the tab was opened from, replying ``fileSaved``.  These
tests drive a REAL :class:`RemoteAccessServer` over real ``wss://``
(and the real UDS for the drop gate) — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect

from kiss.server.sorcar import API, validate_command
from kiss.server.web_server import _file_version
from kiss.tests.server.test_content_tab_file_links import (
    _PY_SOURCE,
    _no_verify_ssl,
    _ServerHarness,
    harness,  # noqa: F401  (module fixture used by param name)
)


async def _ws_request(
    harness: _ServerHarness, payloads: list[dict], reply_type: str,
) -> list[dict]:
    """Authenticate over wss://, send every payload on ONE connection,
    return the *reply_type* replies in order (one per payload)."""
    async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "auth_ok":
                break
        replies: list[dict] = []
        for payload in payloads:
            await ws.send(json.dumps(payload))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg.get("type") == reply_type:
                    replies.append(msg)
                    break
        return replies


def _run_list(
    harness: _ServerHarness, coro: Coroutine[Any, Any, list[dict]],
) -> list[dict]:
    """Run *coro* on the harness loop and return its list result."""
    return asyncio.run_coroutine_threadsafe(coro, harness.loop).result(60)


async def _first_reply(
    harness: _ServerHarness, payload: dict, reply_type: str,
) -> dict:
    return (await _ws_request(harness, [payload], reply_type))[0]


def _save(harness: _ServerHarness, payload: dict) -> dict:
    payload = {"type": "saveFile", **payload}
    return harness.run(_first_reply(harness, payload, "fileSaved"))


def _open(harness: _ServerHarness, path: str) -> dict:
    return harness.run(
        _first_reply(harness, {"type": "openFile", "path": path}, "fileContent"),
    )


class TestCatalog:
    """The command is part of the server API and validated like the rest."""

    def test_save_file_is_catalogued(self) -> None:
        spec = API["saveFile"]
        assert spec.handler == "save_file"
        assert set(spec.required) == {"path", "content"}

    def test_missing_content_is_rejected(self) -> None:
        err = validate_command({"type": "saveFile", "path": "x.py"})
        assert err is not None and "content" in err
        assert (
            validate_command({"type": "saveFile", "path": "x.py", "content": ""})
            is None
        )


class TestSaveFile:
    """``saveFile`` → ``fileSaved`` over a real ``wss://`` connection."""

    def test_saves_text_and_reports_new_version(self, harness) -> None:
        target = harness.work_dir / "edit_me.py"
        target.write_text(_PY_SOURCE)
        opened = _open(harness, str(target))
        st = target.stat()
        assert opened["version"] == f"{st.st_mtime_ns}:{st.st_size}"
        assert opened["version"] == _file_version(st)
        new_text = _PY_SOURCE + "\nprint(greet('web'))\n"
        reply = _save(harness, {
            "path": str(target),
            "content": new_text,
            "version": opened["version"],
            "tabId": "chat-1",
            "token": "content-7",
        })
        assert reply["ok"] is True, reply
        assert "error" not in reply
        assert reply["path"] == str(target)
        assert reply["name"] == "edit_me.py"
        assert reply["tabId"] == "chat-1"
        assert reply["token"] == "content-7"
        assert target.read_text() == new_text
        assert reply["version"] == _file_version(target.stat())
        assert reply["version"] != opened["version"]

    def test_stale_version_is_a_conflict_until_forced(self, harness) -> None:
        target = harness.work_dir / "racy.txt"
        target.write_text("original\n")
        opened = _open(harness, str(target))
        # Someone else (the agent, a shell) rewrites the file meanwhile.
        target.write_text("changed on disk\n")
        reply = _save(harness, {
            "path": str(target),
            "content": "my edits\n",
            "version": opened["version"],
        })
        assert reply["ok"] is False
        assert reply["conflict"] is True
        assert "changed on disk" in reply["error"]
        assert target.read_text() == "changed on disk\n"
        forced = _save(harness, {
            "path": str(target),
            "content": "my edits\n",
            "version": opened["version"],
            "force": True,
        })
        assert forced["ok"] is True, forced
        assert "conflict" not in forced
        assert target.read_text() == "my edits\n"

    def test_same_size_rewrite_with_new_mtime_is_a_conflict(
        self, harness,
    ) -> None:
        """A rewrite that keeps the size still moves the mtime, and the
        stamp catches it."""
        target = harness.work_dir / "same_size.txt"
        target.write_text("aaaa\n")
        opened = _open(harness, str(target))
        target.write_text("bbbb\n")
        os.utime(target, ns=(target.stat().st_mtime_ns + 5_000_000,) * 2)
        reply = _save(harness, {
            "path": str(target), "content": "cccc\n",
            "version": opened["version"],
        })
        assert reply["ok"] is False and reply["conflict"] is True
        assert target.read_text() == "bbbb\n"

    def test_matching_version_saves(self, harness) -> None:
        """Only a DIFFERENT stamp is a conflict: an untouched file saves
        with the stamp it was opened with."""
        target = harness.work_dir / "calm.txt"
        target.write_text("a\n")
        opened = _open(harness, str(target))
        reply = _save(harness, {
            "path": str(target), "content": "b\n",
            "version": opened["version"],
        })
        assert reply["ok"] is True
        assert target.read_text() == "b\n"

    def test_without_version_no_conflict_check(self, harness) -> None:
        target = harness.work_dir / "unchecked.txt"
        target.write_text("one\n")
        reply = _save(harness, {"path": str(target), "content": "two\n"})
        assert reply["ok"] is True
        assert target.read_text() == "two\n"

    def test_non_string_version_values_are_ignored(self, harness) -> None:
        target = harness.work_dir / "bogus.txt"
        target.write_text("one\n")
        for bad in (True, 123, 1.5, None, [1], ""):
            reply = _save(harness, {
                "path": str(target), "content": "two\n", "version": bad,
            })
            assert reply["ok"] is True, (bad, reply)
        assert target.read_text() == "two\n"

    def test_missing_file_is_not_created(self, harness) -> None:
        target = harness.work_dir / "never_existed.py"
        reply = _save(harness, {"path": str(target), "content": "x = 1\n"})
        assert reply["ok"] is False
        assert "File not found" in reply["error"]
        assert not target.exists()

    def test_directory_is_refused(self, harness) -> None:
        folder = harness.work_dir / "a_folder"
        folder.mkdir(exist_ok=True)
        reply = _save(harness, {"path": str(folder), "content": "nope"})
        assert reply["ok"] is False
        assert "directory" in reply["error"].lower()
        assert folder.is_dir()

    def test_non_string_content_is_refused(self, harness) -> None:
        target = harness.work_dir / "typed.txt"
        target.write_text("keep\n")
        for bad in (123, ["a"], {"k": "v"}, True):
            reply = _save(harness, {"path": str(target), "content": bad})
            assert reply["ok"] is False, (bad, reply)
            assert "not text" in reply["error"]
        assert target.read_text() == "keep\n"

    def test_oversized_content_is_refused(self, harness) -> None:
        target = harness.work_dir / "small.txt"
        target.write_text("small\n")
        reply = _save(harness, {
            "path": str(target), "content": "x" * 2_500_000,
        })
        assert reply["ok"] is False
        assert "too large" in reply["error"]
        assert target.read_text() == "small\n"

    def test_relative_path_resolves_against_work_dir(self, harness) -> None:
        target = harness.work_dir / "rel.txt"
        target.write_text("before\n")
        reply = _save(harness, {
            "path": "rel.txt",
            "content": "after\n",
            "workDir": str(harness.work_dir),
        })
        assert reply["ok"] is True
        assert reply["path"] == str(target)
        assert target.read_text() == "after\n"

    def test_non_string_ids_are_blanked(self, harness) -> None:
        target = harness.work_dir / "ids.txt"
        target.write_text("x\n")
        reply = _save(harness, {
            "path": str(target), "content": "y\n", "tabId": 7, "token": [1],
        })
        assert reply["ok"] is True
        assert reply["tabId"] == ""
        assert reply["token"] == ""

    def test_empty_path_gets_no_reply_but_later_commands_do(
        self, harness,
    ) -> None:
        """An empty path is dropped silently (like openFile); the
        connection stays healthy for the next command."""
        target = harness.work_dir / "after_empty.txt"
        target.write_text("z\n")

        async def _probe() -> list[dict]:
            async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if msg.get("type") == "auth_ok":
                        break
                await ws.send(json.dumps(
                    {"type": "saveFile", "path": "", "content": "ignored"},
                ))
                await ws.send(json.dumps(
                    {"type": "saveFile", "path": str(target), "content": "w\n"},
                ))
                got: list[dict] = []
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if msg.get("type") == "fileSaved":
                        got.append(msg)
                        return got

        got = harness.run(_probe())
        assert len(got) == 1
        assert got[0]["path"] == str(target)
        assert target.read_text() == "w\n"

    def test_executable_bit_survives_the_atomic_replace(self, harness) -> None:
        script = harness.work_dir / "tool.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o755)
        before = script.stat()
        reply = _save(harness, {
            "path": str(script), "content": "#!/bin/sh\necho bye\n",
        })
        assert reply["ok"] is True
        after = script.stat()
        assert stat.S_IMODE(after.st_mode) == 0o755
        # Atomic publish: a new inode replaced the old one.
        assert after.st_ino != before.st_ino
        assert script.read_text() == "#!/bin/sh\necho bye\n"

    def test_crlf_and_unicode_are_written_byte_for_byte(self, harness) -> None:
        target = harness.work_dir / "win.txt"
        target.write_bytes(b"a\r\nb\r\n")
        content = "caf\u00e9 \u2014 line\r\nnext\r\n"
        reply = _save(harness, {"path": str(target), "content": content})
        assert reply["ok"] is True
        assert target.read_bytes() == content.encode("utf-8")

    def test_symlink_saves_through_to_its_target(self, harness) -> None:
        real = harness.work_dir / "real_target.txt"
        real.write_text("real\n")
        link = harness.work_dir / "link_to_real.txt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(real)
        reply = _save(harness, {"path": str(link), "content": "via link\n"})
        assert reply["ok"] is True
        assert reply["path"] == str(real)
        assert real.read_text() == "via link\n"
        assert link.is_symlink()

    def test_unwritable_directory_replies_error(self, harness) -> None:
        if os.geteuid() == 0:
            # root ignores directory permission bits; nothing to test.
            return
        locked = harness.work_dir / "locked"
        locked.mkdir(exist_ok=True)
        target = locked / "ro.txt"
        target.write_text("ro\n")
        locked.chmod(0o555)
        try:
            reply = _save(harness, {"path": str(target), "content": "rw\n"})
            assert reply["ok"] is False
            assert "Failed to save" in reply["error"]
            assert target.read_text() == "ro\n"
        finally:
            locked.chmod(0o755)

    def test_directory_listing_reply_has_no_version(self, harness) -> None:
        """A directory is never saveable, so its fileContent carries
        no ``version`` — the client keys the editable surface on it."""
        reply = _open(harness, str(harness.work_dir))
        assert reply["isDirectory"] is True
        assert "version" not in reply

    def test_invalid_utf8_file_opens_read_only(self, harness) -> None:
        """A file that is not valid UTF-8 is shown with replacement
        characters but carries no ``version``: the client then keeps
        its read-only viewer, because saving the U+FFFDs back would
        destroy the bytes the decoder could not represent."""
        target = harness.work_dir / "latin1.txt"
        target.write_bytes(b"prefix-\xff-suffix\n")
        reply = _open(harness, str(target))
        assert "error" not in reply
        assert reply["content"] == "prefix-\ufffd-suffix\n"
        assert "version" not in reply
        assert target.read_bytes() == b"prefix-\xff-suffix\n"

    def test_lone_surrogate_content_gets_an_error_reply(self, harness) -> None:
        """JSON may carry a lone surrogate, which has no UTF-8 form; the
        save is refused with a reply instead of raising without one."""
        target = harness.work_dir / "surrogate.txt"
        target.write_text("safe\n")
        replies = _run_list(harness, _ws_request(
            harness,
            [
                {"type": "saveFile", "path": str(target), "content": "\ud800"},
                {"type": "saveFile", "path": str(target), "content": "next\n"},
            ],
            "fileSaved",
        ))
        assert replies[0]["ok"] is False
        assert "not valid text" in replies[0]["error"]
        assert replies[1]["ok"] is True
        assert target.read_text() == "next\n"

    def test_concurrent_saves_with_one_stale_stamp_yield_one_winner(
        self, harness,
    ) -> None:
        """Many clients saving the same file with the SAME opening
        stamp: exactly one write succeeds, every other reply is a
        conflict, and the file holds exactly the winner's text."""
        target = harness.work_dir / "race.txt"
        target.write_text("start\n")
        opened = _open(harness, str(target))
        n = 16

        async def _one(i: int) -> dict:
            body = f"writer-{i}\n" + ("x" * 1_500_000)
            return (await _ws_request(
                harness,
                [{
                    "type": "saveFile",
                    "path": str(target),
                    "content": body,
                    "version": opened["version"],
                    "token": f"w-{i}",
                }],
                "fileSaved",
            ))[0]

        async def _all() -> list[dict]:
            return list(await asyncio.gather(*(_one(i) for i in range(n))))

        replies = _run_list(harness, _all())
        winners = [r for r in replies if r["ok"]]
        losers = [r for r in replies if not r["ok"]]
        assert len(winners) == 1, replies
        assert len(losers) == n - 1
        assert all(r["conflict"] is True for r in losers)
        winner_idx = int(winners[0]["token"].split("-")[1])
        assert target.read_text().startswith(f"writer-{winner_idx}\n")
        assert winners[0]["version"] == _file_version(target.stat())

    def test_wrong_string_version_is_a_conflict(self, harness) -> None:
        """Any stamp that is not the file's current one is refused —
        including a garbage string."""
        target = harness.work_dir / "garbage_stamp.txt"
        target.write_text("g\n")
        reply = _save(harness, {
            "path": str(target), "content": "h\n", "version": "not-a-stamp",
        })
        assert reply["ok"] is False and reply["conflict"] is True
        assert target.read_text() == "g\n"


class TestUdsDropGate:
    """A VS Code window (UDS) edits files in real editors: its
    ``saveFile`` is dropped and never touches the disk."""

    def test_uds_delivered_save_is_dropped(self, harness) -> None:
        target = harness.work_dir / "uds_guard.txt"
        target.write_text("untouched\n")

        async def _probe() -> list[dict]:
            reader, writer = await asyncio.open_unix_connection(
                str(Path(harness.tmpdir) / "sorcar.sock"),
            )
            try:
                for cmd in (
                    {
                        "type": "saveFile",
                        "path": str(target),
                        "content": "smuggled\n",
                        "tabId": "u-1",
                    },
                    # Positive control, sent LAST: getInputHistory IS
                    # answered over UDS.  Its reply arriving proves the
                    # saveFile before it was processed — and dropped.
                    {"type": "getInputHistory", "tabId": "u-1"},
                ):
                    writer.write((json.dumps(cmd) + "\n").encode())
                await writer.drain()
                got: list[dict] = []
                deadline = asyncio.get_event_loop().time() + 15
                while asyncio.get_event_loop().time() < deadline:
                    line = await asyncio.wait_for(reader.readline(), 15)
                    if not line:
                        break
                    ev = json.loads(line)
                    got.append(ev)
                    if ev.get("type") == "inputHistory":
                        break
                return got
            finally:
                writer.close()
                await writer.wait_closed()

        events = harness.run(_probe())
        types = [e.get("type") for e in events]
        assert "inputHistory" in types, events
        assert "fileSaved" not in types, events
        assert target.read_text() == "untouched\n"
