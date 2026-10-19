# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the Git Commit button of a tab picked onto a registered machine.

A real local daemon and a real remote daemon (``box``, a subprocess
with its own home and working directory).  An ``autocommitAction``
stamped ``machine`` commits the folder on the machine, not the local
daemon's, and the outcome comes back to the local client as the
usual toast plus ``autocommit_done``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from kiss.tests.server.test_remote_machine_settings import (
    RemoteMachineCase,
    _as_another_client,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


def _make_repo(path: Path) -> None:
    """Turn the existing folder *path* into a git repo with one commit."""
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("one\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")


def _head(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD").strip()


def _subject(path: Path) -> str:
    """The first non-empty line of HEAD's commit message, as the outcome toast names it."""
    return next(
        line.strip() for line in _git(path, "log", "-1", "--format=%B").splitlines()
        if line.strip()
    )


class TestGitCommitOnMachine(RemoteMachineCase):
    """``autocommitAction`` stamped ``machine`` commits on that machine."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        _make_repo(self.box_work)
        _make_repo(self.home_work)
        (self.home_work / "f.txt").write_text("home change\n")

    async def _commit_outcome(self, cmd: dict[str, Any]) -> list[dict[str, Any]]:
        """Send *cmd* and return every message up to and including its ``autocommit_done``."""
        await self._send(cmd)
        seen: list[dict[str, Any]] = []
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            seen.append(msg)
            if msg.get("type") == "autocommit_done":
                return seen
        raise AssertionError(f"no autocommit_done: {seen}")

    def _assert_home_untouched(self, home_before: str) -> None:
        self.assertEqual(_head(self.home_work), home_before, "the local repo must not be committed")
        self.assertEqual(_git(self.home_work, "status", "--porcelain").strip(), "M f.txt")

    async def test_commit_lands_on_the_machine(self) -> None:
        (self.box_work / "f.txt").write_text("one\ntwo\n")
        (self.box_work / "new.txt").write_text("on the box\n")
        box_before, home_before = _head(self.box_work), _head(self.home_work)

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-box",
            "workDir": str(self.box_work), "machine": "box",
        })

        notes = [m for m in seen if m.get("type") == "notification"]
        self.assertEqual(notes[0]["message"], "Committing on box…")
        self.assertEqual(notes[0]["tabId"], "t-box")
        self.assertEqual(notes[0]["id"], "manual-commit:t-box")
        done = seen[-1]
        self.assertTrue(done["success"], done)
        self.assertTrue(done["committed"], done)
        self.assertTrue(done["manual"], done)
        self.assertEqual(done["tabId"], "t-box")
        self.assertEqual(done["workDir"], str(self.box_work))
        self.assertTrue(done["message"].startswith("box: Committed: "), done)
        self.assertEqual(notes[-1]["message"], done["message"])
        self.assertEqual(notes[-1]["severity"], "info")
        self.assertIn("commitMessage", done)

        self.assertNotEqual(_head(self.box_work), box_before, "the box repo must have a new commit")
        self.assertEqual(_git(self.box_work, "status", "--porcelain").strip(), "")
        self.assertIn("new.txt", _git(self.box_work, "show", "--stat", "--format=%B", "HEAD"))
        self._assert_home_untouched(home_before)

    async def test_empty_folder_is_the_machines_own(self) -> None:
        (self.box_work / "f.txt").write_text("one\ntwo\n")
        box_before, home_before = _head(self.box_work), _head(self.home_work)

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-box", "workDir": "", "machine": "box",
        })

        done = seen[-1]
        self.assertTrue(done["committed"], done)
        self.assertEqual(done["workDir"], str(self.box_work))
        self.assertNotEqual(_head(self.box_work), box_before)
        self._assert_home_untouched(home_before)

    async def test_clean_machine_folder_is_a_noop(self) -> None:
        box_before, home_before = _head(self.box_work), _head(self.home_work)

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-box",
            "workDir": str(self.box_work), "machine": "box",
        })

        done = seen[-1]
        self.assertTrue(done["success"], done)
        self.assertFalse(done["committed"], done)
        self.assertEqual(done["message"], "box: Nothing to commit.")
        self.assertEqual(_head(self.box_work), box_before)
        self._assert_home_untouched(home_before)

    async def test_unknown_machine_is_reported(self) -> None:
        home_before = _head(self.home_work)

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-nowhere",
            "workDir": str(self.home_work), "machine": "nowhere",
        })

        done = seen[-1]
        self.assertFalse(done["success"], done)
        self.assertFalse(done["committed"], done)
        self.assertEqual(done["tabId"], "t-nowhere")
        self.assertEqual(done["message"], "no machine named 'nowhere'")
        notes = [m for m in seen if m.get("type") == "notification"]
        self.assertEqual(notes[-1]["severity"], "error")
        self.assertEqual(notes[-1]["message"], done["message"])
        self._assert_home_untouched(home_before)

    async def test_another_tabs_outcome_on_the_machine_is_not_taken(self) -> None:
        """A commit of another tab on the machine, by its own client, is not this tab's answer."""
        # A pre-commit hook makes the box's commit of this tab's folder
        # take seconds, during which another of the box's clients
        # commits a clean repo for another tab: the box broadcasts that
        # instant ``autocommit_done`` to its browsers while the link is
        # still waiting, and it must not pass for this tab's outcome.
        hook = self.box_work / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nsleep 3\n")
        hook.chmod(hook.stat().st_mode | 0o111)
        (self.box_work / "f.txt").write_text("one\ntwo\n")
        other = Path(self.tmpdir) / "box-other"
        other.mkdir()
        _make_repo(other)
        box_before = _head(self.box_work)

        await self._send({
            "type": "autocommitAction", "tabId": "t-box",
            "workDir": str(self.box_work), "machine": "box",
        })
        self.assertEqual((await self._next("notification"))["message"], "Committing on box…")
        await asyncio.to_thread(
            _as_another_client, self.remote.url,
            [{"type": "autocommitAction", "tabId": "t-other", "workDir": str(other)}],
            "autocommit_done",
        )
        done = await self._next("autocommit_done")

        self.assertEqual(done["tabId"], "t-box")
        self.assertTrue(done["committed"], done)
        self.assertTrue(done["message"].startswith("box: Committed: "), done)
        self.assertNotEqual(_head(self.box_work), box_before)
        self.assertEqual(_git(other, "status", "--porcelain").strip(), "")

    async def test_a_bound_tab_is_told_once(self) -> None:
        """A tab bound to the machine relays its tab-stamped events; the outcome comes once."""
        await self._send({
            "type": "resumeSession", "tabId": "t-box", "machine": "box", "id": "no-such-chat",
        })
        remote_tabs = self.server._server_api.remote_tabs
        for _ in range(200):
            if remote_tabs.machine_of("t-box") == "box":
                break
            await asyncio.sleep(0.05)
        self.assertEqual(remote_tabs.machine_of("t-box"), "box", "the tab is bound to the box")
        (self.box_work / "f.txt").write_text("one\ntwo\n")

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-box",
            "workDir": str(self.box_work), "machine": "box",
        })
        self.assertTrue(seen[-1]["committed"], seen[-1])
        # The relay would deliver the machine's tab-stamped toasts and
        # outcome on this connection too, a moment after the link's
        # reply: give it that moment, then check the whole sequence is
        # this daemon's alone.
        seen += await self._drain(1.5)
        toasts = [m["message"] for m in seen if m.get("type") == "notification"]
        self.assertEqual(
            toasts, ["Committing on box…", f"box: Committed: {_subject(self.box_work)}"],
        )
        dones = [m for m in seen if m.get("type") == "autocommit_done"]
        self.assertEqual(len(dones), 1, dones)

    async def test_a_commit_already_running_on_the_machine_is_reported(self) -> None:
        """The machine's own in-flight commit of the tab answers the link at once."""
        hook = self.box_work / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nsleep 3\n")
        hook.chmod(hook.stat().st_mode | 0o111)
        (self.box_work / "f.txt").write_text("one\ntwo\n")
        box_before = _head(self.box_work)
        # The box's own client starts the commit; its first toast
        # ("Auto-generating commit message…") says it is under way.
        await asyncio.to_thread(
            _as_another_client, self.remote.url,
            [{"type": "autocommitAction", "tabId": "t-box", "workDir": str(self.box_work)}],
            "notification",
        )

        seen = await self._commit_outcome({
            "type": "autocommitAction", "tabId": "t-box",
            "workDir": str(self.box_work), "machine": "box",
        })

        done = seen[-1]
        self.assertFalse(done["success"], done)
        self.assertEqual(done["message"], "box: A Git Commit of this tab is already running.")
        self.assertEqual(
            _head(self.box_work), box_before, "answered while the box's own commit still ran",
        )

    async def _drain(self, seconds: float) -> list[dict[str, Any]]:
        """Every message that arrives on the local connection within *seconds*."""
        deadline = asyncio.get_running_loop().time() + seconds
        seen: list[dict[str, Any]] = []
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                return seen
            try:
                line = await asyncio.wait_for(self.reader.readline(), timeout=left)
            except TimeoutError:
                return seen
            assert line, "local connection closed"
            seen.append(json.loads(line.decode("utf-8")))
