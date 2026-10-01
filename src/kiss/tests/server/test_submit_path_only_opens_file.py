# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A path-only ``submit`` opens the file, not a task — on every surface.

The daemon owns the one submit path: the remote webapp sends its
webview's ``submit`` over WSS, the VS Code extension host forwards its
webview's ``submit`` over the local WSS endpoint, and ``_handle_submit``
classifies both.  A prompt that is nothing but the path of an existing
regular file is a request to open that file: the submitting connection
gets ``promptOpened`` and then the file — as ``openResolvedFile`` (the
resolved path, which the extension host opens in a real editor tab) for
a local client, as ``fileContent`` for a browser — and no task starts.

These tests drive a real ``RemoteAccessServer`` over its local WSS
endpoint and over a remote WSS connection and assert the daemon's reply frames.  No test
makes a paid LLM call: the prompts that must still start a task use a
model name absent from ``get_available_models()``, so the worker
returns at ``task_runner``'s "No model available" guard.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from websockets.asyncio.client import connect

from kiss.server import agent_state
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_server_liveness_and_run_refusals import (
    _UNAVAILABLE_MODEL,
    _no_verify_ssl,
    _ServerHarness,
)

_OPEN_REPLIES = ("openResolvedFile", "fileContent")


def _submit_cmd(prompt: str, tab_id: str, work_dir: str = "") -> dict[str, Any]:
    """The ``submit`` frame ``media/main.js`` sends for *prompt*."""
    cmd: dict[str, Any] = {
        "type": "submit",
        "prompt": prompt,
        "tabId": tab_id,
        "model": _UNAVAILABLE_MODEL,
    }
    if work_dir:
        cmd["workDir"] = work_dir
    return cmd


def _is_open_reply_or_task_end(msg: dict[str, Any], tab_id: str) -> bool:
    """Stop condition: the file reply arrived, or the tab's task ended."""
    return msg.get("type") in _OPEN_REPLIES or (
        msg.get("type") == "status"
        and msg.get("running") is False
        and msg.get("tabId") == tab_id
    )


class TestPathOnlySubmitOverLocal(_ServerHarness):
    """A VS Code window's ``submit`` (local) is answered ``openResolvedFile``."""

    async def _submit(
        self,
        writer: LocalWriter,
        prompt: str,
        tab_id: str,
        work_dir: str = "",
    ) -> None:
        await self._send(writer, _submit_cmd(prompt, tab_id, work_dir))

    async def _frames_after_submit(
        self, reader: LocalReader, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Frames up to the file reply or the task's end."""
        return await self._collect_frames(
            reader, lambda m: _is_open_reply_or_task_end(m, tab_id),
        )

    async def test_relative_file_path_opens_without_a_task(self) -> None:
        """The exact prompt of the reported task opens the file natively."""
        target = self.work_dir / "src" / "seas" / "review_paper_sea.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('review paper SEA')\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-path-only"

        await self._submit(writer, "./src/seas/review_paper_sea.py", tab_id)
        frames = await self._frames_after_submit(reader, tab_id)

        opens = [f for f in frames if f.get("type") == "openResolvedFile"]
        self.assertEqual(
            len(opens), 1,
            f"BUG: a path-only submit did not open the file: {frames}",
        )
        reply = opens[0]
        self.assertEqual(reply["path"], str(target.resolve()))
        self.assertEqual(reply["tabId"], tab_id)
        self.assertNotIn("error", reply)
        self.assertNotIn(
            "content", reply,
            "a VS Code window opens the path itself; it must not be sent "
            "the file's content",
        )
        self.assertFalse(
            [f for f in frames if f.get("type") == "fileContent"],
            f"a local client must never get a browser fileContent: {frames}",
        )
        self.assertFalse(
            [f for f in frames if f.get("type") == "status"],
            f"a path-only submit must not touch the running state: {frames}",
        )
        types = [f.get("type") for f in frames]
        self.assertLess(
            types.index("promptOpened"), types.index("openResolvedFile"),
            "the webview must be told the prompt opened a file (and lift "
            f"its task claim) before the host opens it: {types}",
        )
        acks = [f for f in frames if f.get("type") == "promptOpened"]
        self.assertEqual(acks[0].get("tabId"), tab_id)
        self.assertIsNone(
            agent_state.find_by_tab(tab_id),
            "BUG: the path-only prompt started an agent task",
        )

    async def test_absolute_and_quoted_paths_open(self) -> None:
        """Absolute, ``~``-free, whitespace-padded prompts open too."""
        target = self.work_dir / "notes.md"
        target.write_text("# notes\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-abs-path"

        await self._submit(writer, f"  {target}  ", tab_id)
        frames = await self._frames_after_submit(reader, tab_id)

        opens = [f for f in frames if f.get("type") == "openResolvedFile"]
        self.assertEqual(len(opens), 1, f"no openResolvedFile in {frames}")
        self.assertEqual(opens[0]["path"], str(target.resolve()))
        self.assertIsNone(agent_state.find_by_tab(tab_id))

    async def test_directory_prompt_still_starts_a_task(self) -> None:
        """A one-word prompt naming a folder (``src``) is a task."""
        (self.work_dir / "src").mkdir()
        reader, writer = await self._connect()
        tab_id = "tab-dir-prompt"

        await self._submit(writer, "src", tab_id)
        frames = await self._frames_after_submit(reader, tab_id)

        self.assertFalse(
            [f for f in frames if f.get("type") in _OPEN_REPLIES],
            f"a directory name must not open as a file: {frames}",
        )
        self.assertTrue(
            any(
                f.get("type") == "status" and f.get("running") is True
                and f.get("tabId") == tab_id
                for f in frames
            ),
            f"the directory-named prompt did not start a task: {frames}",
        )
        self.assertIsNotNone(agent_state.find_by_tab(tab_id))

    async def test_missing_path_and_multiline_prompt_start_tasks(self) -> None:
        """A path that does not exist, or a multi-line prompt, runs."""
        target = self.work_dir / "exists.txt"
        target.write_text("x\n", encoding="utf-8")
        reader, writer = await self._connect()

        await self._submit(writer, "./no/such/file.py", "tab-missing")
        frames = await self._frames_after_submit(reader, "tab-missing")
        self.assertFalse([f for f in frames if f.get("type") in _OPEN_REPLIES])
        self.assertIsNotNone(agent_state.find_by_tab("tab-missing"))

        await self._submit(writer, "exists.txt\nand then edit it", "tab-multi")
        frames = await self._frames_after_submit(reader, "tab-multi")
        self.assertFalse([f for f in frames if f.get("type") in _OPEN_REPLIES])
        self.assertIsNotNone(agent_state.find_by_tab("tab-multi"))

    async def test_running_tab_treats_the_path_as_a_follow_up(self) -> None:
        """A path typed into a tab whose task runs steers that task.

        The webview sends a running tab's prompt as ``appendUserMessage``
        when it knows the tab runs; a ``submit`` that races the status
        must be treated the same by the daemon, or a follow-up that
        happens to name a file — "look at notes.md" typed as just
        ``notes.md`` — would open the file and never reach the worker.
        """
        target = self.work_dir / "notes.md"
        target.write_text("# notes\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-busy"
        state, _exited = self._register_starting_task(tab_id, "task-busy")

        await self._submit(writer, "notes.md", tab_id)
        frames = await self._collect_frames(
            reader,
            lambda m: m.get("type") in (*_OPEN_REPLIES, "promptOpened")
            or (m.get("type") == "prompt" and m.get("tabId") == tab_id),
            timeout=5.0,
        )

        self.assertFalse(
            [
                f for f in frames
                if f.get("type") in (*_OPEN_REPLIES, "promptOpened")
            ],
            f"BUG: a running tab's follow-up was opened as a file: {frames}",
        )
        self.assertEqual(
            list(state.pending_user_messages), ["notes.md"],
            "the path must be queued for the running task as a follow-up",
        )

    async def test_path_resolves_against_the_submit_work_dir(self) -> None:
        """A tab pinned to another folder opens files relative to it."""
        other = self.work_dir.parent / "other"
        other.mkdir()
        target = other / "pinned.txt"
        target.write_text("pinned\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-pinned"

        await self._submit(writer, "pinned.txt", tab_id, work_dir=str(other))
        frames = await self._frames_after_submit(reader, tab_id)

        opens = [f for f in frames if f.get("type") == "openResolvedFile"]
        self.assertEqual(len(opens), 1, f"no openResolvedFile in {frames}")
        self.assertEqual(opens[0]["path"], str(target.resolve()))


class TestPathOnlySubmitOverWss(_ServerHarness):
    """A browser's ``submit`` (WSS) is answered with the ``fileContent``."""

    async def _frames_after_submit(
        self, ws: Any, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Frames up to the file reply or the task's end (10 s cap)."""
        frames: list[dict[str, Any]] = []
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 10
        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
            except TimeoutError:
                break
            msg = json.loads(raw)
            frames.append(msg)
            if _is_open_reply_or_task_end(msg, tab_id):
                break
        return frames

    async def test_relative_file_path_opens_as_content(self) -> None:
        """The same prompt over WSS opens the file in a content tab."""
        target = self.work_dir / "src" / "seas" / "review_paper_sea.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('review paper SEA')\n", encoding="utf-8")
        tab_id = "tab-path-only-web"

        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(auth.get("type"), "auth_ok")
            await ws.send(
                json.dumps(_submit_cmd("./src/seas/review_paper_sea.py", tab_id))
            )
            frames = await self._frames_after_submit(ws, tab_id)

        contents = [f for f in frames if f.get("type") == "fileContent"]
        self.assertEqual(
            len(contents), 1,
            f"BUG: a browser's path-only submit did not open the file: {frames}",
        )
        reply = contents[0]
        self.assertEqual(reply["path"], str(target.resolve()))
        self.assertEqual(reply["name"], "review_paper_sea.py")
        self.assertEqual(reply["tabId"], tab_id)
        self.assertEqual(reply["content"], "print('review paper SEA')\n")
        self.assertIn("version", reply, "the opened file must be editable")
        self.assertFalse(
            [f for f in frames if f.get("type") == "openResolvedFile"],
            f"a browser has no editor for a resolved path: {frames}",
        )
        self.assertFalse(
            [f for f in frames if f.get("type") == "status"],
            f"a path-only submit must not touch the running state: {frames}",
        )
        types = [f.get("type") for f in frames]
        self.assertLess(types.index("promptOpened"), types.index("fileContent"))
        self.assertIsNone(agent_state.find_by_tab(tab_id))
