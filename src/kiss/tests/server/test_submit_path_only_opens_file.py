# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A path-only ``submit`` from the remote webapp opens the file, not a task.

The VS Code extension host turns a prompt that is nothing but the path
of an existing regular file into an editor open
(``SorcarSidebarView`` ``case 'submit'``, tested by
``test/submitWorktreePathOpen.test.js``).  The remote webapp's
``submit`` reaches the daemon instead, whose ``_handle_submit`` used to
build a ``run`` for every prompt: typing
``./src/kiss/agents/seas/review_paper/review_paper_sea.py`` launched a
real agent task (classifier, worktree, an ``xdg-open`` directive that a
headless daemon cannot honour).

These tests drive a real ``RemoteAccessServer`` over its UDS and assert
the daemon's reply frames, so they exercise the same dispatch path a
browser's WSS ``submit`` takes.  No test makes a paid LLM call: the
prompts that must still start a task use a model name absent from
``get_available_models()``, so the worker returns at ``task_runner``'s
"No model available" guard.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kiss.server import agent_state
from kiss.tests.server.test_server_liveness_and_run_refusals import (
    _UNAVAILABLE_MODEL,
    _ServerHarness,
)


class TestPathOnlySubmitOpensFile(_ServerHarness):
    """``submit`` whose prompt is an existing file replies ``fileContent``."""

    async def _submit(
        self,
        writer: asyncio.StreamWriter,
        prompt: str,
        tab_id: str,
        work_dir: str = "",
    ) -> None:
        cmd: dict[str, Any] = {
            "type": "submit",
            "prompt": prompt,
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        }
        if work_dir:
            cmd["workDir"] = work_dir
        await self._send(writer, cmd)

    async def _frames_after_submit(
        self, reader: asyncio.StreamReader, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Frames up to the ``fileContent`` reply or the task's end."""
        return await self._collect_frames(
            reader,
            lambda m: m.get("type") == "fileContent"
            or (
                m.get("type") == "status"
                and m.get("running") is False
                and m.get("tabId") == tab_id
            ),
        )

    async def test_relative_file_path_opens_without_a_task(self) -> None:
        """The exact prompt of the reported task opens the file."""
        target = self.work_dir / "src" / "seas" / "review_paper_sea.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('review paper SEA')\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-path-only"

        await self._submit(writer, "./src/seas/review_paper_sea.py", tab_id)
        frames = await self._frames_after_submit(reader, tab_id)

        contents = [f for f in frames if f.get("type") == "fileContent"]
        self.assertEqual(
            len(contents), 1,
            f"BUG: a path-only submit did not open the file: {frames}",
        )
        reply = contents[0]
        self.assertEqual(reply["path"], str(target.resolve()))
        self.assertEqual(reply["name"], "review_paper_sea.py")
        self.assertEqual(reply["tabId"], tab_id)
        self.assertEqual(reply["content"], "print('review paper SEA')\n")
        self.assertIn("version", reply, "the opened file must be editable")
        self.assertFalse(
            [f for f in frames if f.get("type") == "status"],
            f"a path-only submit must not touch the running state: {frames}",
        )
        types = [f.get("type") for f in frames]
        self.assertLess(
            types.index("promptOpened"), types.index("fileContent"),
            "the webview must be told the prompt opened a file (and lift "
            f"its task claim) before the file arrives: {types}",
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

        contents = [f for f in frames if f.get("type") == "fileContent"]
        self.assertEqual(len(contents), 1, f"no fileContent in {frames}")
        self.assertEqual(contents[0]["path"], str(target.resolve()))
        self.assertIsNone(agent_state.find_by_tab(tab_id))

    async def test_directory_prompt_still_starts_a_task(self) -> None:
        """A one-word prompt naming a folder (``src``) is a task."""
        (self.work_dir / "src").mkdir()
        reader, writer = await self._connect()
        tab_id = "tab-dir-prompt"

        await self._submit(writer, "src", tab_id)
        frames = await self._frames_after_submit(reader, tab_id)

        self.assertFalse(
            [f for f in frames if f.get("type") == "fileContent"],
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
        self.assertFalse([f for f in frames if f.get("type") == "fileContent"])
        self.assertIsNotNone(agent_state.find_by_tab("tab-missing"))

        await self._submit(writer, "exists.txt\nand then edit it", "tab-multi")
        frames = await self._frames_after_submit(reader, "tab-multi")
        self.assertFalse([f for f in frames if f.get("type") == "fileContent"])
        self.assertIsNotNone(agent_state.find_by_tab("tab-multi"))

    async def test_running_tab_treats_the_path_as_a_follow_up(self) -> None:
        """A path typed into a tab whose task runs steers that task.

        The extension host gives its running tabs precedence over the
        shortcut (``_runningTabs`` is checked first); the daemon must do
        the same, or a follow-up that happens to name a file — "look at
        notes.md" typed as just ``notes.md`` — would open the file and
        never reach the worker.
        """
        target = self.work_dir / "notes.md"
        target.write_text("# notes\n", encoding="utf-8")
        reader, writer = await self._connect()
        tab_id = "tab-busy"
        state, _exited = self._register_starting_task(tab_id, "task-busy")

        await self._submit(writer, "notes.md", tab_id)
        frames = await self._collect_frames(
            reader,
            lambda m: m.get("type") in ("fileContent", "promptOpened")
            or (m.get("type") == "prompt" and m.get("tabId") == tab_id),
            timeout=5.0,
        )

        self.assertFalse(
            [f for f in frames if f.get("type") in ("fileContent", "promptOpened")],
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

        contents = [f for f in frames if f.get("type") == "fileContent"]
        self.assertEqual(len(contents), 1, f"no fileContent in {frames}")
        self.assertEqual(contents[0]["path"], str(target.resolve()))
        self.assertEqual(contents[0]["content"], "pinned\n")
