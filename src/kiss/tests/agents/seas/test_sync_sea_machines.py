# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test of ``/sync`` fanning out to the registered machines.

A real local daemon and a real remote daemon (``box``), both on the
scripted model of :mod:`kiss.tests.server.test_run_agent_machine`, and
three real git clones of one bare origin: ``mine`` (the local tab's
work dir), ``theirs`` (the box's checkout, with a commit of its own)
and a colleague's clone that pushed first.  ``/sync`` typed into the
local tab syncs ``mine``, sends ``/sync`` to the box as a sub-task
whose message says "this machine only", and syncs ``mine`` once more;
the test checks the git state of every clone and what each model was
told (the box, which has a machine registered too, is not asked to
fan out again).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.sync import sync_sea
from kiss.tests.agents.seas.conftest import (
    HERMETIC_GIT_WITH_IDENTITY,
    commit,
    git,
    remote_and_clones,
)
from kiss.tests.agents.seas.sea_contract import tool_names, tool_results
from kiss.tests.server.test_run_agent_machine import TwoDaemonsCase, _DelegatingModel, _finish


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    """A scripted answer: one tool call."""
    return {"pause": 0.0, "calls": [(name, arguments)]}


def _user_text(body: dict[str, Any]) -> str:
    """The task of a chat-completions request: its first user message without the
    ``# Task`` header the daemon puts above the prompt."""
    content = next(m["content"] for m in body["messages"] if m["role"] == "user")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content)
    return str(content).removeprefix("\n# Task\n").strip()


class TestSyncOnMachines(TwoDaemonsCase):
    """``/sync`` on the local tab, then on the box, then here again."""

    async def asyncSetUp(self) -> None:
        # The box's daemon is a child process: it inherits the identity for its
        # merge commit.  A cleanup, not asyncTearDown, restores the variables:
        # unittest skips the tear-down when the set-up fails.
        env = pytest.MonkeyPatch()
        self.addCleanup(env.undo)
        for key, value in HERMETIC_GIT_WITH_IDENTITY.items():
            env.setenv(key, value)
        await super().asyncSetUp()
        # The box has a machine of its own: the sub-task must not sync it.
        (self.remote.home / "machines.json").write_text(json.dumps({"machines": [
            {"name": "back", "url": "https://127.0.0.1:1", "password": "", "ssh": ""},
        ]}))

    def _clones(self) -> tuple[Path, Path, str, str]:
        """A bare origin with ``main``, clones ``mine``/``theirs``/``colleague``.

        Returns ``mine``, ``theirs``, the colleague's pushed commit and
        the box's own unpushed commit.
        """
        mine, theirs, colleague = remote_and_clones(
            Path(self.tmpdir), "mine", "theirs", "colleague")
        pushed = commit(colleague, "colleague", {"a.txt": "a\n"})
        git(colleague, "push", "-q", "origin", "main")
        own = commit(theirs, "box", {"b.txt": "b\n"})
        return mine, theirs, pushed, own

    def _box_has_opened(self, *dirs: Path) -> None:
        """Record *dirs* as directories the box's daemon has worked in (its ``config.json``).

        The sub-task sent to the box runs in the box daemon's own
        working directory, which is not a checkout; the box's agent
        finds the checkout among these.  Written after the daemon
        started: it reads the file on every lookup and rewrites it only
        when a tab adopts a directory, which no step of the test does.
        """
        config = self.remote.home / "config.json"
        data = json.loads(config.read_text()) if config.exists() else {}
        data["recent_work_dirs"] = [{"path": str(d), "ts": 1.0} for d in dirs]
        config.write_text(json.dumps(data))

    async def test_sync_runs_here_then_on_the_box_then_here_again(self) -> None:
        mine, theirs, pushed, own = self._clones()
        origin_url = git(mine, "remote", "get-url", "origin")
        self._box_has_opened(theirs)
        # The sub-task names the repository by URL, not by this machine's
        # path: the box keeps its checkout (``theirs``) elsewhere, and with
        # no work_dir option the sub-task runs in the box daemon's own
        # folder (``KISS_WORKDIR`` for the test's child daemon), which is
        # no checkout at all, as on a box whose daemon runs from $HOME.
        box_work_dir = os.environ["KISS_WORKDIR"]
        self.assertNotEqual(box_work_dir, str(theirs))
        box_task = f"origin/main of {origin_url} {sync_sea.THIS_MACHINE_ONLY}"
        _DelegatingModel.script = [
            _call("git_sync", repo=str(mine), ref="origin/main"),
            _call("run_agent", agent="sync", task=box_task, machine="box", max_budget="2"),
            # The box's agent: its work dir, the ref and the URL from its message.
            _call("git_sync", repo=box_work_dir, ref="origin/main", url=origin_url),
            _finish("<p>box: merged origin/main into main and pushed.</p>", pause=1.0),
            _call("git_sync", repo=str(mine), ref="origin/main"),
            _finish("<p>main fast-forwarded twice; box merged and pushed.</p>"),
        ]
        tab = "tab-sync"
        seen, child = await self._delegate(tab, prompt="/sync", work_dir=str(mine))
        self.assertTrue(child, f"no new_tab; the model saw {_DelegatingModel.tool_results}")
        result = next(m for m in seen if m["type"] == "result" and m.get("tabId") == tab)
        self.assertIn("box merged and pushed", json.dumps(result))

        # Git: mine fast-forwarded to the colleague's commit, the box
        # merged it into its own commit and pushed, mine got the merge.
        head = git(theirs, "rev-parse", "HEAD")
        self.assertEqual(git(theirs, "rev-parse", "HEAD^1"), own)
        self.assertEqual(git(theirs, "rev-parse", "HEAD^2"), pushed)
        self.assertEqual(git(mine, "rev-parse", "HEAD"), head)
        self.assertEqual(git(mine, "ls-remote", "origin", "refs/heads/main").split()[0], head)

        # The local model: the sync system prompt, the task listing the
        # box, git_sync and run_agent among its tools, and the reports.
        requests = _DelegatingModel.requests
        self.assertEqual(len(requests), 6, [_user_text(r)[:60] for r in requests])
        first = requests[0]
        self.assertTrue(first["messages"][0]["content"].startswith(sync_sea.SYSTEM_PROMPT))
        self.assertEqual(
            _user_text(first), f"{sync_sea.ADD_TO_PROMPT}\n\n{sync_sea.MACHINES_PROMPT}box.",
        )
        names = tool_names(first)
        self.assertLessEqual({"git_sync", "run_agent", "Bash", "Edit", "finish"}, names)
        reports = tool_results(requests[5])
        self.assertEqual(len(reports), 3, reports)
        self.assertIn(f"Fast-forwarded to origin/main: now at {pushed[:12]}", reports[0])
        self.assertIn("Budget: $0.0000/$4.00", reports[0], "one share per checkout")
        self.assertIn("machine=box", reports[1])
        self.assertIn("success: true", reports[1])
        self.assertIn("ran: sync ", reports[1])
        self.assertIn(f"Fast-forwarded to origin/main: now at {head[:12]}", reports[2])

        # The box's model: the same SEA on the box's checkout, told to
        # sync this machine only although it has a machine registered.
        box = requests[2]
        self.assertTrue(box["messages"][0]["content"].startswith(sync_sea.SYSTEM_PROMPT))
        self.assertEqual(_user_text(box), f"{box_task}\n\n{sync_sea.ADD_TO_PROMPT}")
        # The box's report: the checkout found from the URL, the merge, and
        # one checkout's share, not the $4 the box's own registry would give it.
        box_report = tool_results(requests[3])[0]
        self.assertTrue(box_report.startswith(
            f"{box_work_dir} is not a git checkout with a remote 'origin'; syncing the "
            f"checkout of that repository at {theirs} instead.\n"
            f"Checkout {theirs.resolve().as_posix()}, branch main, syncing with origin/main "
            f"(origin = {origin_url})."
        ), box_report)
        self.assertIn(f"Merged origin/main: now at {head[:12]}", box_report)
        self.assertIn("Budget: $0.0000/$2.00", box_report)
        transcript = next(
            m for m in seen if m["type"] == "task_events" and m.get("tabId") == child
        )
        settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
        self.assertEqual(settings["settings"]["sea"], "sync")
        self.assertEqual(settings["settings"]["tool_profile"], "shell+edit+agents")
        # No work dir was forced on the box: the sub-task ran in its daemon's,
        # the one its system prompt named and the scripted git_sync call used.
        self.assertEqual(
            os.path.realpath(settings["settings"]["work_dir"]), os.path.realpath(box_work_dir),
        )
        self.assertIn(settings["settings"]["work_dir"], box["messages"][0]["content"])
        self.assertIn(sync_sea.THIS_MACHINE_ONLY, transcript["task"])
        self.assertIn(origin_url, transcript["task"])
