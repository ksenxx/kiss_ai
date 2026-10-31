# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test of ``/sync`` fanning out to the registered machines.

A real local daemon and a real remote daemon (``box``), both on the
scripted model of :mod:`kiss.tests.server.test_run_agent_machine`, and
three real git clones of one bare origin: ``mine`` (the local tab's
work dir, with an uncommitted file), ``theirs`` (the box's checkout,
with a commit of its own and an uncommitted file) and a colleague's
clone that pushed first.  ``/sync`` typed into the local tab learns the
repository's URL with ``remote_url``, sends ``/sync`` to the box as a
sub-task whose message says "this machine only", syncs ``mine``, and
sends ``/sync`` to the box once more, so the box ends up with what
``mine`` pushed; the test checks the git state of every clone after
each step and what each model was told (the box, which has a machine
registered too, is not asked to fan out again).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.sync import sync_sea
from kiss.tests.agents.seas.conftest import commit, git, hermetic_git_env, remote_and_clones
from kiss.tests.agents.seas.sea_contract import tool_names, tool_result_text, tool_results
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
    """``/sync`` on the box, then on the local tab, then on the box again."""

    async def asyncSetUp(self) -> None:
        # The box's daemon is a child process: it inherits the identity for its
        # merge commit.  A cleanup, not asyncTearDown, restores the variables:
        # unittest skips the tear-down when the set-up fails.
        env = pytest.MonkeyPatch()
        self.addCleanup(env.undo)
        hermetic_git_env(env)
        await super().asyncSetUp()
        # The box has a machine of its own: the sub-task must not sync it.
        (self.remote.home / "machines.json").write_text(
            json.dumps(
                {
                    "machines": [
                        {"name": "back", "url": "https://127.0.0.1:1", "password": "", "ssh": ""},
                    ]
                }
            )
        )

    def _clones(self) -> tuple[Path, Path, str, str]:
        """A bare origin with ``main``, clones ``mine``/``theirs``/``colleague``.

        Returns ``mine``, ``theirs``, the colleague's pushed commit and
        the box's own unpushed commit; the box's checkout also has an
        uncommitted file, ``c.txt``, for its sync to commit, and
        ``mine`` an uncommitted ``d.txt`` for the local sync to commit.
        """
        mine, theirs, colleague = remote_and_clones(
            Path(self.tmpdir), "mine", "theirs", "colleague"
        )
        pushed = commit(colleague, "colleague", {"a.txt": "a\n"})
        git(colleague, "push", "-q", "origin", "main")
        own = commit(theirs, "box", {"b.txt": "b\n"})
        (theirs / "c.txt").write_text("left uncommitted on the box\n", encoding="utf-8")
        (mine / "d.txt").write_text("left uncommitted here\n", encoding="utf-8")
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

    async def test_sync_runs_on_the_box_then_here_then_on_the_box_again(self) -> None:
        mine, theirs, pushed, own = self._clones()
        seed = git(mine, "rev-parse", "HEAD")
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
        box_sync = _call("git_sync", repo=box_work_dir, ref="origin/main", url=origin_url)
        to_box = _call("run_agent", agent="sync", task=box_task, machine="box", max_budget="2")
        _DelegatingModel.script = [
            # The machines come first, so the URL the sub-task names comes
            # from remote_url, not from a git_sync report.
            _call("remote_url", repo=str(mine), ref="origin/main"),
            to_box,
            # The box's agent: its work dir, the ref and the URL from its message.
            box_sync,
            _finish("<p>box: committed c.txt, merged origin/main into main and pushed.</p>"),
            _call("git_sync", repo=str(mine), ref="origin/main"),
            to_box,
            box_sync,
            _finish("<p>box: fast-forwarded to origin/main.</p>"),
            _finish("<p>box synced twice; main committed d.txt, merged and pushed.</p>"),
        ]
        tab = "tab-sync"
        seen, child = await self._delegate(tab, prompt="/sync", work_dir=str(mine))
        self.assertTrue(child, f"no new_tab; the model saw {_DelegatingModel.tool_results}")
        result = next(m for m in seen if m["type"] == "result" and m.get("tabId") == tab)
        self.assertIn("box synced twice", json.dumps(result))
        # Two sub-tasks ran on the box, one before this checkout's sync and one after.
        spawned = [m for m in seen if m["type"] == "new_tab" and m.get("parent_tab_id") == tab]
        self.assertEqual(len(spawned), 2, spawned)
        self.assertNotEqual(spawned[0]["task_id"], spawned[1]["task_id"])

        # Git, in the order the syncs ran.  First the box: it committed its
        # uncommitted c.txt on top of its own commit, merged the colleague's
        # commit in and pushed that merge.  Then mine: it committed d.txt,
        # merged the box's merge in and pushed.  Then the box again: it
        # fast-forwarded to mine's merge.  Every clone and origin end at it.
        head = git(mine, "rev-parse", "HEAD")
        local = git(mine, "rev-parse", "HEAD^1")
        self.assertEqual(git(mine, "rev-parse", f"{local}^"), seed)
        self.assertEqual(git(mine, "show", "--format=", "--name-only", local), "d.txt")
        box_merge = git(mine, "rev-parse", "HEAD^2")
        gathered = git(mine, "rev-parse", f"{box_merge}^1")
        self.assertEqual(git(mine, "rev-parse", f"{gathered}^"), own)
        self.assertEqual(git(mine, "show", "--format=", "--name-only", gathered), "c.txt")
        self.assertEqual(git(mine, "rev-parse", f"{box_merge}^2"), pushed)
        self.assertEqual(git(theirs, "rev-parse", "HEAD"), head)
        self.assertEqual(git(theirs, "status", "--porcelain"), "")
        self.assertEqual(git(mine, "status", "--porcelain"), "")
        self.assertEqual(
            (mine / "c.txt").read_text(encoding="utf-8"), "left uncommitted on the box\n"
        )
        self.assertEqual((theirs / "d.txt").read_text(encoding="utf-8"), "left uncommitted here\n")
        self.assertEqual(git(mine, "ls-remote", "origin", "refs/heads/main").split()[0], head)

        # The local model: the sync system prompt, the task listing the
        # box, remote_url, git_sync and run_agent among its tools, and the
        # reports in the order of the rounds: URL, box, here, box.
        requests = _DelegatingModel.requests
        self.assertEqual(len(requests), 9, [_user_text(r)[:60] for r in requests])
        first = requests[0]
        self.assertTrue(first["messages"][0]["content"].startswith(sync_sea.SYSTEM_PROMPT))
        self.assertEqual(
            _user_text(first),
            f"{sync_sea.ADD_TO_PROMPT}\n\n{sync_sea.MACHINES_PROMPT}box.",
        )
        names = tool_names(first)
        self.assertLessEqual(
            {"remote_url", "git_sync", "run_agent", "Bash", "Edit", "finish"}, names
        )
        reports = tool_results(requests[8])
        self.assertEqual(len(reports), 4, reports)
        self.assertEqual(tool_result_text(reports[0]), origin_url)
        self.assertIn(
            "Budget: $0.0000/$6.00", reports[0], "one share here, two per machine"
        )
        for report in (reports[1], reports[3]):
            self.assertIn("machine=box", report)
            self.assertIn("success: true", report)
            self.assertIn("ran: sync ", report)
        self.assertIn("committed c.txt, merged origin/main", reports[1])
        self.assertIn(
            f"Committed 1 uncommitted change(s) on main as {local[:12]}: d.txt", reports[2]
        )
        self.assertIn(f"Merged origin/main: now at {head[:12]}", reports[2])
        self.assertIn("Pushed main to origin:", reports[2])
        self.assertIn("fast-forwarded to origin/main", reports[3])

        # The box's model, both rounds: the same SEA on the box's checkout,
        # told to sync this machine only although it has a machine registered.
        for box in (requests[2], requests[6]):
            self.assertTrue(box["messages"][0]["content"].startswith(sync_sea.SYSTEM_PROMPT))
            self.assertEqual(_user_text(box), f"{box_task}\n\n{sync_sea.ADD_TO_PROMPT}")
        # The box's first report: the checkout found from the URL, the commit
        # and the merge, and one checkout's share, not the $6 the box's own
        # registry would give it.
        found = (
            f"{box_work_dir} is not a git checkout with a remote 'origin'; syncing the "
            f"checkout of that repository at {theirs} instead.\n"
            f"Checkout {theirs.resolve().as_posix()}, branch main, syncing with origin/main "
            f"(origin = {origin_url})."
        )
        box_report = tool_results(requests[3])[0]
        self.assertTrue(box_report.startswith(found), box_report)
        self.assertIn(
            f"Committed 1 uncommitted change(s) on main as {gathered[:12]}: c.txt\n", box_report
        )
        self.assertIn(f"Merged origin/main: now at {box_merge[:12]}", box_report)
        self.assertIn("Budget: $0.0000/$2.00", box_report)
        # The box's second report: a fast-forward to what mine pushed, nothing to push.
        box_report = tool_results(requests[7])[0]
        self.assertTrue(box_report.startswith(found), box_report)
        self.assertNotIn("Committed", box_report)
        self.assertIn(f"Fast-forwarded to origin/main: now at {head[:12]}", box_report)
        self.assertIn("Pushed main to origin: Everything up-to-date", box_report)
        self.assertIn("Budget: $0.0000/$2.00", box_report)
        transcript = next(m for m in seen if m["type"] == "task_events" and m.get("tabId") == child)
        settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
        self.assertEqual(settings["settings"]["sea"], "sync")
        self.assertEqual(settings["settings"]["tool_profile"], "shell+edit+agents")
        # No work dir was forced on the box: the sub-task ran in its daemon's,
        # the one its system prompt named and the scripted git_sync call used.
        self.assertEqual(
            os.path.realpath(settings["settings"]["work_dir"]),
            os.path.realpath(box_work_dir),
        )
        self.assertIn(settings["settings"]["work_dir"], requests[2]["messages"][0]["content"])
        self.assertIn(sync_sea.THIS_MACHINE_ONLY, transcript["task"])
        self.assertIn(origin_url, transcript["task"])
