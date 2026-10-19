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

from kiss.agents.seas.sync import sync_sea
from kiss.tests.agents.seas.conftest import HERMETIC_GIT_WITH_IDENTITY, commit, git
from kiss.tests.server.test_run_agent_machine import TwoDaemonsCase, _DelegatingModel, _finish


def _restore_env(saved: dict[str, str | None]) -> None:
    """Put the variables of *saved* back; a ``None`` value removes the variable."""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
        self.addCleanup(
            _restore_env, {key: os.environ.get(key) for key in HERMETIC_GIT_WITH_IDENTITY},
        )
        os.environ.update(HERMETIC_GIT_WITH_IDENTITY)
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
        root = Path(self.tmpdir)
        origin = root / "origin.git"
        git(root, "init", "-q", "--bare", "-b", "main", str(origin))
        seed = root / "seed"
        git(root, "clone", "-q", str(origin), str(seed))
        git(seed, "checkout", "-q", "-b", "main")
        commit(seed, "seed", {"README.md": "hello\n"})
        git(seed, "push", "-q", "origin", "main")
        mine, theirs, colleague = (root / name for name in ("mine", "theirs", "colleague"))
        for clone in (mine, theirs, colleague):
            git(root, "clone", "-q", str(origin), str(clone))
        pushed = commit(colleague, "colleague", {"a.txt": "a\n"})
        git(colleague, "push", "-q", "origin", "main")
        own = commit(theirs, "box", {"b.txt": "b\n"})
        return mine, theirs, pushed, own

    async def test_sync_runs_here_then_on_the_box_then_here_again(self) -> None:
        mine, theirs, pushed, own = self._clones()
        local_task = f"origin/main {sync_sea.THIS_MACHINE_ONLY}"
        _DelegatingModel.script = [
            _call("git_sync", repo=str(mine), ref="origin/main"),
            _call("run_agent", agent="sync", task=local_task, machine="box",
                  max_budget="2", options=json.dumps({"work_dir": str(theirs)})),
            _call("git_sync", repo=str(theirs), ref="origin/main"),  # the box's agent
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
        names = {t["function"]["name"] for t in first["tools"]}
        self.assertLessEqual({"git_sync", "run_agent", "Bash", "Edit", "finish"}, names)
        reports = [str(m["content"]) for m in requests[5]["messages"] if m["role"] == "tool"]
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
        self.assertEqual(_user_text(box), f"{local_task}\n\n{sync_sea.ADD_TO_PROMPT}")
        # One checkout's share, not the $4 the box's own registry would give it.
        box_report = next(str(m["content"]) for m in requests[3]["messages"] if m["role"] == "tool")
        self.assertIn(f"Merged origin/main: now at {head[:12]}", box_report)
        self.assertIn("Budget: $0.0000/$2.00", box_report)
        transcript = next(
            m for m in seen if m["type"] == "task_events" and m.get("tabId") == child
        )
        settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
        self.assertEqual(settings["settings"]["sea"], "sync")
        self.assertEqual(settings["settings"]["tool_profile"], "shell+edit+agents")
        self.assertEqual(settings["settings"]["work_dir"], str(theirs))
        self.assertIn(sync_sea.THIS_MACHINE_ONLY, transcript["task"])
