# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: rsi7d's ``replay_in_clone`` / ``replay_in_place`` replay a past task through the daemon.

A real :class:`~kiss.server.web_server.RemoteAccessServer` on a temp
local WSS endpoint runs the replay; the model is a local HTTP stand-in
speaking the OpenAI wire format, reached through the daemon's
``custom_endpoint`` setting, so the child agent makes a genuine tool
call: it writes a file into its work dir, which must be the clone at
the task's base commit (``replay_in_clone``) or this task's directory
(``replay_in_place``) and never the task's repository.

Not covered: the "daemon's installed kiss package predates
``agent_dispatch.dispatch_result``" branch (``_legacy_daemon_error``).  It
is reachable only when the daemon runs an older installed package than
the checkout whose SEA file it loads (the VS Code extension's bundled
copy before a release), which a test in this checkout cannot arrange
without deleting the attribute from the imported module.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.git_worktree import USER_PROMPT_HEADING
from kiss.agents.sorcar.persistence import _add_task
from kiss.core import vscode_config
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    StandInModelServer,
    finish_response,
    run_git,
    tool_call_response,
)
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness

_DEMO_SEA = '''"""Demo SEA replayed in a clone."""

from kiss.agents.seas.base.base_sea import BaseSea

SYSTEM_PROMPT = "You are the demo agent. Do exactly what the task says and finish."


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        """Replace the default prompt."""
        return SYSTEM_PROMPT


'''


class ReplayInCloneTest(DaemonLocalHarness):
    """``replay_in_clone`` dispatches this checkout's SEA into a clone of the task's repo."""

    def setUp(self) -> None:
        # Every start/mutation registers its undo at once: cleanups run
        # LIFO even when a later step of setUp raises (tearDown does not).
        super().setUp()
        self.addCleanup(super().tearDown)
        saved_endpoint = cron_agent._daemon_endpoint_file
        # _dispatch goes through this daemon.
        cron_agent._daemon_endpoint_file = str(self.endpoint_file)
        self.addCleanup(setattr, cron_agent, "_daemon_endpoint_file", saved_endpoint)
        self.requests: list[dict[str, Any]] = []
        self.standin = StandInModelServer(self._respond)
        self.addCleanup(self.standin.stop)
        vscode_config.CONFIG_PATH.write_text(
            json.dumps({
                "custom_endpoint": self.standin.url,
                "custom_api_key": "kiss-test-key",
                "use_web_browser": False,
                "auto_classify": False,
                "auto_commit_mode": False,
                "is_worktree": False,
            }),
            encoding="utf-8",
        )
        # A fake KISS checkout with an editable demo SEA; rsi7d resolves it through the cwd.
        # Resolved because the SEA reports paths under os.getcwd(), which is the real path
        # (/private/var/... on macOS, where tempdirs live behind the /var symlink).
        self.checkout = (Path(self.tmpdir) / "checkout").resolve()
        seas = self.checkout / "src" / "kiss" / "agents" / "seas"
        (seas / "rsi7d").mkdir(parents=True)
        (seas / "demo").mkdir()
        shutil.copy(Path(sea.__file__), seas / "rsi7d" / "rsi7d_sea.py")
        (seas / "demo" / "demo_sea.py").write_text(_DEMO_SEA, encoding="utf-8")
        saved_cwd = os.getcwd()
        os.chdir(self.checkout)
        self.addCleanup(os.chdir, saved_cwd)

    def tearDown(self) -> None:
        """Nothing to do here: setUp registered every undo as a cleanup.

        The cleanups run LIFO (cwd, stand-in server, endpoint file, then
        the base harness's ``tearDown``, which removes the tmpdir the
        cwd pointed into), the order the explicit ``tearDown`` had.
        """

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """First step: write a marker file into the work dir; second step: finish."""
        self.requests.append(request)
        if any(m.get("role") == "tool" for m in request.get("messages", [])):
            return finish_response("replayed in the clone")
        return tool_call_response(
            "Bash",
            {"command": "printf 'replayed\\n' > replayed.txt", "description": "leave a mark"},
        )

    def test_replay_runs_the_seas_of_this_checkout_in_a_clone_at_the_tasks_commit(self) -> None:
        """The replay edits the clone at the pre-task commit and comes back with its task id."""
        repo = Path(self.repo)
        base = run_git(repo, "rev-parse", "HEAD").stdout.strip()
        task = f"Add a results section to {repo}/seed.txt"
        (repo / "seed.txt").write_text("seed\nresults\n", encoding="utf-8")
        run_git(repo, "add", "-A")
        run_git(repo, "commit", "-q", "-m", f"docs: add results{USER_PROMPT_HEADING}{task}")
        now_ms = int(time.time() * 1000)
        task_id, _chat = _add_task(
            task,
            extra={
                "model": "model-a", "work_dir": str(repo), "sea": "demo_sea",
                "startTs": now_ms - 60_000, "endTs": now_ms, "cost": 0.5, "steps": 3,
            },
        )

        out = json.loads(
            sea.replay_in_clone(task_id, max_budget=1.0, timeout=120, model=STANDIN_MODEL)
        )

        clone = Path(out["clone"])
        assert clone == self.checkout / sea.REPLAY_DIR / f"demo-{task_id[:8]}"
        assert out["commit"] == base and out["commit_source"].startswith("first parent")
        assert out["task"] == f"Add a results section to {clone}/seed.txt"
        assert (clone / "seed.txt").read_text(encoding="utf-8") == "seed\n"  # pre-task state
        assert (clone / "replayed.txt").read_text(encoding="utf-8") == "replayed\n"
        assert not (repo / "replayed.txt").exists()
        assert run_git(repo, "status", "--porcelain").stdout == ""  # the repo is untouched
        result = out["result"]
        assert result["success"] is True and "replayed in the clone" in result["summary"]
        assert result["steps"] >= 2

        assert out["replay_task_id"], out
        row = sea._task_row(out["replay_task_id"])
        assert row is not None, out
        assert row["task"] == out["task"] and row["work_dir"] == str(clone)
        assert row["model"] == STANDIN_MODEL and not row["is_worktree"]
        assert row["cost"] > 0 and row["steps"] == result["steps"]
        agentic = [r for r in self.requests if r.get("tools")]
        assert len(agentic) == 2, [list(r) for r in self.requests]
        system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
        assert str(system["content"]).startswith("You are the demo agent.")
        user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
        assert out["task"] in str(user["content"])

    def test_replay_of_a_plain_sorcar_run_uses_this_checkouts_system_prompt(self) -> None:
        """A run without a SEA replays as a plain task on the checkout's (patched) SYSTEM.md."""
        # The scope over KISS Sorcar itself needs the checkout to be a git repository.
        run_git(self.checkout, "init", "-q")
        marker = "You are the patched Sorcar prompt under test. Finish quickly."
        (self.checkout / "src" / "kiss" / "SYSTEM.md").write_text(
            "{{IDENTITY}}\n\n" + marker + "\n", encoding="utf-8"
        )
        repo = Path(self.repo)
        base = run_git(repo, "rev-parse", "HEAD").stdout.strip()
        # No auto-commit to go back from: the replay starts at the HEAD of the
        # task's start, so the task must have started after the seed commit.
        now_ms = int(time.time() * 1000)
        task_id, _chat = _add_task(
            f"Leave a mark in {repo}",
            extra={
                "model": "model-a", "work_dir": str(repo),
                "startTs": now_ms + 2_000, "endTs": now_ms + 5_000, "cost": 0.2, "steps": 2,
            },
        )

        out = json.loads(
            sea.replay_in_clone(task_id, max_budget=1.0, timeout=120, model=STANDIN_MODEL)
        )

        clone = Path(out["clone"])
        assert out["sea"] == sea.SORCAR and out["commit"] == base
        assert out["sea_file"] == str((self.checkout / "src" / "kiss" / "SYSTEM.md").resolve())
        assert clone == self.checkout / sea.REPLAY_DIR / f"sorcar-{task_id[:8]}"
        assert out["task"] == f"Leave a mark in {clone}"
        assert (clone / "replayed.txt").read_text(encoding="utf-8") == "replayed\n"
        assert not (repo / "replayed.txt").exists()
        assert out["result"]["success"] is True and out["replay_task_id"], out
        row = sea._task_row(out["replay_task_id"])
        assert row is not None and row["work_dir"] == str(clone) and not row["sea"], row
        agentic = [r for r in self.requests if r.get("tools")]
        assert len(agentic) == 2, [list(r) for r in self.requests]
        system = str(next(m for m in agentic[0]["messages"] if m["role"] == "system")["content"])
        assert system.startswith("You are KISS Sorcar") and marker in system, system[:300]
        assert "{{IDENTITY}}" not in system

    def test_replay_in_place_runs_this_checkouts_sea_here_on_the_seas_own_prompt(self) -> None:
        """No clone: the verbatim task runs in this directory on the SEA's prompt, fresh chat."""
        repo = Path(self.repo)
        task = "Say hello and finish"
        now_ms = int(time.time() * 1000)
        task_id, _chat = _add_task(
            task,
            extra={
                "model": "model-a", "work_dir": str(repo), "sea": "demo_sea",
                "startTs": now_ms - 60_000, "endTs": now_ms, "cost": 0.5, "steps": 3,
            },
        )
        assert sea.replay_in_place("no-such-task", max_budget=1.0).startswith(
            "Error: unknown task id"
        )
        assert sea.replay_in_place(task_id, max_budget=1.0, name="nosuchsea").startswith("Error")

        out = json.loads(
            sea.replay_in_place(task_id, max_budget=1.0, timeout=120, model=STANDIN_MODEL)
        )

        demo_file = self.checkout / "src" / "kiss" / "agents" / "seas" / "demo" / "demo_sea.py"
        assert out["sea"] == "demo" and out["sea_file"] == str(demo_file)
        assert out["work_dir"] == str(self.checkout) and out["task"] == task
        assert out["model"] == STANDIN_MODEL and "clone" not in out
        assert (self.checkout / "replayed.txt").read_text(encoding="utf-8") == "replayed\n"
        assert not (repo / "replayed.txt").exists()
        result = out["result"]
        assert result["success"] is True and "replayed in the clone" in result["summary"]
        assert out["replay_task_id"], out
        row = sea._task_row(out["replay_task_id"])
        assert row is not None, out
        assert row["task"] == task and row["work_dir"] == str(self.checkout)
        assert row["model"] == STANDIN_MODEL and not row["is_worktree"]
        agentic = [r for r in self.requests if r.get("tools")]
        assert len(agentic) == 2, [list(r) for r in self.requests]
        system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
        assert str(system["content"]).startswith("You are the demo agent.")
        user = str(next(m for m in agentic[0]["messages"] if m["role"] == "user")["content"])
        assert task in user
