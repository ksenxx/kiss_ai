# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A finished worktree task's stale ``work_dir`` must not be re-created.

Bug report: ``.kiss-worktrees/`` kept filling up with empty, unregistered
``kiss_wt-*`` directories whose birth time was *after* the owning task
had finished.  Two ``mkdir(parents=True)`` calls produced them:

* ``RelentlessAgent._reset`` creates ``self.work_dir`` at the start of
  every run.  The task-update side channel ("What has task X done so
  far") starts a run with the parent agent's ``work_dir`` — for a
  finished worktree task, the removed worktree directory.
* ``agent_dispatch.dispatch_result`` creates the ``run_agent`` sub-task's
  ``work_dir``, inherited from the calling task the same way.

Both now route the path through ``remap_vanished_worktree``: a path under
a torn-down ``.kiss-worktrees/kiss_wt-*`` directory becomes the parent
repository's equivalent, so the run happens where the merged work lives
and no husk appears.  Every test here uses a real git repository whose
worktree was really added and removed; the ``run_agent`` test goes
end-to-end through the tool and a local daemon stand-in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SLUG_PREFIX,
    _WORKTREE_SUBDIR,
)
from kiss.agents.sorcar.relentless_agent import RelentlessAgent, resolve_work_dir
from kiss.agents.sorcar.useful_tools import remap_vanished_worktree
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.agents.sorcar.test_dispatch_timeout import _LocalDaemon, _send_event


def _git(repo: Path, *args: str) -> None:
    """Run git in *repo*, asserting success."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"


@pytest.fixture
def torn_down_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A repo (with a ``sub/`` dir) whose worktree was added, then removed.

    Returns:
        ``(repo, wt)`` where ``wt`` no longer exists on disk.
    """
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "sub" / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    wt = repo / _WORKTREE_SUBDIR / f"{_WORKTREE_SLUG_PREFIX}1700000000-feedface"
    _git(repo, "worktree", "add", "-b", "kiss/wt-1700000000-feedface", str(wt))
    _git(repo, "worktree", "remove", "--force", str(wt))
    assert not wt.exists()
    return repo, wt


class TestRemapVanishedWorktree:
    """The shared remap helper."""

    def test_stale_path_maps_to_parent_repo(self, torn_down_worktree) -> None:
        repo, wt = torn_down_worktree
        assert remap_vanished_worktree(wt / "sub") == repo / "sub"
        assert remap_vanished_worktree(wt) == repo

    def test_existing_path_is_unchanged(self, torn_down_worktree) -> None:
        repo, _wt = torn_down_worktree
        assert remap_vanished_worktree(repo / "sub") == repo / "sub"

    def test_live_worktree_path_is_unchanged(self, torn_down_worktree) -> None:
        """A worktree that still exists is authoritative, not stale."""
        repo, _wt = torn_down_worktree
        live = repo / _WORKTREE_SUBDIR / f"{_WORKTREE_SLUG_PREFIX}1700000001-cafebabe"
        _git(repo, "worktree", "add", "-b", "kiss/wt-1700000001-cafebabe", str(live))
        assert remap_vanished_worktree(live / "sub") == live / "sub"
        # A not-yet-existing file inside a live worktree stays there too.
        assert remap_vanished_worktree(live / "new.txt") == live / "new.txt"

    def test_missing_path_outside_any_worktree_is_unchanged(self, tmp_path: Path) -> None:
        missing = tmp_path / "nowhere" / "deep"
        assert remap_vanished_worktree(missing) == missing

    def test_nested_vanished_worktrees_are_all_peeled(self, torn_down_worktree) -> None:
        """A repo inside a removed worktree, whose own worktree is removed too.

        Peeling only the innermost segment would return a path inside
        the (equally gone) outer worktree and resurrect that one.
        """
        repo, wt = torn_down_worktree
        inner = (
            wt / "child" / _WORKTREE_SUBDIR / f"{_WORKTREE_SLUG_PREFIX}1700000002-0badf00d"
        )
        assert remap_vanished_worktree(inner / "src") == repo / "child" / "src"


class TestResetDoesNotResurrectWorktree:
    """``resolve_work_dir`` / ``_reset`` land in the parent repo."""

    def test_resolve_work_dir_remaps_stale_worktree(self, torn_down_worktree) -> None:
        repo, wt = torn_down_worktree
        assert resolve_work_dir(str(wt / "sub")) == str((repo / "sub").resolve())
        assert resolve_work_dir(str(wt)) == str(repo.resolve())

    def test_resolve_work_dir_keeps_ordinary_paths(self, torn_down_worktree) -> None:
        repo, _wt = torn_down_worktree
        assert resolve_work_dir(str(repo / "sub")) == str((repo / "sub").resolve())

    def test_reset_with_stale_work_dir_creates_no_husk(self, torn_down_worktree) -> None:
        """The side-channel scenario: a run started with the removed path."""
        repo, wt = torn_down_worktree
        agent = RelentlessAgent("side-channel")
        agent._reset(
            model_name="m", max_sub_sessions=1, max_steps=1, max_budget=0.01,
            work_dir=str(wt), docker_image=None,
        )
        assert agent.work_dir == str(repo.resolve())
        assert not wt.exists(), "the removed worktree directory was re-created"
        assert sorted(p.name for p in (repo / _WORKTREE_SUBDIR).iterdir()) == []

    def test_failed_session_summary_after_teardown_creates_no_husk(
        self, torn_down_worktree,
    ) -> None:
        """A worktree removed mid-run must not come back as ``kiss_wt-*/tmp``.

        ``_summarize_failed_session`` writes the failed executor's
        trajectory under ``<work_dir>/tmp`` before summarizing.  The
        agent is reset into a LIVE worktree, the worktree is removed
        (a concurrent discard), and the summarizer runs with a model
        name that cannot resolve, so it takes its ``"Agent failed"``
        path without any LLM call — the trajectory write happens first
        either way.
        """
        repo, _wt = torn_down_worktree
        live = repo / _WORKTREE_SUBDIR / f"{_WORKTREE_SLUG_PREFIX}1700000003-deadc0de"
        _git(repo, "worktree", "add", "-b", "kiss/wt-1700000003-deadc0de", str(live))
        agent = RelentlessAgent("mid-run")
        agent._reset(
            model_name="no-such-model", max_sub_sessions=1, max_steps=1,
            max_budget=0.01, work_dir=str(live), docker_image=None,
        )
        assert agent.work_dir == str(live.resolve())
        _git(repo, "worktree", "remove", "--force", str(live))
        assert not live.exists()

        summary = agent._summarize_failed_session(KISSAgent("executor"), 0, RuntimeError("boom"))

        assert summary == "Agent failed: boom"
        assert not live.exists(), "the summarizer re-created the removed worktree"
        assert (repo / "tmp").is_dir(), "the trajectory scratch dir landed elsewhere"


class _RecordingDaemon(_LocalDaemon):
    """A daemon stand-in that answers the ``run`` at once and records it."""

    def __init__(self) -> None:
        super().__init__("kiss_no_husk_")

    async def _handle(self, ws: ServerConnection) -> None:
        tab_id = await self._read_run(ws)
        if tab_id is None:
            return
        await _send_event(ws, {"type": "status", "running": True, "tabId": tab_id})
        await _send_event(ws, {
            "type": "result", "tabId": tab_id, "taskId": "task-1", "success": True,
            "text": "done", "cost": "$0.0010", "total_tokens": 1, "step_count": 1,
        })
        await _send_event(ws, {"type": "status", "running": False, "tabId": tab_id})
        try:
            await ws.recv()
        except ConnectionClosed:
            pass


def test_run_agent_from_stale_worktree_creates_no_husk(
    torn_down_worktree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent`` dispatched from a finished worktree task runs in the repo.

    End-to-end through the real tool and the ``KISS_SORCAR_LOCAL``
    endpoint: the recorded ``run`` command carries the parent repo as
    ``workDir`` and the removed worktree directory stays absent.
    """
    repo, wt = torn_down_worktree
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _RecordingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    try:
        out = make_run_agent_tool(str(wt / "sub"))("say hi", str(script))
        assert yaml.safe_load(out) == {"success": True, "summary": "done"}
        assert daemon.run_cmd is not None, "the daemon stand-in saw no run command"
        assert daemon.run_cmd["workDir"] == str(repo / "sub")
    finally:
        daemon.close()
    assert not wt.exists(), "the removed worktree directory was re-created"
