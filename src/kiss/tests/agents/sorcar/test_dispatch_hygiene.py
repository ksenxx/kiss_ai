# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for dispatch hygiene (WP7 of the cost levers).

* parent-repo paths in sub-agent task text are rewritten to the active
  worktree at dispatch (``run_agent`` / ``run_parallel``);
* the Bash guard's refusal suggests the rewritten command;
* ``run_agent`` with a generic name runs the plain sub-agent; a
  misspelled name gets a useful hint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    _run_agent,
    available_channels,
    resolve_agent,
)
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _bash_parent_repo_guard,
    rewrite_parent_repo_paths,
)


@pytest.fixture
def worktree(tmp_path: Path) -> tuple[str, str]:
    """A fake repo with a live ``.kiss-worktrees/kiss_wt-x`` worktree."""
    repo = tmp_path / "repo"
    wt = repo / ".kiss-worktrees" / "kiss_wt-x"
    wt.mkdir(parents=True)
    (repo / ".kiss-worktrees" / "kiss_wt-other").mkdir()
    return str(repo), str(wt)


class TestRewrite:
    def test_rewrites_repo_paths_but_not_other_worktrees(self, worktree) -> None:
        repo, wt = worktree
        text = (
            f"Edit {repo}/src/a.py, run `ls {repo}`; keep {repo}/.kiss-worktrees/kiss_wt-other/b "
            f"and {wt}/c and {repo}2/d (a different directory)."
        )
        out = rewrite_parent_repo_paths(text, wt)
        assert out == (
            f"Edit {wt}/src/a.py, run `ls {wt}`; keep {repo}/.kiss-worktrees/kiss_wt-other/b "
            f"and {wt}/c and {repo}2/d (a different directory)."
        )

    def test_no_worktree_means_no_change(self, worktree, tmp_path: Path) -> None:
        repo, _wt = worktree
        text = f"cat {repo}/x"
        assert rewrite_parent_repo_paths(text, repo) == text
        assert rewrite_parent_repo_paths(text, None) == text
        assert rewrite_parent_repo_paths(text, str(tmp_path / "elsewhere")) == text

    def test_bash_guard_suggests_rewritten_command(self, worktree) -> None:
        repo, wt = worktree
        err = _bash_parent_repo_guard(f"sed -n 1,5p {repo}/README.md", wt)
        assert err is not None
        assert err.endswith(f"Suggested command: sed -n 1,5p {wt}/README.md")
        assert _bash_parent_repo_guard(f"sed -n 1,5p {wt}/README.md", wt) is None

    def test_bash_tool_refusal_includes_suggestion(self, worktree) -> None:
        repo, wt = worktree
        tools = UsefulTools(work_dir=wt)
        out = tools.Bash(f"echo hi > {repo}/note.txt", "write in parent repo")
        assert "Suggested command:" in out and f"{wt}/note.txt" in out
        assert not (Path(repo) / "note.txt").exists()


class TestUnknownAgentHints:
    def test_generic_name_runs_the_plain_sub_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A generic label means what an empty ``agent`` means: the plain sub-agent.

        ``general``, ``Agent``, ``analysis``, ... resolve to the
        bundled ``sorcar_sea.py`` instead of erroring, so the dispatch
        reaches the daemon.  Both endpoint sources are pointed at a
        daemon that does not exist (an absent ``KISS_SORCAR_LOCAL``
        file, no endpoint recorded by an in-process cron scheduler), so
        the reply is the sorcar agent's could-not-run error — the same
        text an empty ``agent`` produces — not an unknown-agent hint.
        A reviewer name asks for a toolset, not an agent, and is refused
        with the spelling of that intent (U4).
        """
        from kiss.agents.sorcar import cron_agent

        monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
        monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
        expected = _run_agent("", "review it", "")
        assert expected.startswith("Error: the sorcar agent task could not run:")
        for name in ("general", "Agent", "sorcar", "analysis", "LLM"):
            assert resolve_agent(name, "") == (DEFAULT_AGENT_PATH, "sorcar")
            out = _run_agent("", "review it", name)
            assert out == expected
            assert "unknown agent" not in out and "Commands:" not in out
        for name in ("code-review", " Reviewer "):
            out = _run_agent("", "review it", name)
            assert out.startswith(f"Error: {name!r} is not an agent.")
            assert 'pass tool_profile="review"' in out
        # ``worker`` is a base class, not a generic label: naming it is the usual error.
        for name in ("worker", "subagent", "helper"):
            assert str(resolve_agent(name, "")).startswith(f"Error: unknown agent '{name}'")

    def test_misspelled_channel_gets_a_suggestion(self) -> None:
        channels = available_channels()
        if not channels:
            pytest.skip("no channel agents installed")
        target = channels[0]
        typo = target[:-1] + "x" if len(target) > 3 else target + "x"
        out = _run_agent("", "x", typo)
        assert out.startswith(f"Error: unknown agent {typo!r}")
        assert f"Did you mean {target!r}?" in out
        assert "Commands:" in out and target in out

    def test_nonsense_name_has_no_suggestion(self) -> None:
        out = _run_agent("", "x", "zzqqxxjjvv")
        assert "Did you mean" not in out and "Commands:" in out
