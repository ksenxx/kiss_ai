# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/sync`` agent (:mod:`kiss.agents.seas.sync`).

The git tests run :func:`kiss.agents.seas.sync.git_sync.sync` against
real repositories: a bare "origin" and clones of it under ``tmp_path``,
with other clones playing the colleagues whose pushes the sync has to
merge.  The agent-level test runs a real :class:`ChatSorcarAgent`
ReAct loop against the scripted local chat-completions server
(:mod:`kiss.tests.agents.sorcar.local_model_server`) configured as the
daemon configures it from the SEA's methods; the only replaced
boundary is the LLM endpoint, the ``git_sync`` tool really fetches,
merges and pushes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync import git_sync, sync_sea
from kiss.agents.seas.sync.git_sync import GitSyncError, sync
from kiss.agents.seas.sync.sync_sea import SyncSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import resolve_tool_profile
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters, system_message
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sync_sea.__file__).resolve()

# A fixed identity and no user or system git config: the same settings
# for the test's own git calls and for the ones :func:`sync` spawns.
# Otherwise a machine without a global ``user.name`` (a fresh CI VM)
# fails the merge commit with "Committer identity unknown", and Git for
# Windows' system ``core.autocrlf=true`` against the user's ``false``
# makes the two sides disagree on which files are modified.
_HERMETIC_GIT = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

_GIT_ENV = {**os.environ, **_HERMETIC_GIT}


@pytest.fixture(autouse=True)
def _hermetic_git_for_the_product(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the git commands :func:`sync` spawns with :data:`_HERMETIC_GIT` too."""
    for key, value in _HERMETIC_GIT.items():
        monkeypatch.setenv(key, value)


def _git(repo: Path, *args: str) -> str:
    """Run ``git *args`` in *repo* (must succeed) and return its stripped stdout."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=_GIT_ENV,
    ).stdout.strip()


def _commit(repo: Path, name: str, text: str) -> str:
    """Write *text* to *name* in *repo*, commit it and return the commit sha."""
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", f"{name}: {text.strip()[:20]}")
    return _git(repo, "rev-parse", "HEAD")


def _remote_and_clones(tmp_path: Path, *names: str) -> list[Path]:
    """Create a bare ``origin`` with one commit on ``main`` and return one clone per name."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    _git(seed, "checkout", "-q", "-b", "main")
    _commit(seed, "README.md", "hello\n")
    _git(seed, "push", "-q", "origin", "main")
    clones = []
    for name in names:
        clone = tmp_path / name
        _git(tmp_path, "clone", "-q", str(origin), str(clone))
        clones.append(clone)
    return clones


def _remote_main(repo: Path) -> str:
    """Return the sha ``origin``'s ``main`` is at, as *repo* sees it after a fresh fetch."""
    return _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]


def _pre_push_hook(repo: Path, body: str) -> None:
    """Install a ``pre-push`` hook running the shell *body* in *repo*."""
    hooks = Path(_git(repo, "rev-parse", "--git-path", "hooks"))
    if not hooks.is_absolute():
        hooks = repo / hooks
    hooks.mkdir(exist_ok=True)
    hook = hooks / "pre-push"
    hook.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    hook.chmod(0o755)


def _race_once(repo: Path, other: Path, branch: str, marker: Path) -> Path:
    """Make *repo*'s first push race: *other* pushes a commit to *branch* from the hook.

    The hook plays the colleague who pushes between our fetch and our
    push, exactly once (*marker* records that it did).
    """
    _pre_push_hook(
        repo,
        f"if [ ! -e '{marker}' ]; then\n"
        f"  touch '{marker}'\n"
        f"  cd '{other}' && echo theirs > a.txt && git add a.txt && git commit -q -m race "
        f"&& git push -q origin HEAD:{branch}\n"
        "fi\nexit 0\n",
    )
    return marker


def test_sea_methods_follow_the_user_contract() -> None:
    """The SEA's methods pin the run: shell+edit tools, its own prompt, a default task."""
    sea = SyncSea()
    assert isinstance(sea, WorkerSea)
    assert "origin/main" in sea.description()
    assert "`/sync`" in sea.description()
    assert sea.prompt("") == sync_sea.ADD_TO_PROMPT
    assert sea.prompt("   ") == sync_sea.ADD_TO_PROMPT
    assert sea.prompt("upstream/dev") == "upstream/dev\n\n" + sync_sea.ADD_TO_PROMPT
    assert "origin/main when none is named" in sync_sea.ADD_TO_PROMPT
    prompt = sea.system_prompt("ASSEMBLED")
    assert prompt == sync_sea.SYSTEM_PROMPT
    assert "ASSEMBLED" not in prompt
    assert "`git_sync`" in prompt
    assert "git merge --abort" in prompt
    assert sea.settings({"model": "m"}) == {
        "model": "m",
        "tool_profile": "shell+edit",
        "max_budget": sync_sea.MAX_BUDGET_USD,
        "locked": ["tool_profile"],
    }
    assert sea_commands.base_settings([sea]) == {
        "tool_profile": "shell+edit",
        "max_budget": sync_sea.MAX_BUDGET_USD,
        "locked": ["tool_profile"],
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    profile = resolve_tool_profile("shell+edit")
    assert profile is not None
    assert {"Bash", "Read", "Edit", "Write"} <= profile
    # The chained ``tools``: the worker drops the worktree tools, the
    # subclass adds ``git_sync``.
    def create_worktree() -> None: ...
    assert sea_commands.base_tools([sea], [print, create_worktree]) == [print, sync_sea.git_sync]
    assert_no_removed_getters(sync_sea)


def test_slash_sync_resolves_bare_and_with_a_ref() -> None:
    """``/sync`` alone and ``/sync <ref>`` both run this file; ``/sync help`` is answered."""
    assert sea_commands.get_command("sync") == _SEA_PATH
    assert sea_commands.slash_command_task("/sync") == ("", _SEA_PATH)
    assert sea_commands.slash_command_task("/sync upstream/dev") == ("upstream/dev", _SEA_PATH)
    assert sea_commands.slash_command_task("/sync help") is None
    assert sea_commands.help_text_if_command("/sync help") == SyncSea().description()
    run = sea_commands.evaluate_sea([sea_commands.load_sea(_SEA_PATH)], "")
    assert run.prompt == sync_sea.ADD_TO_PROMPT
    assert sea_commands.sea_settings(_SEA_PATH) == sea_commands.base_settings([SyncSea()])


def test_sync_fast_forwards_when_the_branch_has_no_own_commits(tmp_path: Path) -> None:
    """A colleague's push is fast-forwarded in; the push then has nothing to send."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    pushed = _commit(theirs, "a.txt", "theirs\n")
    _git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    # The checkout is named the way git prints it (forward slashes on Windows too).
    assert f"Checkout {mine.resolve().as_posix()}, branch main, syncing with origin/main." in report
    assert "main is 0 ahead and 1 behind." in report
    assert f"Fast-forwarded to origin/main: now at {pushed[:12]} (1 files changed)." in report
    assert "Pushed main to origin: Everything up-to-date" in report
    assert _git(mine, "rev-parse", "HEAD") == pushed
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"
    # An empty commit upstream changes no file: nothing to grep for markers.
    _git(theirs, "commit", "-q", "--allow-empty", "-m", "empty")
    _git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "(0 files changed)." in report
    assert "WARNING" not in report


def test_sync_merges_and_pushes_when_both_sides_moved(tmp_path: Path) -> None:
    """Own commits plus a remote move give a merge commit that is pushed to origin."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "a.txt", "theirs\n")
    _git(theirs, "push", "-q", "origin", "main")
    own = _commit(mine, "b.txt", "mine\n")
    (mine / "untracked.txt").write_text("scratch\n", encoding="utf-8")
    report = sync(str(mine))
    assert "main is 1 ahead and 1 behind." in report
    assert "Merged origin/main: now at " in report
    assert "(1 files changed)." in report  # the merge brought a.txt
    assert "uncommitted change" not in report  # untracked files are not changes
    head = _git(mine, "rev-parse", "HEAD")
    assert _git(mine, "rev-parse", "HEAD^1") == own
    assert "Pushed main to origin:" in report
    assert _remote_main(mine) == head
    assert (mine / "untracked.txt").read_text(encoding="utf-8") == "scratch\n"


def test_sync_pushes_when_there_is_nothing_to_merge(tmp_path: Path) -> None:
    """A branch that is only ahead is pushed without a merge; a dirty tree is reported."""
    (mine,) = _remote_and_clones(tmp_path, "mine")
    own = _commit(mine, "b.txt", "mine\n")
    (mine / "README.md").write_text("edited, not committed\n", encoding="utf-8")
    report = sync(str(mine))
    assert "1 uncommitted change(s) in the working tree, left as they are." in report
    assert "main is 1 ahead and 0 behind." in report
    assert "Nothing to merge: origin/main is already part of main." in report
    assert _remote_main(mine) == own
    assert (mine / "README.md").read_text(encoding="utf-8") == "edited, not committed\n"


def test_sync_leaves_a_conflicted_merge_in_place_and_finishes_after_resolution(
    tmp_path: Path,
) -> None:
    """Conflicts stop the sync with the files listed; once committed, a second sync pushes."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "README.md", "theirs\n")
    _git(theirs, "push", "-q", "origin", "main")
    _commit(mine, "README.md", "mine\n")
    with pytest.raises(GitSyncError) as info:
        sync(str(mine))
    message = str(info.value)
    assert "stopped on 1 conflicted file(s); the merge is left in progress:\n  README.md" in message
    assert "`git commit --no-edit`, then call git_sync again to push." in message
    assert "<<<<<<<" in (mine / "README.md").read_text(encoding="utf-8")
    # A second call while the merge is unresolved is refused, not retried.
    with pytest.raises(GitSyncError, match="A merge is already in progress"):
        sync(str(mine))
    (mine / "README.md").write_text("mine and theirs\n", encoding="utf-8")
    _git(mine, "add", "README.md")
    _git(mine, "commit", "-q", "--no-edit")
    report = sync(str(mine))
    assert "Nothing to merge" in report
    assert _remote_main(mine) == _git(mine, "rev-parse", "HEAD")


def test_sync_refuses_a_merge_that_would_overwrite_local_edits(tmp_path: Path) -> None:
    """git refuses to merge over an uncommitted edit of a file the remote changed."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "README.md", "theirs\n")
    _git(theirs, "push", "-q", "origin", "main")
    (mine / "README.md").write_text("edited, not committed\n", encoding="utf-8")
    with pytest.raises(GitSyncError, match="git merge origin/main did not start"):
        sync(str(mine))
    assert (mine / "README.md").read_text(encoding="utf-8") == "edited, not committed\n"
    assert not (mine / ".git" / "MERGE_HEAD").exists()


def test_sync_warns_about_conflict_markers_the_remote_committed(tmp_path: Path) -> None:
    """Markers that arrive from origin are reported, not mistaken for this merge's."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "notes.rst", "<<<<<<< HEAD\nleft\n=======\nright\n>>>>>>> theirs\n")
    _commit(theirs, "title.rst", "Title\n=======\n")
    _git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "WARNING: conflict markers are committed in the merged files" in report
    assert "Pushed main to origin: Everything up-to-date" in report
    assert "notes.rst" in report.split("WARNING", 1)[1]
    assert "title.rst" not in report.split("WARNING", 1)[1]


def test_sync_retries_the_round_when_the_remote_moves_during_the_push(tmp_path: Path) -> None:
    """A push the remote rejects because it moved is retried from the fetch, once per round."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(mine, "b.txt", "mine\n")
    marker = _race_once(mine, theirs, "main", tmp_path / "raced")
    report = sync(str(mine))
    assert marker.exists()
    assert "origin/main moved since the fetch; merging it, then pushing again." in report
    assert report.count("Fetched origin/main") == 2
    assert "main is 1 ahead and 1 behind." in report
    assert "Merged origin/main" in report
    assert _remote_main(mine) == _git(mine, "rev-parse", "HEAD")
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"


def test_sync_reports_a_push_the_remote_refuses(tmp_path: Path) -> None:
    """A push rejected for a reason other than a race is reported with git's words, no retry."""
    (mine,) = _remote_and_clones(tmp_path, "mine")
    _commit(mine, "b.txt", "mine\n")
    _pre_push_hook(mine, "echo 'pushes are frozen today' >&2\nexit 1\n")
    with pytest.raises(GitSyncError) as info:
        sync(str(mine))
    message = str(info.value)
    assert "Nothing to merge" in message
    assert "git push origin HEAD failed:" in message
    assert "pushes are frozen today" in message
    assert message.count("Fetched origin/main") == 1


def test_sync_refuses_what_it_cannot_sync(tmp_path: Path) -> None:
    """A bad ref, a missing directory, a plain folder, a detached HEAD and a dead remote."""
    with pytest.raises(GitSyncError, match="must be <remote>/<branch>, not 'main'"):
        sync(str(tmp_path), "main")
    with pytest.raises(GitSyncError, match="is not a directory"):
        sync(str(tmp_path / "missing"))
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitSyncError, match="git rev-parse --show-toplevel failed"):
        sync(str(plain))
    (mine,) = _remote_and_clones(tmp_path, "mine")
    _git(mine, "checkout", "-q", "--detach")
    with pytest.raises(GitSyncError, match="git symbolic-ref --short HEAD failed"):
        sync(str(mine))
    _git(mine, "checkout", "-q", "main")
    _git(mine, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    with pytest.raises(GitSyncError, match="git fetch origin .* failed"):
        sync(str(mine))


def test_agent_run_calls_git_sync_and_reports(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees the shell, file and sync tools and syncs.

    The scripted model calls ``git_sync`` on the work dir and finishes
    with the report; the test checks the tools offered on every step,
    the system prompt, the real report in the tool-result message, and
    that origin really received the branch.
    """
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "a.txt", "theirs\n")
    _git(theirs, "push", "-q", "origin", "main")
    own = _commit(mine, "b.txt", "mine\n")
    script = [
        tool_call_body("git_sync", {"repo": str(mine), "ref": "origin/main"}, prompt_tokens=500),
        finish_body("<p>Merged origin/main into main and pushed.</p>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([SyncSea()], "")
    assert run.prompt == sync_sea.ADD_TO_PROMPT
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("sync-sea-test")
        result = agent.run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(mine),
            max_steps=4,
            max_budget=run.settings["max_budget"],
            model_config={"base_url": url, "api_key": "local"},
            system_prompt_hook=run.system_prompt_hook,
            tools_hook=run.tools_hook,
            tool_profile=run.settings["tool_profile"],
            web_tools=run.settings["use_web_tools"],
            use_memory=run.settings["use_memory"],
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert "pushed" in parsed["summary"]
    assert _git(mine, "rev-parse", "HEAD^1") == own
    assert _remote_main(mine) == _git(mine, "rev-parse", "HEAD")

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert {"git_sync", "Bash", "Read", "Edit", "Write", "finish"} <= names, names
        assert not {"create_worktree", "commit_and_merge", "go_to_url", "memory_search"} & names
        system = system_message(request)
        assert system.startswith(sync_sea.SYSTEM_PROMPT), system[:200]
        assert f"Work dir: {mine}" in system
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1, agentic[1]["messages"]
    assert "Merged origin/main: now at" in str(tool_results[0]["content"])
    assert "Pushed main to origin:" in str(tool_results[0]["content"])


def test_git_sync_tool_returns_the_failure_report_instead_of_raising(tmp_path: Path) -> None:
    """The tool hands the model the error text (conflicts included) as its result."""
    missing = tmp_path / "missing"
    assert sync_sea.git_sync(str(missing)) == f"{missing} is not a directory"
    assert git_sync.PUSH_ATTEMPTS == 3


def test_sync_merges_into_a_feature_branch_of_a_single_branch_clone(tmp_path: Path) -> None:
    """origin/main is merged into ``feature`` and ``feature`` is pushed, from a single-branch clone.

    ``git clone --single-branch`` fetches only ``feature``; the explicit
    refspec still creates ``origin/main`` so the merge has a ref.
    """
    (seed,) = _remote_and_clones(tmp_path, "seed2")
    _git(seed, "checkout", "-q", "-b", "feature")
    _commit(seed, "f.txt", "feature\n")
    _git(seed, "push", "-q", "origin", "feature")
    _git(seed, "checkout", "-q", "main")
    _commit(seed, "a.txt", "main moved\n")
    _git(seed, "push", "-q", "origin", "main")
    mine = tmp_path / "mine"
    _git(tmp_path, "clone", "-q", "--single-branch", "--branch", "feature",
         str(tmp_path / "origin.git"), str(mine))
    assert _git(mine, "branch", "-r") == "origin/feature"
    report = sync(str(mine))
    assert "branch feature, syncing with origin/main." in report
    assert "Fetched origin/main: feature is 1 ahead and 1 behind." in report
    assert "Merged origin/main" in report
    assert "Pushed feature to origin:" in report
    head = _git(mine, "rev-parse", "HEAD")
    assert _git(mine, "ls-remote", "origin", "refs/heads/feature").split()[0] == head
    assert (mine / "a.txt").read_text(encoding="utf-8") == "main moved\n"


def test_sync_merges_the_remote_copy_of_the_branch_when_its_push_races(tmp_path: Path) -> None:
    """A rejected push of ``feature`` merges ``origin/feature`` too, not only origin/main again."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _git(mine, "checkout", "-q", "-b", "feature")
    _commit(mine, "f.txt", "feature\n")
    _git(mine, "push", "-q", "origin", "feature")
    _commit(mine, "g.txt", "more\n")
    _git(theirs, "fetch", "-q", "origin", "feature")
    _git(theirs, "checkout", "-q", "-b", "feature", "origin/feature")
    marker = _race_once(mine, theirs, "feature", tmp_path / "raced")
    report = sync(str(mine))
    assert marker.exists()
    assert "origin/feature moved since the fetch; merging it, then pushing again." in report
    assert "Fetched origin/feature: feature is 1 ahead and 1 behind." in report
    assert "Merged origin/feature" in report
    assert report.count("Fetched origin/main") == 2
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"
    head = _git(mine, "rev-parse", "HEAD")
    assert _git(mine, "ls-remote", "origin", "refs/heads/feature").split()[0] == head


def test_sync_warns_about_markers_in_a_file_with_a_non_ascii_name(tmp_path: Path) -> None:
    """Paths come from ``-z`` output, so a name git would quote still reaches the grep."""
    mine, theirs = _remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "r\u00e9sum\u00e9.txt", "<<<<<<< HEAD\nleft\n=======\nright\n>>>>>>> x\n")
    _git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "WARNING" in report
    assert "r\u00e9sum\u00e9.txt" in report.split("WARNING", 1)[1]


def test_sync_reports_git_that_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``git`` on PATH the tool reports the failure instead of raising ``OSError``."""
    (mine,) = _remote_and_clones(tmp_path, "mine")
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(GitSyncError, match="git rev-parse --show-toplevel could not run"):
        sync(str(mine))
    assert "could not run" in sync_sea.git_sync(str(mine))
