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

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync import git_sync, sync_sea
from kiss.agents.seas.sync.git_sync import GitSyncError, sync
from kiss.agents.seas.sync.sync_sea import SyncSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.machines import Machine, save_machines
from kiss.agents.sorcar.sorcar_agent import resolve_tool_profile
from kiss.core.config import kiss_home
from kiss.tests.agents.seas.conftest import (
    commit,
    git,
    hermetic_git_env,
    remote_and_clones,
    run_sea,
)
from kiss.tests.agents.seas.sea_contract import (
    assert_no_removed_getters,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body
from kiss.tests.conftest import posix_only

_SEA_PATH = Path(sync_sea.__file__).resolve()


@pytest.fixture(autouse=True)
def _hermetic_git_for_the_product(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the git commands :func:`sync` spawns under :data:`HERMETIC_GIT_WITH_IDENTITY` too.

    The test's own :func:`git` calls read the environment on every call,
    so both sides commit as ``t <t@t>`` with no user or system config.
    ``GIT_CEILING_DIRECTORIES`` keeps git's repository search inside
    *tmp_path*: under an in-repo ``--basetemp`` a plain folder of the test
    would otherwise resolve to the enclosing checkout, which a real
    :func:`sync` commits and pushes.
    """
    hermetic_git_env(monkeypatch)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


def set_known_work_dirs(work_dir: str, recent: Sequence[str]) -> None:
    """Write *work_dir* and *recent* (most recent first) as the ``work_dir`` and
    ``recent_work_dirs`` keys of the session's ``config.json``; a test that calls this
    requests :func:`restored_config` so the file goes back afterwards."""
    config = kiss_home() / "config.json"
    data = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
    data["work_dir"] = work_dir
    data["recent_work_dirs"] = [{"path": path, "ts": 1.0} for path in recent]
    config.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def restored_config() -> Iterator[None]:
    """Put the session's ``config.json`` back to what it was when the test ends."""
    config = kiss_home() / "config.json"
    before = config.read_bytes() if config.exists() else None
    yield
    if before is None:
        config.unlink(missing_ok=True)
    else:
        config.write_bytes(before)


def _commit(repo: Path, name: str, text: str) -> str:
    """Write *text* to *name* in *repo*, commit it and return the commit sha."""
    return commit(repo, f"{name}: {text.strip()[:20]}", {name: text})


def _remote_main(repo: Path) -> str:
    """Return the sha ``origin``'s ``main`` is at (asked of the remote, nothing fetched)."""
    return git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]


def _git_path(repo: Path, name: str) -> Path:
    """Return the absolute path of *name* (``hooks``, ``info/exclude``, ...) in *repo*'s git dir."""
    path = Path(git(repo, "rev-parse", "--git-path", name))
    return path if path.is_absolute() else repo / path


def _hook(repo: Path, name: str, body: str) -> None:
    """Install the git hook *name* (``pre-push``, ``pre-commit``) running shell *body* in *repo*."""
    hooks = _git_path(repo, "hooks")
    hooks.mkdir(exist_ok=True)
    hook = hooks / name
    hook.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    hook.chmod(0o755)


def _race_once(repo: Path, other: Path, branch: str, marker: Path) -> None:
    """Make *repo*'s first push race: *other* pushes a commit to *branch* from the hook.

    The hook plays the colleague who pushes between our fetch and our
    push, exactly once (*marker* records that it did).
    """
    _hook(
        repo,
        "pre-push",
        f"if [ ! -e '{marker}' ]; then\n"
        f"  touch '{marker}'\n"
        f"  cd '{other}' && echo theirs > a.txt && git add a.txt && git commit -q -m race "
        f"&& git push -q origin HEAD:{branch}\n"
        "fi\nexit 0\n",
    )


def test_sea_methods_follow_the_user_contract() -> None:
    """The SEA's methods pin the run: shell+edit tools, its own prompt, a default task."""
    sea = SyncSea()
    assert isinstance(sea, WorkerSea)
    assert "origin/main" in sea.description()
    assert sea.description().startswith("Commits the uncommitted changes")
    assert "`/sync`" in sea.description()
    assert sea.prompt("") == sync_sea.ADD_TO_PROMPT
    assert sea.prompt("   ") == sync_sea.ADD_TO_PROMPT
    assert sea.prompt("upstream/dev") == "upstream/dev\n\n" + sync_sea.ADD_TO_PROMPT
    assert "origin/main when none is named" in sync_sea.ADD_TO_PROMPT
    prompt = sea.system_prompt("ASSEMBLED")
    assert prompt == sync_sea.SYSTEM_PROMPT
    assert "ASSEMBLED" not in prompt
    assert "`git_sync`" in prompt
    # The sub-task sent to a machine names the repository by URL, never by this
    # machine's path (the checkout lives elsewhere there); when the message names
    # none, the URL comes from `remote_url`, since the machines are synced before
    # this checkout is.
    assert f'{sync_sea.THIS_MACHINE_ONLY}"' in prompt
    assert "`remote_url`" in prompt
    assert prompt.index("`remote_url`") < prompt.index("`git_sync` once")
    # A URL the message names (`/sync origin/main of <url>`) is the one the machines
    # get; `remote_url` is for a message that names none.
    assert re.search(r"when it names\s+none, what `remote_url` returns", prompt)
    assert sea.settings({"model": "m"}) == {
        "model": "m",
        "tool_profile": "shell+edit+agents",
        "max_budget": sync_sea.MAX_BUDGET_USD,
        "locked": ["tool_profile"],
    }
    assert sea_commands.base_settings([sea]) == {
        "tool_profile": "shell+edit+agents",
        "max_budget": sync_sea.MAX_BUDGET_USD,
        "locked": ["tool_profile"],
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    profile = resolve_tool_profile("shell+edit+agents")
    assert profile is not None
    assert {"Bash", "Read", "Edit", "Write", "run_agent"} <= profile

    # The chained ``tools``: the worker drops the worktree tools, the
    # subclass adds ``git_sync`` and ``remote_url``.
    def create_worktree() -> None: ...

    assert sea_commands.base_tools([sea], [print, create_worktree]) == [
        print,
        sync_sea.git_sync,
        sync_sea.remote_url,
    ]
    assert_no_removed_getters(sync_sea)


def test_slash_sync_resolves_bare() -> None:
    """``/sync`` alone runs this file on the empty text; ``prompt()`` makes it the playbook."""
    assert sea_commands.slash_command_task("/sync") == ("", _SEA_PATH)
    run = sea_commands.evaluate_sea([sea_commands.load_sea(_SEA_PATH)], "")
    assert run.prompt == sync_sea.ADD_TO_PROMPT


@pytest.mark.usefixtures("empty_machines")
def test_prompt_lists_the_registered_machines_and_the_budget_covers_them() -> None:
    """With machines registered the task names them in order, unless it says
    "this machine only" (what the sub-task sent to a machine says); the budget
    is one share for this checkout and two per machine (synced before and after it)."""
    sea = SyncSea()
    save_machines([Machine("box", url="https://box:8787"), Machine("lab", ssh="u@lab")])
    listed = f"\n\n{sync_sea.ADD_TO_PROMPT}\n\n{sync_sea.MACHINES_PROMPT}box, lab."
    assert sea.prompt("") == listed.strip()
    assert sea.prompt("upstream/dev") == "upstream/dev" + listed
    assert sea.prompt("preamble\n/sync upstream/dev") == "preamble\n/sync upstream/dev" + listed
    local = f"origin/main {sync_sea.THIS_MACHINE_ONLY}"
    assert sea.prompt(local) == f"{local}\n\n{sync_sea.ADD_TO_PROMPT}"
    assert sea.prompt(f"/sync {sync_sea.THIS_MACHINE_ONLY}") == (
        f"/sync {sync_sea.THIS_MACHINE_ONLY}\n\n{sync_sea.ADD_TO_PROMPT}"
    )
    assert sync_sea.machine_names("") == ["box", "lab"]
    assert sync_sea.machine_names(local) == []
    assert sea.settings({})["max_budget"] == 5 * sync_sea.MAX_BUDGET_USD
    run = sea_commands.evaluate_sea([sea_commands.load_sea(_SEA_PATH)], "")
    assert run.prompt == listed.strip()
    assert run.settings["max_budget"] == 5 * sync_sea.MAX_BUDGET_USD
    save_machines([])
    assert sea.prompt("") == sync_sea.ADD_TO_PROMPT
    assert sea.settings({})["max_budget"] == sync_sea.MAX_BUDGET_USD


def test_sync_fast_forwards_when_the_branch_has_no_own_commits(tmp_path: Path) -> None:
    """A colleague's push is fast-forwarded in; the push then has nothing to send."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    pushed = _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    # The checkout is named the way git prints it (forward slashes on Windows too).
    origin = git(mine, "remote", "get-url", "origin")
    assert (
        f"Checkout {mine.resolve().as_posix()}, branch main, syncing with origin/main "
        f"(origin = {origin})."
    ) in report
    assert "main is 0 ahead and 1 behind." in report
    assert f"Fast-forwarded to origin/main: now at {pushed[:12]} (1 files changed)." in report
    assert "Pushed main to origin: Everything up-to-date" in report
    assert git(mine, "rev-parse", "HEAD") == pushed
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"
    # An empty commit upstream changes no file: nothing to grep for markers.
    git(theirs, "commit", "-q", "--allow-empty", "-m", "empty")
    git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "(0 files changed)." in report
    assert "WARNING" not in report


def test_sync_merges_and_pushes_when_both_sides_moved(tmp_path: Path) -> None:
    """Own commits plus a remote move give a merge commit that is pushed to origin."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    own = _commit(mine, "b.txt", "mine\n")
    # A new file is committed before the merge (`git add -A`); an ignored one is not.
    (mine / "untracked.txt").write_text("scratch\n", encoding="utf-8")
    (mine / "ignored.log").write_text("noise\n", encoding="utf-8")
    exclude = _git_path(mine, "info/exclude")
    exclude.parent.mkdir(exist_ok=True)
    exclude.write_text("*.log\n", encoding="utf-8")
    report = sync(str(mine))
    assert "Committed 1 uncommitted change(s) on main as " in report
    assert ": untracked.txt\n" in report
    assert "ignored.log" not in report
    assert "main is 2 ahead and 1 behind." in report
    assert "Merged origin/main: now at " in report
    assert "(1 files changed)." in report  # the merge brought a.txt
    head = git(mine, "rev-parse", "HEAD")
    gathered = git(mine, "rev-parse", "HEAD^1")
    assert git(mine, "rev-parse", f"{gathered}^") == own
    assert git(mine, "log", "-1", "--format=%s", gathered) == git_sync.commit_message(
        1, "main", git_sync._hostname()
    )
    assert git(mine, "show", "--format=", "--name-only", gathered) == "untracked.txt"
    assert "Pushed main to origin:" in report
    assert _remote_main(mine) == head
    assert (mine / "untracked.txt").read_text(encoding="utf-8") == "scratch\n"
    assert git(mine, "status", "--porcelain", "--ignored") == "!! ignored.log"


def test_sync_pushes_when_there_is_nothing_to_merge(tmp_path: Path) -> None:
    """A branch that is only ahead is pushed without a merge; an edit is committed first."""
    (mine,) = remote_and_clones(tmp_path, "mine")
    own = _commit(mine, "b.txt", "mine\n")
    (mine / "README.md").write_text("edited, not committed\n", encoding="utf-8")
    (mine / "b.txt").unlink()
    report = sync(str(mine))
    head = git(mine, "rev-parse", "HEAD")
    assert f"Committed 2 uncommitted change(s) on main as {head[:12]}: README.md, b.txt" in report
    assert "left as they are" not in report
    assert "main is 2 ahead and 0 behind." in report
    assert "Nothing to merge: origin/main is already part of main." in report
    assert git(mine, "rev-parse", "HEAD^") == own
    assert _remote_main(mine) == head
    assert git(mine, "status", "--porcelain") == ""
    assert (mine / "README.md").read_text(encoding="utf-8") == "edited, not committed\n"
    assert not (mine / "b.txt").exists()
    # A clean tree commits nothing and the report says nothing about it.
    report = sync(str(mine))
    assert "Committed" not in report
    assert git(mine, "rev-parse", "HEAD") == head


def test_sync_leaves_a_conflicted_merge_in_place_and_finishes_after_resolution(
    tmp_path: Path,
) -> None:
    """Conflicts stop the sync with the files listed; once committed, a second sync pushes."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "README.md", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
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
    git(mine, "add", "README.md")
    git(mine, "commit", "-q", "--no-edit")
    report = sync(str(mine))
    assert "Nothing to merge" in report
    assert _remote_main(mine) == git(mine, "rev-parse", "HEAD")


def test_sync_commits_a_local_edit_of_a_file_the_remote_changed_then_conflicts(
    tmp_path: Path,
) -> None:
    """An uncommitted edit no longer blocks the merge: it is committed, and the conflict
    it makes with the remote's change is reported after the commit line."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "README.md", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    (mine / "README.md").write_text("edited, not committed\n", encoding="utf-8")
    with pytest.raises(GitSyncError) as info:
        sync(str(mine))
    message = str(info.value)
    gathered = git(mine, "rev-parse", "HEAD")
    assert message.startswith(
        f"Committed 1 uncommitted change(s) on main as {gathered[:12]}: README.md\n"
        "Merging origin/main stopped on 1 conflicted file(s)"
    )
    assert "did not start" not in message
    assert git(mine, "show", f"{gathered}:README.md") == "edited, not committed"
    assert "<<<<<<<" in (mine / "README.md").read_text(encoding="utf-8")
    assert git(mine, "rev-parse", "-q", "--verify", "MERGE_HEAD")


def test_sync_refuses_to_commit_over_unmerged_paths(tmp_path: Path) -> None:
    """Conflicted files left without a MERGE_HEAD (a failed stash pop) are not swept into
    the commit, markers and all: the sync stops and says what to do."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "README.md", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    _commit(mine, "README.md", "mine\n")
    git(mine, "fetch", "-q", "origin")
    git(mine, "merge", "origin/main", check=False)
    (mine / ".git" / "MERGE_HEAD").unlink()  # what a failed stash pop leaves behind
    (mine / "other.txt").write_text("new\n", encoding="utf-8")
    with pytest.raises(GitSyncError) as info:
        sync(str(mine))
    assert str(info.value) == (
        "1 unmerged path(s) in the working tree, left from an earlier merge: README.md\n"
        "Resolve them, `git add` the files and `git commit`, then call git_sync again."
    )
    assert git(mine, "rev-parse", "HEAD") == git(mine, "rev-parse", "main")
    assert "<<<<<<<" in (mine / "README.md").read_text(encoding="utf-8")
    assert "?? other.txt" in git(mine, "status", "--porcelain")


@pytest.mark.parametrize(
    "name",
    [
        "mine",
        pytest.param(
            "mine ", marks=posix_only("Windows strips a trailing space from a directory name")
        ),
    ],
)
def test_sync_commits_the_whole_tree_from_a_subdirectory_whatever_git_is_configured(
    tmp_path: Path, name: str
) -> None:
    """Run in a subdirectory with ``diff.relative`` on, the unmerged-path check and the
    file list still cover the whole tree; ``status.showUntrackedFiles=no`` does not
    hide a new file from the commit; a checkout name ending in a space is kept."""
    mine, theirs = remote_and_clones(tmp_path, name, "theirs")
    assert mine.name == name
    git(mine, "config", "diff.relative", "true")
    git(mine, "config", "status.showUntrackedFiles", "no")
    sub = mine / "sub"
    sub.mkdir()
    # Only an untracked file, which this configuration hides from a plain `git status`.
    (sub / "new.txt").write_text("new\n", encoding="utf-8")
    assert git(mine, "status", "--porcelain") == ""
    report = sync(str(sub))
    first = git(mine, "rev-parse", "HEAD")
    assert f"Committed 1 uncommitted change(s) on main as {first[:12]}: sub/new.txt" in report
    assert f"Checkout {mine.resolve().as_posix()}, branch main" in report
    # A tracked edit outside sub/ is seen and listed from sub/ although diffs are relative.
    (mine / "README.md").write_text("root edit\n", encoding="utf-8")
    report = sync(str(sub))
    head = git(mine, "rev-parse", "HEAD")
    assert f"Committed 1 uncommitted change(s) on main as {head[:12]}: README.md" in report
    assert git(mine, "rev-parse", "HEAD^") == first
    assert git(mine, "status", "--porcelain", "--untracked-files=all") == ""
    assert _remote_main(mine) == head
    # An unresolved conflict at the root is seen from sub/ too, and nothing is committed.
    git(theirs, "fetch", "-q", "origin")
    git(theirs, "reset", "-q", "--hard", "origin/main")
    _commit(theirs, "README.md", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    own = _commit(mine, "README.md", "mine again\n")
    git(mine, "fetch", "-q", "origin")
    git(mine, "merge", "origin/main", check=False)
    (mine / ".git" / "MERGE_HEAD").unlink()
    (sub / "more.txt").write_text("more\n", encoding="utf-8")
    with pytest.raises(GitSyncError, match="1 unmerged path\\(s\\) .*: README.md"):
        sync(str(sub))
    assert git(mine, "rev-parse", "HEAD") == own
    assert "<<<<<<<" in (mine / "README.md").read_text(encoding="utf-8")
    assert "?? sub/more.txt" in git(mine, "status", "--porcelain", "--untracked-files=all")


def test_sync_reports_the_edits_inside_a_submodule_it_cannot_commit(tmp_path: Path) -> None:
    """``git add -A`` in the superproject stages a submodule's new HEAD, not edits inside
    it: those are reported as left in place instead of silently passing as committed."""
    (mine,) = remote_and_clones(tmp_path, "mine")
    lib_origin = tmp_path / "lib.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(lib_origin))
    lib_seed = tmp_path / "lib_seed"
    git(tmp_path, "clone", "-q", str(lib_origin), str(lib_seed))
    git(lib_seed, "checkout", "-q", "-b", "main")
    commit(lib_seed, "lib: v1", {"lib.txt": "v1\n"})
    git(lib_seed, "push", "-q", "origin", "main")
    git(mine, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(lib_origin), "lib")
    git(mine, "commit", "-q", "-m", "add lib")
    (mine / "lib" / "lib.txt").write_text("edited inside the submodule\n", encoding="utf-8")
    before = git(mine, "rev-parse", "HEAD")
    report = sync(str(mine))
    assert "Committed" not in report
    assert (
        "1 change(s) `git add -A` cannot stage from this checkout (edits inside a "
        "submodule?), left as they are: lib\n"
    ) in report
    assert git(mine, "rev-parse", "HEAD") == before
    assert _remote_main(mine) == before
    assert (mine / "lib" / "lib.txt").read_text(encoding="utf-8") == "edited inside the submodule\n"
    # With a top-level edit as well, the commit line comes first, the leftover second.
    (mine / "README.md").write_text("edited\n", encoding="utf-8")
    report = sync(str(mine))
    head = git(mine, "rev-parse", "HEAD")
    assert (
        f"Committed 1 uncommitted change(s) on main as {head[:12]}: README.md\n"
        "1 change(s) `git add -A` cannot stage from this checkout"
    ) in report
    assert git(mine, "show", "--format=", "--name-only", head) == "README.md"


def test_sync_reports_a_commit_that_fails(tmp_path: Path) -> None:
    """A pre-commit hook that refuses the commit stops the sync before the fetch."""
    (mine,) = remote_and_clones(tmp_path, "mine")
    _hook(mine, "pre-commit", "echo 'lint says no' >&2\nexit 1\n")
    before = git(mine, "rev-parse", "HEAD")
    (mine / "README.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(GitSyncError, match=r"git commit -q -m .* failed:\nlint says no"):
        sync(str(mine))
    assert git(mine, "rev-parse", "HEAD") == before
    assert (mine / "README.md").read_text(encoding="utf-8") == "edited\n"


def test_commit_message_and_the_report_name_the_host_and_cap_the_file_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit says where the changes were left; the report lists at most 20 files."""
    assert git_sync.commit_message(3, "main", "my-laptop") == (
        "sync: commit 3 uncommitted change(s) on main (my-laptop)"
    )
    assert git_sync._hostname()  # whatever the machine is called, never empty
    monkeypatch.setattr(git_sync.socket, "gethostname", lambda: "")
    assert git_sync._hostname() == "unknown"
    monkeypatch.setattr(git_sync.socket, "gethostname", lambda: "box")
    (mine,) = remote_and_clones(tmp_path, "mine")
    git(mine, "checkout", "-q", "-b", "feature")
    for index in range(git_sync.LISTED_FILES + 2):
        (mine / f"f{index:02d}.txt").write_text(f"{index}\n", encoding="utf-8")
    report = sync(str(mine), "origin/main")
    shown = ", ".join(f"f{index:02d}.txt" for index in range(git_sync.LISTED_FILES))
    assert "Committed 22 uncommitted change(s) on feature as " in report
    assert f": {shown} ... and 2 more\n" in report
    assert git(mine, "log", "-1", "--format=%s") == (
        "sync: commit 22 uncommitted change(s) on feature (box)"
    )
    assert "Pushed feature to origin:" in report
    assert git(mine, "ls-remote", "origin", "refs/heads/feature").split()[0] == git(
        mine, "rev-parse", "HEAD"
    )


def test_sync_warns_about_conflict_markers_the_remote_committed(tmp_path: Path) -> None:
    """Markers that arrive from origin are reported, not mistaken for this merge's."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "notes.rst", "<<<<<<< HEAD\nleft\n=======\nright\n>>>>>>> theirs\n")
    _commit(theirs, "title.rst", "Title\n=======\n")
    git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "WARNING: conflict markers are committed in the merged files" in report
    assert "Pushed main to origin: Everything up-to-date" in report
    assert "notes.rst" in report.split("WARNING", 1)[1]
    assert "title.rst" not in report.split("WARNING", 1)[1]


def test_sync_retries_the_round_when_the_remote_moves_during_the_push(tmp_path: Path) -> None:
    """A push the remote rejects because it moved is retried from the fetch, once per round."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(mine, "b.txt", "mine\n")
    marker = tmp_path / "raced"
    _race_once(mine, theirs, "main", marker)
    report = sync(str(mine))
    assert marker.exists()
    assert "origin/main moved since the fetch; merging it, then pushing again." in report
    assert report.count("Fetched origin/main") == 2
    assert "main is 1 ahead and 1 behind." in report
    assert "Merged origin/main" in report
    assert _remote_main(mine) == git(mine, "rev-parse", "HEAD")
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"


def test_sync_reports_a_push_the_remote_refuses(tmp_path: Path) -> None:
    """A push rejected for a reason other than a race is reported with git's words, no retry."""
    (mine,) = remote_and_clones(tmp_path, "mine")
    _commit(mine, "b.txt", "mine\n")
    _hook(mine, "pre-push", "echo 'pushes are frozen today' >&2\nexit 1\n")
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
    (mine,) = remote_and_clones(tmp_path, "mine")
    git(mine, "checkout", "-q", "--detach")
    with pytest.raises(GitSyncError, match="git symbolic-ref --short HEAD failed"):
        sync(str(mine))
    git(mine, "checkout", "-q", "main")
    git(mine, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    with pytest.raises(GitSyncError, match="git fetch origin .* failed"):
        sync(str(mine))


def test_agent_run_calls_git_sync_and_reports(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees the shell, file and sync tools and syncs.

    The scripted model calls ``git_sync`` on the work dir and finishes
    with the report; the test checks the tools offered on every step,
    the system prompt, the real report in the tool-result message, and
    that origin really received the branch.
    """
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    own = _commit(mine, "b.txt", "mine\n")
    (mine / "notes.txt").write_text("not committed yet\n", encoding="utf-8")
    script = [
        tool_call_body("git_sync", {"repo": str(mine), "ref": "origin/main"}, prompt_tokens=500),
        finish_body("<p>Merged origin/main into main and pushed.</p>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([SyncSea()], "")
    assert run.prompt == sync_sea.ADD_TO_PROMPT
    parsed, agentic = run_sea(run, script, mine, max_steps=4)
    assert parsed["success"] is True
    assert "pushed" in parsed["summary"]
    # The tool committed notes.txt on main (between own and the merge) and pushed it.
    assert git(mine, "rev-parse", "HEAD^1^") == own
    assert git(mine, "show", "--format=", "--name-only", "HEAD^1") == "notes.txt"
    assert git(mine, "status", "--porcelain") == ""
    assert _remote_main(mine) == git(mine, "rev-parse", "HEAD")

    assert len(agentic) == 2, [list(r) for r in agentic]
    for request in agentic:
        names = tool_names(request)
        assert {
            "git_sync", "remote_url", "Bash", "Read", "Edit", "Write", "run_agent", "finish",
        } <= names, names
        assert not {"create_worktree", "commit_and_merge", "go_to_url", "memory_search"} & names
        system = system_message(request)
        assert system.startswith(sync_sea.SYSTEM_PROMPT), system[:200]
        assert f"Work dir: {mine}" in system
    results = tool_results(agentic[1])
    assert len(results) == 1, agentic[1]["messages"]
    assert "Committed 1 uncommitted change(s) on main as " in results[0]
    assert "Merged origin/main: now at" in results[0]
    assert "Pushed main to origin:" in results[0]


def test_git_sync_tool_returns_the_failure_report_instead_of_raising(tmp_path: Path) -> None:
    """The tool hands the model the error text (conflicts included) as its result."""
    missing = tmp_path / "missing"
    assert sync_sea.git_sync(str(missing)) == f"{missing} is not a directory"


def test_remote_url_tool_names_the_repository_without_its_credentials(tmp_path: Path) -> None:
    """What the sub-task sent to a machine names: the ref's remote URL, token dropped.

    The agent calls it before the first machine is synced (the local
    ``git_sync``, whose report also names the URL, comes after the
    machines now); a failure is returned as text, like ``git_sync``'s.
    """
    (mine,) = remote_and_clones(tmp_path, "mine")
    origin = git(mine, "remote", "get-url", "origin")
    assert sync_sea.remote_url(str(mine)) == origin
    assert sync_sea.remote_url(str(mine), "origin/dev") == origin
    # From a subdirectory of the checkout too (the work dir of a tab opened there).
    (mine / "sub").mkdir()
    assert sync_sea.remote_url(str(mine / "sub")) == origin
    git(mine, "remote", "add", "upstream", "https://oauth2:TOKEN@host/me/kiss.git")
    assert sync_sea.remote_url(str(mine), "upstream/main") == "https://host/me/kiss.git"
    assert git_sync.remote_url(str(mine), "upstream/main") == "https://host/me/kiss.git"
    assert sync_sea.remote_url(str(mine), "nosuch/main") == (
        f"{mine} is not a git checkout with a remote 'nosuch'"
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    assert sync_sea.remote_url(str(plain)) == (
        f"{plain} is not a git checkout with a remote 'origin'"
    )
    missing = tmp_path / "missing"
    assert sync_sea.remote_url(str(missing)) == (
        f"{missing} is not a git checkout with a remote 'origin'"
    )
    assert sync_sea.remote_url(str(mine), "main") == (
        "the ref to merge must be <remote>/<branch>, not 'main'"
    )
    with pytest.raises(GitSyncError, match="not a git checkout with a remote 'origin'"):
        git_sync.remote_url(str(plain))


def test_sync_merges_into_a_feature_branch_of_a_single_branch_clone(tmp_path: Path) -> None:
    """origin/main is merged into ``feature`` and ``feature`` is pushed, from a single-branch clone.

    ``git clone --single-branch`` fetches only ``feature``; the explicit
    refspec still creates ``origin/main`` so the merge has a ref.
    """
    (seed,) = remote_and_clones(tmp_path, "seed2")
    git(seed, "checkout", "-q", "-b", "feature")
    _commit(seed, "f.txt", "feature\n")
    git(seed, "push", "-q", "origin", "feature")
    git(seed, "checkout", "-q", "main")
    _commit(seed, "a.txt", "main moved\n")
    git(seed, "push", "-q", "origin", "main")
    mine = tmp_path / "mine"
    git(
        tmp_path,
        "clone",
        "-q",
        "--single-branch",
        "--branch",
        "feature",
        str(tmp_path / "origin.git"),
        str(mine),
    )
    assert git(mine, "branch", "-r") == "origin/feature"
    report = sync(str(mine))
    assert "branch feature, syncing with origin/main (origin = " in report
    assert "Fetched origin/main: feature is 1 ahead and 1 behind." in report
    assert "Merged origin/main" in report
    assert "Pushed feature to origin:" in report
    head = git(mine, "rev-parse", "HEAD")
    assert git(mine, "ls-remote", "origin", "refs/heads/feature").split()[0] == head
    assert (mine / "a.txt").read_text(encoding="utf-8") == "main moved\n"


def test_sync_merges_the_remote_copy_of_the_branch_when_its_push_races(tmp_path: Path) -> None:
    """A rejected push of ``feature`` merges ``origin/feature`` too, not only origin/main again."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    git(mine, "checkout", "-q", "-b", "feature")
    _commit(mine, "f.txt", "feature\n")
    git(mine, "push", "-q", "origin", "feature")
    _commit(mine, "g.txt", "more\n")
    git(theirs, "fetch", "-q", "origin", "feature")
    git(theirs, "checkout", "-q", "-b", "feature", "origin/feature")
    marker = tmp_path / "raced"
    _race_once(mine, theirs, "feature", marker)
    report = sync(str(mine))
    assert marker.exists()
    assert "origin/feature moved since the fetch; merging it, then pushing again." in report
    assert "Fetched origin/feature: feature is 1 ahead and 1 behind." in report
    assert "Merged origin/feature" in report
    assert report.count("Fetched origin/main") == 2
    assert (mine / "a.txt").read_text(encoding="utf-8") == "theirs\n"
    head = git(mine, "rev-parse", "HEAD")
    assert git(mine, "ls-remote", "origin", "refs/heads/feature").split()[0] == head


def test_sync_warns_about_markers_in_a_file_with_a_non_ascii_name(tmp_path: Path) -> None:
    """Paths come from ``-z`` output, so a name git would quote still reaches the grep."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "r\u00e9sum\u00e9.txt", "<<<<<<< HEAD\nleft\n=======\nright\n>>>>>>> x\n")
    git(theirs, "push", "-q", "origin", "main")
    report = sync(str(mine))
    assert "WARNING" in report
    assert "r\u00e9sum\u00e9.txt" in report.split("WARNING", 1)[1]


def test_sync_reports_git_that_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``git`` on PATH the tool reports the failure instead of raising ``OSError``."""
    (mine,) = remote_and_clones(tmp_path, "mine")
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(GitSyncError, match="git rev-parse --show-toplevel could not run"):
        sync(str(mine))
    assert "could not run" in sync_sea.git_sync(str(mine))


def test_repo_key_identifies_a_repository_whatever_the_url_form() -> None:
    """The https, ssh, scp-like and local forms of one repository compare equal.

    What tells repositories apart is kept: the path's case, an explicit
    port, a numeric directory in the scp-like form (which has no port).
    """
    key = ("github.com", "me/kiss")
    assert git_sync.repo_key("https://github.com/me/kiss.git") == key
    assert git_sync.repo_key("HTTPS://GitHub.com/me/kiss/") == key
    assert git_sync.repo_key("git@github.com:me/kiss.git") == key
    assert git_sync.repo_key("GIT@GitHub.COM:me/kiss.GIT") == key
    assert git_sync.repo_key("ssh://git@github.com/me/kiss") == key
    assert git_sync.repo_key("https://user:token@github.com/me/kiss") == key
    assert git_sync.repo_key("https://github.com/me/other.git") != key
    assert git_sync.repo_key("https://gitlab.com/me/kiss.git") != key
    assert git_sync.repo_key("https://github.com/Me/Kiss.git") != key
    assert git_sync.repo_key("ssh://git@host:2222/team/r.git") == ("host:2222", "team/r")
    assert git_sync.repo_key("ssh://git@host:2222/team/r") != git_sync.repo_key(
        "ssh://git@host:3333/team/r"
    )
    assert git_sync.repo_key("git@host:123/r.git") == ("host", "123/r")
    assert git_sync.repo_key("git@host:123/r.git") != git_sync.repo_key("git@host:456/r.git")
    assert git_sync.repo_key("alice@host:r.git") == ("host", "r")
    # Only a user before the host is dropped; an ``@`` in the path stays.
    assert git_sync.repo_key("git@server:team@other/repo.git") == ("server", "team@other/repo")
    assert git_sync.repo_key("other:repo.git") == ("other", "repo")
    assert git_sync.repo_key("git@[::1]:repo.git") == ("[::1]", "repo")
    assert git_sync.repo_key("ssh://git@[::1]:2222/repo.git") == ("[::1]:2222", "repo")
    assert git_sync.repo_key("ssh://git@[::1]/repo.git") == ("[::1]", "repo")
    # A local path is the directory as written: ``r`` and ``r.git`` are two.
    assert git_sync.repo_key("/tmp/x/origin.git") == ("", "/tmp/x/origin.git")
    assert git_sync.repo_key("/tmp/x/origin.git/") == git_sync.repo_key("/tmp/x/origin.git")
    assert git_sync.repo_key("/tmp/x/origin.git") != git_sync.repo_key("/tmp/x/origin")
    assert git_sync.repo_key("/tmp/x/Origin") != git_sync.repo_key("/tmp/x/origin")
    assert git_sync.repo_key("./rel/repo") == ("", "./rel/repo")
    assert git_sync.repo_key("/") == ("", "/")
    windows = git_sync.repo_key("C:\\repos\\r.git")
    assert windows == git_sync.repo_key("C:/repos/r.git/") == ("", "C:/repos/r.git")
    assert git_sync.repo_key("C:/repos/r.git") != git_sync.repo_key("C:/repos/r")


def test_public_url_drops_the_credentials_a_url_carries() -> None:
    """Reports and messages to machines show URLs without ``user:token@``."""
    assert (
        git_sync.public_url("https://oauth2:TOKEN@host/me/kiss.git") == "https://host/me/kiss.git"
    )
    assert git_sync.public_url("https://user@host/me/kiss") == "https://host/me/kiss"
    assert git_sync.public_url("https://github.com/me/kiss.git") == "https://github.com/me/kiss.git"
    assert git_sync.public_url("ssh://git@host:2222/r.git") == "ssh://host:2222/r.git"
    assert git_sync.public_url("git@github.com:me/kiss.git") == "git@github.com:me/kiss.git"
    assert git_sync.public_url("/tmp/x/origin.git") == "/tmp/x/origin.git"
    assert git_sync.public_url("https://user:p@ss@host/r") == "https://host/r"


def test_known_work_dirs_lists_the_configured_folder_then_the_recent_ones(
    tmp_path: Path,
    restored_config: None,
) -> None:
    """``config.json``'s ``work_dir`` first, the recent dirs next, no duplicates or blanks."""
    kiss, other = tmp_path / "kiss", tmp_path / "other"
    kiss.mkdir()
    other.mkdir()
    set_known_work_dirs(str(kiss), [str(kiss), "", str(other), str(tmp_path / "gone")])
    assert git_sync.known_work_dirs() == [str(kiss), str(other)]
    set_known_work_dirs("", [])
    assert git_sync.known_work_dirs() == []


def test_sync_with_a_url_syncs_the_work_dir_when_it_is_a_clone_of_it(tmp_path: Path) -> None:
    """A machine's work dir that is a clone of the repository is synced in place."""
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    pushed = _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    origin = git(mine, "remote", "get-url", "origin")
    # The URL may be spelt differently from the clone's remote (a trailing slash here).
    report = sync(str(mine), "origin/main", origin + "/")
    assert report.startswith(f"Checkout {mine.resolve().as_posix()}, branch main")
    assert f"Fast-forwarded to origin/main: now at {pushed[:12]}" in report
    assert git(mine, "rev-parse", "HEAD") == pushed


def test_sync_with_a_url_finds_the_checkout_among_the_daemons_work_dirs(
    tmp_path: Path,
    restored_config: None,
) -> None:
    """The sub-task sent to a machine runs where its daemon runs; the clone may be elsewhere.

    The work dir is the machine's home (not a checkout), then a clone of
    another repository: in both cases the clone of the named repository
    among the daemon's known work dirs is synced, and the report says so.
    A path of the sending machine that does not exist here is fine too.
    """
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    pushed = _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    origin = git(mine, "remote", "get-url", "origin")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other,) = remote_and_clones(other_dir, "clone")
    home = tmp_path / "home"
    home.mkdir()
    set_known_work_dirs(str(tmp_path / "missing"), [str(home), str(other), str(mine)])
    report = sync(str(home), "origin/main", origin)
    assert report.startswith(
        f"{home} is not a git checkout with a remote 'origin'; syncing the checkout of that "
        f"repository at {mine} instead.\nCheckout {mine.resolve().as_posix()}, branch main"
    )
    assert f"Fast-forwarded to origin/main: now at {pushed[:12]}" in report
    assert git(mine, "rev-parse", "HEAD") == pushed
    # The work dir is a clone of another repository: same thing, said differently.
    report = sync(str(other), "origin/main", origin)
    other_url = git(other, "remote", "get-url", "origin")
    assert report.startswith(
        f"{other} is a clone of {other_url}, not of {origin}; syncing the checkout of that "
        f"repository at {mine} instead."
    )
    assert "Pushed main to origin: Everything up-to-date" in report
    # The sending machine's path that does not exist here.
    report = sync("/Users/someone/work/kiss", "origin/main", origin)
    assert report.startswith("/Users/someone/work/kiss is not a git checkout")
    assert f"at {mine} instead." in report.splitlines()[0]
    # Nothing known is a clone of the repository: a clear error, nothing synced.
    set_known_work_dirs(str(home), [str(other)])
    with pytest.raises(GitSyncError) as info:
        sync(str(home), "origin/main", origin)
    message = str(info.value)
    assert message.startswith(f"{home} is not a git checkout with a remote 'origin', and no ")
    assert f"checkout of {origin} either." in message
    assert "work_dir option" in message
    assert sync_sea.git_sync(str(home), url=origin) == message
    # Without a URL the work dir is taken as is, as before.
    with pytest.raises(GitSyncError, match="git rev-parse --show-toplevel failed"):
        sync(str(home))


def test_sync_names_the_discovered_checkout_when_its_merge_stops_on_conflicts(
    tmp_path: Path,
    restored_config: None,
) -> None:
    """A conflict in a checkout found from the URL is reported with that checkout's path.

    The agent's shell and file tools work in its work dir; the report
    tells it where the conflicted files and the merge in progress are.
    """
    mine, theirs = remote_and_clones(tmp_path, "mine", "theirs")
    _commit(theirs, "a.txt", "theirs\n")
    git(theirs, "push", "-q", "origin", "main")
    _commit(mine, "a.txt", "mine\n")
    origin = git(mine, "remote", "get-url", "origin")
    home = tmp_path / "home"
    home.mkdir()
    set_known_work_dirs(str(mine), [])
    with pytest.raises(GitSyncError) as info:
        sync(str(home), "origin/main", origin)
    message = str(info.value)
    assert message.startswith(
        f"{home} is not a git checkout with a remote 'origin'; syncing the checkout of that "
        f"repository at {mine} instead.\nMerging origin/main stopped on 1 conflicted file(s)"
    )
    assert "  a.txt\n" in message
    assert message.endswith(
        f"(The files above are relative to {mine}, not to the work dir: edit them there and "
        f"run git with `-C {mine}`.)"
    )
    assert git(mine, "rev-parse", "-q", "--verify", "MERGE_HEAD")
    # Resolved there, the next call (same discovery) commits nothing new and pushes.
    (mine / "a.txt").write_text("both\n", encoding="utf-8")
    git(mine, "add", "a.txt")
    git(mine, "commit", "-q", "--no-edit")
    report = sync(str(home), "origin/main", origin)
    assert report.startswith(f"{home} is not a git checkout")
    assert "Pushed main to origin:" in report
    assert git(mine, "rev-parse", "HEAD") == _remote_main(mine)
