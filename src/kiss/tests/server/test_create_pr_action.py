# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Create PR button (``createPrAction`` and its ``pr_done``).

A real daemon is driven over its local connection, in real git
repositories whose remote is a bare repository on this disk, so a
press pushes for real and never reaches a Git host (the one test of a
logged-in ``gh`` lets it ask GitHub about a repository GitHub cannot
have).

Not covered, because nothing here may stand in for GitHub:

* a ``gh pr create`` that succeeds and an open pull request that is
  reused (``pr_flow._reused_or_created``) need a repository on a real
  GitHub that the tests could write to;
* ``_create_pr``'s last-resort ``except`` needs git, gh or the
  description writer to raise, which none does on any input.

Every test but one runs with ``gh`` logged out (an empty
``GH_CONFIG_DIR`` and no token in the environment), whatever account
the machine running the tests has.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from urllib.parse import parse_qs

import pytest
from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.git_worktree import _reclaim_process_lock, repo_lock
from kiss.core.utils import rmtree_force
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.test_remote_machine_settings import REPLY_SECONDS, RemoteMachineCase

TAB = "tab-pr"
ON_ANOTHER_HOST = "Open the pull request on your Git host."
NOT_INCLUDED = " (uncommitted changes are not included)"
# An owner github.com cannot have (a name there holds no "--"), so a test
# that lets gh ask about one of its repositories reaches nobody's.
NO_SUCH_OWNER = "kiss--no-such-owner"
GH_LOGIN_ENV = (
    "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_REPO",
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout.strip()


def _commit(work: Path, name: str, message: str) -> None:
    """Commit a new file *name* in *work* under *message*."""
    (work / name).write_text(name + "\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", message)


def _repo(root: Path, name: str, base: str = "main") -> tuple[Path, Path]:
    """Make ``root/name`` a repository with one commit on *base*, pushed to a new bare ``origin``.

    Returns:
        The repository and its bare remote.
    """
    work, bare = root / name, root / f"{name}.git"
    _git(root, "init", "-q", "--bare", "-b", base, str(bare))
    _git(root, "init", "-q", "-b", base, str(work))
    _git(work, "config", "user.name", "T")
    _git(work, "config", "user.email", "t@example.com")
    _commit(work, "base.txt", "base")
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-q", "-u", "origin", base)
    return work, bare


def _branch(work: Path, name: str, *subjects: str) -> None:
    """Check out a new branch *name* at ``main`` and commit one file under each of *subjects*."""
    _git(work, "checkout", "-q", "-b", name, "main")
    for number, subject in enumerate(subjects):
        _commit(work, f"{name.replace('/', '-')}-{number}.txt", subject)


def _checkout_under_the_lock(work: Path, branch: str) -> None:
    """Check *branch* out in *work* as another tab's git step does: under the repository's lock."""
    with repo_lock(work):
        _git(work, "checkout", "-q", branch)


def _outcome(done: dict[str, Any]) -> tuple[bool, bool, str, str]:
    """What a ``pr_done`` says: ``(success, created, url, message)``."""
    return done["success"], done["created"], done["url"], done["message"]


async def _press(
    case: Any, work: Path | str, tab: str = TAB, **fields: Any,
) -> list[dict[str, Any]]:
    """Press Create PR for *tab* in *work* on *case*'s connection.

    Returns:
        The press's toasts in order and, last, its ``pr_done``.
    """
    cmd = {"type": "createPrAction", "tabId": tab, "workDir": str(work), **fields}
    case.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
    await case.writer.drain()
    seen: list[dict[str, Any]] = []
    async with asyncio.timeout(REPLY_SECONDS):
        while not seen or seen[-1]["type"] != "pr_done":
            line = await case.reader.readline()
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") in ("notification", "pr_done") and msg.get("tabId") == tab:
                seen.append(msg)
    return seen


class _NoGhLogin:
    """Tests of a daemon that finds ``gh`` logged out."""

    @pytest.fixture(autouse=True)
    def _log_gh_out(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.monkeypatch = monkeypatch
        monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "no-gh-login"))
        for name in GH_LOGIN_ENV:
            monkeypatch.delenv(name, raising=False)


class _DaemonCase(IsolatedAsyncioTestCase):
    """A daemon serving its local connection, and a folder for the test's repositories."""

    async def asyncSetUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=str(self.root),
            url_file=self.root / "remote-url.json",
            local_endpoint_file=self.root / "sorcar-local.json",
        )
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_private_async()
        self.reader, self.writer = await open_local_connection(self.server)
        self.addCleanup(self.writer.close)


class TestCreatePr(_NoGhLogin, _DaemonCase):
    """One press pushes the branch's commits and says how to open their pull request."""

    async def test_a_branch_is_pushed_and_its_pull_request_left_to_the_git_host(self) -> None:
        work, bare = _repo(self.root, "feature")
        _branch(work, "feature/x", "Add x")
        (work / "draft.txt").write_text("not committed\n")

        toast, done = await _press(self, work)

        message = f"Pushed feature/x to origin. {ON_ANOTHER_HOST}{NOT_INCLUDED}"
        self.assertEqual(_outcome(done), (True, False, "", message))
        self.assertEqual(done["workDir"], str(work))
        self.assertEqual(
            (toast["id"], toast["severity"], toast["message"]),
            (f"create-pr:{TAB}", "info", message),
        )
        self.assertEqual(_git(bare, "rev-parse", "feature/x"), _git(work, "rev-parse", "HEAD"))
        upstream = _git(work, "rev-parse", "--abbrev-ref", "feature/x@{u}")
        self.assertEqual(upstream, "origin/feature/x")
        self.assertNotIn("draft.txt", _git(bare, "ls-tree", "--name-only", "feature/x"))
        # The outcome is the tab's alone: no chat's transcript holds it.
        th._flush_chat_events()
        stored = th._get_db().execute(
            "SELECT COUNT(*) FROM events WHERE event_json LIKE ?", (f"%{work}%",),
        ).fetchone()[0]
        self.assertEqual(stored, 0)

    async def test_commits_on_the_base_branch_go_to_a_branch_of_their_own(self) -> None:
        # ``origin/HEAD`` names the remote's default branch, whatever it is called.
        work, bare = _repo(self.root, "onbase", base="trunk")
        _git(work, "remote", "set-head", "origin", "trunk")
        _commit(work, "fix.txt", "Fix: the Thing!! (v2)")
        head = "pr/fix-the-thing-v2-" + _git(work, "rev-parse", "--short", "HEAD")

        done = (await _press(self, work))[-1]

        message = f"Pushed {head} to origin. {ON_ANOTHER_HOST}"
        self.assertEqual(_outcome(done), (True, False, "", message))
        self.assertEqual(_git(bare, "rev-parse", head), _git(work, "rev-parse", "HEAD"))
        # The remote's base branch and the local branch's upstream are as they were.
        self.assertEqual(_git(bare, "log", "-1", "--format=%s", "trunk"), "base")
        self.assertEqual(_git(work, "rev-parse", "--abbrev-ref", "trunk@{u}"), "origin/trunk")

    async def test_a_tag_named_like_the_branch_changes_nothing(self) -> None:
        # git abbreviates a branch that shares its name with a tag to ``heads/<name>``.
        # Commits on the base branch still go to a branch of their own.
        work, bare = _repo(self.root, "tagged-base")
        _git(work, "tag", "main")
        _commit(work, "fix.txt", "Fix it")
        head = "pr/fix-it-" + _git(work, "rev-parse", "--short", "HEAD")

        done = (await _press(self, work))[-1]

        message = f"Pushed {head} to origin. {ON_ANOTHER_HOST}"
        self.assertEqual(_outcome(done), (True, False, "", message))
        pushed = _git(bare, "rev-parse", f"refs/heads/{head}")
        self.assertEqual(pushed, _git(work, "rev-parse", "HEAD"))
        self.assertEqual(_git(bare, "log", "-1", "--format=%s", "refs/heads/main"), "base")

        # Any other branch is pushed, and tracked, under its own name.
        work, bare = _repo(self.root, "tagged-feature")
        _branch(work, "feature", "Add feature")
        _git(work, "tag", "feature", "main")
        done = (await _press(self, work, tab="tab-feature"))[-1]
        self.assertEqual(done["message"], f"Pushed feature to origin. {ON_ANOTHER_HOST}")
        pushed = _git(bare, "rev-parse", "refs/heads/feature")
        self.assertEqual(pushed, _git(work, "rev-parse", "HEAD"))
        self.assertEqual(_git(work, "config", "branch.feature.merge"), "refs/heads/feature")

    async def test_a_branch_with_nothing_new_is_left_alone(self) -> None:
        work, bare = _repo(self.root, "nothing")

        done = (await _press(self, work))[-1]

        self.assertEqual(_outcome(done), (
            True, False, "",
            "Nothing to open a pull request for: main has no commits that origin/main lacks.",
        ))
        self.assertEqual(_git(bare, "branch", "--format=%(refname:short)"), "main")

    async def test_a_folder_that_cannot_be_pushed_says_why(self) -> None:
        plain = self.root / "plain"
        plain.mkdir()
        detached, _ = _repo(self.root, "detached")
        _git(detached, "checkout", "-q", "--detach")
        no_remote, _ = _repo(self.root, "no-remote")
        _git(no_remote, "remote", "remove", "origin")
        no_base, _ = _repo(self.root, "no-base")
        _git(no_base, "update-ref", "-d", "refs/remotes/origin/main")
        reasons = {
            plain: "Not a git repository.",
            detached: "Check out a branch first: HEAD is detached.",
            no_remote: "This repository has no remote to push to.",
            no_base: (
                "Could not find the default branch of origin (no origin/HEAD, origin/main or "
                "origin/master). Run: git fetch origin && git remote set-head origin -a"
            ),
        }
        for number, (work, reason) in enumerate(reasons.items()):
            with self.subTest(folder=work.name):
                toast, done = await _press(self, work, tab=f"tab-{number}")
                self.assertEqual(_outcome(done), (False, False, "", reason))
                self.assertEqual((toast["severity"], toast["message"]), ("error", reason))

    async def test_the_remote_is_the_branchs_own_then_origin_then_the_only_one(self) -> None:
        work, origin = _repo(self.root, "remotes")
        fork = self.root / "fork.git"
        _git(self.root, "init", "-q", "--bare", "-b", "main", str(fork))
        _git(work, "remote", "add", "fork", str(fork))
        _git(work, "fetch", "-q", "fork")
        _git(work, "push", "-q", "fork", "main")
        _branch(work, "to-fork", "For the fork")
        _git(work, "config", "branch.to-fork.remote", "fork")

        done = (await _press(self, work))[-1]
        self.assertEqual(done["message"], f"Pushed to-fork to fork. {ON_ANOTHER_HOST}")
        self.assertEqual(_git(origin, "branch", "--format=%(refname:short)"), "main")

        _branch(work, "to-origin", "For origin")
        done = (await _press(self, work, tab="tab-origin"))[-1]
        self.assertEqual(done["message"], f"Pushed to-origin to origin. {ON_ANOTHER_HOST}")

        _git(work, "remote", "rename", "origin", "upstream")
        _git(work, "remote", "remove", "fork")
        _branch(work, "to-the-only-one", "For the only remote")
        done = (await _press(self, work, tab="tab-only"))[-1]
        self.assertEqual(done["message"], f"Pushed to-the-only-one to upstream. {ON_ANOTHER_HOST}")

    async def test_a_failed_push_reports_gits_reason(self) -> None:
        work, bare = _repo(self.root, "rejected")
        _branch(work, "f", "Add f")
        _git(work, "push", "-q", "-u", "origin", "f")
        _git(work, "commit", "-q", "--amend", "-m", "Add f, reworded")
        (work / "draft.txt").write_text("not committed\n")

        done = (await _press(self, work))[-1]
        rejected = f"Push failed: ! [rejected] f -> f (non-fast-forward){NOT_INCLUDED}"
        self.assertEqual(_outcome(done), (False, False, "", rejected))

        # A remote that is gone fails before any branch is looked at.  (git's object files
        # are read-only, which stops a plain ``shutil.rmtree`` on Windows.)
        rmtree_force(bare)
        done = (await _press(self, work, tab="tab-gone"))[-1]
        gone = f"Push failed: fatal: '{bare}' does not appear to be a git repository"
        self.assertEqual(done["message"], gone + NOT_INCLUDED)

    async def test_a_checkout_switched_while_the_description_is_written_is_not_pushed(self) -> None:
        # Two commits on the base branch itself, and a branch with one more that is not for review.
        work, bare = _repo(self.root, "switched")
        _commit(work, "one.txt", "Add one")
        _commit(work, "two.txt", "Add two")
        _git(work, "checkout", "-q", "-b", "private")
        _commit(work, "private.txt", "Private notes")
        _git(work, "checkout", "-q", "main")
        lock = repo_lock(work)
        another_tab = threading.Thread(target=_checkout_under_the_lock, args=(work, "private"))

        # The press's first step takes the repository's lock and then waits for
        # the one held here.  Another tab's checkout queues behind that step, so
        # it runs while the description of the two commits is being written.
        with _reclaim_process_lock(work):
            press = asyncio.ensure_future(_press(self, work))
            while lock.acquire(blocking=False):
                lock.release()
                await asyncio.sleep(0.01)
            another_tab.start()
        writing, toast, done = await press
        another_tab.join()

        moved = (
            "The checkout changed while the pull request was being prepared. "
            "Press Create PR again."
        )
        self.assertEqual(writing["message"], "Writing the pull request description…")
        self.assertEqual((toast["severity"], toast["message"]), ("error", moved))
        self.assertEqual(_outcome(done), (False, False, "", moved))
        self.assertEqual(_git(work, "rev-parse", "--abbrev-ref", "HEAD"), "private")
        self.assertEqual(_git(bare, "branch", "--format=%(refname:short)"), "main")
        self.assertEqual(_git(bare, "log", "-1", "--format=%s", "main"), "base")

    def _on_github(self, name: str, url: str) -> tuple[Path, Path]:
        """Make a repository whose ``origin`` is *url*, which git rewrites to a bare one here.

        ``gh`` is also taken off the ``PATH``; git stays on it through
        a link of its own, wherever the two are installed.
        """
        work, bare = _repo(self.root, name)
        _git(work, "config", "remote.origin.url", url)
        _git(work, "config", f"url.{bare}.insteadOf", url)
        tools = self.root / f"{name}-bin"
        tools.mkdir()
        git = shutil.which("git")
        assert git is not None
        (tools / "git").symlink_to(git)
        kept = [f for f in os.environ["PATH"].split(os.pathsep) if not (Path(f) / "gh").exists()]
        self.monkeypatch.setenv("PATH", os.pathsep.join([str(tools), *kept]))
        return work, bare

    async def test_a_github_remote_gets_the_page_that_opens_the_pull_request(self) -> None:
        work, bare = self._on_github("widgets", "https://github.com/acme/widgets.git")
        _branch(work, "feature/y", "Add y & z\n\nWhy: because.\nSecond line.")

        toast, done = await _press(self, work)

        url = (
            "https://github.com/acme/widgets/compare/main...feature/y"
            "?quick_pull=1&title=Add+y+%26+z&body=Why%3A+because.%0ASecond+line."
        )
        message = f"Pushed feature/y. Open the pull request page to finish: {url}"
        self.assertEqual(_outcome(done), (True, False, url, message))
        self.assertEqual(toast["message"], message)
        self.assertEqual(_git(bare, "rev-parse", "feature/y"), _git(work, "rev-parse", "HEAD"))

    async def test_the_pull_request_page_is_the_pushed_repositorys(self) -> None:
        # origin is fetched from one repository on github.com and pushed to three others,
        # one of them on github.com as well (bare repositories here, for git).
        work, fetched = self._on_github("fetched", "https://github.com/acme/widgets.git")
        mirror, fork, backup = (self.root / f"{name}.git" for name in ("mirror", "fork", "backup"))
        for bare in (mirror, fork, backup):
            _git(self.root, "clone", "-q", "--bare", str(fetched), str(bare))
        _git(work, "config", f"url.{fork}.insteadOf", "git@github.com:me/widgets-fork.git")
        for address in (str(mirror), "git@github.com:me/widgets-fork.git", str(backup)):
            _git(work, "config", "--add", "remote.origin.pushurl", address)
        _branch(work, "f", "Add f")

        done = (await _press(self, work))[-1]

        # The page is in the repository on github.com that has the branch.
        url = "https://github.com/me/widgets-fork/compare/main...f?quick_pull=1&title=Add+f&body="
        message = f"Pushed f. Open the pull request page to finish: {url}"
        self.assertEqual(_outcome(done), (True, False, url, message))
        head = _git(work, "rev-parse", "HEAD")
        for bare in (mirror, fork, backup):
            self.assertEqual(_git(bare, "rev-parse", "refs/heads/f"), head)
        self.assertEqual(_git(fetched, "branch", "--format=%(refname:short)"), "main")

    async def test_a_cleared_push_address_names_no_repository(self) -> None:
        version = re.search(r"(\d+)\.(\d+)", _git(self.root, "version"))
        assert version is not None
        if (int(version[1]), int(version[2])) < (2, 46):
            self.skipTest("an empty pushurl clears the ones before it from git 2.46 on")
        work, bare = self._on_github("cleared", "git@github.com:acme/widgets.git")
        # git pushes to the last address only: the empty one clears the first.
        for address in ("git@github.com:gone/widgets.git", "", "git@github.com:acme/widgets.git"):
            _git(work, "config", "--add", "remote.origin.pushurl", address)
        _branch(work, "f", "Add f")

        done = (await _press(self, work))[-1]

        page = "https://github.com/acme/widgets/compare/main...f?"
        self.assertTrue(done["url"].startswith(page), done)
        self.assertEqual(_git(bare, "rev-parse", "refs/heads/f"), _git(work, "rev-parse", "HEAD"))

    async def test_a_long_description_is_cut_to_fit_the_pages_address(self) -> None:
        work, _bare = self._on_github("long", "git@github.com:acme/widgets.git")
        body = "é long body " * 1000
        _branch(work, "f", "Add f\n\n" + body)

        done = (await _press(self, work))[-1]

        page, _, query = done["url"].partition("?")
        self.assertEqual(page, "https://github.com/acme/widgets/compare/main...f")
        # As much of the body as the address has room for, and no more.
        self.assertTrue(5990 < len(done["url"]) < 6000, len(done["url"]))
        sent = parse_qs(query)
        self.assertEqual((sent["quick_pull"], sent["title"]), (["1"], ["Add f"]))
        self.assertTrue(body.startswith(sent["body"][0]))

    async def test_several_commits_get_a_written_description(self) -> None:
        work, bare = self._on_github("several", "https://github.com/acme/widgets")
        _branch(work, "f", "Add one", "Add two")

        writing, toast, done = await _press(self, work)

        self.assertEqual(writing["message"], "Writing the pull request description…")
        self.assertEqual((writing["id"], toast["id"]), (f"create-pr:{TAB}",) * 2)
        self.assertEqual(toast["message"], done["message"])
        # The written subject is the title; no commit's own subject is taken for it.
        title = parse_qs(done["url"].partition("?")[2])["title"][0]
        self.assertTrue(title.strip())
        self.assertNotIn(title, ("Add one", "Add two"))
        self.assertEqual(_git(bare, "rev-parse", "f"), _git(work, "rev-parse", "HEAD"))

    async def test_a_second_press_of_the_tab_is_refused_while_one_runs(self) -> None:
        work, bare = _repo(self.root, "busy")
        _branch(work, "f", "Add f")
        backend = self.server._agent_server
        with backend._state_lock:
            backend._pr_tabs.add(TAB)

        seen = await _press(self, work)

        # The refusal is the pressing connection's alone: no toast is broadcast.
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            _outcome(seen[0]),
            (False, False, "", "A pull request of this tab is already being created."),
        )
        self.assertEqual(_git(bare, "branch", "--format=%(refname:short)"), "main")
        # Once the tab's press has ended the button works again.
        with backend._state_lock:
            backend._pr_tabs.discard(TAB)
        self.assertTrue((await _press(self, work))[-1]["success"])
        self.assertEqual(_git(bare, "rev-parse", "f"), _git(work, "rev-parse", "HEAD"))

    async def test_a_press_another_daemon_forwards_is_answered_on_its_link_alone(self) -> None:
        work, bare = _repo(self.root, "linked")
        _branch(work, "f", "Add one", "Add two")
        plain = self.root / "plain"
        plain.mkdir()
        # Another daemon's link: a tab of its picked onto this machine presses through it.
        endpoint = local_endpoint.read_endpoint(self.root / "sorcar-local.json")
        assert endpoint is not None
        ws = await connect(
            endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint), compression=None,
            close_timeout=2.0,
        )
        self.addAsyncCleanup(ws.close)
        await ws.send(json.dumps({"type": "auth", "token": endpoint.token, "link": True}))
        self.assertEqual(json.loads(await ws.recv())["type"], "auth_ok")
        link = SimpleNamespace(reader=LocalReader(ws), writer=LocalWriter(ws))

        seen = await _press(link, work)

        # The outcome alone, as the reply: that daemon shows its tab's toasts itself.
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            _outcome(seen[0]), (True, False, "", f"Pushed f to origin. {ON_ANOTHER_HOST}"),
        )
        self.assertEqual(_git(bare, "rev-parse", "f"), _git(work, "rev-parse", "HEAD"))
        # This daemon's own pages heard nothing of it, not even that the description
        # was being written: the next the tab hears here is a press of its own.
        seen = await _press(self, plain)
        self.assertEqual([msg["message"] for msg in seen], ["Not a git repository."] * 2)

    async def test_a_press_whose_thread_cannot_start_leaves_the_button_usable(self) -> None:
        if not thread_start_can_be_starved():
            self.skipTest("RLIMIT_NPROC cannot starve Thread.start on this host")
        work, bare = _repo(self.root, "starved")
        _branch(work, "f", "Add f")
        cmd = {"type": "createPrAction", "tabId": TAB, "workDir": str(work)}

        # Thread exhaustion for real: threads count against the lowered limit.
        with nproc_limit_lowered_to_one(), self.assertRaises(RuntimeError):
            self.server._agent_server._handle_command(cmd)

        # Nothing was pushed, and the tab is not left refusing the next press.
        self.assertEqual(_git(bare, "branch", "--format=%(refname:short)"), "main")
        self.assertTrue((await _press(self, work))[-1]["success"])
        self.assertEqual(_git(bare, "rev-parse", "f"), _git(work, "rev-parse", "HEAD"))


class TestCreatePrWithGhLoggedIn(_DaemonCase):
    """What a ``gh`` that is logged in adds, as far as a repository GitHub lacks can show."""

    @pytest.fixture(autouse=True)
    def _point_gh_at_another_repository(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GH_REPO", f"{NO_SUCH_OWNER}/gh-env")

    async def test_gh_is_asked_about_the_pushed_repository_and_its_failure_kept(self) -> None:
        logged_in = shutil.which("gh") is not None and subprocess.run(
            ["gh", "auth", "status"], capture_output=True, check=False,
        ).returncode == 0
        if not logged_in:
            self.skipTest("needs a gh that is logged in on this machine")
        # origin is on github.com (git pushes to the bare repository here).  gh's
        # own settings name two other repositories: the one in its environment
        # and a remote marked as its default.
        work, bare = _repo(self.root, "named")
        pushed_to = f"{NO_SUCH_OWNER}/pushed-to"
        _git(work, "config", "remote.origin.url", f"https://github.com/{pushed_to}.git")
        _git(work, "config", f"url.{bare}.insteadOf", f"https://github.com/{pushed_to}.git")
        _git(work, "remote", "add", "upstream", f"https://github.com/{NO_SUCH_OWNER}/gh-default")
        _git(work, "config", "remote.upstream.gh-resolved", "base")
        _branch(work, "f", "Add f")

        done = (await _press(self, work))[-1]

        page = f"https://github.com/{pushed_to}/compare/main...f?quick_pull=1&title=Add+f&body="
        self.assertEqual(_outcome(done)[:3], (True, False, page))
        # GitHub has no such repository, and says which one it was asked about.
        self.assertRegex(
            done["message"],
            rf"^Pushed f\. Open the pull request page to finish: {re.escape(page)}"
            rf" \(gh failed: .*'{pushed_to}'.*\)$",
        )
        self.assertEqual(_git(bare, "rev-parse", "f"), _git(work, "rev-parse", "HEAD"))


class TestCreatePrOnMachine(_NoGhLogin, RemoteMachineCase):
    """``createPrAction`` stamped ``machine`` pushes and reports on that machine."""

    async def test_the_pull_request_is_the_machines(self) -> None:
        box, box_bare = _repo(Path(self.tmpdir), self.box_work.name)
        home, home_bare = _repo(Path(self.tmpdir), self.home_work.name)
        _branch(box, "f", "Add one", "Add two")
        _branch(home, "f", "Add f")

        # No folder named: the machine's own configured one.  The machine
        # writes the description of the two commits; the toasts are this
        # daemon's, so that step shows none.
        creating, toast, done = await _press(self, "", machine="box")

        message = f"box: Pushed f to origin. {ON_ANOTHER_HOST}"
        self.assertEqual(creating["message"], "Creating the pull request on box…")
        self.assertEqual((toast["severity"], toast["message"]), ("info", message))
        self.assertEqual(_outcome(done), (True, False, "", message))
        self.assertEqual(done["workDir"], str(box))
        self.assertEqual(_git(box_bare, "rev-parse", "f"), _git(box, "rev-parse", "HEAD"))
        self.assertEqual(_git(home_bare, "branch", "--format=%(refname:short)"), "main")

        # A machine nobody registered: the link's error is the outcome.
        seen = await _press(self, home, tab="tab-nowhere", machine="nowhere")
        self.assertEqual(_outcome(seen[-1]), (False, False, "", "no machine named 'nowhere'"))
        self.assertEqual(seen[-2]["severity"], "error")
        self.assertEqual(_git(home_bare, "branch", "--format=%(refname:short)"), "main")
