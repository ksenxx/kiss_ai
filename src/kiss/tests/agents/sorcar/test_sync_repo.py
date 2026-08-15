# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``scripts/sync-repo.sh``.

``./sorcar-cloud`` no longer copies the project folder to the remote host: it
syncs both checkouts through ``origin``, so a deployment can commit and push
like any clone and an agent's work on the server comes back to the laptop.
The tests below run the real script over three real repositories -- a bare
``origin.git``, a clone standing in for the laptop, and a folder standing in
for the server -- with a fake ``ssh`` on ``PATH`` that executes the remote
half locally.

What has to hold after a sync, and is asserted here: both working trees hold
the same commit on every branch, uncommitted work on either side survives as
a commit, a branch that only one side has appears on the other, a branch that
diverged is merged, and one that cannot be merged is reported instead of
being lost or forced.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "sync-repo.sh"

_FAKE_SSH = """#!/bin/bash
# Stand-in for ssh: run the command locally, keeping stdin (the script the
# driver pipes to "bash -s") attached.
shift                       # drop the user@host argument
exec bash -c "$*"
"""


class SyncRepoTest(unittest.TestCase):
    """The laptop's checkout and the deployment must end up identical."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.origin = self.tmp / "origin.git"
        self.local = self.tmp / "laptop"
        self.remote = self.tmp / "server" / "project"
        bindir = self.tmp / "bin"
        bindir.mkdir()
        fake_ssh = bindir / "ssh"
        fake_ssh.write_text(_FAKE_SSH)
        fake_ssh.chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bindir}:{self.env['PATH']}"
        # A sandbox HOME keeps the developer's ~/.gitconfig out of the way.
        self.env["HOME"] = str(self.tmp / "home")
        (self.tmp / "home").mkdir()
        self.env["GIT_CONFIG_NOSYSTEM"] = "1"

        self.git(self.tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        self.git(self.tmp, "clone", "-q", str(self.origin), str(self.local))
        self.git(self.local, "config", "user.name", "Laptop")
        self.git(self.local, "config", "user.email", "laptop@example.com")
        self.write(self.local, "README.md", "hello\n")
        self.write(self.local, ".gitignore", "ignored.txt\n")
        self.commit(self.local, "first commit")
        self.git(self.local, "push", "-q", "-u", "origin", "main")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- helpers ----------------------------------------------------------
    def git(self, repo: Path, *args: str) -> str:
        """Run git in *repo* and return its stripped stdout."""
        done = subprocess.run(
            ["git", "-C", str(repo), *args], env=self.env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, f"git {args}: {done.stderr}")
        return done.stdout.strip()

    def write(self, repo: Path, name: str, text: str) -> None:
        """Write *text* to *name* inside *repo*, creating parents."""
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, repo: Path, message: str) -> str:
        """Commit everything in *repo* and return the new commit's sha."""
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def sync(self, branch: str = "main",
             **overrides: str) -> subprocess.CompletedProcess[str]:
        """Run the real script the way ./sorcar-cloud runs it."""
        return subprocess.run(
            ["bash", str(_SCRIPT), "user@server", str(self.local),
             str(self.remote), branch],
            env={**self.env, **overrides}, capture_output=True, text=True,
            timeout=600,
        )

    def sync_ok(self, branch: str = "main",
                **overrides: str) -> subprocess.CompletedProcess[str]:
        """Run the script and require it to succeed."""
        done = self.sync(branch, **overrides)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done

    def tip(self, repo: Path, ref: str) -> str:
        """Return the commit *ref* points at in *repo*."""
        return self.git(repo, "rev-parse", ref)

    def branches(self, repo: Path) -> set[str]:
        """Return the local branch names of *repo*."""
        return set(self.git(
            repo, "for-each-ref", "--format=%(refname:short)",
            "refs/heads").split())

    def assert_in_sync(self, *branches: str) -> None:
        """Both checkouts and origin must agree on every named branch."""
        for branch in branches:
            local = self.tip(self.local, branch)
            self.assertEqual(self.tip(self.remote, branch), local,
                             f"{branch} differs between the two checkouts")
            self.assertEqual(self.tip(self.origin, branch), local,
                             f"{branch} differs on origin")
        self.assertEqual(self.git(self.local, "status", "--porcelain"), "")
        self.assertEqual(self.git(self.remote, "status", "--porcelain"), "")

    # --- tests ------------------------------------------------------------
    def test_first_deploy_creates_the_remote_checkout(self) -> None:
        """An empty folder on the server becomes a clone of this branch."""
        self.write(self.local, "src/app.py", "print('hi')\n")
        self.write(self.local, "ignored.txt", "build noise\n")

        self.sync_ok()

        self.assertTrue((self.remote / ".git").is_dir())
        self.assertEqual((self.remote / "src" / "app.py").read_text(),
                         "print('hi')\n")
        self.assertEqual(self.git(self.remote, "symbolic-ref", "--short", "HEAD"),
                         "main")
        # The uncommitted file travelled as a commit; the ignored one did not.
        self.assertFalse((self.remote / "ignored.txt").exists())
        self.assertTrue((self.local / "ignored.txt").exists())
        self.assert_in_sync("main")

    def test_second_run_changes_nothing(self) -> None:
        """A deploy with nothing to carry is a no-op, not a new commit."""
        self.sync_ok()
        before = self.tip(self.local, "main")

        self.sync_ok()

        self.assertEqual(self.tip(self.local, "main"), before)
        self.assert_in_sync("main")

    def test_work_done_on_the_server_comes_back(self) -> None:
        """An agent's commits and leftover edits reach the laptop."""
        self.sync_ok()
        self.write(self.remote, "server_work.py", "# written on the server\n")
        self.commit(self.remote, "work from the agent")
        self.write(self.remote, "still_editing.py", "# not committed there\n")

        self.sync_ok()

        self.assertTrue((self.local / "server_work.py").exists())
        self.assertTrue((self.local / "still_editing.py").exists())
        self.assert_in_sync("main")

    def test_both_sides_moved_on_is_merged(self) -> None:
        """Commits made on both machines end up on both machines."""
        self.sync_ok()
        self.write(self.remote, "on_server.txt", "server\n")
        self.commit(self.remote, "server side")
        self.write(self.local, "on_laptop.txt", "laptop\n")
        self.commit(self.local, "laptop side")

        self.sync_ok()

        for repo in (self.local, self.remote):
            self.assertTrue((repo / "on_server.txt").exists(), repo)
            self.assertTrue((repo / "on_laptop.txt").exists(), repo)
        self.assert_in_sync("main")

    def test_a_conflict_on_the_deployed_branch_stops_the_deploy(self) -> None:
        """Neither side's commit may be dropped, forced away or glossed over."""
        self.sync_ok()
        self.write(self.remote, "README.md", "written on the server\n")
        server_tip = self.commit(self.remote, "server edit")
        self.write(self.local, "README.md", "written on the laptop\n")
        laptop_tip = self.commit(self.local, "laptop edit")

        done = self.sync()

        output = done.stdout + done.stderr
        self.assertNotEqual(done.returncode, 0, output)
        self.assertIn("conflicting changes", output)
        self.assertIn("git merge origin/main", output)   # what to do about it
        # Both commits still exist, each on its own machine, both trees clean.
        self.assertEqual(self.tip(self.local, "main"), laptop_tip)
        self.assertEqual(self.tip(self.remote, "main"), server_tip)
        self.assertEqual(self.git(self.local, "status", "--porcelain"), "")
        self.assertEqual(self.git(self.remote, "status", "--porcelain"), "")

    def test_a_conflict_elsewhere_does_not_stop_the_deploy(self) -> None:
        """A branch nobody is deploying may lag behind, but must be named."""
        self.git(self.local, "checkout", "-q", "-b", "side")
        self.write(self.local, "side.txt", "start\n")
        self.commit(self.local, "side start")
        self.git(self.local, "push", "-q", "-u", "origin", "side")
        self.git(self.local, "checkout", "-q", "main")
        self.sync_ok()
        # Both machines change the same line of the same file on "side".
        self.git(self.remote, "checkout", "-q", "side")
        self.write(self.remote, "side.txt", "server\n")
        server_tip = self.commit(self.remote, "side on the server")
        self.git(self.remote, "checkout", "-q", "main")
        self.git(self.local, "checkout", "-q", "side")
        self.write(self.local, "side.txt", "laptop\n")
        laptop_tip = self.commit(self.local, "side on the laptop")
        self.git(self.local, "checkout", "-q", "main")

        done = self.sync_ok()

        output = done.stdout + done.stderr
        self.assertIn("side", output)
        self.assertIn("not in sync", output)
        # Each machine keeps its own commit on the branch that could not be
        # merged, and origin keeps the one that got there first.
        self.assertEqual(self.tip(self.local, "side"), laptop_tip)
        self.assertEqual(self.tip(self.remote, "side"), server_tip)
        self.assertEqual(self.tip(self.origin, "side"), laptop_tip)
        self.assert_in_sync("main")

    def test_a_local_only_branch_is_mirrored_everywhere(self) -> None:
        """A branch that only the laptop has appears on origin and the server."""
        self.git(self.local, "branch", "feature-x")
        self.git(self.local, "checkout", "-q", "feature-x")
        self.write(self.local, "feature.txt", "new idea\n")
        self.commit(self.local, "the idea")
        self.git(self.local, "checkout", "-q", "main")

        self.sync_ok()

        self.assertIn("feature-x", self.branches(self.remote))
        self.assert_in_sync("main", "feature-x")

    def test_a_rejected_side_branch_does_not_block_the_deployed_branch(self) -> None:
        """An archival branch must not make a valid deployment impossible.

        Git servers can reject one historical ref (for example, because it
        contains an oversized blob).  Sending that ref and the checked-out
        deployment branch in one push lets the rejection reject both.  The
        branch being deployed must travel independently; auxiliary failures
        remain warnings.
        """
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text(
            "#!/bin/bash\n"
            "while read -r old new ref; do\n"
            "  [ \"$ref\" != refs/heads/unpushable ] || exit 1\n"
            "done\n"
        )
        hook.chmod(0o755)
        self.git(self.local, "branch", "unpushable")
        self.write(self.local, "deploy-me.txt", "new main content\n")

        done = self.sync_ok()

        self.assertIn("could not be pushed", (done.stdout + done.stderr).lower())
        self.assertEqual((self.remote / "deploy-me.txt").read_text(), "new main content\n")
        self.assertNotIn("unpushable", self.branches(self.remote))
        self.assert_in_sync("main")

    def test_a_rejected_branch_is_not_retried_in_the_same_deploy(self) -> None:
        """A permanent rejection must not make pass three repeat a slow push.

        The third pass exists to collect work the server pushed.  Retrying a
        local ref that origin already rejected in pass one cannot collect
        anything; on a large historical branch it only adds another long,
        silent upload to every deployment.
        """
        rejected = self.tmp / "rejected-pushes"
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text(
            "#!/bin/bash\n"
            f"echo attempted >> {rejected}\n"
            "while read -r old new ref; do\n"
            "  [ \"$ref\" != refs/heads/unpushable ] || exit 1\n"
            "done\n"
        )
        hook.chmod(0o755)
        self.git(self.local, "branch", "unpushable")

        done = self.sync_ok()

        self.assertIn("could not be pushed", (done.stdout + done.stderr).lower())
        self.assertEqual(rejected.read_text().splitlines(), ["attempted"])
        self.assertNotIn("unpushable", self.branches(self.remote))
        self.assert_in_sync("main")

    def test_a_branch_only_origin_has_is_created_on_both(self) -> None:
        """Somebody else's branch lands in both checkouts."""
        # Pushing a ref without creating it locally is what another machine
        # having a branch looks like from here.
        self.git(self.local, "push", "-q", "origin", "main:refs/heads/hotfix")
        self.assertNotIn("hotfix", self.branches(self.local))

        self.sync_ok()

        self.assertIn("hotfix", self.branches(self.local))
        self.assertIn("hotfix", self.branches(self.remote))
        self.assert_in_sync("main", "hotfix")

    def test_a_branch_nobody_has_checked_out_is_synced_too(self) -> None:
        """Divergence on a branch that is not in any working tree is merged."""
        self.git(self.local, "checkout", "-q", "-b", "topic")
        self.write(self.local, "topic.txt", "topic\n")
        self.commit(self.local, "topic start")
        self.git(self.local, "push", "-q", "-u", "origin", "topic")
        self.git(self.local, "checkout", "-q", "main")
        self.sync_ok()
        # The server advances "topic" while both machines have "main" checked
        # out, so the merge on the laptop has to happen off to the side.
        self.git(self.remote, "checkout", "-q", "topic")
        self.write(self.remote, "from_server.txt", "server\n")
        self.commit(self.remote, "topic on the server")
        self.git(self.remote, "checkout", "-q", "main")
        self.git(self.local, "checkout", "-q", "topic")
        self.write(self.local, "from_laptop.txt", "laptop\n")
        self.commit(self.local, "topic on the laptop")
        self.git(self.local, "checkout", "-q", "main")

        self.sync_ok()

        self.assert_in_sync("main", "topic")
        files = self.git(self.local, "ls-tree", "--name-only", "topic")
        self.assertIn("from_server.txt", files)
        self.assertIn("from_laptop.txt", files)
        # The throw-away worktree used for that merge (on the server, where
        # "topic" was the one that had diverged) is gone again on both sides.
        for repo in (self.local, self.remote):
            self.assertEqual(
                [line for line in self.git(repo, "worktree", "list").splitlines()
                 if str(repo) not in line], [], repo)

    def test_the_agents_worktree_branches_stay_local(self) -> None:
        """kiss/wt-* branches belong to a running task, not to origin."""
        self.git(self.local, "branch", "kiss/wt-1234-abcd")

        done = self.sync_ok()

        self.assertIn("kiss/wt-1234-abcd", done.stdout + done.stderr)
        self.assertNotIn("kiss/wt-1234-abcd", self.branches(self.remote))
        self.assertEqual(
            self.git(self.origin, "for-each-ref", "--format=%(refname:short)",
                     "refs/heads/kiss"), "")

    def test_an_emptied_checkout_stops_the_deploy(self) -> None:
        """A folder that lost its files must not commit that loss to origin."""
        self.write(self.local, "a.txt", "a\n")
        self.write(self.local, "b.txt", "b\n")
        self.write(self.local, "c.txt", "c\n")
        self.commit(self.local, "three files")
        self.sync_ok()
        good = self.tip(self.origin, "main")
        for name in ("README.md", ".gitignore", "a.txt", "b.txt", "c.txt"):
            (self.remote / name).unlink()

        done = self.sync()

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Refusing to commit", done.stdout + done.stderr)
        self.assertEqual(self.tip(self.origin, "main"), good)

    def test_a_folder_full_of_files_is_adopted_not_deleted(self) -> None:
        """The first sync over an old file-copy deploy keeps its edits."""
        # What the previous sorcar-cloud left behind: the files, no .git.
        self.remote.mkdir(parents=True)
        for name in ("README.md", ".gitignore"):
            shutil.copy(self.local / name, self.remote / name)
        (self.remote / "README.md").write_text("edited on the server\n")
        (self.remote / "left_behind.txt").write_text("only on the server\n")

        self.sync_ok()

        self.assertEqual((self.remote / "README.md").read_text(),
                         "edited on the server\n")
        self.assertEqual((self.local / "README.md").read_text(),
                         "edited on the server\n")
        self.assertTrue((self.local / "left_behind.txt").exists())
        self.assert_in_sync("main")

    def test_deploying_another_branch_moves_the_server_checkout(self) -> None:
        """Switching branches keeps what the server was still working on."""
        self.sync_ok()
        self.write(self.remote, "unsaved.txt", "still editing on main\n")
        self.git(self.local, "checkout", "-q", "-b", "release")
        self.write(self.local, "release.txt", "shipping\n")
        self.commit(self.local, "release notes")

        self.sync_ok("release")

        self.assertEqual(
            self.git(self.remote, "symbolic-ref", "--short", "HEAD"), "release")
        self.assertTrue((self.remote / "release.txt").exists())
        # The unsaved file was committed on main before the switch, so it is
        # still on the laptop's main and nothing was thrown away.
        self.assertIn("unsaved.txt",
                      self.git(self.local, "ls-tree", "--name-only", "main"))
        self.assert_in_sync("main", "release")

    def test_replacing_every_file_is_still_refused(self) -> None:
        """Adding files must not buy permission to delete the old ones."""
        self.write(self.local, "a.txt", "a\n")
        self.write(self.local, "b.txt", "b\n")
        self.commit(self.local, "two more files")
        self.sync_ok()
        good = self.tip(self.origin, "main")
        for name in ("README.md", ".gitignore", "a.txt", "b.txt"):
            (self.remote / name).unlink()
        for name in ("w.txt", "x.txt", "y.txt", "z.txt"):
            (self.remote / name).write_text("new\n")

        done = self.sync()

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Refusing to commit", done.stdout + done.stderr)
        self.assertEqual(self.tip(self.origin, "main"), good)

    def test_a_detached_head_with_edits_is_refused(self) -> None:
        """Work on a detached HEAD could not reach origin, so say so."""
        self.sync_ok()
        self.git(self.local, "checkout", "-q", "--detach", "HEAD")
        self.write(self.local, "wandering.txt", "on no branch\n")

        done = self.sync()

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("detached HEAD", done.stdout + done.stderr)
        self.assertTrue((self.local / "wandering.txt").exists())

    def test_a_detached_head_on_the_server_is_refused(self) -> None:
        """An agent's edits must not become a commit no branch points at."""
        self.sync_ok()
        self.git(self.remote, "checkout", "-q", "--detach", "HEAD")
        self.write(self.remote, "agent_work.txt", "written on the server\n")

        done = self.sync()

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("detached HEAD", done.stdout + done.stderr)
        self.assertEqual((self.remote / "agent_work.txt").read_text(),
                         "written on the server\n")

    def test_commits_only_a_detached_head_holds_are_rescued(self) -> None:
        """Leaving that HEAD behind would make them unreachable.

        A committed detached HEAD is clean, so nothing stops the deploy --
        and the deploy then checks out the branch it was asked for.  With
        no branch pointing at those commits, nothing does: they stop being
        reachable and git eventually collects them.  Somebody's work would
        be gone with no branch, no push and no warning saying where.  A
        branch is created for them instead, and travels like any other.
        """
        self.sync_ok()
        self.git(self.remote, "checkout", "-q", "--detach", "HEAD")
        self.write(self.remote, "agent_work.txt", "written on the server\n")
        self.git(self.remote, "add", "-A")
        self.git(self.remote, "-c", "user.name=Agent",
                 "-c", "user.email=agent@example.com",
                 "commit", "-q", "-m", "work an agent committed detached")
        orphan = self.tip(self.remote, "HEAD")

        done = self.sync_ok()

        self.assertIn("detached HEAD held", done.stdout + done.stderr)
        rescued = [b for b in self.branches(self.remote)
                   if b.startswith("sorcar-rescued-")]
        self.assertEqual(len(rescued), 1)
        # On the server, on origin and back on the laptop.
        self.assertEqual(self.tip(self.remote, rescued[0]), orphan)
        self.assertEqual(self.tip(self.origin, rescued[0]), orphan)
        self.assertEqual(self.tip(self.local, rescued[0]), orphan)
        # And the deployment is on the branch it was asked for.
        self.assertEqual(self.git(self.remote, "rev-parse", "--abbrev-ref", "HEAD"),
                         "main")

    def test_a_detached_head_a_branch_already_has_is_left_alone(self) -> None:
        """Nothing is at risk there, so no branch is invented for it."""
        self.sync_ok()
        self.git(self.remote, "checkout", "-q", "--detach", "main")

        self.sync_ok()

        self.assertEqual([b for b in self.branches(self.remote)
                          if b.startswith("sorcar-rescued-")], [])

    def test_a_local_tag_naming_another_commit_does_not_stop_the_deploy(self) -> None:
        """The branches are what a deploy needs; the tag is its owner's business."""
        self.git(self.local, "tag", "v1")
        self.git(self.local, "push", "-q", "origin", "v1")
        # The same name on another commit here, which is what makes fetching
        # origin's tags fail.
        self.write(self.local, "later.txt", "later\n")
        self.commit(self.local, "a second commit")
        self.git(self.local, "tag", "-f", "v1")
        moved = self.tip(self.local, "v1")

        done = self.sync_ok()

        # The tagged fetch really did fail, and the deploy carried on anyway.
        self.assertIn("tags were not fetched", done.stdout + done.stderr)
        self.assertEqual(self.tip(self.local, "main"), self.tip(self.remote, "main"))
        # The local tag was not moved, and origin's was not moved either.
        self.assertEqual(self.tip(self.local, "v1"), moved)
        self.assertNotEqual(self.tip(self.origin, "v1"), moved)

    def test_a_quote_in_the_author_name_survives_the_ssh_line(self) -> None:
        """The remote command line is built, not interpolated by hope."""
        self.sync_ok()
        self.write(self.remote, "agent_work.txt", "left uncommitted\n")

        self.sync_ok(SORCAR_GIT_NAME="Bob O'Brien",
                     SORCAR_GIT_EMAIL="bob@example.com")

        self.assertEqual(self.git(self.remote, "log", "-1", "--format=%an"),
                         "Bob O'Brien")
        self.assertTrue((self.local / "agent_work.txt").exists())
        self.assert_in_sync("main")

    def test_a_single_branch_clone_still_learns_every_branch(self) -> None:
        """A narrow fetch refspec must not hide origin's other branches."""
        self.git(self.local, "config", "remote.origin.fetch",
                 "+refs/heads/main:refs/remotes/origin/main")
        self.git(self.local, "push", "-q", "origin", "main:refs/heads/elsewhere")

        self.sync_ok()

        self.assertIn("elsewhere", self.branches(self.local))
        self.assertIn("elsewhere", self.branches(self.remote))

    def test_a_branch_busy_in_another_worktree_is_reported(self) -> None:
        """A running task's branch is left alone, and the deploy goes on."""
        busy = self.tmp / "busy-worktree"
        self.git(self.local, "worktree", "add", "-q", "-b", "busy", str(busy))
        (busy / "task.txt").write_text("a task is working here\n")
        self.git(busy, "add", "-A")
        self.git(busy, "commit", "-q", "-m", "task work")

        done = self.sync_ok()

        self.assertIn("another worktree", done.stdout + done.stderr)
        self.assertEqual(
            self.git(self.origin, "for-each-ref", "--format=%(refname:short)",
                     "refs/heads/busy"), "")
        self.assertNotIn("busy", self.branches(self.remote))
        self.assert_in_sync("main")

    def test_a_folder_that_is_not_a_repository_is_refused(self) -> None:
        """``--sync`` without a branch is for an existing checkout only."""
        plain = self.tmp / "not-a-repo"
        plain.mkdir()

        done = subprocess.run(
            ["bash", str(_SCRIPT), "--sync", str(plain)],
            env=self.env, capture_output=True, text=True, timeout=120)

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("not a git repository", done.stdout + done.stderr)

    def test_a_repository_without_origin_is_refused(self) -> None:
        """There is nothing to sync through without an origin."""
        lonely = self.tmp / "lonely"
        self.git(self.tmp, "init", "-q", str(lonely))

        done = subprocess.run(
            ["bash", str(_SCRIPT), "--sync", str(lonely)],
            env=self.env, capture_output=True, text=True, timeout=120)

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no 'origin' remote", done.stdout + done.stderr)

    def test_a_repository_in_the_middle_of_a_merge_is_refused(self) -> None:
        """Deploying on top of a half-finished merge would make a mess."""
        self.sync_ok()
        self.write(self.local, "README.md", "laptop\n")
        self.commit(self.local, "laptop edit")
        self.git(self.local, "checkout", "-q", "-b", "other", "HEAD~1")
        self.write(self.local, "README.md", "other\n")
        self.commit(self.local, "other edit")
        merge = subprocess.run(
            ["git", "-C", str(self.local), "merge", "main"],
            env=self.env, capture_output=True, text=True)
        self.assertNotEqual(merge.returncode, 0, "the merge should conflict")

        done = self.sync()

        self.assertNotEqual(done.returncode, 0)
        self.assertIn("middle of a merge", done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
