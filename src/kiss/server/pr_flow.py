# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Create PR flow mixin for the backend server.

Owns the Create PR button of a chat tab: the commits of the tab's
branch that its remote's default branch lacks are pushed, and a pull
request is opened for them, by GitHub's ``gh`` CLI when it is logged
in; otherwise the outcome carries the page that opens one.
Uncommitted changes are never part of it, nor is anything the checkout
gained after the press read it.

The twin of the manual Git Commit of ``merge_flow.py``, with which it
shares the toasts, the outcome report and the forwarding to a
registered machine.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, quote_plus, urlencode

from kiss.agents.sorcar.commit_message import generate_commit_message_from_diff
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
    _git_command,
    _reclaim_process_lock,
    repo_lock,
)
from kiss.agents.sorcar.useful_tools import remap_vanished_worktree
from kiss.server import sidebar_panels
from kiss.server.merge_flow import _git_error_line, _machine_outcome, _press_toast, _report_press

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)

PR_TOAST = "create-pr"
"""The kind of a Create PR press's toasts (``merge_flow._press_toast``)."""

PR_URL_LIMIT = 6000
"""A compare URL stays under this many characters: GitHub refuses one that is too long."""

PR_TITLE_LIMIT = 256
"""GitHub's longest pull request title, in characters."""

GH_TIMEOUT_SECONDS = 120.0
"""Seconds one ``gh`` call gets: each is a request or two to GitHub."""

_DIRTY_NOTE = " (uncommitted changes are not included)"

_MOVED = "The checkout changed while the pull request was being prepared. Press Create PR again."

_GITHUB_URL = re.compile(
    r"(?:[a-z+]+://)?(?:[^@/]+@)?github\.com(?::\d+)?[:/]+(?P<repo>[^/]+/[^/]+?)(?:\.git)?/*",
    re.IGNORECASE,
)
"""A remote URL on github.com, in its https or ssh form; ``repo`` is ``<owner>/<name>``."""


@dataclass(frozen=True)
class PrOutcome:
    """How one Create PR press ended: the fields of its ``pr_done`` event.

    Attributes:
        success: Whether the press did what it could; a pushed branch
            whose pull request is still to be opened counts.
        message: One human-readable line.
        created: Whether this press opened a new pull request.
        url: The pull request's page, or the page that opens one;
            ``""`` when there is none.
    """

    success: bool
    message: str
    created: bool = False
    url: str = ""


@dataclass(frozen=True)
class _PullRequest:
    """A pull request to open: pushed branch ``head`` into ``base``, both on ``remote``."""

    remote: str
    base: str
    head: str
    title: str
    body: str


def _push_remote(work_path: Path, branch: str) -> str:
    """Return the remote *branch* is pushed to, or ``""`` when there is none to choose.

    The branch's configured remote (``branch.<name>.remote``), else
    ``origin``, else the repository's only remote.
    """
    remotes = _git("remote", cwd=work_path).stdout.split()
    configured = GitWorktreeOps._load_branch_config(work_path, branch, "remote")
    for name in (configured, "origin"):
        if name in remotes:
            return name
    return remotes[0] if len(remotes) == 1 else ""


def _default_branch(work_path: Path, remote: str) -> str:
    """Return *remote*'s default branch as this repository knows it, or ``""``.

    ``refs/remotes/<remote>/HEAD`` names it in a clone.  A repository
    that was not cloned has no such ref: ``main`` or ``master`` is
    taken then, whichever the remote has.
    """
    prefix = f"refs/remotes/{remote}/"
    head = _git("symbolic-ref", "--quiet", prefix + "HEAD", cwd=work_path).stdout.strip()
    for name in filter(None, (head.removeprefix(prefix), "main", "master")):
        if _git("show-ref", "--verify", "--quiet", prefix + name, cwd=work_path).returncode == 0:
            return name
    return ""


def _title_and_body(description: str) -> tuple[str, str]:
    """Split a commit message into a pull request's title (its subject) and body (the rest)."""
    title, _, body = description.strip().partition("\n")
    return title.strip()[:PR_TITLE_LIMIT], body.strip()


def _pr_branch(work_path: Path, title: str) -> str:
    """Name the remote branch for commits made on the base branch itself.

    Returns:
        ``pr/<first words of the title>-<short sha of HEAD>``.
    """
    sha = _git("rev-parse", "--short", "HEAD", cwd=work_path).stdout.strip()
    words = re.findall(r"[a-z0-9]+", title[:40].lower())
    return "pr/" + "-".join([*words, sha])


def _push(
    work_path: Path, remote: str, branch: str, head: str,
) -> subprocess.CompletedProcess[str]:
    """Push the checked-out *branch* to *remote* as *head*.

    Under its own name the branch also starts tracking the remote one.
    Under another name (:func:`_pr_branch`) only ``HEAD``'s commits are
    sent and the local branch is left as it is.  Both are named by
    their full ref: a tag of the same name makes the short one
    ambiguous.  A remote that asks for a password nobody can type here
    fails at once.
    """
    ref = f"refs/heads/{head}"
    target = ("-u", remote, ref) if head == branch else (remote, f"HEAD:{ref}")
    return _git("push", *target, cwd=work_path, env={"GIT_TERMINAL_PROMPT": "0"})


def _push_error_line(push: subprocess.CompletedProcess[str]) -> str:
    """Return the line of a failed ``git push`` that says why it failed.

    A push the remote rejects reports the reason per ref
    (`` ! [rejected]  feature -> feature (non-fast-forward)``) under a
    ``To <url>`` line that says nothing; any other failure names its
    reason first.
    """
    for line in push.stderr.splitlines():
        if line.startswith(" ! "):
            return " ".join(line.split())
    return _git_error_line(push, "git push failed")


def _gh(work_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run GitHub's ``gh`` CLI in *work_path*.

    ``gh`` reads the repository through git, so it gets the environment
    of this daemon's own git calls: one that cannot point git at
    another repository.

    Returns:
        The completed call; one that could not start (no ``gh`` on the
        ``PATH``) or did not end within :data:`GH_TIMEOUT_SECONDS` is a
        failed call whose error output says so.
    """
    cmd = ["gh", *args]
    _, env = _git_command((), work_path)
    try:
        return subprocess.run(
            cmd, cwd=work_path, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, errors="replace", timeout=GH_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))


def _gh_pull_request(work_path: Path, repo: str, pr: _PullRequest) -> PrOutcome:
    """Find the open pull request of ``pr.head`` in GitHub's *repo*, or create one, with ``gh``.

    Each call names *repo*, the repository that was pushed to.  Left to
    choose, ``gh`` takes the one its own settings name (``GH_REPO``, a
    remote marked as its default), which may be another.

    Returns:
        The pull request, or a failed outcome whose message is gh's
        error line.
    """
    found = _gh(
        work_path, "pr", "list", f"--repo={repo}", f"--head={pr.head}", "--state=open",
        "--json=url", "--jq=.[].url",
    )
    if found.returncode == 0:  # pragma: no cover — GitHub answers for its own repositories only
        return _reused_or_created(work_path, repo, pr, found.stdout.split())
    return PrOutcome(False, _git_error_line(found, "gh pr list failed"))


def _reused_or_created(  # pragma: no cover — runs on GitHub's answer (see _gh_pull_request)
    work_path: Path, repo: str, pr: _PullRequest, open_urls: list[str],
) -> PrOutcome:
    """Return the first of *open_urls*, the head branch's open pull requests, or create one."""
    if open_urls:
        return PrOutcome(True, f"Pull request updated: {open_urls[0]}", url=open_urls[0])
    made = _gh(
        work_path, "pr", "create", f"--repo={repo}", f"--base={pr.base}", f"--head={pr.head}",
        f"--title={pr.title}", f"--body={pr.body}",
    )
    if made.returncode != 0:
        return PrOutcome(False, _git_error_line(made, "gh pr create failed"))
    # gh prints the new pull request's URL last.
    url = made.stdout.strip().rpartition("\n")[2]
    return PrOutcome(True, f"Pull request created: {url}", created=True, url=url)


def _github_repo(work_path: Path, remote: str) -> str:
    """Return ``<owner>/<name>`` when a push to *remote* goes to github.com, else ``""``.

    ``git push`` sends to every ``remote.<name>.pushurl`` or, where
    none is set, to every ``remote.<name>.url``: the first of them on
    github.com names the repository.  The raw values decide, not ``git
    remote get-url``: that one applies the ``url.<base>.insteadOf``
    rewrites, which say where git connects, not where the repository
    lives.
    """
    urls: list[str] = []
    for key in ("pushurl", "url"):
        found = _git("config", "--get-all", f"remote.{remote}.{key}", cwd=work_path)
        urls = found.stdout.splitlines()
        # To git an empty value clears the ones before it.
        while "" in urls:
            urls = urls[urls.index("") + 1:]
        if urls:
            break
    matches = (_GITHUB_URL.fullmatch(url) for url in urls)
    return next((match["repo"] for match in matches if match), "")


def _encoded_prefix(text: str, room: int) -> str:
    """Return the longest prefix of *text* that URL-encodes to at most *room* characters."""
    used = 0
    for end, char in enumerate(text):
        used += len(quote_plus(char, errors="replace"))
        if used > room:
            return text[:end]
    return text


def _compare_url(repo: str, pr: _PullRequest) -> str:
    """Return the github.com page that opens *pr* in *repo*, title and body filled in.

    The body is cut so that the URL stays under :data:`PR_URL_LIMIT`
    characters.
    """
    base, head = quote(pr.base, errors="replace"), quote(pr.head, errors="replace")
    page = f"https://github.com/{repo}/compare/{base}...{head}?"
    query = {"quick_pull": 1, "title": pr.title, "body": ""}
    room = PR_URL_LIMIT - 1 - len(page + urlencode(query, errors="replace"))
    query["body"] = _encoded_prefix(pr.body, room)
    return page + urlencode(query, errors="replace")


def _open_pull_request(work_path: Path, pr: _PullRequest) -> PrOutcome:
    """Open the pull request of the pushed ``pr.head``, or say where to open it.

    Any host but github.com gets the news of the push alone.  On
    github.com a ``gh`` that is logged in finds or creates the pull
    request; without one, or when it fails (its error line is kept in
    the message), the outcome carries the page that opens the pull
    request.
    """
    repo = _github_repo(work_path, pr.remote)
    if not repo:
        return PrOutcome(
            True, f"Pushed {pr.head} to {pr.remote}. Open the pull request on your Git host.",
        )
    gh_note = ""
    if _gh(work_path, "auth", "status").returncode == 0:
        opened = _gh_pull_request(work_path, repo, pr)
        if opened.success:  # pragma: no cover — only GitHub's answer succeeds (_gh_pull_request)
            return opened
        gh_note = f" (gh failed: {opened.message})"
    url = _compare_url(repo, pr)
    return PrOutcome(
        True, f"Pushed {pr.head}. Open the pull request page to finish: {url}{gh_note}", url=url,
    )


class _PrFlowMixin:
    """The Create PR button: push the tab's branch and open its pull request."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str

    def _broadcast_pr_done(
        self, tab_id: str, outcome: PrOutcome, work_dir: str = "", reply_to: str = "",
    ) -> None:
        """Report the outcome of a Create PR press.

        The outcome is a toast ``notification``, info on success and
        error on failure, plus the ``pr_done`` event that re-arms the
        button.  Neither is part of a chat's transcript.

        Args:
            tab_id: Frontend tab identifier.
            outcome: How the press ended.
            work_dir: The working directory the press targeted
                (carried as ``workDir``).
            reply_to: The connection id of a daemon-link peer whose
                forwarded press this is: the event goes to that
                connection alone, as the reply to its request, and
                nothing is broadcast here.
        """
        event: dict[str, Any] = {
            "type": "pr_done",
            "success": outcome.success,
            "created": outcome.created,
            "url": outcome.url,
            "message": outcome.message,
            "tabId": tab_id,
            "workDir": work_dir,
        }
        _report_press(self.printer, PR_TOAST, event, reply_to)

    def _create_pr(self, tab_id: str, work_dir: str = "", reply_to: str = "") -> None:
        """Push the tab's branch and open a pull request for it.

        Serves the Create PR button and reports through
        :meth:`_broadcast_pr_done`.

        Args:
            tab_id: The tab that pressed the button.
            work_dir: The tab's working directory (preferred over the
                daemon-wide ``self.work_dir``); a path under a
                now-deleted ``.kiss-worktrees/kiss_wt-*`` checkout is
                remapped to the parent repository.
            reply_to: The connection id of a daemon-link peer that
                forwarded the command (another daemon's tab picked
                onto this machine), or ``""``.
        """
        requested_dir = work_dir or self.work_dir
        try:
            outcome = self._pr_outcome(tab_id, requested_dir, reply_to)
        except Exception as e:  # pragma: no cover — unexpected git/gh/LLM error
            logger.debug("Create PR failed", exc_info=True)
            outcome = PrOutcome(False, str(e))
        self._broadcast_pr_done(tab_id, outcome, requested_dir, reply_to)

    def _pr_outcome(self, tab_id: str, requested_dir: str, reply_to: str) -> PrOutcome:
        """Run one Create PR press in *requested_dir* and return how it ended.

        Args:
            tab_id: The tab that pressed the button.
            requested_dir: The working directory the press targets.
            reply_to: A daemon-link peer's connection id, or ``""``:
                that daemon shows the press's toasts, so none is sent
                from here.
        """
        work_path = remap_vanished_worktree(Path(requested_dir))
        repo = GitWorktreeOps.discover_repo(work_path)
        if repo is None:
            return PrOutcome(False, "Not a git repository.")
        # The two locks a manual commit takes (see ``_commit_main_tree``),
        # held around the git steps only: reading what to push here, the
        # push below.  Writing the description may take minutes.
        with repo_lock(repo), _reclaim_process_lock(repo):
            branch = GitWorktreeOps.current_branch(work_path)
            if branch is None:
                return PrOutcome(False, "Check out a branch first: HEAD is detached.")
            commit = GitWorktreeOps.head_sha(work_path)
            remote = _push_remote(work_path, branch)
            if not remote:
                return PrOutcome(False, "This repository has no remote to push to.")
            base = _default_branch(work_path, remote)
            if not base:
                return PrOutcome(
                    False,
                    f"Could not find the default branch of {remote} (no {remote}/HEAD, "
                    f"{remote}/main or {remote}/master). Run: git fetch {remote} && "
                    f"git remote set-head {remote} -a",
                )
            theirs = f"refs/remotes/{remote}/{base}"
            ahead = _git("rev-list", "--count", f"{theirs}..HEAD", cwd=work_path).stdout.strip()
            dirty = _DIRTY_NOTE if GitWorktreeOps.has_uncommitted_changes(work_path) else ""
            if ahead == "0":
                return PrOutcome(
                    True,
                    f"Nothing to open a pull request for: {branch} has no commits that "
                    f"{remote}/{base} lacks.{dirty}",
                )
            # One commit describes itself; several (or one without a
            # message) are described from their diff.
            description = ""
            if ahead == "1":
                description = _git("log", "-1", "--format=%B", cwd=work_path).stdout.strip()
            diff_text = "" if description else GitWorktreeOps.prompt_diff(
                work_path, f"{theirs}...HEAD",
            )
        if not description:
            if not reply_to:
                _press_toast(
                    self.printer, PR_TOAST, tab_id, "info",
                    "Writing the pull request description…",
                )
            description = generate_commit_message_from_diff(diff_text)
        title, body = _title_and_body(description)
        with repo_lock(repo), _reclaim_process_lock(repo):
            # Other git work may have run since the locks were let go: only the
            # commits read above, still the ones checked out, are pushed.
            now = GitWorktreeOps.current_branch(work_path), GitWorktreeOps.head_sha(work_path)
            if now != (branch, commit):
                return PrOutcome(False, _MOVED + dirty)
            head = _pr_branch(work_path, title) if branch == base else branch
            push = _push(work_path, remote, branch, head)
        if push.returncode != 0:
            return PrOutcome(False, "Push failed: " + _push_error_line(push) + dirty)
        opened = _open_pull_request(work_path, _PullRequest(remote, base, head, title, body))
        return replace(opened, message=opened.message + dirty)

    def _create_pr_on_machine(self, tab_id: str, machine: str, work_dir: str) -> None:
        """Run the tab's Create PR on registered *machine* and report its outcome here.

        The Create PR button of a tab whose drop-down names another
        machine opens the pull request of that machine's folder: the
        command goes to its daemon
        (:func:`~kiss.server.sidebar_panels.machine_create_pr`), which
        pushes and opens the pull request there, and its ``pr_done`` is
        re-broadcast to this daemon's clients through
        :meth:`_broadcast_pr_done`.

        Args:
            tab_id: The tab that pressed the button.
            machine: The registered machine's name.
            work_dir: The folder on the machine; empty for the
                machine's own configured folder.
        """
        _press_toast(
            self.printer, PR_TOAST, tab_id, "info", f"Creating the pull request on {machine}…",
        )
        reply = sidebar_panels.machine_create_pr(
            machine, {"type": "createPrAction", "tabId": tab_id, "workDir": work_dir},
        )
        success, message = _machine_outcome(machine, reply)
        outcome = PrOutcome(
            success, message, bool(reply.get("created")), str(reply.get("url") or ""),
        )
        self._broadcast_pr_done(tab_id, outcome, str(reply.get("workDir") or work_dir))
