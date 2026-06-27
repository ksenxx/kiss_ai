# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the auto-commit lifecycle notifications.

When the agent finishes a task and the worktree auto-commit path
runs, the chat webview should receive two ``notification`` toasts:

* ``"Generating commit message"`` immediately before the (typically
  slow) LLM call that generates the commit message — so the user
  sees that something is happening during the wait.
* ``"Committed <subject>"`` immediately after the commit lands in
  git, where ``<subject>`` is the first non-empty line of the
  committed message — so the user sees the LLM-generated subject
  the moment the commit is created.

These tests drive the real
:meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._auto_commit_worktree`
path against on-disk git repositories with a fake printer that
records every :meth:`broadcast` call.  The LLM call inside the
commit-message helper is forced through its ``except Exception``
fallback by patching :class:`~kiss.core.kiss_agent.KISSAgent.run` to
raise (the same trick used by ``test_autocommit_user_prompt.py``)
so the tests stay hermetic.

The "generating" notification must arrive BEFORE the commit lands;
the "committed" notification must arrive AFTER.  Both invariants
are asserted via subprocess ``git log`` snapshots taken from inside
the fake broadcast hook.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import auto_commit_changes
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
    )
    return path


def _git_config_user(repo: Path) -> None:
    """Set git user.email/name on *repo* so commits can be created."""
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "T"],
        check=True,
    )


def _head_sha(repo: Path) -> str:
    """Return HEAD's SHA in *repo* (empty string if no HEAD)."""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


def _head_message(repo: Path) -> str:
    """Return HEAD's full commit message in *repo*."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip()


class _LLMUnavailable:
    """Force the LLM call inside ``generate_commit_message_from_diff``
    through its ``except Exception`` fallback so tests don't need a
    real model.

    Patches :class:`kiss.core.kiss_agent.KISSAgent` to a class whose
    ``run`` raises.  Reverts on exit.
    """

    def __enter__(self) -> _LLMUnavailable:
        import kiss.core.kiss_agent as kiss_agent_mod

        self._orig = kiss_agent_mod.KISSAgent

        class _RaisingAgent:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass

            def run(self, *_a: Any, **_kw: Any) -> str:
                raise RuntimeError("no LLM in test")

        kiss_agent_mod.KISSAgent = _RaisingAgent  # type: ignore[misc, assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        import kiss.core.kiss_agent as kiss_agent_mod

        kiss_agent_mod.KISSAgent = self._orig  # type: ignore[misc]


class _RecordingPrinter:
    """Minimal printer stub that records every ``broadcast`` event.

    The webview consumes ``broadcast`` events via the VS Code
    extension host; for tests we only need a ``broadcast`` method
    that captures the event payload alongside the HEAD SHA observed
    at broadcast time (so we can prove ordering relative to the
    commit).
    """

    def __init__(self, repo: Path) -> None:
        self._repo = repo
        self.events: list[dict[str, Any]] = []
        # HEAD SHA at the moment each event was broadcast.  ``""``
        # means "no HEAD yet" (an unborn branch).
        self.head_at_event: list[str] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.head_at_event.append(_head_sha(self._repo))


def _notification_events(printer: _RecordingPrinter) -> list[tuple[dict[str, Any], str]]:
    """Return ``[(event, head_sha_at_event), ...]`` only for ``type=notification`` events."""
    return [
        (ev, head)
        for ev, head in zip(printer.events, printer.head_at_event, strict=True)
        if ev.get("type") == "notification"
    ]


def _setup_worktree_agent(tmp: Path, branch_slug: str) -> tuple[
    WorktreeSorcarAgent, Path, Path
]:
    """Build a real on-disk worktree backed by a ``WorktreeSorcarAgent``.

    Returns ``(agent, repo, wt_dir)``.  The agent's ``_wt`` is
    populated so ``_auto_commit_worktree`` can run end-to-end.
    """
    repo = _make_repo(tmp / "repo")
    branch = f"kiss/wt-test-{branch_slug}"
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    _git_config_user(wt_dir)

    agent = WorktreeSorcarAgent("test")
    agent._wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
        baseline_commit=None,
    )
    agent._tab_id = "tab-xyz"
    return agent, repo, wt_dir


class TestAutoCommitChangesNotifyFn:
    """``auto_commit_changes`` invokes ``notify_fn`` at the two
    documented life-cycle points and never raises when the hook does.
    """

    def test_generating_fires_before_commit_and_committed_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")
            head_before = _head_sha(repo)
            (repo / "f.txt").write_text("hi\n")

            calls: list[tuple[str, str, str]] = []

            def notify(stage: str, subject: str) -> None:
                calls.append((stage, subject, _head_sha(repo)))

            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt="add f.txt",
                    message_fn=lambda d, p: "feat: add f.txt\n\ndetails",
                    notify_fn=notify,
                )

            assert created is True
            assert [c[0] for c in calls] == ["generating", "committed"]
            # "generating" must fire BEFORE the commit lands.
            assert calls[0][2] == head_before
            # "committed" must carry the first line of the message
            # and fire AFTER the commit lands.
            assert calls[1][1] == "feat: add f.txt"
            head_after = _head_sha(repo)
            assert calls[1][2] == head_after
            assert head_after != head_before

    def test_no_commit_skips_message_fn_and_notifications(self) -> None:
        """Bug 1 (gpt-5.5 review): when nothing is staged, do NOT fire
        the misleading "Generating commit message" toast and do NOT
        invoke the (slow, token-costing) ``message_fn``.  The whole
        auto-commit short-circuits silently.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")
            head_before = _head_sha(repo)

            calls: list[tuple[str, str]] = []

            def notify(stage: str, subject: str) -> None:
                calls.append((stage, subject))

            msg_fn_called = []

            def message_fn(_d: Path, _p: str | None) -> str:
                msg_fn_called.append(True)
                return "feat: noop"

            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt=None,
                    message_fn=message_fn,
                    notify_fn=notify,
                )

            assert created is False
            # No changes ⇒ no "generating" toast (it would be misleading
            # because nothing is going to be committed).
            assert calls == []
            # No changes ⇒ no LLM call (saves tokens and latency).
            assert msg_fn_called == []
            assert _head_sha(repo) == head_before

    def test_notify_fn_none_does_not_break_flow(self) -> None:
        """Legacy callers (e.g. the cli_repl path) pass no ``notify_fn``
        — the commit must still happen."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")
            (repo / "f.txt").write_text("hello\n")

            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt=None,
                    message_fn=lambda d, p: "feat: hello",
                )

            assert created is True
            assert _head_message(repo) == "feat: hello"

    def test_notify_fn_exception_is_swallowed(self) -> None:
        """A buggy UI hook must not block or rollback the commit."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")
            (repo / "f.txt").write_text("hi\n")

            def notify(_stage: str, _subject: str) -> None:
                raise RuntimeError("buggy UI hook")

            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt=None,
                    message_fn=lambda d, p: "feat: ok",
                    notify_fn=notify,
                )

            assert created is True
            assert _head_message(repo) == "feat: ok"

    def test_committed_subject_is_first_nonempty_line(self) -> None:
        """The "committed" toast carries only the subject line of the
        multi-line commit message — not the body."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")
            (repo / "f.txt").write_text("hi\n")

            calls: list[tuple[str, str]] = []

            def notify(stage: str, subject: str) -> None:
                calls.append((stage, subject))

            message = "\n\nfeat: real subject\n\nbody line 1\nbody line 2\n"
            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt=None,
                    message_fn=lambda d, p: message,
                    notify_fn=notify,
                )

            assert created is True
            committed = [s for stage, s in calls if stage == "committed"]
            assert committed == ["feat: real subject"]

    def test_empty_tree_does_not_invoke_message_fn(self) -> None:
        """Bug 1 (gpt-5.5 review): an empty tree must short-circuit
        BEFORE the (slow, token-costing) ``message_fn`` call.  Otherwise
        we burn LLM tokens on a commit that will never happen and the
        user sees a misleading "Generating commit message" toast.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            repo = _make_repo(tmp / "repo")

            invocations: list[str] = []

            def message_fn(_d: Path, _p: str | None) -> str:
                invocations.append("called")
                return "feat: should not happen"

            with _LLMUnavailable():
                created = auto_commit_changes(
                    repo,
                    user_prompt=None,
                    message_fn=message_fn,
                )

            assert created is False
            assert invocations == []  # message_fn was NOT called.


class TestWorktreeAutoCommitBroadcasts:
    """``WorktreeSorcarAgent._auto_commit_worktree`` routes the two
    life-cycle notifications through ``self.printer.broadcast`` with
    the chat-webview-expected payload shape.
    """

    def test_worktree_emits_generating_then_committed_with_tab_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, repo, wt_dir = _setup_worktree_agent(tmp, "notif-happy")
            head_before = _head_sha(wt_dir)
            (wt_dir / "new.txt").write_text("hello world\n")

            printer = _RecordingPrinter(wt_dir)
            agent.printer = printer  # type: ignore[assignment]

            with _LLMUnavailable():
                # The LLM-fallback path produces:
                #   "kiss: auto-commit agent changes" (no user prompt)
                assert agent._auto_commit_worktree() is True

            notifs = _notification_events(printer)
            assert len(notifs) == 2

            gen_ev, gen_head = notifs[0]
            committed_ev, committed_head = notifs[1]

            # Shape: every notification carries type/id/severity/message/tabId.
            for ev in (gen_ev, committed_ev):
                assert ev["type"] == "notification"
                assert ev["severity"] == "info"
                assert ev["tabId"] == "tab-xyz"
                assert isinstance(ev["id"], str) and ev["id"]
            # Bug 2 (gpt-5.5 review): the two events MUST share the
            # same notification id so the webview updates the toast in
            # place (Generating → Committed) instead of stacking two
            # toasts and leaving the misleading "Generating commit
            # message" lingering until its own auto-dismiss timer fires.
            assert gen_ev["id"] == committed_ev["id"]

            # Content + ordering.
            assert gen_ev["message"] == "Generating commit message"
            assert gen_head == head_before  # fired BEFORE the commit
            # The "Generating commit message" toast MUST be sticky:
            # the webview's transient auto-dismiss timer (~5 s for
            # info severity) would otherwise hide it mid-LLM-call and
            # mislead the user into thinking the commit had stalled.
            # The follow-up "Committed <subject>" event reuses the
            # same id but omits sticky, so the toast reverts to a
            # normal transient that fades on its own.
            assert gen_ev.get("sticky") is True
            assert "sticky" not in committed_ev or not committed_ev["sticky"]
            assert committed_ev["message"].startswith("Committed ")
            head_after = _head_sha(wt_dir)
            assert committed_head == head_after  # fired AFTER the commit
            assert head_after != head_before
            # Subject in the toast matches the actual git log subject.
            subject = _head_message(wt_dir).splitlines()[0].strip()
            assert committed_ev["message"] == f"Committed {subject}"

    def test_worktree_no_commit_emits_no_notifications(self) -> None:
        """Bug 1 (gpt-5.5 review): worktree path must NOT emit the
        misleading "Generating commit message" toast when there are no
        staged changes to commit.  Otherwise the webview shows a toast
        that never resolves to a "Committed" message.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-empty")
            head_before = _head_sha(wt_dir)
            # No changes in wt_dir ⇒ commit_staged returns False.

            printer = _RecordingPrinter(wt_dir)
            agent.printer = printer  # type: ignore[assignment]

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is False

            # No notifications at all — the empty path is silent.
            assert _notification_events(printer) == []
            assert _head_sha(wt_dir) == head_before

    def test_worktree_no_printer_does_not_crash(self) -> None:
        """When no printer is attached (e.g. ``--no-printer`` CLI
        invocations) the auto-commit must still succeed and silently
        skip the broadcasts."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-noprn")
            (wt_dir / "new.txt").write_text("hi\n")
            agent.printer = None  # type: ignore[assignment]

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            # And the commit landed.  The LLM-unavailable path inside
            # ``generate_commit_message_from_diff`` swallows the
            # exception and returns the ``"kiss: auto-commit agent work"``
            # fallback, so the outer ``except`` in
            # :func:`auto_commit_changes` does NOT fire.
            assert _head_message(wt_dir).startswith(
                "kiss: auto-commit agent work"
            )

    def test_worktree_printer_without_broadcast_does_not_crash(self) -> None:
        """A printer object that lacks ``broadcast`` (e.g. plain
        terminal printer) must not raise."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-nobcast")
            (wt_dir / "new.txt").write_text("hi\n")

            class _NoBroadcast:
                pass

            agent.printer = _NoBroadcast()  # type: ignore[assignment]

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

    def test_worktree_disabled_emits_no_notifications(self) -> None:
        """When ``auto_commit_enabled`` is False the path short-circuits
        before staging — neither notification is sent."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-off")
            (wt_dir / "new.txt").write_text("hi\n")
            agent.auto_commit_enabled = False

            printer = _RecordingPrinter(wt_dir)
            agent.printer = printer  # type: ignore[assignment]

            assert agent._auto_commit_worktree() is False
            assert _notification_events(printer) == []

    def test_worktree_generating_and_committed_share_id(self) -> None:
        """Bug 2 (gpt-5.5 review): the "generating" and "committed"
        toast events must share the same notification id so
        ``media/main.js`` updates the existing toast in place instead of
        stacking two notifications.  Otherwise the misleading
        "Generating commit message" lingers next to the new
        "Committed <subject>" toast until its own auto-dismiss timer.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-sameid")
            (wt_dir / "new.txt").write_text("hi\n")

            printer = _RecordingPrinter(wt_dir)
            agent.printer = printer  # type: ignore[assignment]

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = _notification_events(printer)
            assert len(notifs) == 2
            assert notifs[0][0]["id"] == notifs[1][0]["id"]
            assert notifs[0][0]["message"] == "Generating commit message"
            assert notifs[1][0]["message"].startswith("Committed ")

    def test_worktree_no_commit_does_not_call_llm(self) -> None:
        """Bug 1 (gpt-5.5 review) — worktree integration check: when
        nothing is staged the worktree path must NOT invoke the
        LLM-backed commit-message generator (it costs tokens for
        nothing) and must NOT broadcast the "Generating commit
        message" toast.
        """
        # Spy on ``_generate_commit_message`` to confirm it is never
        # called when the worktree has no changes.
        import kiss.agents.sorcar.sorcar_agent as sa

        original = sa._generate_commit_message
        calls: list[tuple[Path, str | None]] = []

        def spy(commit_dir: Path, user_prompt: str | None) -> str:
            calls.append((commit_dir, user_prompt))
            return original(commit_dir, user_prompt)

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            agent, _repo, wt_dir = _setup_worktree_agent(tmp, "notif-llmskip")

            printer = _RecordingPrinter(wt_dir)
            agent.printer = printer  # type: ignore[assignment]

            sa._generate_commit_message = spy  # type: ignore[assignment]
            try:
                with _LLMUnavailable():
                    assert agent._auto_commit_worktree() is False
            finally:
                sa._generate_commit_message = original  # type: ignore[assignment]

            assert calls == []
            assert _notification_events(printer) == []
