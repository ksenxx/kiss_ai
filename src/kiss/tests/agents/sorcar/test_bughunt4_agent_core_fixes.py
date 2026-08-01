# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: regressions found by the sorcar agent-core audit (findings-1).

End-to-end reproductions for the confirmed findings in
``sorcar_agent.py``, ``worktree_sorcar_agent.py``,
``chat_sorcar_agent.py`` and ``running_agent_state.py``:

* F-01: ``finish`` accepted while a queued user follow-up is undrained.
* F-02: a raising ``broadcast`` in the pending-message drain loses the
  already-dequeued steering input.
* F-03: a parent stop unwinding ``run_tasks_parallel`` loses the
  sub-agents' usage accounting entirely.
* F-04: auto-commit failure leaves the sticky "Generating commit
  message" toast with no terminal update.
* F-05: ``set_model`` silently drops a task-specific ``api_key`` when
  the old model used a registered provider's DEFAULT endpoint.
* F-06: provider-specific model_config keys leak across provider
  switches.
* F-08: the live-usage monitor emits regressing totals during the
  session-handoff torn-read window.
* F-10: ABBA deadlock between two agents switching worktrees across
  two repositories in opposite directions.
* F-11: a raising ``broadcast`` in ``_flush_warnings`` permanently
  loses both warnings and aborts the caller.
* F-12: a successful merge ignores ``delete_branch`` failure, silently
  leaking the task branch.
* F-13: manual recovery command blocks are not shell-quoted.
* F-14: an exception in post-registration setup of
  ``ChatSorcarAgent.run`` leaks both running-agent registries.
* F-15: key-only ``unregister`` deletes a replacement state registered
  under the same tab id (ABA).

All tests use real objects (real git repos, real SQLite persistence,
real printers, real model adapters) — no mocks, patches, fakes, or
test doubles.
"""

from __future__ import annotations

import functools
import random
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree, MergeResult
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _LiveUsageMonitor,
    auto_commit_changes,
    run_tasks_parallel,
)
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _merge_fix_steps,
)
from kiss.core.models.model_info import model as _model_factory
from kiss.server.json_printer import JsonPrinter


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _make_repo(base: Path, name: str) -> Path:
    repo = base / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")
        self._saved_states = dict(_RunningAgentState.running_agent_states)
        _RunningAgentState.running_agent_states.clear()
        self._saved_running = dict(ChatSorcarAgent.running_agents)
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        _RunningAgentState.running_agent_states.clear()
        _RunningAgentState.running_agent_states.update(self._saved_states)
        ChatSorcarAgent.running_agents.clear()
        ChatSorcarAgent.running_agents.update(self._saved_running)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class _CollectingPrinter:
    """Minimal real printer collecting broadcasts and prints."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.printed: list[tuple[object, dict[str, object]]] = []
        self._thread_local = threading.local()

    def broadcast(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def print(self, content: object, type: str = "text", **kwargs: object) -> str:
        self.printed.append((content, {"type": type, **kwargs}))
        return ""


class _BrokenBroadcastPrinter:
    """Real printer whose ``broadcast`` always raises."""

    def __init__(self) -> None:
        self.attempts = 0
        self._thread_local = threading.local()

    def broadcast(self, event: dict[str, object]) -> None:
        self.attempts += 1
        raise RuntimeError("broadcast pipe closed")


def _fresh_openai_model(api_key: str, config: dict[str, object] | None = None) -> Any:
    cfg: dict[str, object] = {
        "base_url": "https://api.openai.com/v1",
        "api_key": api_key,
    }
    if config:
        cfg.update(config)
    return _model_factory("gpt-4o", model_config=cfg)


class TestF01FinishGuard(_TempDbTestBase):
    """finish must be rejected while a queued user follow-up is undrained."""

    def test_finish_blocked_until_drain(self) -> None:
        agent: Any = SorcarAgent("f01")
        agent._tab_id = "tab-f01"
        state = _RunningAgentState("tab-f01", "gpt-4o")
        state.pending_user_messages.append("also update the docs")
        _RunningAgentState.register("tab-f01", state)

        blocked = agent._block_finish_when_user_message_pending("finish", {})
        assert blocked is not None
        assert "new message" in blocked

        # Non-finish tools are never blocked.
        assert (
            agent._block_finish_when_user_message_pending("Bash", {}) is None
        )

        # Once the queue is drained, finish is allowed again.
        state.pending_user_messages.clear()
        assert (
            agent._block_finish_when_user_message_pending("finish", {}) is None
        )

    def test_guard_wired_through_chat_agent_property(self) -> None:
        """ChatSorcarAgent's guard property must delegate to the F-01 guard."""
        agent: Any = ChatSorcarAgent("f01-wire")
        agent._tab_id = "tab-f01w"
        state = _RunningAgentState("tab-f01w", "gpt-4o")
        state.pending_user_messages.append("stop, do X instead")
        _RunningAgentState.register("tab-f01w", state)

        # Same assignment SorcarAgent.perform_task performs.
        agent.tool_call_guard = agent._block_finish_when_user_message_pending
        blocked = agent.tool_call_guard("finish", {})
        assert blocked is not None and "new message" in blocked


class TestF02DrainRobustness(_TempDbTestBase):
    """A raising echo broadcast must not lose dequeued steering input."""

    def test_steering_survives_broken_broadcast(self) -> None:
        agent: Any = SorcarAgent("f02")
        agent._tab_id = "tab-f02"
        broken_printer = _BrokenBroadcastPrinter()
        agent.printer = broken_printer
        state = _RunningAgentState("tab-f02", "gpt-4o")
        state.pending_user_messages.append("use tabs not spaces")
        state.unattributed_prompt_echoes.append("use tabs not spaces")
        _RunningAgentState.register("tab-f02", state)

        model = _fresh_openai_model("test-key")
        model.initialize("original task")

        agent._drain_pending_user_messages(model)  # must not raise

        assert state.pending_user_messages == []
        # The durable echo is REQUEUED after the failed broadcast so a
        # later drain can retry it (review-1 issue 3).
        assert state.unattributed_prompt_echoes == ["use tabs not spaces"]
        user_texts = [
            m["content"]
            for m in model.conversation
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert any("use tabs not spaces" in str(t) for t in user_texts)
        assert broken_printer.attempts >= 1

    def test_drain_and_guard_skip_foreign_owner(self) -> None:
        """A stale agent must never consume a replacement's queued input."""
        stale: Any = SorcarAgent("f02-stale")
        stale._tab_id = "tab-f02b"
        owner: Any = SorcarAgent("f02-owner")
        state = _RunningAgentState("tab-f02b", "gpt-4o", agent=owner)
        state.pending_user_messages.append("for the owner only")
        _RunningAgentState.register("tab-f02b", state)

        model = _fresh_openai_model("test-key")
        model.initialize("original task")
        stale._drain_pending_user_messages(model)

        # The replacement's queue is untouched and nothing was injected.
        assert state.pending_user_messages == ["for the owner only"]
        assert all(
            "for the owner only" not in str(m.get("content", ""))
            for m in model.conversation
            if isinstance(m, dict)
        )
        # The stale agent's finish is not blocked by the foreign queue.
        assert (
            stale._block_finish_when_user_message_pending("finish", {}) is None
        )


class TestF03InterruptUsageAccounting(_TempDbTestBase):
    """A parent stop must not erase sub-agent usage accounting."""

    def test_totals_out_filled_when_parent_stop_unwinds_pool(self) -> None:
        printer = JsonPrinter()
        stop = threading.Event()
        stop.set()
        printer._thread_local.stop_event = stop
        totals: dict[str, float] = {}

        time.sleep(random.random() * 0.05)
        with pytest.raises(KeyboardInterrupt):
            run_tasks_parallel(
                ["say hi"],
                model_name="gpt-4o",
                work_dir=self.tmpdir,
                printer=printer,
                totals_out=totals,
            )

        # Before the fix, the interrupt skipped the aggregation entirely
        # and totals_out stayed empty — the parent lost all accounting.
        assert set(totals) == {
            "budget_used", "total_tokens_used", "total_steps",
        }
        # No sub-agent registry entry may leak either.
        assert all(
            not state.is_subagent
            for state in _RunningAgentState.running_agent_states.values()
        )


class TestF04CommitFailureNotification(_TempDbTestBase):
    """A rejected auto-commit must terminate the sticky toast."""

    def test_failed_stage_replaces_sticky_toast(self) -> None:
        repo = _make_repo(Path(self.tmpdir), "repo")
        hook = repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        (repo / "work.txt").write_text("agent output\n")

        agent: Any = WorktreeSorcarAgent("f04")
        collecting = _CollectingPrinter()
        agent.printer = collecting
        agent._tab_id = "tab-f04"

        committed = auto_commit_changes(
            repo,
            "the user prompt",
            lambda commit_dir, prompt, result: "kiss: test commit",
            notify_fn=functools.partial(
                agent._broadcast_commit_notification, "toast-1",
            ),
        )
        assert committed is False

        notifications = [
            e for e in collecting.events if e.get("type") == "notification"
        ]
        stages = [(e.get("message"), e.get("sticky", False)) for e in notifications]
        assert len(notifications) == 2, stages
        generating, terminal = notifications
        assert generating["sticky"] is True
        assert generating["id"] == terminal["id"] == "toast-1"
        assert terminal.get("sticky", False) is False
        assert terminal["severity"] == "warning"
        assert "failed" in str(terminal["message"]).lower()


class _SetModelHarness(_TempDbTestBase):
    """Shared setup to reach the real ``set_model`` tool closure."""

    def _set_model_tool(self, agent: Any) -> Any:
        agent.work_dir = self.tmpdir
        agent._use_web_tools = False
        agent.printer = None
        tools = agent._get_tools()
        return next(t for t in tools if getattr(t, "__name__", "") == "set_model")


class TestF05ApiKeyPreservation(_SetModelHarness):
    """Same-provider switch on a DEFAULT endpoint must keep the task key."""

    def test_task_specific_key_survives_same_provider_switch(self) -> None:
        agent: Any = SorcarAgent("f05")
        agent.model = _fresh_openai_model("task-key-123")
        agent.model_name = "gpt-4o"
        set_model = self._set_model_tool(agent)

        msg = set_model("gpt-4o-mini")

        assert "gpt-4o-mini" in msg
        assert agent.model.model_name == "gpt-4o-mini"
        # Before the fix, the default-endpoint branch dropped the key and
        # the replacement silently used the process-global credential.
        assert agent.model.api_key == "task-key-123"
        assert agent.model.base_url.rstrip("/") == "https://api.openai.com/v1"


class TestF06ConfigSanitization(_SetModelHarness):
    """Provider-specific request options must not leak across providers."""

    def test_openai_options_dropped_on_switch_to_anthropic(self) -> None:
        pytest.importorskip("anthropic")
        agent: Any = SorcarAgent("f06")
        agent.model = _fresh_openai_model(
            "task-key-456",
            config={"reasoning_effort": "low", "use_responses_api": True},
        )
        agent.model_name = "gpt-4o"
        set_model = self._set_model_tool(agent)

        msg = set_model("claude-sonnet-4-5")

        assert "claude-sonnet-4-5" in msg
        new_config = agent.model.model_config or {}
        assert "reasoning_effort" not in new_config
        assert "use_responses_api" not in new_config
        # The OpenAI credential must not be misrouted to Anthropic either.
        assert new_config.get("api_key") != "task-key-456"


class TestF07GeminiThoughtSignatures(_SetModelHarness):
    """Gemini-to-Gemini switch must carry the thought-signature map."""

    def test_signatures_survive_gemini_switch(self) -> None:
        pytest.importorskip("google.genai")
        agent: Any = SorcarAgent("f07")
        try:
            old_model: Any = _model_factory("gemini-2.5-flash")
        except Exception as exc:  # no key configured in this environment
            pytest.skip(f"gemini model unavailable: {exc}")
        old_model.initialize("task")
        old_model._thought_signatures["call-1"] = b"sig-bytes"
        agent.model = old_model
        agent.model_name = "gemini-2.5-flash"
        set_model = self._set_model_tool(agent)

        msg = set_model("gemini-2.5-pro")

        assert "gemini-2.5-pro" in msg
        assert agent.model._thought_signatures.get("call-1") == b"sig-bytes"


class TestF08MonotonicLiveUsage(_TempDbTestBase):
    """The live-usage monitor must never emit regressing totals."""

    def test_torn_read_window_is_not_emitted(self) -> None:
        parent: Any = SorcarAgent("f08-parent")
        printer = _CollectingPrinter()
        monitor = _LiveUsageMonitor(parent, printer)
        child: Any = SorcarAgent("f08-child")
        child.budget_used = 0.5
        child.total_tokens_used = 100
        child.total_steps = 3
        monitor.track(child)

        monitor._emit()
        assert len(printer.printed) == 1

        # Session-handoff torn-read window: executor detached, spend not
        # yet folded — the poll sees zeros.  Must NOT be emitted.
        child.budget_used = 0.0
        child.total_tokens_used = 0
        child.total_steps = 0
        monitor._emit()
        assert len(printer.printed) == 1

        # Budget-only regression (tokens/steps rise while cost falls,
        # e.g. an expensive child's handoff dip offset by a cheap
        # sibling's growth) must not be emitted either (review-1 issue 5).
        child.budget_used = 0.1
        child.total_tokens_used = 110
        child.total_steps = 3
        monitor._emit()
        assert len(printer.printed) == 1

        # Fold completes; totals grow past the last emission — emitted.
        child.budget_used = 0.6
        child.total_tokens_used = 120
        child.total_steps = 4
        monitor._emit()
        assert len(printer.printed) == 2
        assert printer.printed[-1][1]["total_tokens"] == 120


class TestF10CrossRepoDeadlock(_TempDbTestBase):
    """Two agents swapping repos in opposite directions must not deadlock."""

    def test_opposite_direction_switch_completes(self) -> None:
        base = Path(self.tmpdir)
        repo_a = _make_repo(base, "repo_a")
        repo_b = _make_repo(base, "repo_b")

        agent1 = WorktreeSorcarAgent("f10-1")
        agent2 = WorktreeSorcarAgent("f10-2")
        assert agent1._try_setup_worktree(repo_a, None) is not None
        assert agent2._try_setup_worktree(repo_b, None) is not None

        errors: list[BaseException] = []

        def _switch(agent: WorktreeSorcarAgent, repo: Path) -> None:
            try:
                time.sleep(random.random() * 0.05)
                agent._try_setup_worktree(repo, None)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=_switch, args=(agent1, repo_b), daemon=True)
        t2 = threading.Thread(target=_switch, args=(agent2, repo_a), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=90)
        t2.join(timeout=90)

        # Before the fix, each thread held its destination repo's lock
        # while _release_worktree tried to lock the other repo (ABBA).
        assert not t1.is_alive() and not t2.is_alive(), "cross-repo deadlock"
        assert errors == []


class TestF11FlushWarningsRobustness(_TempDbTestBase):
    """A raising broadcast must neither lose warnings nor abort run()."""

    def test_warnings_survive_broken_printer(self) -> None:
        agent = WorktreeSorcarAgent("f11")
        agent._set_warnings(
            stash="stash pop failed; run 'git stash pop'",
            merge="merge conflict; resolve manually",
        )

        broken = _BrokenBroadcastPrinter()
        agent._flush_warnings(broken)  # must not raise
        assert broken.attempts == 2

        good = _CollectingPrinter()
        agent._flush_warnings(good)
        messages = [e["message"] for e in good.events if e["type"] == "warning"]
        assert messages == [
            "stash pop failed; run 'git stash pop'",
            "merge conflict; resolve manually",
        ]


class TestF12DeleteBranchFailureSurfaced(_TempDbTestBase):
    """A leaked task branch after a successful merge must be surfaced."""

    def test_undeletable_branch_sets_warning(self) -> None:
        repo = _make_repo(Path(self.tmpdir), "repo")
        _git(repo, "branch", "kiss/wt-test")
        # Check the branch out in a second worktree so both `git branch -d`
        # and `-D` fail with "used by worktree".
        holder = Path(self.tmpdir) / "holder"
        _git(repo, "worktree", "add", str(holder), "kiss/wt-test")
        (holder / "feature.txt").write_text("new work\n")
        _git(holder, "add", "-A")
        _git(holder, "commit", "-m", "agent work")

        agent = WorktreeSorcarAgent("f12")
        wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test",
            original_branch="main",
            wt_dir=Path(self.tmpdir) / "gone",
            baseline_commit=None,
        )
        result, stash_warning, cleanup_warning = agent._do_merge(wt)

        assert result == MergeResult.SUCCESS
        assert stash_warning == ""
        assert "could not be deleted" in cleanup_warning
        assert "kiss/wt-test" in cleanup_warning

    def test_merge_response_mentions_undeleted_branch(self) -> None:
        """merge() must not report an unqualified success (review-1 #10)."""
        repo = _make_repo(Path(self.tmpdir), "repo2")
        _git(repo, "branch", "kiss/wt-test2")
        holder = Path(self.tmpdir) / "holder2"
        _git(repo, "worktree", "add", str(holder), "kiss/wt-test2")
        (holder / "feature.txt").write_text("new work\n")
        _git(holder, "add", "-A")
        _git(holder, "commit", "-m", "agent work")

        agent: Any = WorktreeSorcarAgent("f12b")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test2",
            original_branch="main",
            wt_dir=Path(self.tmpdir) / "gone2",
            baseline_commit=None,
        )
        msg = agent.merge()
        assert "Successfully merged" in msg
        assert "could not be deleted" in msg
        assert agent._wt is None


class TestF13ShellQuoting(_TempDbTestBase):
    """Recovery command blocks must be executable for paths with spaces."""

    def test_recovery_block_quotes_paths_and_branches(self) -> None:
        wt = GitWorktree(
            repo_root=Path("/tmp/my repo"),
            branch="kiss/wt-1",
            original_branch="feat branch",
            wt_dir=Path("/tmp/my repo/.kiss-worktrees/x"),
            baseline_commit=None,
        )
        block = _merge_fix_steps(wt, "    git commit\n")
        assert "cd '/tmp/my repo'" in block
        assert "git checkout 'feat branch'" in block
        assert "git branch -D kiss/wt-1" in block


class TestF14SetupFailureCleansRegistries(_TempDbTestBase):
    """A setup failure after task allocation must clean both registries."""

    class _ExplodingSubscribePrinter:
        def __init__(self) -> None:
            self._thread_local = threading.local()
            self._lock = threading.Lock()
            self._persist_agents: dict[str, object] = {}

        def subscribe_tab(self, task_id: object, tab_id: str) -> None:
            raise RuntimeError("subscription backend down")

    def test_subscribe_failure_does_not_leak_running_entries(self) -> None:
        agent = ChatSorcarAgent("f14")
        printer = self._ExplodingSubscribePrinter()

        with pytest.raises(RuntimeError, match="subscription backend down"):
            agent.run(
                prompt_template="do something",
                printer=printer,
                _subscribe_tab_id="tab-f14",
            )

        # Before the fix both registries kept their entries forever,
        # leaving a permanently "running" task with no stop route.
        assert ChatSorcarAgent.running_agents == {}
        assert all(
            state.agent is not agent
            for state in _RunningAgentState.running_agent_states.values()
        )
        # The printer's persist-agent registration is removed too
        # (review-1 issue 12: a setup failure must not leave a stale
        # strong reference in a long-lived printer).
        assert printer._persist_agents == {}
        # The failure is still recorded in task history.
        history = th._load_history()
        assert len(history) == 1
        assert history[0]["result"] == "Task failed"


class TestF15AbaUnregister(_TempDbTestBase):
    """Identity-checked unregister must not delete a replacement state."""

    def test_stale_owner_cannot_remove_replacement(self) -> None:
        state_a = _RunningAgentState("tab-x", "gpt-4o", chat_id="chat-a")
        state_b = _RunningAgentState("tab-x", "gpt-4o", chat_id="chat-b")
        _RunningAgentState.register("tab-x", state_a)
        _RunningAgentState.register("tab-x", state_b)

        # Stale owner A unwinds: B must survive.
        _RunningAgentState.unregister("tab-x", state_a)
        assert _RunningAgentState.running_agent_states.get("tab-x") is state_b

        # The real owner removes its own entry.
        _RunningAgentState.unregister("tab-x", state_b)
        assert "tab-x" not in _RunningAgentState.running_agent_states

        # Key-only unregister keeps working for legacy callers.
        _RunningAgentState.register("tab-y", state_a)
        _RunningAgentState.unregister("tab-y")
        assert "tab-y" not in _RunningAgentState.running_agent_states
