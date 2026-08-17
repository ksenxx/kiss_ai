# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The per-run Auto-commit toggle must reach the agent.

Two consolidation defects are pinned here, both reproduced end to end
with a **real** :class:`~kiss.server.server.VSCodeServer`, a **real**
:class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`,
a **real** temporary git repository and a **real** local
OpenAI-compatible SSE endpoint that the agent genuinely talks to.  No
mock, patch, fake or test double is used, and no paid model call is
ever made.

* **K2-1** — ``task_runner`` adopted the wire's ``autoCommit`` value
  into ``state.auto_commit_mode`` and used it for its *own* post-task
  decision, but never passed it to ``agent.run(...)``.  The agent
  therefore fell back to the *persisted* ``auto_commit_mode`` config
  value for every automatic worktree cleanup it performs.  With the
  persisted value ``true`` and the wire value ``false``, a second run
  on the same tab published the first run's work onto the user's
  branch through ``_try_setup_worktree`` →
  ``_retire_previous_worktree`` — exactly the "publish work I declined
  to merge" failure the toggle exists to prevent.

* **K2-2** — ``_release_worktree_without_merging`` records the recovery
  instructions with ``agent._set_warnings(merge=...)``, and
  ``WorktreeSorcarAgent.run`` flushed them *before* ``super().run()``,
  i.e. before the run's ``task_history`` id existed and before the tab
  was subscribed to it.  The resulting ``{"type": "warning"}`` event
  carried no ``tabId``, so the printer's task fan-out had no targets
  and the event was dropped: the user was never told that the work
  they declined now lives only on a ``kiss/wt-*`` branch.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.core import config as config_module
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    run_git,
    tool_call_response,
    wait_for,
)

#: Marker put in the first run's prompt so the stand-in can tell the
#: two runs apart and script a different tool call for each.
_WRITE_MARKER = "K2-WRITE-A-FILE"

#: File the first run creates inside its worktree.  Its presence on the
#: user's branch is the observable symptom of K2-1.
_PUBLISHED_FILE = "k2-declined-work.txt"

#: Marker that makes the stand-in end the run with ``success: false``,
#: i.e. a genuinely failed task that still left changes behind.
_FAIL_MARKER = "K2-FAIL-THE-TASK"


class _FakeCredentials:
    """Blank every real provider key, leaving one unusable OpenAI key.

    Two model calls in the paths under test are *not* routed through
    the run command's ``modelConfig``: the commit-message generator
    (``generate_commit_message_from_diff``) and the follow-up
    suggestion.  Both build their own agent from
    ``config.DEFAULT_CONFIG`` and pick a model with
    :func:`~kiss.core.models.model_info.get_fast_model`.  Blanking the
    developer's real keys and leaving a single bogus OpenAI key makes
    those helpers resolve to an OpenAI model that cannot authenticate,
    so they take their documented fallback path instead of billing a
    real account, while the runner's ``get_available_models()`` gate
    still admits the stand-in's model name.
    """

    _KEY_FIELDS = (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
        "MOONSHOT_API_KEY",
    )

    def __init__(self) -> None:
        """Swap in the bogus credential set."""
        keys = config_module.DEFAULT_CONFIG
        self._saved = {
            name: getattr(keys, name) for name in self._KEY_FIELDS
        }
        for name in self._KEY_FIELDS:
            setattr(keys, name, "")
        keys.OPENAI_API_KEY = "kiss-k2-standin-key"

    def restore(self) -> None:
        """Put the developer's real credentials back."""
        keys = config_module.DEFAULT_CONFIG
        for name, value in self._saved.items():
            setattr(keys, name, value)


class _WorktreeRunnerHarness(unittest.TestCase):
    """A real server + agent + git repo + local model endpoint."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-k2-")
        self.repo = self.home.repo
        # The persisted setting is deliberately the OPPOSITE of the
        # per-run wire toggle every test sends: that difference is the
        # whole point of K2-1.
        self.home.write_config(
            auto_commit_mode=True,
            is_worktree=True,
            max_budget=5.0,
            use_web_browser=False,
        )
        self.credentials = _FakeCredentials()
        self.calls: list[str] = []
        self._calls_lock = threading.Lock()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.repo)

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
            agent = state.agent
            if agent is not None and getattr(agent, "_wt", None) is not None:
                try:
                    agent.discard()
                except Exception:
                    pass
        self.standin.stop()
        self.credentials.restore()
        self.home.cleanup()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Script the agent: write a file on the first run, else finish."""
        text = request_text(request)
        with self._calls_lock:
            self.calls.append(text)
            seen_write = sum(1 for t in self.calls if _WRITE_MARKER in t)
        if _WRITE_MARKER in text and seen_write == 1:
            return tool_call_response(
                "Bash",
                {
                    "command": f"printf 'declined\\n' > {_PUBLISHED_FILE}",
                    "description": "create the file the user declines to merge",
                },
            )
        if _FAIL_MARKER in text:
            return tool_call_response(
                "finish",
                {"success": "false", "summary_in_html": "k2 failed"},
            )
        return finish_response("k2 done")

    def _run(
        self,
        prompt: str,
        *,
        tab_id: str = "k2-tab",
        auto_commit: bool,
        use_worktree: bool = True,
    ) -> None:
        """Drive one real run through the real task runner, and block."""
        self.server._run_task(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": prompt,
                "model": STANDIN_MODEL,
                "workDir": str(self.repo),
                "useWorktree": use_worktree,
                "useParallel": False,
                "autoCommit": auto_commit,
                "webTools": False,
                "maxBudget": 5.0,
                "modelConfig": self.standin.model_config,
            },
        )

    def _tab_agent(self, tab_id: str = "k2-tab") -> Any:
        """Return the live agent the runner created for *tab_id*."""
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
        assert state is not None, "the runner never registered a state"
        return state.agent

    def _tracked_on_head(self, name: str) -> bool:
        """True when *name* is committed on the repo's checked-out branch."""
        return run_git(self.repo, "cat-file", "-e", f"HEAD:{name}").returncode == 0


class TestPerRunAutoCommitReachesAgent(_WorktreeRunnerHarness):
    """K2-1: ``autoCommit: false`` on the wire must bind the agent."""

    def test_declined_work_is_not_published_by_the_next_run(self) -> None:
        """A second worktree run must not merge the first run's work."""
        self._run(f"{_WRITE_MARKER}: create the file", auto_commit=False)
        agent = self._tab_agent()
        self.assertTrue(
            getattr(agent, "_wt_pending", False),
            "the first run should leave a worktree pending review",
        )

        self._run("second run, nothing to do", auto_commit=False)

        self.assertFalse(
            self._tracked_on_head(_PUBLISHED_FILE),
            "the declined worktree work was published onto the user's "
            "branch: the per-run autoCommit toggle never reached the "
            "agent, so it used the persisted config value instead",
        )

    def test_agent_binds_the_wire_toggle_not_the_persisted_config(self) -> None:
        """The agent's effective toggle is the one the run carried."""
        self._run("plain run", auto_commit=False)
        self.assertFalse(
            self._tab_agent().auto_commit_enabled,
            "agent.auto_commit_enabled ignored the run's autoCommit:false",
        )

    def test_wire_toggle_true_still_enables_auto_commit(self) -> None:
        """``autoCommit: true`` keeps the merging behaviour intact.

        The symmetric case: the persisted setting says *false* and the
        wire says *true*, so the run's work must land on the user's
        branch.  Asserted on git state rather than on the agent object
        because a fully auto-committed run ends with its worktree
        merged and its agent released.
        """
        self.home.write_config(auto_commit_mode=False)
        self._run(f"{_WRITE_MARKER}: create the file", auto_commit=True)
        self.assertTrue(
            self._tracked_on_head(_PUBLISHED_FILE),
            "the run's work never reached the user's branch: the "
            "per-run autoCommit:true was ignored in favour of the "
            "persisted auto_commit_mode:false",
        )


class TestMainTreeCommitObeysTheRunToggle(_WorktreeRunnerHarness):
    """A run without a worktree commits only when it was asked to."""

    def test_autocommit_off_leaves_the_users_checkout_uncommitted(self) -> None:
        """``useWorktree:false, autoCommit:false`` must not commit."""
        self._run(
            f"{_WRITE_MARKER}: create the file",
            auto_commit=False,
            use_worktree=False,
        )

        self.assertTrue(
            (self.repo / _PUBLISHED_FILE).is_file(),
            "the run never produced the file it was asked to write",
        )
        self.assertFalse(
            self._tracked_on_head(_PUBLISHED_FILE),
            "the run committed in the user's own checkout even though "
            "the run carried autoCommit:false",
        )

    def test_autocommit_on_commits_the_users_checkout(self) -> None:
        """The symmetric case still commits, so the toggle is real."""
        self.home.write_config(auto_commit_mode=False)
        self._run(
            f"{_WRITE_MARKER}: create the file",
            auto_commit=True,
            use_worktree=False,
        )

        self.assertTrue(
            self._tracked_on_head(_PUBLISHED_FILE),
            "autoCommit:true no longer commits a non-worktree run",
        )

    def test_failed_run_does_not_commit_partial_work(self) -> None:
        """A task that reports failure leaves its half-done work alone."""
        self._run(
            f"{_WRITE_MARKER} {_FAIL_MARKER}: create the file, then fail",
            auto_commit=True,
            use_worktree=False,
        )

        self.assertTrue(
            (self.repo / _PUBLISHED_FILE).is_file(),
            "the run never produced the file it was asked to write",
        )
        self.assertFalse(
            self._tracked_on_head(_PUBLISHED_FILE),
            "a failed task's partial changes were committed to the "
            "user's checkout",
        )


class TestPendingWorktreeRetirementObeysTheCurrentRun(_WorktreeRunnerHarness):
    """Retiring a carried-over worktree uses THIS run's toggle."""

    def _leave_pending_worktree(self, auto_commit: bool) -> Any:
        """Run a failing worktree task, leaving its worktree pending."""
        self._run(
            f"{_WRITE_MARKER} {_FAIL_MARKER}: create the file, then fail",
            auto_commit=auto_commit,
        )
        agent = self._tab_agent()
        self.assertTrue(
            getattr(agent, "_wt_pending", False),
            "the failed run should leave a worktree pending review",
        )
        return agent

    def test_toggling_off_preserves_the_carried_over_worktree(self) -> None:
        """Auto-commit off at run time must not commit the old worktree."""
        agent = self._leave_pending_worktree(auto_commit=True)
        wt_dir = agent._wt.wt_dir
        branch = agent._wt_branch
        self.assertTrue((wt_dir / _PUBLISHED_FILE).is_file())

        # The user switches Auto-commit off and starts a run directly
        # on the main tree, which retires the pending worktree.
        self._run("follow-up run", auto_commit=False, use_worktree=False)

        committed = run_git(
            self.repo, "cat-file", "-e", f"{branch}:{_PUBLISHED_FILE}",
        )
        self.assertNotEqual(
            committed.returncode,
            0,
            "the pending worktree was committed even though the run "
            "that retired it carried autoCommit:false",
        )
        self.assertTrue(
            (wt_dir / _PUBLISHED_FILE).is_file(),
            "the work was neither committed nor left on disk",
        )

    def test_toggling_on_commits_the_carried_over_worktree(self) -> None:
        """Auto-commit on at run time commits the old worktree's work."""
        agent = self._leave_pending_worktree(auto_commit=False)
        wt_dir = agent._wt.wt_dir
        branch = agent._wt_branch

        self._run("follow-up run", auto_commit=True, use_worktree=False)

        committed = run_git(
            self.repo, "cat-file", "-e", f"{branch}:{_PUBLISHED_FILE}",
        )
        self.assertEqual(
            committed.returncode,
            0,
            "the run asked for auto-commit, but the retired worktree's "
            f"work never landed on {branch}",
        )
        self.assertFalse(
            wt_dir.exists(),
            "a committed worktree directory must be removed",
        )


class TestReleaseWarningReachesTheUser(_WorktreeRunnerHarness):
    """K2-2: the "your work is only on kiss/wt-*" warning must arrive."""

    def test_release_warning_is_broadcast_to_the_tab(self) -> None:
        """Releasing a pending worktree must tell the user where it went."""
        self._run(f"{_WRITE_MARKER}: create the file", auto_commit=False)
        agent = self._tab_agent()
        branch = agent._wt_branch
        self.assertTrue(branch, "the first run should own a worktree branch")

        # A run on ANOTHER tab that writes the main working tree makes
        # the pending worktree unmergeable, which is what triggers
        # `_release_worktree_without_merging`.
        holder = threading.Event()
        released = threading.Event()
        self._occupy_main_tree(holder, released)
        try:
            self._run(
                "second run on the same tab",
                auto_commit=False,
            )
        finally:
            holder.set()
            released.wait(timeout=30)

        warnings = [
            event
            for event in self.printer.events_of_type("warning")
            if branch in str(event.get("message", ""))
        ]
        self.assertTrue(
            warnings,
            "the user was never told their declined work now lives only "
            f"on {branch}: the warning was flushed before the tab was "
            "subscribed and dropped by the printer fan-out",
        )
        self.assertTrue(
            all(event.get("tabId") for event in warnings),
            "a release warning without a tabId is dropped by every client",
        )

    def _occupy_main_tree(
        self, holder: threading.Event, released: threading.Event,
    ) -> None:
        """Register a real non-worktree task holding the main tree."""
        state = agent_state.AgentState(
            "k2-main-tree-holder",
            chat_id="k2-holder-chat",
            tab_id="k2-other-tab",
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.use_worktree = False
        state.is_running_non_wt = True
        state.is_task_active = True

        def _worker() -> None:
            holder.wait(timeout=60)
            with agent_state.STATE_LOCK:
                state.is_task_active = False
                state.is_running_non_wt = False
            released.set()

        thread = threading.Thread(
            target=_worker, name="k2-main-tree-holder", daemon=True,
        )
        state.task_thread = thread
        agent_state.register(state)
        thread.start()
        self.addCleanup(holder.set)
        self.addCleanup(lambda: wait_for(released.is_set, timeout=30))


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
