# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A worktree a run left unmerged is parked, never published, by the next run.

Reproduced end to end with a **real** :class:`~kiss.server.server.VSCodeServer`,
a **real** :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent`,
a **real** temporary git repository and a **real** local OpenAI-compatible
SSE endpoint that the agent genuinely talks to.  No mock, patch, fake or
test double is used, and no paid model call is ever made.

The agent decides about worktrees itself through its ``create_worktree``
and ``commit_and_merge`` tools.  A run that calls ``create_worktree``,
writes a file and finishes **without** ``commit_and_merge`` leaves the
worktree pending on the tab (the server offers the Merge / Discard /
Do-nothing bar).  Two things are pinned here:

* **K2-1** — the next run on the same tab must not publish that work
  onto the user's branch: ``task_runner`` retires the pending worktree
  through ``_retire_previous_worktree``, which commits it to its own
  ``kiss/wt-*`` branch and removes the worktree directory, and never
  squash-merges it.

* **K2-2** — the "your work now lives only on ``kiss/wt-*``" warning the
  retire records with ``agent._set_warnings(merge=...)`` must reach the
  tab: ``ChatSorcarAgent.run`` flushes it before ``super().run()``,
  i.e. before the run's ``task_history`` id exists, so the
  ``{"type": "warning"}`` event has to be stamped with the tab id by
  the agent's own routing or the printer's fan-out drops it.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.core import config as config_module
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.conftest import run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

#: Marker put in a run's prompt so the stand-in scripts a worktree run:
#: ``create_worktree`` → write a file → ``finish`` (never merged).
_WRITE_MARKER = "K2-WRITE-A-FILE"

#: File the worktree run creates.  Its presence on the user's branch is
#: the observable symptom of K2-1.
_PUBLISHED_FILE = "k2-declined-work.txt"

#: Marker that makes the stand-in end the run with ``success: false``,
#: i.e. a genuinely failed task that still left changes behind.
_FAIL_MARKER = "K2-FAIL-THE-TASK"

#: Marker of a follow-up run that just finishes.
_PLAIN_MARKER = "K2-PLAIN-RUN"

_MARKERS = (_WRITE_MARKER, _PLAIN_MARKER)


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


def _current_run_marker(request: dict[str, Any]) -> str:
    """Return the marker of the run the *request* belongs to.

    The chat keeps earlier runs' messages, so the marker is read from
    the **last** user message that carries one, not from the whole
    transcript.
    """
    for message in reversed(request.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if not isinstance(content, str):
            continue
        for marker in _MARKERS:
            if marker in content:
                return marker
        if _FAIL_MARKER in content:
            return _FAIL_MARKER
    return ""


class _WorktreeRunnerHarness(unittest.TestCase):
    """A real server + agent + git repo + local model endpoint."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-k2-")
        self.repo = self.home.repo
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.credentials = _FakeCredentials()
        self.calls: list[tuple[str, str]] = []
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
        """Script the agent per run.

        A write run (``_WRITE_MARKER``) calls ``create_worktree``, then
        writes ``_PUBLISHED_FILE`` with ``Bash``, then ``finish``es
        without ``commit_and_merge`` (the first ``finish`` is refused
        once by the worktree gate, the second passes).  A plain run
        just finishes.
        """
        text = request_text(request)
        marker = _current_run_marker(request)
        with self._calls_lock:
            self.calls.append((marker, text))
            step = sum(1 for m, _ in self.calls if m == marker)
        if marker == _WRITE_MARKER and step == 1:
            return tool_call_response("create_worktree", {})
        if marker == _WRITE_MARKER and step == 2:
            return tool_call_response(
                "Bash",
                {
                    "command": f"printf 'declined\\n' > {_PUBLISHED_FILE}",
                    "description": "create the file the user never merges",
                },
            )
        if _FAIL_MARKER in text:
            return tool_call_response(
                "finish",
                {"success": "false", "summary_in_html": "k2 failed"},
            )
        return finish_response("k2 done")

    def _run(self, prompt: str, *, tab_id: str = "k2-tab") -> None:
        """Drive one real run through the real task runner, and block."""
        self.server._run_task(
            {
                "type": "run",
                "tabId": tab_id,
                "prompt": prompt,
                "model": STANDIN_MODEL,
                "workDir": str(self.repo),
                "isParallel": False,
                "useWebTools": False,
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

    def _tracked_on(self, ref: str, name: str) -> bool:
        """True when *name* is committed on *ref* in the repo."""
        return run_git(self.repo, "cat-file", "-e", f"{ref}:{name}").returncode == 0

    def _leave_pending_worktree(self, prompt: str) -> tuple[Any, str, Any]:
        """Run *prompt* and assert it left its worktree pending on the tab.

        Returns:
            The agent, its ``kiss/wt-*`` branch and the worktree dir.
        """
        self._run(prompt)
        agent = self._tab_agent()
        self.assertIsNotNone(agent, "the runner released the tab's agent")
        self.assertTrue(
            getattr(agent, "_wt_pending", False),
            "the run should leave a worktree pending review",
        )
        branch = agent._wt_branch
        wt_dir = agent._wt.wt_dir
        self.assertTrue(branch, "the run should own a worktree branch")
        self.assertTrue(
            (wt_dir / _PUBLISHED_FILE).is_file(),
            "the run never produced the file inside its worktree",
        )
        self.assertFalse(
            (self.repo / _PUBLISHED_FILE).exists(),
            "the run wrote into the user's checkout instead of the worktree",
        )
        return agent, branch, wt_dir


class TestPendingWorktreeIsOfferedNotMerged(_WorktreeRunnerHarness):
    """A run that never calls ``commit_and_merge`` leaves its work pending."""

    def test_finished_run_leaves_worktree_pending_with_bar(self) -> None:
        """The server offers the Merge / Discard bar instead of merging."""
        _agent, branch, _wt_dir = self._leave_pending_worktree(
            f"{_WRITE_MARKER}: create the file",
        )
        self.assertFalse(
            self._tracked_on("HEAD", _PUBLISHED_FILE),
            "the run's unmerged work was published onto the user's branch",
        )
        done = [
            event
            for event in self.printer.events_of_type("worktree_done")
            if event.get("tabId") == "k2-tab"
        ]
        self.assertEqual(
            len(done), 1, f"expected one worktree_done bar for the tab, got {done}",
        )
        self.assertEqual(done[0].get("branch"), branch)
        self.assertIn(_PUBLISHED_FILE, done[0].get("changedFiles", []))

    def test_failed_run_leaves_worktree_pending(self) -> None:
        """A task that reports failure keeps its half-done work pending."""
        self._leave_pending_worktree(
            f"{_WRITE_MARKER} {_FAIL_MARKER}: create the file, then fail",
        )
        self.assertFalse(
            self._tracked_on("HEAD", _PUBLISHED_FILE),
            "a failed task's partial changes were merged into the user's branch",
        )


class TestNextRunParksThePendingWorktree(_WorktreeRunnerHarness):
    """K2-1: the next run retires the carried-over worktree without merging."""

    def test_declined_work_is_not_published_by_the_next_run(self) -> None:
        """The pending work is committed to its branch, never to the user's."""
        agent, branch, wt_dir = self._leave_pending_worktree(
            f"{_WRITE_MARKER}: create the file",
        )

        self._run(f"{_PLAIN_MARKER}: second run, nothing to do")

        self.assertFalse(
            self._tracked_on("HEAD", _PUBLISHED_FILE),
            "the unmerged worktree work was published onto the user's "
            "branch by the next run",
        )
        self.assertTrue(
            self._tracked_on(branch, _PUBLISHED_FILE),
            f"the retired worktree's work never landed on {branch}",
        )
        self.assertFalse(
            wt_dir.exists(),
            "a retired worktree directory must be removed",
        )
        self.assertFalse(
            getattr(agent, "_wt_pending", False),
            "the retired worktree is still pending on the agent",
        )


class TestRetireWarningReachesTheUser(_WorktreeRunnerHarness):
    """K2-2: the "your work is only on kiss/wt-*" warning must arrive."""

    def test_retire_warning_is_broadcast_to_the_tab(self) -> None:
        """Retiring a pending worktree must tell the user where it went."""
        _agent, branch, _wt_dir = self._leave_pending_worktree(
            f"{_WRITE_MARKER}: create the file",
        )

        self._run(f"{_PLAIN_MARKER}: second run on the same tab")

        warnings = [
            event
            for event in self.printer.events_of_type("warning")
            if branch in str(event.get("message", ""))
        ]
        self.assertTrue(
            warnings,
            "the user was never told their unmerged work now lives only "
            f"on {branch}: the warning was flushed before the tab was "
            "subscribed and dropped by the printer fan-out",
        )
        self.assertTrue(
            all(event.get("tabId") == "k2-tab" for event in warnings),
            f"a retire warning must be stamped with the tab id: {warnings}",
        )


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
