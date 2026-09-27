# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Spare worktrees are prepared at daemon start and at every submit.

Submit-path latency: a full-checkout ``git worktree add`` takes seconds
on a large repository, and a worktree run used to pay for it inline
whenever the spare pool was empty — always for the first worktree task
after a daemon start, and for any run whose classifier verdict came
from the cache (no LLM wait to overlap the checkout with).

* ``VSCodeServer.prewarm_worktree_pool`` (called by ``kiss-web`` at
  start) creates a spare for the daemon's work dir.
* ``_run_task_inner`` schedules ``worktree_pool.prewarm_async`` right
  before classification (maintenance-free, so it never squash-merges
  anything into the main tree), whether or not a classifier call
  follows.
* A run that reaches ``_acquire_task_worktree`` while that refill is
  still checking out waits on ``repo_lock`` (held by the refill) and
  then consumes the spare, never running a second checkout.

Everything here is real: a real :class:`VSCodeServer`, a real
:class:`WorktreeSorcarAgent`, a real temporary git repository and a
real local OpenAI-compatible endpoint that answers both the
classifier's and the agent's requests.  No mock, patch, fake or test
double is used.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.task_classifier import (
    _DISABLE_ENV as _CLASSIFIER_DISABLE_ENV,
)
from kiss.agents.sorcar.task_classifier import (
    clear_classification_cache,
)
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
    tool_call_response,
)

_CLASSIFIER_MARKER = "You are a task classifier"
_KEY_FIELDS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
)


def _verdict_body(is_development: bool) -> dict[str, Any]:
    """A chat-completions body carrying the classifier verdict JSON."""
    return {
        "id": "chatcmpl-kiss-classifier",
        "object": "chat.completion",
        "created": 0,
        "model": STANDIN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "is_simple": not is_development,
                        "is_development": is_development,
                    }),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


class _PrewarmHarness(unittest.TestCase):
    """Server + agent + repo + endpoint, with the pool and classifier ON."""

    #: Verdict the stand-in classifier returns; tests override.
    development_verdict = False
    #: Whether the agent's first step writes a file (keeps the worktree
    #: pending for inspection) or finishes immediately.
    agent_writes_file = False

    def setUp(self) -> None:
        self._saved_env = {
            name: os.environ.get(name)
            for name in (worktree_pool._DISABLE_ENV, _CLASSIFIER_DISABLE_ENV)
        }
        os.environ[worktree_pool._DISABLE_ENV] = "0"
        os.environ[_CLASSIFIER_DISABLE_ENV] = "0"
        self.home = IsolatedKissHome(prefix="kiss-prewarm-")
        self.repo = self.home.repo
        self.home.write_config(
            auto_commit_mode=False,
            is_worktree=True,
            max_budget=5.0,
            use_web_browser=False,
            classify_tasks=True,
        )
        clear_classification_cache()
        keys = config_module.DEFAULT_CONFIG
        self._saved_keys = {name: getattr(keys, name) for name in _KEY_FIELDS}
        for name in _KEY_FIELDS:
            setattr(keys, name, "")
        keys.OPENAI_API_KEY = "kiss-prewarm-standin-key"
        self.classifier_seen: list[dict[str, Any]] = []
        self._lock = threading.Lock()
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
        worktree_pool.discard_all()
        self.standin.stop()
        keys = config_module.DEFAULT_CONFIG
        for name, value in self._saved_keys.items():
            setattr(keys, name, value)
        clear_classification_cache()
        self.home.cleanup()
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _pool_snapshot(self) -> dict[str, Any]:
        """What the pool holds for the repo right now (thread-safe read)."""
        key = worktree_pool._repo_key(self.repo)
        with worktree_pool._pool_lock:
            refill = worktree_pool._refill_threads.get(key)
            return {
                "spare": worktree_pool._spares.get(key),
                "refilling": key in worktree_pool._prewarming
                or (refill is not None and refill.is_alive()),
            }

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if _CLASSIFIER_MARKER in text:
            # Observe the pool AT THE MOMENT the classifier request
            # arrives: the prewarm must already be underway (or done).
            with self._lock:
                self.classifier_seen.append(self._pool_snapshot())
            return _verdict_body(self.development_verdict)
        if self.agent_writes_file and "PREWARM-WRITE" in text:
            calls = sum(
                1 for e in self.printer.events_of_type("tool_call")
                if e.get("name") == "Bash"
            )
            if calls == 0:
                return tool_call_response(
                    "Bash",
                    {
                        "command": "printf 'x\\n' > prewarm-written.txt",
                        "description": "leave a change in the worktree",
                    },
                )
        return finish_response("prewarm done")

    def _run(self, prompt: str, *, use_worktree: bool, classify: bool | None = None) -> None:
        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": "prewarm-tab",
            "prompt": prompt,
            "model": STANDIN_MODEL,
            "workDir": str(self.repo),
            "useWorktree": use_worktree,
            "useParallel": False,
            "autoCommit": False,
            "webTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        }
        if classify is not None:
            cmd["classifyTasks"] = classify
        self.server._run_task(cmd)

    def _join_refill(self) -> None:
        key = worktree_pool._repo_key(self.repo)
        with worktree_pool._pool_lock:
            thread = worktree_pool._refill_threads.get(key)
        if thread is not None:
            thread.join(timeout=120)

    def _agent(self) -> Any:
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab("prewarm-tab")
        assert state is not None, "the runner never registered a state"
        return state.agent


class TestPrewarmOverlapsClassifier(_PrewarmHarness):
    """A spare is being prepared while the classifier is still out."""

    def test_prewarm_starts_before_the_classifier_request(self) -> None:
        """By the time the classifier request reaches the model, the pool
        is already refilling (or holds a spare) for the run's repo."""
        self._run("say hello", use_worktree=True)
        self.assertEqual(len(self.classifier_seen), 1)
        seen = self.classifier_seen[0]
        self.assertTrue(
            seen["spare"] is not None or seen["refilling"],
            f"no spare preparation was in flight at classification time: {seen}",
        )

    def test_simple_verdict_leaves_the_spare_pooled_for_the_next_task(self) -> None:
        """A run the verdict keeps out of a worktree does not consume the
        spare; it stays ready for the next development task."""
        self._run("say hello", use_worktree=True)
        self._join_refill()
        spare = worktree_pool.take_spare(self.repo)
        self.assertIsNotNone(spare, "the submit-time prewarm produced no spare")
        branch, wt_dir = spare  # type: ignore[misc]
        self.assertTrue(branch.startswith("kiss/wt-"))
        self.assertTrue(wt_dir.is_dir())
        # The direct run itself never touched a worktree.
        self.assertIsNone(getattr(self._agent(), "_wt", None))
        # Maintenance-free prewarm: nothing was merged into the main
        # branch and the checkout is untouched.
        self.assertEqual(
            (self.repo / "seed.txt").read_text(encoding="utf-8"), "seed\n",
        )


class TestDevelopmentRunConsumesThePrewarmedSpare(_PrewarmHarness):
    """A development verdict runs inside the spare prepared at submit."""

    development_verdict = True
    agent_writes_file = True

    def test_worktree_run_uses_the_spare_created_during_classification(self) -> None:
        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        agent = self._agent()
        wt = getattr(agent, "_wt", None)
        assert wt is not None, "the development verdict should leave a pending worktree"
        self.assertTrue((wt.wt_dir / "prewarm-written.txt").is_file())
        # The spare prepared during classification was consumed: the
        # pool holds a different (refilled) spare or is refilling it.
        self._join_refill()
        with worktree_pool._pool_lock:
            pooled = worktree_pool._spares.get(worktree_pool._repo_key(self.repo))
        self.assertTrue(pooled is None or pooled[0] != wt.branch)
        seen = self.classifier_seen[0]
        self.assertTrue(seen["spare"] is not None or seen["refilling"])


class TestPrewarmGating(_PrewarmHarness):
    """A spare is prepared at submit only for a user who has worktrees
    on, in a repo that is not itself a kiss worktree."""

    def test_no_prewarm_when_worktree_off_and_classifier_off(self) -> None:
        self._run("say hello", use_worktree=False, classify=False)
        self.assertEqual(self.classifier_seen, [], "classifier must not run")
        self._join_refill()
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_no_prewarm_when_the_user_turned_worktrees_off(self) -> None:
        """Worktrees off in the client: the classifier still runs (its
        verdict picks the system prompt but can no longer force a
        worktree on a pinned-off run) and no spare checkout is put on
        disk the user did not ask for."""
        self._run("say hello", use_worktree=False)
        self.assertEqual(len(self.classifier_seen), 1)
        self.assertFalse(
            self.classifier_seen[0]["spare"] is not None
            or self.classifier_seen[0]["refilling"],
            "a spare was being prepared although worktrees are off",
        )
        self._join_refill()
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_no_prewarm_for_a_run_inside_a_kiss_worktree(self) -> None:
        """A run whose work dir is itself a kiss worktree (a nested
        sub-agent run) keeps the acquire-then-refill path: no spare is
        nested under a worktree that will be removed."""
        assert worktree_pool.prewarm(self.repo) is True
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        _branch, nested_dir = spare
        self.server.work_dir = str(nested_dir)
        self.server._run_task({
            "type": "run",
            "tabId": "prewarm-tab",
            "prompt": "say hello",
            "model": STANDIN_MODEL,
            "workDir": str(nested_dir),
            "useWorktree": True,
            "useParallel": False,
            "autoCommit": False,
            "webTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        })
        self.assertEqual(len(self.classifier_seen), 1)
        self._join_refill()
        self.assertEqual(worktree_pool.spare_branches(), set())
        self.assertFalse((nested_dir / ".kiss-worktrees").exists())

    def test_classifier_off_still_pools_a_spare(self) -> None:
        """With no classifier wait to overlap, a worktree run still
        leaves a spare pooled for the next task."""
        self._run("say hello", use_worktree=True, classify=False)
        self.assertEqual(self.classifier_seen, [])
        self._join_refill()
        self.assertIsNotNone(worktree_pool.take_spare(self.repo))

    def test_cached_verdict_still_prewarms(self) -> None:
        """A memoised verdict means no LLM wait, but the submit-time
        prewarm still runs: a direct run leaves a spare pooled."""
        self._run("say hello", use_worktree=True)
        self.assertEqual(len(self.classifier_seen), 1)
        self._join_refill()
        worktree_pool.discard_all()
        self.assertIsNone(worktree_pool.take_spare(self.repo))

        self._run("say hello", use_worktree=True)
        self.assertEqual(
            len(self.classifier_seen), 1, "the second run must hit the verdict cache",
        )
        self._join_refill()
        self.assertIsNotNone(worktree_pool.take_spare(self.repo))

    def test_non_git_work_dir_is_harmless(self) -> None:
        plain_dir = Path(self.home.tmpdir) / "not-a-repo"
        plain_dir.mkdir()
        self.server.work_dir = str(plain_dir)
        self.server._run_task({
            "type": "run",
            "tabId": "prewarm-tab",
            "prompt": "say hello",
            "model": STANDIN_MODEL,
            "workDir": str(plain_dir),
            "useWorktree": True,
            "useParallel": False,
            "autoCommit": False,
            "webTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        })
        self.assertEqual(len(self.classifier_seen), 1)
        self.assertEqual(worktree_pool.spare_branches(), set())
        results = self.printer.events_of_type("result")
        self.assertTrue(results and results[-1].get("success") is not False)


class TestDevelopmentVerdictRespectsWorktreePin(_PrewarmHarness):
    """A development verdict cannot promote a run whose client pinned
    worktrees off — the channel-dispatch guarantee: ``run_agent``
    sends ``use_worktree=False`` for a channel sub-task in a scratch
    directory while leaving classification enabled, so the verdict may
    pick the system prompt but must never put a worktree there."""

    development_verdict = True
    agent_writes_file = True

    def test_pinned_off_run_stays_in_the_main_tree(self) -> None:
        self._run("PREWARM-WRITE: create a file", use_worktree=False)
        # The classifier really ran (classification was NOT disabled) …
        self.assertEqual(len(self.classifier_seen), 1)
        # … but its development verdict created no worktree: the run
        # executed directly in the repo and the file landed there.
        agent = self._agent()
        self.assertIsNone(getattr(agent, "_wt", None))
        self.assertEqual(self.printer.events_of_type("worktree_created"), [])
        self.assertTrue((self.repo / "prewarm-written.txt").is_file())
        results = self.printer.events_of_type("result")
        self.assertTrue(results and results[-1].get("success") is not False)


def _record_checkouts(repo: Path) -> Path:
    """Install a ``post-checkout`` hook that logs each new worktree.

    ``git worktree add`` runs the hook inside the new checkout, so the
    log lists worktree branches in creation order.  The hook also
    sleeps, standing in for a large repository's slow checkout, so a
    run really starts while the submit-time spare is still being
    created.  Returns the log.
    """
    log = repo.parent / "checkouts.log"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(f"#!/bin/sh\nsleep 1\ngit symbolic-ref --short HEAD >> '{log}'\n")
    hook.chmod(0o755)
    return log


class TestRunWaitsForTheInFlightSpare(_PrewarmHarness):
    """A worktree run with no classifier wait consumes the spare the
    submit-time prewarm is still creating instead of running a second
    checkout inline."""

    development_verdict = True
    agent_writes_file = True

    def _assert_run_used_first_checkout(self, log: Path) -> None:
        """The run's worktree is the first one created after submit:
        the prewarm's spare, not a second inline checkout."""
        wt = getattr(self._agent(), "_wt", None)
        assert wt is not None, "the run should leave a pending worktree"
        self.assertTrue((wt.wt_dir / "prewarm-written.txt").is_file())
        self._join_refill()
        created = log.read_text(encoding="utf-8").split()
        self.assertEqual(created[0], wt.branch, created)

    def test_cached_development_verdict_uses_the_submit_time_spare(self) -> None:
        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        self.assertEqual(len(self.classifier_seen), 1)
        self._agent().discard()
        self._join_refill()
        worktree_pool.discard_all()
        self.printer.captured.clear()
        log = _record_checkouts(self.repo)

        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        self.assertEqual(
            len(self.classifier_seen), 1, "the second run must hit the verdict cache",
        )
        self._assert_run_used_first_checkout(log)

    def test_classifier_off_run_uses_the_submit_time_spare(self) -> None:
        log = _record_checkouts(self.repo)
        self._run("PREWARM-WRITE: create a file", use_worktree=True, classify=False)
        self.assertEqual(self.classifier_seen, [])
        self._assert_run_used_first_checkout(log)


class TestDaemonStartPrewarm(_PrewarmHarness):
    """``VSCodeServer.prewarm_worktree_pool`` (run by ``kiss-web`` at
    start) leaves a spare ready for the daemon's work dir."""

    def test_start_creates_a_spare_for_the_work_dir(self) -> None:
        thread = self.server.prewarm_worktree_pool()
        assert thread is not None
        thread.join(timeout=120)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None, "no spare after the daemon-start prewarm"
        self.assertTrue(spare[1].is_dir())
        # Maintenance-free: the main checkout is untouched.
        self.assertEqual((self.repo / "seed.txt").read_text(encoding="utf-8"), "seed\n")

    def test_nothing_when_worktrees_are_off(self) -> None:
        self.home.write_config(is_worktree=False)
        self.assertIsNone(self.server.prewarm_worktree_pool())
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_nothing_when_the_pool_is_disabled(self) -> None:
        os.environ[worktree_pool._DISABLE_ENV] = "1"
        self.assertIsNone(self.server.prewarm_worktree_pool())
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_nothing_for_a_non_git_work_dir(self) -> None:
        plain_dir = Path(self.home.tmpdir) / "not-a-repo"
        plain_dir.mkdir()
        self.server.work_dir = str(plain_dir)
        self.assertIsNone(self.server.prewarm_worktree_pool())

    def test_nothing_for_a_work_dir_inside_a_kiss_worktree(self) -> None:
        assert worktree_pool.prewarm(self.repo) is True
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        self.server.work_dir = str(spare[1])
        self.assertIsNone(self.server.prewarm_worktree_pool())
        self.assertEqual(worktree_pool.spare_branches(), set())
