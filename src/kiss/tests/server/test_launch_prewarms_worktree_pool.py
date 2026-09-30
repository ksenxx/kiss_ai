# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Spare worktrees are prepared at daemon start and when a worktree run ends.

A full-checkout ``git worktree add`` takes seconds to tens of seconds
on a large repository.  The pool keeps one spare per repository so a
launch only hard-resets it, and refills:

* at daemon start (``VSCodeServer.prewarm_worktree_pool``, called by
  ``kiss-web``), and
* when a worktree run ends (``WorktreeSorcarAgent.run``), never at
  submit or consume time, so the checkout's disk traffic does not
  compete with the launch it would run beside;
* holding ``repo_lock`` only for the millisecond-scale registration
  (``git worktree add --no-checkout``), not for the checkout, so a
  launch, stop or merge in that window no longer stalls behind it.

The daemon also tells the user what a launch is doing (``launch_phase``
events: "Classifying task…", "Preparing worktree…") until the agent's
first output.

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
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import repo_lock
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


class TestNoRefillAtSubmit(_PrewarmHarness):
    """Submitting a task prepares no spare: the pool is refilled when a
    worktree task ends (and at daemon start), never beside a launch."""

    def test_classifier_sees_an_idle_pool(self) -> None:
        self._run("say hello", use_worktree=True)
        self.assertEqual(len(self.classifier_seen), 1)
        seen = self.classifier_seen[0]
        self.assertIsNone(seen["spare"])
        self.assertFalse(seen["refilling"], f"a refill ran beside the launch: {seen}")

    def test_direct_run_leaves_the_pool_empty(self) -> None:
        """A run the verdict keeps out of a worktree ends no worktree
        run, so nothing schedules a refill."""
        self._run("say hello", use_worktree=True)
        self._join_refill()
        self.assertIsNone(getattr(self._agent(), "_wt", None))
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())
        self.assertEqual(
            (self.repo / "seed.txt").read_text(encoding="utf-8"), "seed\n",
        )

    def test_launch_phases_are_shown_and_cleared(self) -> None:
        """The user sees "Classifying task…" while the classifier is
        out and the line is cleared when the agent takes over."""
        self._run("say hello", use_worktree=True)
        events = self.printer.events_of_type("launch_phase")
        self.assertEqual([e["text"] for e in events], ["Classifying task…", ""])
        # Transient: every copy is addressed to the launching tab (the
        # run has no registered task yet, so the printer cannot resolve
        # the tab itself) and never recorded under a taskId.
        self.assertEqual([e.get("tabId") for e in events], ["prewarm-tab"] * 2)
        self.assertFalse(any(e.get("taskId") for e in events))

    def test_no_classifying_phase_when_the_classifier_is_off(self) -> None:
        self._run("say hello", use_worktree=False, classify=False)
        self.assertEqual(self.classifier_seen, [])
        phases = [e["text"] for e in self.printer.events_of_type("launch_phase")]
        self.assertEqual(phases, [""])


class TestRefillWhenTheWorktreeRunEnds(_PrewarmHarness):
    """A development run checks out its worktree inline (the pool was
    empty) and schedules the refill when it ends."""

    development_verdict = True
    agent_writes_file = True

    def test_spare_is_pooled_after_the_run(self) -> None:
        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        agent = self._agent()
        wt = getattr(agent, "_wt", None)
        assert wt is not None, "the development verdict should leave a pending worktree"
        self.assertTrue((wt.wt_dir / "prewarm-written.txt").is_file())
        # Nothing was prepared at submit ...
        self.assertIsNone(self.classifier_seen[0]["spare"])
        self.assertFalse(self.classifier_seen[0]["refilling"])
        # ... and the run's end refilled the pool with a different branch.
        self._join_refill()
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None, "no spare was prepared when the worktree run ended"
        self.assertNotEqual(spare[0], wt.branch)
        self.assertTrue(spare[1].is_dir())
        # The refill's maintenance pass left the pending task worktree alone.
        self.assertTrue((wt.wt_dir / "prewarm-written.txt").is_file())

    def test_worktree_run_shows_every_launch_phase(self) -> None:
        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        events = self.printer.events_of_type("launch_phase")
        self.assertEqual(
            [e["text"] for e in events], ["Classifying task…", "Preparing worktree…", ""],
        )
        self.assertEqual([e.get("tabId") for e in events], ["prewarm-tab"] * 3)
        self.assertFalse(any(e.get("taskId") for e in events))

    def test_consumed_spare_is_replaced_when_the_run_ends(self) -> None:
        """A run that consumed a pooled spare also refills at its end."""
        assert worktree_pool.prewarm(self.repo) is True
        (pooled,) = worktree_pool.spare_branches()
        self._run("PREWARM-WRITE: create a file", use_worktree=True)
        wt = getattr(self._agent(), "_wt", None)
        assert wt is not None
        self.assertEqual(wt.branch, pooled)
        self._join_refill()
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None
        self.assertNotEqual(spare[0], pooled)


class TestPrewarmGating(_PrewarmHarness):
    """Only a worktree run in a repo that is not itself a kiss worktree
    ever ends with a refill."""

    def test_no_spare_when_worktree_off_and_classifier_off(self) -> None:
        self._run("say hello", use_worktree=False, classify=False)
        self.assertEqual(self.classifier_seen, [], "classifier must not run")
        self._join_refill()
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_no_spare_when_the_user_turned_worktrees_off(self) -> None:
        """Worktrees off in the client: the classifier still runs (its
        verdict picks the system prompt but can no longer force a
        worktree on a pinned-off run) and no spare checkout is put on
        disk the user did not ask for."""
        self._run("say hello", use_worktree=False)
        self.assertEqual(len(self.classifier_seen), 1)
        self._join_refill()
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_no_spare_for_a_run_inside_a_kiss_worktree(self) -> None:
        """A run whose work dir is itself a kiss worktree (a nested
        sub-agent run) gets no worktree and so no refill: no spare is
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


def _slow_down_index_writes(repo: Path, seconds: float) -> None:
    """Install a ``post-index-change`` hook that sleeps.

    ``git reset --hard`` — the step that populates a spare — rewrites
    the index and so runs this hook; the sleep stands in for the
    checkout time of a large repository, so a test can observe what
    the pool does while a spare is still being populated.
    """
    hook = repo / ".git" / "hooks" / "post-index-change"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(f"#!/bin/sh\nsleep {seconds}\n")
    hook.chmod(0o755)


class TestCheckoutHoldsNoLock(_PrewarmHarness):
    """While a refill populates its spare, ``repo_lock`` is free, so a
    launch, stop or merge in that window no longer waits for the
    checkout; the half-built spare is meanwhile fenced off from the
    reclaim pass and from consumers."""

    def test_repo_lock_is_free_while_the_spare_is_populated(self) -> None:
        _slow_down_index_writes(self.repo, 2.0)
        thread = threading.Thread(target=worktree_pool.prewarm, args=(self.repo,))
        thread.start()
        try:
            key = worktree_pool._repo_key(self.repo)
            deadline = time.monotonic() + 30
            building: str | None = None
            while time.monotonic() < deadline:
                with worktree_pool._pool_lock:
                    building = worktree_pool._building.get(key)
                if building is not None:
                    break
                time.sleep(0.01)
            assert building is not None, "the refill never registered its spare"
            # The spare is registered but unpopulated: reported to the
            # reclaim exclusion set, not yet consumable.
            self.assertIn(building, worktree_pool.spare_branches())
            self.assertIsNone(worktree_pool.take_spare(self.repo))
            started = time.monotonic()
            with repo_lock(self.repo):
                pass
            waited = time.monotonic() - started
            self.assertLess(waited, 1.0, f"repo_lock was held during the checkout ({waited:.1f}s)")
        finally:
            thread.join(timeout=120)
        self.assertNotIn(key, worktree_pool._building)
        spare = worktree_pool.take_spare(self.repo)
        assert spare is not None, "the refill published no spare"
        self.assertEqual(spare[0], building)
        self.assertTrue((spare[1] / "seed.txt").is_file())


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
