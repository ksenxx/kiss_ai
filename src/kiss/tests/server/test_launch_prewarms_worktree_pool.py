# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The task runner starts preparing a spare worktree before classifying.

Submit-path latency: the pre-run classifier costs one LLM round trip
(~2 s on the large models), and a development verdict then paid a
full-checkout ``git worktree add`` (~1.5 s on a real repository) on
top of it whenever the spare pool was empty — always for the first
worktree task of a daemon session, since the pool used to refill only
AFTER a worktree task had consumed a spare.

``_run_task_inner`` now schedules ``worktree_pool.prewarm_async``
right before ``classify_task_for_run`` (maintenance-free, so it never
squash-merges anything into the main tree), so the checkout overlaps
the classifier's wait and the run's own worktree acquisition finds a
ready spare.

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
    """No spare is prepared unless a classifier wait can be overlapped
    for a user who has worktrees on."""

    def test_no_prewarm_when_worktree_off_and_classifier_off(self) -> None:
        self._run("say hello", use_worktree=False, classify=False)
        self.assertEqual(self.classifier_seen, [], "classifier must not run")
        self._join_refill()
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

    def test_no_prewarm_when_the_user_turned_worktrees_off(self) -> None:
        """Worktrees off in the client: the classifier still runs (and
        could force a worktree inline, as before) but no spare checkout
        is put on disk the user did not ask for."""
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

    def test_classifier_off_keeps_the_acquire_then_refill_path(self) -> None:
        """With no classifier wait to overlap, a worktree run acquires
        its worktree as before and the refill still pools a spare."""
        self._run("say hello", use_worktree=True, classify=False)
        self.assertEqual(self.classifier_seen, [])
        self._join_refill()
        self.assertIsNotNone(worktree_pool.take_spare(self.repo))

    def test_cached_verdict_schedules_no_prewarm(self) -> None:
        """A memoised verdict means no LLM wait, hence no submit-time
        prewarm: the pool stays empty for a direct run."""
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
        self.assertIsNone(worktree_pool.take_spare(self.repo))
        self.assertEqual(worktree_pool.spare_branches(), set())

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
