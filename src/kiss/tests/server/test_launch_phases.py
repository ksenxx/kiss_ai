# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The launch phase line ("Classifying task…") around a task launch.

A full-chain run: ``VSCodeServer._run_task`` with the classifier ON,
a stand-in model endpoint answering the classifier call and the
agent's turn, an isolated ``~/.kiss`` and a real git repo.  While the
classifier (a blocking model round trip) is out the user sees
"Classifying task…"; the line is cleared when the agent takes over.
With the classifier off no phase is shown, only the clear.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from typing import Any

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


def _verdict_body() -> dict[str, Any]:
    """A chat-completions body carrying a "simple task" classifier verdict."""
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
                    "content": json.dumps({"is_simple": True}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


class TestLaunchPhases(unittest.TestCase):
    """Server + agent + repo + endpoint, with the classifier ON."""

    def setUp(self) -> None:
        self._saved_env = os.environ.get(_CLASSIFIER_DISABLE_ENV)
        os.environ[_CLASSIFIER_DISABLE_ENV] = "0"
        self.home = IsolatedKissHome(prefix="kiss-launch-phase-")
        self.repo = self.home.repo
        self.home.write_config(
            max_budget=5.0,
            use_web_browser=False,
            classify_tasks=True,
        )
        clear_classification_cache()
        keys = config_module.DEFAULT_CONFIG
        self._saved_keys = {name: getattr(keys, name) for name in _KEY_FIELDS}
        for name in _KEY_FIELDS:
            setattr(keys, name, "")
        keys.OPENAI_API_KEY = "kiss-launch-phase-standin-key"
        self.classifier_calls = 0
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
        self.standin.stop()
        keys = config_module.DEFAULT_CONFIG
        for name, value in self._saved_keys.items():
            setattr(keys, name, value)
        clear_classification_cache()
        self.home.cleanup()
        if self._saved_env is None:
            os.environ.pop(_CLASSIFIER_DISABLE_ENV, None)
        else:
            os.environ[_CLASSIFIER_DISABLE_ENV] = self._saved_env

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        if _CLASSIFIER_MARKER in request_text(request):
            with self._lock:
                self.classifier_calls += 1
            return _verdict_body()
        return finish_response("launch phase done")

    def _run(self, *, classify: bool | None = None) -> None:
        cmd: dict[str, Any] = {
            "type": "run",
            "tabId": "launch-phase-tab",
            "prompt": "say hello",
            "model": STANDIN_MODEL,
            "workDir": str(self.repo),
            "isParallel": False,
            "useWebTools": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        }
        if classify is not None:
            cmd["classifyTasks"] = classify
        self.server._run_task(cmd)

    def test_launch_phases_are_shown_and_cleared(self) -> None:
        """The user sees "Classifying task…" while the classifier is
        out and the line is cleared when the agent takes over."""
        self._run()
        self.assertEqual(self.classifier_calls, 1)
        events = self.printer.events_of_type("launch_phase")
        self.assertEqual([e["text"] for e in events], ["Classifying task…", ""])
        # Transient: every copy is addressed to the launching tab (the
        # run has no registered task yet, so the printer cannot resolve
        # the tab itself) and never recorded under a taskId.
        self.assertEqual([e.get("tabId") for e in events], ["launch-phase-tab"] * 2)
        self.assertFalse(any(e.get("taskId") for e in events))
        results = self.printer.events_of_type("result")
        self.assertTrue(results and results[-1].get("success") is not False)

    def test_no_classifying_phase_when_the_classifier_is_off(self) -> None:
        self._run(classify=False)
        self.assertEqual(self.classifier_calls, 0)
        phases = [e["text"] for e in self.printer.events_of_type("launch_phase")]
        self.assertEqual(phases, [""])


if __name__ == "__main__":
    unittest.main()
