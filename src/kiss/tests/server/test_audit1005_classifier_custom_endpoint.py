# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the pre-run task classifier calls the run's own endpoint.

A settings-panel custom model (a ``~/.kiss/MY_MODELS.json`` entry with
an ``endpoint``) runs against ITS OWN endpoint / key: the task runner
resolves ``custom_model_config(model)`` for the run.  The classifier
that decides the run's worktree mode BEFORE the run used to be handed
a different configuration — the global config's, resolved a second
time, without the custom endpoint — so a custom-endpoint model was
classified by name against the default provider, which fails soft (no
worktree demotion) or waits out the classifier's stall timeout.

``_run_task_inner`` now resolves the model configuration once, and the
classifier and the run share it.

Harness: an isolated ``KISS_HOME`` whose ``MY_MODELS.json`` points a
bundled model name at a local stand-in OpenAI-compatible endpoint
(``StandInModelServer``) that records every request; a real
``VSCodeServer`` runs the task through ``_cmd_run`` with the real
``WorktreeSorcarAgent`` the server creates.  The stand-in answers the
classifier's request (no ``tools``) with a verdict and the agent's
first request (``tools`` attached) with a ``finish`` call, so the
test sees BOTH model calls of the run arrive at the custom endpoint.
No mocks or patches.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.task_classifier import clear_classification_cache
from kiss.core.models import model_info
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

_TAB = "tab-a1005-classifier-endpoint"
_WAIT_S = 90.0
_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"
_SUMMARY = "<p>audit1005 run finished at the custom endpoint</p>"


class _CapturePrinter(JsonPrinter):
    """Real printer recording every broadcast."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        super().broadcast(event)


def _verdict(request: dict[str, Any]) -> dict[str, Any]:
    """Answer the classifier's chat-completions request with a verdict."""
    return {
        "id": "chatcmpl-audit1005",
        "object": "chat.completion",
        "created": 0,
        "model": request.get("model", STANDIN_MODEL),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps({"is_simple": True, "is_development": False}),
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


class _Endpoint:
    """A stand-in endpoint that records the requests it receives.

    The classifier is a plain structured-output call (no ``tools``);
    the agent's tool loop attaches its toolset to every request.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        if request.get("tools"):
            return finish_response(_SUMMARY)
        return _verdict(request)


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """Isolated ``KISS_HOME`` with the LLM classifier enabled."""
    if STANDIN_MODEL not in get_available_models():
        pytest.skip(f"{STANDIN_MODEL} is not configured in this environment")
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-audit1005-classifier-")
    isolated.write_config(classify_with_decisions=False)
    # ``USER_MY_MODELS_PATH`` is bound at import time, so the isolated
    # home's file must be pointed at explicitly (as
    # ``test_settings_custom_models.py`` does) or the entry would land
    # in the developer's real ``~/.kiss/MY_MODELS.json``.
    saved_my_models = model_info.USER_MY_MODELS_PATH
    model_info.USER_MY_MODELS_PATH = isolated.kiss_home / "MY_MODELS.json"
    clear_classification_cache()
    _clear_tab_states()
    try:
        yield isolated
    finally:
        _clear_tab_states()
        clear_classification_cache()
        model_info.USER_MY_MODELS_PATH = saved_my_models
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _clear_tab_states() -> None:
    with agent_state.STATE_LOCK:
        for st in agent_state.snapshot():
            if st.tab_id == _TAB:
                agent_state.unregister(st.task_id, st)


def _wait_idle() -> None:
    deadline = time.time() + _WAIT_S
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            thread = st.task_thread if st is not None else None
        if thread is None:
            return
        thread.join(timeout=max(0.0, deadline - time.time()))
    raise AssertionError("tab never went idle")


def test_classifier_uses_the_custom_models_endpoint(home: IsolatedKissHome) -> None:
    """The classifier's request reaches the MY_MODELS endpoint of the run's model."""
    endpoint = _Endpoint()
    try:
        # The bundled name now runs against the stand-in endpoint.
        error = model_info.save_custom_model(
            STANDIN_MODEL, endpoint=endpoint.server.url, api_key="kiss-test-key",
        )
        assert error is None, error
        assert model_info.custom_model_config(STANDIN_MODEL) == {
            "base_url": endpoint.server.url, "api_key": "kiss-test-key",
        }
        printer = _CapturePrinter()
        server = VSCodeServer(printer=printer)
        server._cmd_run({
            "type": "run",
            "tabId": _TAB,
            "prompt": "classify me against my own endpoint",
            "workDir": str(home.repo),
            "model": STANDIN_MODEL,
            "useWorktree": False,
            "autoCommit": False,
            "useWebTools": False,
            "classifyTasks": True,
        })
        _wait_idle()
        # The real agent ran and finished with the stand-in's summary.
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1 and _SUMMARY in str(results[0].get("text")), results
        assert any(
            e.get("type") == "status" and e.get("running") is False
            for e in printer.events
        ), printer.events
        # Exactly two requests reached the custom endpoint, both for the
        # run's model: the classifier's structured-output call first,
        # then the agent's tool-loop call.
        assert [r.get("model") for r in endpoint.requests] == [STANDIN_MODEL] * 2, (
            endpoint.requests
        )
        classifier, run = endpoint.requests
        assert not classifier.get("tools") and classifier.get("response_format"), classifier
        assert run.get("tools") and not run.get("response_format"), run
        assert "classify me against my own endpoint" in json.dumps(run["messages"])
    finally:
        endpoint.server.stop()
