"""A settings-panel custom model with its own endpoint runs under any name.

``VSCodeServer._get_models`` lists every ``~/.kiss/MY_MODELS.json`` entry
that carries an endpoint, so a user can pick ``my-local-llama`` served
by a local OpenAI-compatible server.  ``_run_task_inner`` then admitted
the run only when the model was in ``get_available_models()`` — the
bundled catalog filtered by configured vendor keys — and refused the
pick with "No model available.  Set at least one API key in the
environment." although the run never needs a vendor key: the entry's
``base_url`` bypasses provider routing in ``model_info.model``.  Past
that gate the run still died at its first LLM call: ``MODEL_INFO`` is
built once at import, so the daemon had no pricing for the new name
(``calculate_cost`` raised) until a restart; ``_lookup_model_info`` now
falls back to the current ``MY_MODELS.json`` for names outside the
loaded catalog.

The run here goes through the real ``_cmd_run`` -> task runner ->
``WorktreeSorcarAgent`` chain against a local stand-in endpoint (no
mocks); the classifier is disabled so the one request is the agent's.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

_TAB = "tab-a1005-custom-model-name"
_MODEL = "audit1005-local-llama"
_WAIT_S = 90.0
_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"
_SUMMARY = "<p>audit1005 custom-named model finished</p>"


class _CapturePrinter(JsonPrinter):
    """Real printer recording every broadcast."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        super().broadcast(event)


class _Endpoint:
    """A stand-in endpoint recording the requests it receives."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        return finish_response(_SUMMARY)


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """Isolated ``KISS_HOME`` whose ``MY_MODELS.json`` is private, classifier off."""
    saved_disable = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "1"
    isolated = IsolatedKissHome("kiss-audit1005-custom-name-")
    isolated.write_config()
    saved_my_models = model_info.USER_MY_MODELS_PATH
    model_info.USER_MY_MODELS_PATH = isolated.kiss_home / "MY_MODELS.json"
    _clear_tab_states()
    try:
        yield isolated
    finally:
        _clear_tab_states()
        model_info.USER_MY_MODELS_PATH = saved_my_models
        if saved_disable is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved_disable
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


def test_custom_named_model_with_endpoint_runs(home: IsolatedKissHome) -> None:
    """A MY_MODELS entry under a non-catalog name is admitted and runs at its endpoint."""
    endpoint = _Endpoint()
    try:
        error = model_info.save_custom_model(
            _MODEL, endpoint=endpoint.server.url, api_key="kiss-test-key",
        )
        assert error is None, error
        # The running process prices the new name at once (zero-priced
        # registry entry) although the loaded catalog does not list it
        # and the vendor-key filter excludes it: the endpoint admits it.
        assert _MODEL not in model_info.MODEL_INFO
        assert model_info.calculate_cost(_MODEL, 1000, 1000) == 0.0
        assert _MODEL not in model_info.get_available_models()
        printer = _CapturePrinter()
        server = VSCodeServer(printer=printer)
        server._cmd_run({
            "type": "run",
            "tabId": _TAB,
            "prompt": "run me on my own endpoint",
            "workDir": str(home.repo),
            "model": _MODEL,
            "useWorktree": False,
            "autoCommit": False,
            "useWebTools": False,
            "classifyTasks": False,
        })
        _wait_idle()
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1, results
        assert "No model available" not in str(results[0].get("text")), results
        assert _SUMMARY in str(results[0].get("text")), results
        assert [r.get("model") for r in endpoint.requests] == [_MODEL], endpoint.requests
        assert "run me on my own endpoint" in json.dumps(endpoint.requests[0]["messages"])
    finally:
        endpoint.server.stop()
        model_info.delete_custom_model(_MODEL)
        with pytest.raises(KISSError):
            model_info.calculate_cost(_MODEL, 1, 1)


def test_unpriced_name_with_explicit_endpoint_is_still_refused(home: IsolatedKissHome) -> None:
    """An endpoint alone does not admit a name nothing can price.

    Without catalog or registry metadata the first LLM call would die in
    ``calculate_cost``; the gate refuses up front and the endpoint is
    never called.
    """
    endpoint = _Endpoint()
    try:
        printer = _CapturePrinter()
        server = VSCodeServer(printer=printer)
        server._cmd_run({
            "type": "run",
            "tabId": _TAB,
            "prompt": "never sent",
            "workDir": str(home.repo),
            "model": "audit1005-unregistered",
            "modelConfig": {"base_url": endpoint.server.url, "api_key": "k"},
            "useWorktree": False,
            "autoCommit": False,
            "useWebTools": False,
            "classifyTasks": False,
        })
        _wait_idle()
        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1, results
        assert results[0]["success"] is False
        assert "No model available" in str(results[0]["text"])
        assert endpoint.requests == []
    finally:
        endpoint.server.stop()
