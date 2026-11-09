# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the adaptive effort verdict is per (endpoint, model).

Audit findings 02/I3 and 02/C1.  Whether an endpoint accepts ``tools``
together with ``reasoning_effort`` was learned at runtime and cached in a
process-global dict keyed by ``base_url`` **alone**, mutated with no
synchronization from a check-then-act that straddles an HTTP round trip.

* **I3** — the capability is a property of the *(endpoint, model)* pair:
  a reasoning model rejects the combination while another model on the
  same gateway accepts it.  With one shared key, whichever model was
  probed first decided for all of them.  Cached ``False`` silently
  stripped ``reasoning_effort`` from every other model (a quality
  regression with no error and no log above ``debug``); cached ``True``
  made the rejecting model skip its recovery retry, so the 400
  propagated and the turn hard-failed.
* **C1** — parallel subagents run in real threads in one process, so two
  probes overlap and the surviving verdict was decided by which HTTP
  response happened to arrive last.

No mocks: one real ``ThreadingHTTPServer`` serves both models, rejecting
the combination for one of them exactly the way a provider does, and a
``threading.Barrier`` inside the handler forces the two probes to be in
flight simultaneously.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from types import FrameType
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.openai_compatible_model import (
    _ADAPTIVE_TOOL_EFFORT_VERDICTS,
    OpenAICompatibleModel,
    _rejects_effort_value,
)
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)

_REJECTING_MODEL = "reject-me"
_ACCEPTING_MODEL = "accept-me"
_DEADLINE = 30.0

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
    }
]


def _ok_body(model_name: str) -> dict[str, Any]:
    """Return a minimal successful completion body.

    Args:
        model_name: The model id to echo back.

    Returns:
        A Chat Completions JSON body.
    """
    return {
        "id": "chatcmpl-ok",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
    }


def _rejection_body() -> dict[str, Any]:
    """Return the 400 body a provider sends for the bad combination."""
    return {
        "error": {
            "message": (
                "reasoning_effort is not supported with tools for this model"
            ),
            "type": "invalid_request_error",
        }
    }


class _PerModelPolicy:
    """Rejects ``tools`` + ``reasoning_effort`` for one model only."""

    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        """Set up the policy.

        Args:
            barrier: When given, every probing request waits on it before
                being answered, so the concurrent probes are guaranteed
                to be in flight at the same time.
        """
        self.barrier = barrier
        self.reject_delay = 0.0
        self.probes = 0
        self.lock = threading.Lock()

    def __call__(self, request: Request) -> Reply:
        """Answer according to the requested model and parameters."""
        model_name = str(request.body.get("model", ""))
        has_effort = "reasoning_effort" in request.body
        has_tools = bool(request.body.get("tools"))
        if has_effort and has_tools:
            with self.lock:
                self.probes += 1
                first_round = self.probes <= 2
            # Only the two racing probes rendezvous; the follow-up turn
            # afterwards must not block on a barrier nobody else reaches.
            if self.barrier is not None and first_round:
                self.barrier.wait(timeout=_DEADLINE)
        if model_name == _REJECTING_MODEL and has_effort and has_tools:
            if self.reject_delay:
                threading.Event().wait(self.reject_delay)
            return Reply(status=400, json_body=_rejection_body())
        return Reply(json_body=_ok_body(model_name))


def _make_model(base_url: str, model_name: str) -> OpenAICompatibleModel:
    """Build a non-streaming model that sends ``reasoning_effort``.

    Args:
        base_url: The scripted server's ``/v1`` root.
        model_name: The model id to send.

    Returns:
        An initialized model.
    """
    model = OpenAICompatibleModel(
        model_name,
        base_url=base_url,
        api_key="test-key",
        model_config={"reasoning_effort": "high"},
    )
    model.initialize("Do the thing.")
    return model


def _run(model: OpenAICompatibleModel) -> None:
    """Run one tool-bearing turn.

    Args:
        model: The model under test.
    """
    model.generate_and_process_with_tools({}, tools_schema=_TOOLS)


def _efforts_by_model(server: ScriptedOpenAIServer) -> list[tuple[str, bool]]:
    """Return ``(model, carried_reasoning_effort)`` for each request.

    Args:
        server: The scripted server.

    Returns:
        One entry per received request, in arrival order.
    """
    return [
        (str(r.body.get("model", "")), "reasoning_effort" in r.body)
        for r in server.requests
    ]


@pytest.fixture
def clean_verdicts() -> Generator[None]:
    """Run with an empty verdict cache and leave it empty afterwards."""
    before = dict(_ADAPTIVE_TOOL_EFFORT_VERDICTS)
    _ADAPTIVE_TOOL_EFFORT_VERDICTS.clear()
    yield
    _ADAPTIVE_TOOL_EFFORT_VERDICTS.clear()
    _ADAPTIVE_TOOL_EFFORT_VERDICTS.update(before)


@pytest.mark.usefixtures("clean_verdicts")
class TestVerdictIsPerModel:
    """One model's verdict must not decide for another on the same host."""

    def test_rejecting_model_first(self) -> None:
        """After a rejection, another model must keep its effort."""
        with ScriptedOpenAIServer(_PerModelPolicy()) as server:
            _run(_make_model(server.base_url, _REJECTING_MODEL))
            _run(_make_model(server.base_url, _ACCEPTING_MODEL))
            observed = _efforts_by_model(server)
        assert observed == [
            (_REJECTING_MODEL, True),
            (_REJECTING_MODEL, False),
            (_ACCEPTING_MODEL, True),
        ], observed

    def test_accepting_model_first(self) -> None:
        """After an acceptance, the other model must still recover."""
        with ScriptedOpenAIServer(_PerModelPolicy()) as server:
            _run(_make_model(server.base_url, _ACCEPTING_MODEL))
            _run(_make_model(server.base_url, _REJECTING_MODEL))
            observed = _efforts_by_model(server)
        assert observed == [
            (_ACCEPTING_MODEL, True),
            (_REJECTING_MODEL, True),
            (_REJECTING_MODEL, False),
        ], observed

    def test_verdicts_are_keyed_by_endpoint_and_model(self) -> None:
        """Both models must end up with their own cached verdict."""
        with ScriptedOpenAIServer(_PerModelPolicy()) as server:
            _run(_make_model(server.base_url, _REJECTING_MODEL))
            _run(_make_model(server.base_url, _ACCEPTING_MODEL))
            base_url = server.base_url
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS[(base_url, _REJECTING_MODEL)] is False
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS[(base_url, _ACCEPTING_MODEL)] is True


@pytest.mark.usefixtures("clean_verdicts")
class TestConcurrentProbesDoNotClobber:
    """Two subagents probing at once must not overwrite each other."""

    def test_parallel_probes_keep_their_own_verdicts(self) -> None:
        """The exact interleaving from the audit, forced with a barrier.

        Both probes are held in the handler until both have arrived, and
        the rejection is then answered last so that its cache write is
        the one that lands after the success.  With a single ``base_url``
        key that made the endpoint look incapable, and the accepting
        model silently lost ``reasoning_effort`` from then on.
        """
        policy = _PerModelPolicy(barrier=threading.Barrier(2))
        policy.reject_delay = 0.4
        errors: list[BaseException] = []

        with ScriptedOpenAIServer(policy) as server:

            def probe(model_name: str) -> None:
                """Run one probing turn, recording any failure."""
                try:
                    _run(_make_model(server.base_url, model_name))
                except BaseException as exc:  # noqa: BLE001 — reported below
                    errors.append(exc)

            threads = [
                threading.Thread(target=probe, args=(name,), daemon=True)
                for name in (_REJECTING_MODEL, _ACCEPTING_MODEL)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=_DEADLINE)
                assert not thread.is_alive()
            assert not errors, errors

            # The follow-up turn is what the audit says goes wrong: the
            # accepting model must still be sending its effort.
            before = len(server.requests)
            _run(_make_model(server.base_url, _ACCEPTING_MODEL))
            follow_up = server.requests[before:]

        assert [
            ("reasoning_effort" in r.body) for r in follow_up
        ] == [True], json.dumps([r.body for r in follow_up])


def _unsupported_value_policy(request: Request) -> Reply:
    """Reject the ``xhigh`` effort level as an invalid value; accept every other request."""
    if request.body.get("reasoning_effort") == "xhigh":
        error = {
            "error": {
                "message": (
                    "Invalid value: 'xhigh'. Supported values are: 'low', 'medium', and 'high'."
                ),
                "type": "invalid_request_error",
                "param": "reasoning_effort",
                "code": "invalid_value",
            }
        }
        return Reply(status=400, json_body=error)
    return Reply(json_body=_ok_body(str(request.body.get("model", ""))))


@pytest.mark.usefixtures("clean_verdicts")
def test_an_unsupported_effort_value_is_no_verdict_on_the_combination() -> None:
    """A 400 for the effort's *value* must not disable every effort value of the pair.

    The endpoint accepts ``tools`` + ``reasoning_effort`` and merely
    refuses ``xhigh``: the turn still recovers (retried without the
    effort, the value named in the log), but the endpoint/model pair is
    left unprobed, so a later model asking for ``high`` sends it.
    """
    with ScriptedOpenAIServer(_unsupported_value_policy) as server:
        too_high = OpenAICompatibleModel(
            _ACCEPTING_MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config={"reasoning_effort": "xhigh"},
        )
        too_high.initialize("Do the thing.")
        _run(too_high)
        assert (server.base_url, _ACCEPTING_MODEL) not in _ADAPTIVE_TOOL_EFFORT_VERDICTS
        _run(_make_model(server.base_url, _ACCEPTING_MODEL))
        efforts = [r.body.get("reasoning_effort") for r in server.requests]
    assert efforts == ["xhigh", None, "high"], efforts


@pytest.mark.usefixtures("clean_verdicts")
def test_a_stopped_turn_records_its_verdict_under_the_endpoint_it_was_sent_to() -> None:
    """A rejection handled after the model moved on is a verdict on the old endpoint.

    The callback-less turn runs on a worker thread.  It is held (a
    profile hook on the worker) right after the old endpoint's rejection
    arrived; the user stops the turn, the next turn moves the model to a
    new endpoint that accepts the combination, and only then does the
    held worker record its verdict.  Read from the model at that moment,
    the key would be the new endpoint's, and its ``True`` would be
    overwritten by a ``False`` it never earned (A04-1a / A04-1f).
    """
    held = threading.Event()
    resume = threading.Event()

    def hold_the_verdict(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code is _rejects_effort_value.__code__:
            held.set()
            resume.wait(_DEADLINE)

    def accept_everything(request: Request) -> Reply:
        return Reply(json_body=_ok_body(str(request.body.get("model", ""))))

    stop = threading.Event()
    outcome: dict[str, BaseException] = {}

    with (
        ScriptedOpenAIServer(_PerModelPolicy()) as old,
        ScriptedOpenAIServer(accept_everything) as new,
    ):
        model = _make_model(old.base_url, _REJECTING_MODEL)

        def stopped_turn() -> None:
            stop_signal.set_thread_stop_event(stop)
            try:
                _run(model)
            except BaseException as exc:  # noqa: BLE001 - reported to the test
                outcome["error"] = exc

        threading.setprofile(hold_the_verdict)
        try:
            turn = threading.Thread(target=stopped_turn, daemon=True)
            turn.start()
            assert held.wait(_DEADLINE), "the old endpoint's rejection never reached the worker"
            stop.set()
            turn.join(_DEADLINE)
        finally:
            threading.setprofile(None)
        assert not turn.is_alive(), "the Stop did not end the turn"
        assert isinstance(outcome.get("error"), KeyboardInterrupt), f"got {outcome!r}"

        model.base_url = new.base_url
        _run(model)
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS.get((new.base_url, _REJECTING_MODEL)) is True

        resume.set()
        for worker in threading.enumerate():
            if worker.name == "model-blocking-request":
                worker.join(_DEADLINE)
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS.get((old.base_url, _REJECTING_MODEL)) is False
        assert _ADAPTIVE_TOOL_EFFORT_VERDICTS.get((new.base_url, _REJECTING_MODEL)) is True
        assert _efforts_by_model(new) == [(_REJECTING_MODEL, True)]
