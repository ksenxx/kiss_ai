# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a dropped Responses stream must close its thinking bracket.

``OpenAICompatibleModel2._consume_stream_events`` opens the thinking
bracket on the first ``response.reasoning_summary_text.delta`` and, like
the Chat Completions transport before its fix, closed it only when the
loop ran to completion.  ``stop_aware_events`` closes the bracket for a
stop and for a stall, but re-raises every other transport failure
untouched, so a provider that dropped the connection mid-reasoning left
``Model._thinking_open`` set and the printer in thinking mode for the
retry ``KISSAgent._run_agentic_loop`` takes in the same run.

No mocks or patches: a real ``ThreadingHTTPServer`` speaks genuine
Responses-API SSE to the real ``openai`` SDK and the real adapter.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "gpt-responses-bracket-under-test"
_STALL_TIMEOUT = 5.0
_DEADLINE = 30.0

_REASONING_THEN_CUT = [
    responses_event(
        "response.reasoning_summary_text.delta",
        {
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "Let me think",
            "sequence_number": 1,
        },
    ),
    responses_event(
        "response.output_text.delta",
        {
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "delta": "never arrives",
            "sequence_number": 2,
        },
    ),
]


def _cut_mid_reasoning(_request: Request) -> Reply:
    """Send one reasoning delta, then drop the connection."""
    return Reply(sse_chunks=_REASONING_THEN_CUT, truncate_after=1)


@pytest.fixture
def cut_server() -> Generator[ScriptedOpenAIServer]:
    """A real endpoint that crashes every stream mid-reasoning."""
    with ScriptedOpenAIServer(_cut_mid_reasoning) as server:
        yield server


def _run_bounded(call: Any) -> BaseException | None:
    """Run *call* on a daemon thread bounded by :data:`_DEADLINE`."""
    outcome: list[BaseException | None] = []

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(f"turn still running after {_DEADLINE}s")
    return outcome[0]


def test_dropped_responses_stream_closes_thinking_bracket(
    cut_server: ScriptedOpenAIServer,
) -> None:
    """The bracket opened by the reasoning delta is closed before the failure propagates."""
    thinking: list[bool] = []
    model = OpenAICompatibleModel2(
        _MODEL,
        base_url=cut_server.base_url,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=lambda _t: None,
        thinking_callback=thinking.append,
    )
    model.initialize("Think, then answer.")

    error = _run_bounded(model.generate)

    assert error is not None, "the cut stream should have failed the turn"
    assert not isinstance(error, (KeyboardInterrupt, TimeoutError)), error
    assert thinking == [True, False], (
        f"the thinking bracket was left unbalanced: {thinking}"
    )
    assert model._thinking_open is False
