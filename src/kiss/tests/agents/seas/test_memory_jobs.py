# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test of the model calls the memory jobs make (``kiss/agents/seas/memory_jobs.py``).

The option parsing (a scheduled run's context included), the refused and
unusable replies, the budget reserve and the charge to the task are
exercised through the two jobs
(``test_memory_consolidate_sea.py``, ``test_memory_quiz_sea.py``) on the
OpenAI-compatible local server.  What that server cannot show is here: a
provider that hands a tool-less caller the text of a reply its output
limit cut, served by the real ``google-genai`` SDK from the local Gemini
endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from kiss.agents.seas.memory_jobs import ModelCalls
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core import config as config_module
from kiss.tests.core.models import gemini_sse_harness
from kiss.tests.core.models.gemini_sse_harness import GeminiScript, chunk, text_part

GEMINI = "gemini-2.5-flash"


@pytest.fixture
def gemini_endpoint() -> Iterator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from gemini_sse_harness.serve()


def test_a_reply_cut_at_the_output_limit_is_no_reply(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    """A page cut in the middle must never be taken for the merged page (or half a list)."""
    url, script = gemini_endpoint
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", url)
    usage = {"promptTokenCount": 100, "candidatesTokenCount": 7}
    caller = ChatSorcarAgent("memory-jobs-test")
    caller.model_name = GEMINI
    caller.max_budget = 1.0
    calls = ModelCalls(caller, "Run the job.")
    assert calls.words == []  # an agent that runs no task has no options

    script.play([chunk([text_part("TITLE: Half a pa")], usage=usage, finish_reason="MAX_TOKENS")])
    assert calls.ask(GEMINI, "Merge the pages.") == ""
    script.play([chunk([text_part("SEPARATE")], usage=usage, finish_reason="STOP")])
    assert calls.ask(GEMINI, "Merge the pages.") == "SEPARATE"
    # Both calls were paid for and are counted.
    assert calls.calls == 2 and calls.cost > 0
    assert calls.spend().startswith(f"Model calls: 2, ${calls.cost:.2f}; written by {GEMINI}, ")
