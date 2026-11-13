# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: GeminiModel must invoke thinking_callback.

Gemini models with thinking enabled return parts where ``part.thought``
is ``True`` for thinking content.  The ``thinking_callback`` must be
invoked with ``True`` at the start of a thinking block and ``False`` at
the end so that the browser UI routes thinking tokens to the thinking
panel rather than the main text area.

Bug reproduction: without the fix, ``_stream_parts()`` calls
``_invoke_token_callback()`` for all parts without checking
``part.thought``, so thinking tokens are broadcast as ``text_delta``
events — thoughts appear outside the thinking panel.

Uses real ``google.genai.types.Part`` objects — no mocks, patches, or
fakes.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from google.genai import errors, types

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import OutputCapError
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import MODEL_INFO
from kiss.scripts import update_models
from kiss.tests.core.models.gemini_sse_harness import GeminiScript, chunk, serve, text_part


class TestGeminiStreamPartsThinkingCallback:
    """Verify _stream_parts invokes thinking_callback for thought parts."""

    def test_thinking_callback_fires_for_thought_parts(self) -> None:
        """thinking_callback must receive True then False around thinking parts."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )

        thinking_part = types.Part(text="Let me think about this.", thought=True)
        text_part = types.Part(text="The answer is 42.")

        m._stream_parts([thinking_part, text_part])

        assert True in thinking_events, (
            "thinking_callback(True) was never called — "
            "thinking tokens leak as text_delta events"
        )
        assert False in thinking_events, (
            "thinking_callback(False) was never called — "
            "thinking panel will never close"
        )
        first_true = thinking_events.index(True)
        last_false = len(thinking_events) - 1 - thinking_events[::-1].index(False)
        assert first_true < last_false

        combined = "".join(tokens)
        assert "Let me think" in combined
        assert "The answer is 42." in combined

    def test_thinking_across_multiple_chunks(self) -> None:
        """Thinking state must carry across multiple _stream_parts calls."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Thinking chunk 1", thought=True)])
        m._stream_parts([types.Part(text=" and chunk 2", thought=True)])
        m._stream_parts([types.Part(text="Final answer.")])

        assert thinking_events[0] is True
        assert False in thinking_events

        combined = "".join(tokens)
        assert "Thinking chunk 1" in combined
        assert "and chunk 2" in combined
        assert "Final answer." in combined

    def test_no_thinking_callback_when_no_thought_parts(self) -> None:
        """thinking_callback must NOT fire when there are no thought parts."""
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Just regular text.")])

        assert thinking_events == [], (
            f"thinking_callback fired unexpectedly: {thinking_events}"
        )

    def test_close_thinking_if_open_closes_open_block(self) -> None:
        """_close_thinking_if_open must close an open thinking block.

        After a streaming loop ends, if the last chunk was a thinking part,
        the thinking block must still be closed.  GeminiModel used to track
        this on a private ``_in_thinking_stream`` duplicate of the base
        ``_thinking_open`` flag; it now uses the base one, so the closing
        helper is the base ``_close_thinking_if_open``.
        """
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Only thinking.", thought=True)])
        m._close_thinking_if_open()

        assert thinking_events == [True, False], (
            f"Expected [True, False] but got {thinking_events}"
        )

    def test_close_thinking_if_open_noop_when_not_thinking(self) -> None:
        """_close_thinking_if_open must be a no-op when no block is open."""
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Regular text.")])
        m._close_thinking_if_open()

        assert thinking_events == [], (
            f"thinking_callback fired unexpectedly: {thinking_events}"
        )


@pytest.fixture
def gemini_endpoint(monkeypatch: pytest.MonkeyPatch) -> Generator[GeminiScript]:
    """A real local Gemini endpoint that streams one thought and one answer."""
    for base_url, script in serve():
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
        script.play(
            [
                chunk([text_part("Weighing the options.", thought=True)]),
                chunk([text_part("Hello.")], finish_reason="STOP"),
            ]
        )
        yield script


def _sent_thinking_config(
    script: GeminiScript, model_name: str, model_config: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str], list[bool]]:
    """Run one streamed turn of *model_name*; return the wire thinkingConfig, tokens, events."""
    tokens: list[str] = []
    events: list[bool] = []
    m = GeminiModel(
        model_name,
        api_key="test-key",
        model_config=model_config,
        token_callback=tokens.append,
        thinking_callback=events.append,
    )
    m.initialize("Say hello.")
    m.generate()
    return script.requests[-1]["generationConfig"]["thinkingConfig"], tokens, events


class TestGeminiThinkingLevel:
    """The thinking level reaches the request with thought summaries kept on."""

    def test_high_alias_sends_base_model_at_high_with_thoughts(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        sent, tokens, events = _sent_thinking_config(gemini_endpoint, "gemini-4-argon-high")
        assert sent == {"include_thoughts": True, "thinking_level": "HIGH"}
        assert "/models/gemini-4-argon:streamGenerateContent" in gemini_endpoint.paths[-1]
        assert tokens == ["Weighing the options.", "Hello."]
        assert events == [True, False]

    def test_model_without_catalog_level_only_asks_for_thoughts(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        sent, _, _ = _sent_thinking_config(gemini_endpoint, "gemini-2.5-flash")
        assert sent == {"include_thoughts": True}

    def test_configured_level_keeps_thoughts_on(self, gemini_endpoint: GeminiScript) -> None:
        sent, _, _ = _sent_thinking_config(
            gemini_endpoint, "gemini-4-argon", {"thinking_config": {"thinking_level": "medium"}}
        )
        assert sent == {"include_thoughts": True, "thinking_level": "MEDIUM"}
        sent, _, _ = _sent_thinking_config(
            gemini_endpoint,
            "gemini-4-argon",
            {"thinking_config": types.ThinkingConfig(include_thoughts=False)},
        )
        assert sent == {"include_thoughts": False, "thinking_level": "HIGH"}

    def test_reasoning_effort_overrides_level_and_budget(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        sent, _, _ = _sent_thinking_config(
            gemini_endpoint,
            "gemini-4-argon-high",
            {"reasoning_effort": "low", "thinking_config": {"thinking_budget": 512}},
        )
        assert sent == {"include_thoughts": True, "thinking_level": "LOW"}
        for effort in ("xhigh", "max"):
            sent, _, _ = _sent_thinking_config(
                gemini_endpoint, "gemini-4-argon", {"reasoning_effort": effort}
            )
            assert sent == {"include_thoughts": True, "thinking_level": "HIGH"}

    @pytest.mark.parametrize(
        ("thinking_config", "retries"),
        [
            ({"thinking_level": "low"}, False),
            (types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW), False),
            ({"thinking_budget": 512}, True),
            (None, True),
        ],
    )
    def test_low_effort_retry_only_when_it_lowers_the_level(
        self, thinking_config: Any, retries: bool
    ) -> None:
        agent = KISSAgent("gemini-retry-under-test")
        agent.printer = None
        agent._low_effort_retried = False
        agent.model = GeminiModel(
            "gemini-4-argon", api_key="test-key", model_config={"thinking_config": thinking_config}
        )
        assert agent._retry_at_low_effort(OutputCapError("cut", "MAX_TOKENS")) is retries

    def test_configured_budget_is_not_mixed_with_catalog_level(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        sent, _, _ = _sent_thinking_config(
            gemini_endpoint, "gemini-4-argon-high", {"thinking_config": {"thinking_budget": 512}}
        )
        assert sent == {"include_thoughts": True, "thinking_budget": 512}

    def test_model_without_thinking_is_asked_again_without_a_thinking_config(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        """Gemini refuses any ``thinkingConfig``, the default ``include_thoughts``
        included, for a model that cannot think (``gemini-2.5-flash-image``):
        the turn is asked again without one, and so is every later turn."""
        gemini_endpoint.refuse_thinking = "Thinking is not enabled for this model"
        gemini_endpoint.play([chunk([text_part("Hello.")], finish_reason="STOP")])
        m = GeminiModel("gemini-2.5-flash-image", api_key="test-key")
        for prompt in ("Say hello.", "Say hello again."):
            m.initialize(prompt)
            assert m.generate()[0] == "Hello."
        sent = ["thinkingConfig" in r.get("generationConfig", {}) for r in gemini_endpoint.requests]
        assert sent == [True, False, False]

    def test_another_refusal_of_the_thinking_config_is_not_retried(
        self, gemini_endpoint: GeminiScript
    ) -> None:
        gemini_endpoint.refuse_thinking = "Thinking level HIGH is not supported"
        m = GeminiModel("gemini-4-argon-high", api_key="test-key")
        m.initialize("Say hello.")
        with pytest.raises(errors.ClientError, match="is not supported"):
            m.generate()
        assert len(gemini_endpoint.requests) == 1

    def test_catalog_ships_gemini_level_aliases(self) -> None:
        for level in ("low", "medium", "high"):
            info = MODEL_INFO[f"gemini-4-argon-{level}"]
            assert (info.alias_of, info.thinking) == ("gemini-4-argon", level)
        assert MODEL_INFO["gemini-4-argon"].thinking == "high"
        assert MODEL_INFO["gemini-2.5-flash"].thinking is None

    def test_update_models_probes_gemini_on_its_own_ladder(
        self, gemini_endpoint: GeminiScript, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DEFAULT_CONFIG, "GEMINI_API_KEY", "test-key")
        assert update_models._thinking_scale_for("gemini-4-argon") == ("low", "medium", "high")
        assert update_models.detect_thinking_level("gemini-4-argon") == "high"
        sent = gemini_endpoint.requests[-1]["generationConfig"]["thinkingConfig"]
        assert sent == {"include_thoughts": True, "thinking_level": "HIGH"}
