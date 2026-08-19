# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests (real LLM calls, no mocks) for context-window handling.

The cheap-reproduction trick: temporarily shrink the *registered* context
length of a real, inexpensive model (``gpt-4o-mini``) in
``kiss.core.models.model_info.MODEL_INFO``.  All API calls remain real
(the provider still enforces its true 128K limit); only KISS's own
bookkeeping sees the small value, so the agent's context-limit logic
triggers after a couple of genuine calls.

The RelentlessAgent recovery half (summarize-and-continue after a
mid-session overflow) lives in
``kiss.tests.agents.sorcar.test_context_window_e2e``, which reuses
``ShrunkContextMixin`` and ``_big_text`` from this file.
"""

import unittest

import pytest

from kiss.core.kiss_agent import (
    CONTEXT_LIMIT_FRACTION,
    KISSAgent,
    _is_context_overflow_error,
)
from kiss.core.kiss_error import ContextWindowExceededError, KISSError
from kiss.core.models import model_info
from kiss.tests.conftest import requires_openai_api_key

TEST_MODEL = "gpt-4o-mini"



class TestOverflowPhraseDetection(unittest.TestCase):
    """Pure tests for provider context-overflow error classification."""

    def test_detects_all_known_provider_phrasings(self) -> None:
        """Every known provider overflow phrasing is detected."""
        provider_errors = [
            "Your input exceeds the context window of this model. "
            "Please adjust your input and try again.",
            "prompt is too long: 213462 tokens > 200000 maximum",
            "Error code: 400 - context_length_exceeded",
            "This model's maximum context length is 128000 tokens.",
            "The input token count (1200000) exceeds the maximum number "
            "of tokens allowed (1048576).",
        ]
        for msg in provider_errors:
            with self.subTest(msg=msg):
                self.assertTrue(_is_context_overflow_error(ValueError(msg)))

    def test_ignores_unrelated_errors(self) -> None:
        """Rate limits, auth errors, etc. are not classified as overflow."""
        for msg in [
            "rate limit exceeded",
            "invalid api key",
            "connection reset",
            "the context window feature is unavailable",
        ]:
            with self.subTest(msg=msg):
                self.assertFalse(_is_context_overflow_error(ValueError(msg)))

    def test_error_type_is_kiss_error(self) -> None:
        """ContextWindowExceededError is a KISSError (relentless-routable)."""
        err = ContextWindowExceededError("too big")
        self.assertIsInstance(err, KISSError)
        self.assertIn("too big", str(err))



class ShrunkContextMixin(unittest.TestCase):
    """Shrinks TEST_MODEL's registered context length for the duration of a test.

    This mutates real configuration (not a mock): every LLM call still goes
    to the real provider, which enforces its true 128K limit.  Only
    ``get_max_context_length`` sees the small value.
    """

    small_context = 6000

    def setUp(self) -> None:
        self._orig_context = model_info.MODEL_INFO[TEST_MODEL].context_length
        model_info.MODEL_INFO[TEST_MODEL].context_length = self.small_context

    def tearDown(self) -> None:
        model_info.MODEL_INFO[TEST_MODEL].context_length = self._orig_context



def _big_text(n_words: int) -> str:
    """Return ``n_words`` words of filler text (~1.3 tokens per word)."""
    return "context filler words for window overflow testing " * (n_words // 8)



@requires_openai_api_key
class TestProactiveContextStop(ShrunkContextMixin):
    """Reproduces defects D1/D2: the agent must stop BEFORE the provider fails."""

    @pytest.mark.slow
    def test_agent_stops_proactively_and_reports_unwrapped_usage(self) -> None:
        """A conversation nearing the (shrunken) context limit raises
        ContextWindowExceededError proactively, and the usage string shows the
        true context size instead of the old modulo-wrapped value."""
        agent = KISSAgent("Proactive-Ctx-Test")

        def note(text: str) -> str:
            """Record a note.

            Args:
                text: The note text.

            Returns:
                Acknowledgement string.
            """
            return f"noted: {text}"

        filler = _big_text(8000)
        with pytest.raises(ContextWindowExceededError) as exc_info:
            agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "Reference material:\n{filler}\n\n"
                    "You MUST call the tool note('step one') first. Only after "
                    "note returns may you call finish. Never call finish first."
                ),
                arguments={"filler": filler},
                tools=[note],
                max_steps=10,
                max_budget=1.0,
                verbose=False,
            )
        self.assertIsNone(exc_info.value.__cause__)
        threshold = CONTEXT_LIMIT_FRACTION * self.small_context
        self.assertGreaterEqual(agent.context_tokens_used, threshold)
        usage = agent._get_usage_info_string()
        self.assertIn(f"Context: {agent.context_tokens_used:,}/6,000", usage)
        self.assertIn(f"Total tokens: {agent.total_tokens_used:,}", usage)
        wrapped = agent.context_tokens_used % self.small_context
        self.assertNotIn(f"Context: {wrapped:,}/6,000", usage)



@requires_openai_api_key
class TestProviderOverflowConversion(unittest.TestCase):
    """Reproduces defect D3: real provider overflow must fail fast and typed."""

    @pytest.mark.slow
    def test_provider_rejection_raises_typed_error_without_retries(self) -> None:
        """A prompt genuinely exceeding gpt-4o-mini's 128K window is rejected by
        the real provider; the agent must raise ContextWindowExceededError on the
        FIRST failure instead of retrying 3 times while growing the conversation."""
        agent = KISSAgent("Provider-Overflow-Test")
        filler = _big_text(200_000)
        with pytest.raises(ContextWindowExceededError) as exc_info:
            agent.run(
                model_name=TEST_MODEL,
                prompt_template="Summarize this:\n{filler}",
                arguments={"filler": filler},
                tools=[],
                max_steps=5,
                max_budget=1.0,
                verbose=False,
            )
        self.assertIsNotNone(exc_info.value.__cause__)
        retry_messages = [
            m for m in agent.messages if "Please try again" in str(m.get("content", ""))
        ]
        self.assertEqual(retry_messages, [])
        self.assertEqual(agent.step_count, 1)
