# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests (real LLM calls, no mocks) for context-window handling.

Reproduces the production failure of task ``9c0bcb00997a4f3fa6715e46a62c66f4``
(2026-07-15), whose RelentlessAgent run repeatedly exhausted the model
context window and finally hard-failed with::

    KISS Error: Agent ... failed with 3 consecutive errors. Last error:
    Your input exceeds the context window of this model.

The cheap-reproduction trick: temporarily shrink the *registered* context
length of a real, inexpensive model (``gpt-4o-mini``) in
``kiss.core.models.model_info.MODEL_INFO``.  All API calls remain real
(the provider still enforces its true 128K limit); only KISS's own
bookkeeping sees the small value, so the agent's context-limit logic
triggers after a couple of genuine calls.
"""

import tempfile
import unittest

import pytest
import yaml

from kiss.core.kiss_agent import (
    CONTEXT_LIMIT_FRACTION,
    KISSAgent,
    _is_context_overflow_error,
)
from kiss.core.kiss_error import ContextWindowExceededError, KISSError
from kiss.core.models import model_info
from kiss.core.relentless_agent import MAX_PROGRESS_CHARS, RelentlessAgent, _capped_progress_text
from kiss.tests.conftest import requires_openai_api_key

TEST_MODEL = "gpt-4o-mini"


class TestOverflowPhraseDetection(unittest.TestCase):
    """Pure tests for provider context-overflow error classification."""

    def test_detects_all_known_provider_phrasings(self) -> None:
        """Every known provider overflow phrasing is detected."""
        provider_errors = [
            # Anthropic (exact production error from task 9c0bcb00...)
            "Your input exceeds the context window of this model. "
            "Please adjust your input and try again.",
            # Anthropic alternative
            "prompt is too long: 213462 tokens > 200000 maximum",
            # OpenAI error code
            "Error code: 400 - context_length_exceeded",
            # OpenAI message
            "This model's maximum context length is 128000 tokens.",
            # Gemini
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
            # Merely mentioning the words must not be misrouted to the
            # context-overflow recovery path.
            "the context window feature is unavailable",
        ]:
            with self.subTest(msg=msg):
                self.assertFalse(_is_context_overflow_error(ValueError(msg)))

    def test_error_type_is_kiss_error(self) -> None:
        """ContextWindowExceededError is a KISSError (relentless-routable)."""
        err = ContextWindowExceededError("too big")
        self.assertIsInstance(err, KISSError)
        self.assertIn("too big", str(err))


class TestCappedProgressText(unittest.TestCase):
    """Pure tests for the continuation-prompt growth cap (defect D5)."""

    def test_many_tiny_summaries_stay_within_hard_cap(self) -> None:
        """Separator overhead is counted: thousands of tiny summaries can't
        push the joined text past the cap."""
        text = _capped_progress_text(["z"] * 4000)
        self.assertLessEqual(len(text), MAX_PROGRESS_CHARS)
        self.assertIn("earlier attempt summaries omitted", text)
        self.assertIn("### Attempt 4000\nz", text)

    def test_all_summaries_kept_when_under_cap(self) -> None:
        """Small summaries are all kept, oldest first, with no omission note."""
        text = _capped_progress_text(["did A", "did B", "did C"])
        self.assertIn("### Attempt 1\ndid A", text)
        self.assertIn("### Attempt 2\ndid B", text)
        self.assertIn("### Attempt 3\ndid C", text)
        self.assertNotIn("omitted", text)
        self.assertLess(text.index("Attempt 1"), text.index("Attempt 3"))

    def test_oldest_summaries_dropped_when_over_cap(self) -> None:
        """Old summaries are dropped first and an omission note is prepended."""
        big = "x" * (MAX_PROGRESS_CHARS // 2)
        text = _capped_progress_text([big, big, big, "latest work"])
        self.assertIn("### Attempt 4\nlatest work", text)
        self.assertIn("### Attempt 3", text)
        self.assertNotIn("### Attempt 1\n", text)
        self.assertIn("2 earlier attempt summaries omitted", text)
        self.assertLessEqual(len(text), MAX_PROGRESS_CHARS, "cap must be hard")

    def test_single_oversized_summary_is_kept_but_truncated(self) -> None:
        """The newest summary is always kept, but hard-truncated to the cap."""
        huge = "y" * (MAX_PROGRESS_CHARS * 2)
        text = _capped_progress_text([huge])
        self.assertIn("### Attempt 1", text)
        self.assertIn("(...summary truncated.)", text)
        self.assertNotIn("omitted", text)
        self.assertLessEqual(len(text), MAX_PROGRESS_CHARS)


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

        # ~8000 words ≈ 10K tokens of prompt >> 0.9 * 6000 threshold, but
        # far below the provider's real 128K limit, so the first call
        # SUCCEEDS and only KISS's proactive check fires at the next step.
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
        # Proactive stop: raised by _check_limits, not chained from a
        # provider rejection.
        self.assertIsNone(exc_info.value.__cause__)
        threshold = CONTEXT_LIMIT_FRACTION * self.small_context
        self.assertGreaterEqual(agent.context_tokens_used, threshold)
        # D1 regression: the usage string must show the TRUE context size
        # (which exceeds the registered max) — never wrapped modulo max.
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
        # ~200K words ≈ 250K tokens — genuinely above the 128K window.
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
        # Reactive path: the provider error is chained as the cause.
        self.assertIsNotNone(exc_info.value.__cause__)
        # Fast failure: no "Please try again" retry messages were appended
        # (the old code retried 3 times, growing the conversation each time).
        retry_messages = [
            m for m in agent.messages if "Please try again" in str(m.get("content", ""))
        ]
        self.assertEqual(retry_messages, [])
        self.assertEqual(agent.step_count, 1)


@requires_openai_api_key
class TestRelentlessRecovery(ShrunkContextMixin):
    """Reproduces defect D4 end-to-end: RelentlessAgent must recover from a
    mid-session context overflow via summarize-and-continue instead of
    hard-failing the whole task (as production task 9c0bcb00... did)."""

    @pytest.mark.slow
    def test_relentless_continues_after_context_overflow(self) -> None:
        """Session 0 overflows the (shrunken) context mid-run; the trajectory
        summarizer produces progress and session 1 finishes the task."""
        agent = RelentlessAgent("Relentless-Ctx-Recovery")

        def load_dataset(part: int) -> str:
            """Load one part of the dataset.

            Args:
                part: Which part to load (1-5).

            Returns:
                The dataset text for that part.
            """
            return _big_text(8000)

        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template=(
                    "If the section '# Task Progress' appears below, a previous "
                    "attempt already loaded the dataset: IMMEDIATELY call "
                    "finish(success=True, is_continue=False, summary='recovered') "
                    "and nothing else.\n"
                    "Otherwise: call load_dataset(1), then load_dataset(2), then "
                    "load_dataset(3), then load_dataset(4), then load_dataset(5), "
                    "one call per step, and only then call finish. Never call "
                    "finish before loading all 5 parts."
                ),
                tools=[load_dataset],
                max_steps=12,
                max_budget=3.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        # The old code hard-failed here with "failed with 3 consecutive
        # errors ... exceeds the context window" and success=False.
        self.assertNotIn("consecutive errors", parsed.get("summary", ""))
        self.assertTrue(parsed["success"], f"expected recovery, got: {parsed}")
        # Recovery really went through the continuation path: the merged
        # summary contains the prior (overflowed) session's summary section.
        self.assertIn("### Previous Session", parsed.get("summary", ""))

    @pytest.mark.slow
    def test_first_step_overflow_hard_fails(self) -> None:
        """If the FIRST provider call already overflows the real 128K window,
        continuing would replay the same oversized prompt forever — the
        relentless agent must hard-fail with the typed error message."""
        model_info.MODEL_INFO[TEST_MODEL].context_length = self._orig_context
        agent = RelentlessAgent("Relentless-Ctx-HardFail")
        filler = _big_text(200_000)
        with tempfile.TemporaryDirectory() as td:
            result = agent.run(
                model_name=TEST_MODEL,
                prompt_template="Summarize this:\n" + filler,
                max_steps=5,
                max_budget=2.0,
                max_sub_sessions=3,
                work_dir=td,
                verbose=False,
            )
        parsed = yaml.safe_load(result)
        self.assertFalse(parsed["success"])
        self.assertFalse(parsed["is_continue"])
        self.assertIn("context window", parsed["summary"].lower())


if __name__ == "__main__":
    unittest.main()
