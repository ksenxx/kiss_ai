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

The core-side halves (overflow phrase classification, proactive stop, and
typed provider-overflow conversion in ``KISSAgent``) live in
``kiss.tests.core.test_context_window_e2e``, which owns the shared
``ShrunkContextMixin`` / ``_big_text`` helpers imported below.
"""

import tempfile
import unittest

import pytest
import yaml

from kiss.agents.sorcar.relentless_agent import (
    MAX_PROGRESS_CHARS,
    RelentlessAgent,
    _capped_progress_text,
)
from kiss.core.models import model_info
from kiss.tests.conftest import requires_openai_api_key
from kiss.tests.core.test_context_window_e2e import (
    TEST_MODEL,
    ShrunkContextMixin,
    _big_text,
)


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
                    "finish(success=True, is_continue=False, "
                    "summary_in_html='recovered') "
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
        self.assertNotIn("consecutive errors", parsed.get("summary", ""))
        self.assertTrue(parsed["success"], f"expected recovery, got: {parsed}")
        self.assertIn("<h3>Previous Session", parsed.get("summary", ""))

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
