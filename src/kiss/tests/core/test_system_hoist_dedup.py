# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the shared system-message hoisting helper.

``AnthropicModel._build_create_kwargs`` and
``GeminiModel._resolve_system_instruction`` used to carry two verbatim
copies of the same ~14-line loop that merges the configured
``system_instruction`` with OpenAI-style ``role="system"`` conversation
messages (which enter a conversation on model hand-off, e.g. via the
Sorcar ``set_model`` tool).  The copies could silently drift apart, so
the loop now lives once in :func:`kiss.core.models.model.merge_system_texts`.

These tests pin the merged behaviour through both adapters' REAL request
builders (no mocks, no network — request shaping happens locally) and
assert the two transports agree on every branch of the shared helper:
configured-only, message-only, both merged in order, content-part lists,
whitespace-only and duplicate skipping, and the all-empty ``None`` case.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import merge_system_texts


def _anthropic_system(
    conversation: list[dict[str, Any]], system_instruction: str | None
) -> str | None:
    """Return the ``system`` param AnthropicModel would send for *conversation*."""
    config = {"system_instruction": system_instruction} if system_instruction else {}
    m = AnthropicModel("claude-sonnet-5", api_key="test-key", model_config=config)
    m.conversation = list(conversation)
    return m._build_create_kwargs().get("system")


def _gemini_system(
    conversation: list[dict[str, Any]], system_instruction: str | None
) -> str | None:
    """Return the system instruction GeminiModel would send for *conversation*."""
    config = {"system_instruction": system_instruction} if system_instruction else {}
    m = GeminiModel("gemini-2.5-pro", api_key="test-key", model_config=config)
    m.conversation = list(conversation)
    return m._resolve_system_instruction()


_USER = {"role": "user", "content": "hello"}


class TestSystemHoistShared(unittest.TestCase):
    """Both adapters must produce the identical merged system instruction."""

    def test_configured_only(self) -> None:
        conv: list[dict[str, Any]] = [_USER]
        self.assertEqual(_anthropic_system(conv, "cfg sys"), "cfg sys")
        self.assertEqual(_gemini_system(conv, "cfg sys"), "cfg sys")

    def test_none_when_nothing_contributes(self) -> None:
        conv: list[dict[str, Any]] = [_USER]
        self.assertIsNone(_anthropic_system(conv, None))
        self.assertIsNone(_gemini_system(conv, None))

    def test_system_messages_merged_in_order_after_configured(self) -> None:
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": "first"},
            _USER,
            {"role": "system", "content": "second"},
        ]
        expected = "cfg\n\nfirst\n\nsecond"
        self.assertEqual(_anthropic_system(conv, "cfg"), expected)
        self.assertEqual(_gemini_system(conv, "cfg"), expected)

    def test_duplicates_and_whitespace_and_non_string_skipped(self) -> None:
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": "cfg"},  # duplicate of configured
            {"role": "system", "content": "   \n"},  # whitespace-only
            {"role": "system", "content": None},  # non-string
            {"role": "system", "content": "extra"},
            {"role": "system", "content": "extra"},  # duplicate of earlier msg
            _USER,
        ]
        expected = "cfg\n\nextra"
        self.assertEqual(_anthropic_system(conv, "cfg"), expected)
        self.assertEqual(_gemini_system(conv, "cfg"), expected)

    def test_content_part_lists_contribute_text_parts_only(self) -> None:
        conv: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                    "not-a-dict",
                    {"type": "text", "text": "part two"},
                ],
            },
            _USER,
        ]
        expected = "part one part two"
        self.assertEqual(_anthropic_system(conv, None), expected)
        self.assertEqual(_gemini_system(conv, None), expected)

    def test_helper_returns_none_for_empty_inputs(self) -> None:
        self.assertIsNone(merge_system_texts(None, []))
        self.assertIsNone(merge_system_texts("", [{"role": "user", "content": "x"}]))


if __name__ == "__main__":
    unittest.main()
