# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for text scanners that used to be quadratic.

The 2026-09-12 outage (see ``test_tool_result_scan_linear.py``) was a
regex that rescanned a 5.8 MB tool result from every offset while
holding the GIL.  The review of that fix found five more scanners with
the same shape — a lazy or greedy run rescanned from every candidate
start — on inputs a user or a tool can make large:

* ``parse_binary_attachments`` — runs on EVERY tool result;
* ``_extract_deepseek_reasoning`` — runs on model output;
* ``parse_task_tags`` / ``contains_task_tags`` — run on user messages;
* ``trailing_identifier`` — runs on the chat input box on every keystroke,
  on the server's event-loop thread;
* the cron silence-token HTML strip — runs on job summaries.

Each test checks that the rewritten scanner keeps the former semantics
on representative inputs and that a pathological input of a few hundred
kilobytes finishes in well under a second (the old code needed tens of
seconds to hours).
"""

import base64
import time

from kiss.agents.sorcar.cron_agent import _is_silent
from kiss.core.models.model import (
    BINARY_ATTACHMENT_CLOSE,
    encode_binary_attachment,
    parse_binary_attachments,
)
from kiss.core.models.openai_compatible_model import _extract_deepseek_reasoning
from kiss.server.autocomplete import trailing_identifier
from kiss.server.task_runner import contains_task_tags, parse_task_tags

# Every scan below must finish far under this; the old code took 3-50 s
# on the same inputs (measured during the 2026-09-12 review).
_MAX_SECONDS = 2.0

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
    "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


def _timed(fn, *args):
    started = time.monotonic()
    result = fn(*args)
    assert time.monotonic() - started < _MAX_SECONDS
    return result


class TestParseBinaryAttachments:
    def test_single_attachment_round_trip(self):
        text = "before " + encode_binary_attachment("image/png", _PNG_1PX) + " after"
        plain, atts = parse_binary_attachments(text)
        assert plain == f"before [attached image/png, {len(_PNG_1PX)} bytes] after"
        assert len(atts) == 1
        assert atts[0].mime_type == "image/png"
        assert atts[0].data == _PNG_1PX

    def test_opener_quoted_inside_block_is_not_a_second_block(self):
        # A Read of a file that quotes the sentinel yields an opener
        # inside another block's span.  The outer payload is not base64
        # (it contains ``<``), so the block is dropped; the inner opener
        # must not then be paired with the same closer — the old scanner
        # did exactly that and fabricated a zero-byte attachment.
        inner = "<<KISS_BINARY_ATTACHMENT mime_type=image/png>>"
        text = (
            "<<KISS_BINARY_ATTACHMENT mime_type=text/plain>>"
            + inner
            + BINARY_ATTACHMENT_CLOSE
            + " tail"
        )
        plain, atts = parse_binary_attachments(text)
        assert atts == []
        assert "[attached" not in plain
        assert plain.endswith(" tail")

    def test_many_unclosed_openers_scan_linearly(self):
        opener = "<<KISS_BINARY_ATTACHMENT mime_type=image/png>>"
        text = opener * 20_000 + "A" * 100_000 + BINARY_ATTACHMENT_CLOSE
        plain, atts = _timed(parse_binary_attachments, text)
        # One block: the first opener through the single closer.
        assert len(atts) == 1
        assert plain.startswith("[attached image/png")

    def test_no_sentinel_returns_input_unchanged(self):
        text = "x" * 300_000
        assert _timed(parse_binary_attachments, text) == (text, [])


class TestDeepseekReasoning:
    def test_reasoning_and_answer_split(self):
        assert _extract_deepseek_reasoning("<think> why </think> ans") == ("why", "ans")

    def test_every_closed_block_removed_first_kept_as_reasoning(self):
        text = "a<think>r1</think>b<think>r2</think>c"
        assert _extract_deepseek_reasoning(text) == ("r1", "abc")

    def test_plain_or_unclosed_content_returned_verbatim(self):
        assert _extract_deepseek_reasoning("Just a plain answer") == (
            "",
            "Just a plain answer",
        )
        assert _extract_deepseek_reasoning("<think>never closed") == (
            "",
            "<think>never closed",
        )

    def test_repeated_unclosed_openers_scan_linearly(self):
        text = "<think>" * 60_000 + "tail"
        assert _timed(_extract_deepseek_reasoning, text) == ("", text)


class TestTaskTags:
    def test_blocks_are_split_and_stripped(self):
        text = "x <task> one </task> y <task></task> <task>two</task>"
        assert parse_task_tags(text) == ["one", "two"]
        assert contains_task_tags(text) is True

    def test_no_or_empty_blocks_fall_back_to_whole_text(self):
        assert parse_task_tags("plain") == ["plain"]
        assert parse_task_tags("<task>  </task>") == ["<task>  </task>"]
        assert contains_task_tags("<task>  </task>") is False
        assert parse_task_tags("<task>unclosed") == ["<task>unclosed"]

    def test_repeated_unclosed_openers_scan_linearly(self):
        text = "<task>" * 60_000 + "tail"
        assert _timed(parse_task_tags, text) == [text]
        assert _timed(contains_task_tags, text) is False


class TestTrailingIdentifier:
    def test_trailing_token_semantics(self):
        assert trailing_identifier("call foo.bar") == "foo.bar"
        assert trailing_identifier("x = np") == "np"
        assert trailing_identifier("ends with space ") == ""
        assert trailing_identifier("one letter a") == ""
        assert trailing_identifier("..abc") == "abc"
        assert trailing_identifier("a.") == "a."
        assert trailing_identifier("...") == ""
        assert trailing_identifier("") == ""
        assert trailing_identifier("unicode wörd") == "wörd"

    def test_long_unbroken_run_scans_linearly(self):
        # A pasted blob that does NOT end in an identifier: the old regex
        # tried every offset and backtracked to the end each time.
        assert _timed(trailing_identifier, "A" * 300_000 + " ") == ""
        # And one that does: the whole blob is the token.
        assert _timed(trailing_identifier, "B" * 300_000) == "B" * 300_000


class TestSilenceTokenHtmlStrip:
    def test_tokens_with_and_without_tags(self):
        assert _is_silent("[SILENT]") is True
        assert _is_silent("<p><b>NO_REPLY</b></p>") is True
        assert _is_silent("<p>hello</p>") is False

    def test_many_lone_angle_brackets_scan_linearly(self):
        assert _timed(_is_silent, "<" * 200_000 + "x") is False
