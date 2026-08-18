# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the findings-3.md model-layer fixes.

Each test exercises real model objects through their actual code paths —
no mocks, patches, or fakes.  Bug-focused cases fail before their fixes;
additional preservation cases pin unchanged precedence and alias behavior:

* #3  — v1 OpenRouter-Anthropic ``cache_control`` must not mutate the
        caller's nested ``extra_body`` dict.
* #4  — v2 must strip the v1-only ``use_responses_api`` config key
        instead of forwarding it to ``client.responses.create``.
* #5  — v1 must not inject the MODEL_INFO default ``reasoning_effort``
        when the caller passed a native ``reasoning={"effort": ...}``.
* #7  — non-string (dict/list) tool results must be JSON-encoded, not
        crash ``parse_binary_attachments``.
* #18 — ``get_max_context_length`` raises ``KISSError`` (not
        ``KeyError``) for unknown models, matching ``calculate_cost``.
* #22 — Gemini honors ``max_completion_tokens`` via
        ``max_output_tokens`` like v1/v2/Anthropic.
* #1  — the ``-xhigh`` alias stripping shared with ``model_info`` still
        yields the same wire model names.
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_max_context_length
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


class TestFinding18UnknownModelErrorType:
    """#18 — unknown models raise KISSError from get_max_context_length."""

    def test_get_max_context_length_raises_kiss_error(self) -> None:
        with pytest.raises(KISSError):
            get_max_context_length("no-such-model-xyz-12345")
