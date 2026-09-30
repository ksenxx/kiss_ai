"""End-to-end tests for the system-prompt cache breakpoint.

A real :class:`KISSAgent` runs the native Anthropic adapter against the
scripted local Messages server and the tests inspect the request bodies:
the static system prefix must arrive as its own ``system`` block carrying
an explicit ``cache_control`` breakpoint, the per-run tail as a plain
block after it, and the top-level automatic ``cache_control`` must stay.
The OpenAI-compatible adapter must send the instruction as one plain
system message with the marker removed, and the transcript shown to the
user must not contain the marker either.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model import (
    SYSTEM_CACHE_BREAK,
    split_system_cache_break,
    strip_system_cache_break,
)
from kiss.core.printer import Printer
from kiss.tests.agents.sorcar import local_model_server as openai_server
from kiss.tests.core import local_anthropic_server as anthropic_server

STATIC = "You are a careful assistant.\n- Always finish.\n"
TAIL = "\n- Work dir: /tmp/task-42\n# Task Settings\n- Model name: x\n"
SYSTEM_WITH_BREAK = STATIC + SYSTEM_CACHE_BREAK + TAIL


class RecordingPrinter(Printer):
    """A real printer that records every event it is asked to render."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record the ``(type, content)`` pair of every print call.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "system_prompt").
            **kwargs: Additional type-specific options (ignored).

        Returns:
            Always the empty string.
        """
        self.events.append((type, str(content)))
        return ""

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (ignored).
        """

    def reset(self) -> None:
        """Reset streaming state (no-op)."""


def test_split_and_strip_helpers() -> None:
    assert split_system_cache_break("plain") == ("plain", "")
    assert split_system_cache_break(SYSTEM_WITH_BREAK) == (STATIC, TAIL)
    twice = STATIC + SYSTEM_CACHE_BREAK + "a" + SYSTEM_CACHE_BREAK + "b"
    assert split_system_cache_break(twice) == (STATIC, "ab")
    assert strip_system_cache_break(SYSTEM_WITH_BREAK) == STATIC + TAIL
    assert strip_system_cache_break("plain") == "plain"


def _finish_script() -> list[dict[str, Any]]:
    return [anthropic_server.tool_use_message("finish", {"result": "<p>ok</p>"}, 20_000, "toolu_1")]


def _run_anthropic(
    monkeypatch: pytest.MonkeyPatch, system_prompt: str, model_config: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], RecordingPrinter]:
    printer = RecordingPrinter()
    script = _finish_script()
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        KISSAgent("cache-break-test").run(
            model_name=anthropic_server.MODEL,
            prompt_template="Finish now.",
            system_prompt=system_prompt,
            tools=[],
            max_steps=3,
            max_budget=50.0,
            verbose=False,
            printer=printer,
            model_config=model_config,
        )
    return list(requests), printer


def test_anthropic_caches_static_prefix_and_sends_tail_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, printer = _run_anthropic(monkeypatch, SYSTEM_WITH_BREAK)
    assert len(requests) == 1
    body = requests[0]
    assert body["system"] == [
        {"type": "text", "text": STATIC, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": TAIL},
    ]
    # The automatic breakpoint keeps caching the growing conversation.
    assert body["cache_control"] == {"type": "ephemeral"}
    # The transcript shows the prompt without the internal marker.
    shown = [text for etype, text in printer.events if etype == "system_prompt"]
    assert shown == [STATIC + TAIL]


def test_anthropic_without_marker_caches_the_whole_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, _printer = _run_anthropic(monkeypatch, STATIC)
    assert requests[0]["system"] == [
        {"type": "text", "text": STATIC, "cache_control": {"type": "ephemeral"}},
    ]


def test_anthropic_with_caching_off_sends_one_plain_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, _printer = _run_anthropic(
        monkeypatch, SYSTEM_WITH_BREAK, model_config={"enable_cache": False}
    )
    body = requests[0]
    assert body["system"] == STATIC + TAIL
    assert "cache_control" not in body


def test_anthropic_marker_at_the_start_leaves_nothing_to_cache() -> None:
    # A marker with no static prefix in front of it (a caller that put the
    # whole prompt in the tail) degrades to a single plain string: an empty
    # text block cannot be cached and would be rejected by the API.
    m = AnthropicModel(
        model_name="claude-sonnet-4-5",
        api_key="test-key",
        model_config={"system_instruction": SYSTEM_CACHE_BREAK + TAIL},
    )
    m.conversation = [{"role": "user", "content": "hi"}]
    assert m._build_create_kwargs()["system"] == TAIL
    # A tail that is only whitespace is dropped rather than sent as an
    # empty block.
    m.model_config["system_instruction"] = STATIC + SYSTEM_CACHE_BREAK + "\n\n"
    assert m._build_create_kwargs()["system"] == [
        {"type": "text", "text": STATIC, "cache_control": {"type": "ephemeral"}},
    ]


def test_openai_compatible_adapter_strips_the_marker() -> None:
    script = [openai_server.finish_body("<p>ok</p>")]
    with openai_server.serve(script) as (url, requests):
        KISSAgent("cache-break-openai").run(
            model_name=openai_server.MODEL,
            prompt_template="Finish now.",
            system_prompt=SYSTEM_WITH_BREAK,
            tools=[],
            max_steps=3,
            max_budget=50.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    system_messages = [m for m in requests[0]["messages"] if m["role"] == "system"]
    assert system_messages == [{"role": "system", "content": STATIC + TAIL}]
