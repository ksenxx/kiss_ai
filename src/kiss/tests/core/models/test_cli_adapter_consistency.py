# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for behaviour the two CLI adapters must share.

Like :mod:`test_cli_subprocess_lifecycle`, every CLI test here runs a
REAL stand-in ``claude`` / ``codex`` executable installed on ``PATH``.

Findings covered (audit ``tmp/audit/03-core-models-c.md``):

* **I5** — ``cc/*`` streamed the raw ``{"tool_calls": ...}`` JSON to the
  user while ``codex/*`` stripped it.
* **I7** — the CLI usage getters returned ``None`` for a JSON ``null``
  field, so the caller's token arithmetic raised ``TypeError`` and the
  step was silently billed ``$0``.
* **C6** — ``stop_on_tool_calls`` truncates the text and then drains the
  stream for the terminal ``result`` event (the only carrier of usage,
  issue #34).  The drain must not be able to fail a step whose tool call
  is already parsed.
* **R5** — the Anthropic and Claude Code cache-creation parsers were the
  same function twice; one shared parser must keep both answers equal.
"""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Usage
from anthropic.types.cache_creation import CacheCreation

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_CODEX_EMITS_A_TOOL_CALL = """
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message",
                               "text": 'Listing now.\\n{"tool_calls": [{"name": '
                                       '"finish", "arguments": {"result": "ok"}}]}'}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 10, "cached_input_tokens": 0,
                                "output_tokens": 5}}), flush=True)
"""

_CLAUDE_EMITS_A_TOOL_CALL = """
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "message_delta",
                      "delta": {"stop_reason": "end_turn"},
                      "usage": {"input_tokens": 10, "output_tokens": 5,
                                "cache_read_input_tokens": 0}}), flush=True)
    for chunk in ["Listing now.\\n",
                  '{"tool_calls": [{"name": "finish", ',
                  '"arguments": {"result": "ok"}}]}',
                  "\\nI will now pretend the tool already ran."]:
        print(json.dumps({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": chunk}}),
              flush=True)
    print(json.dumps({"type": "result", "result": "ignored",
                      "usage": {"input_tokens": 999999,
                                "output_tokens": 999999,
                                "cache_read_input_tokens": 0}}), flush=True)
"""

_CLAUDE_OVERRUNS_THE_DEADLINE_AFTER_A_TOOL_CALL = """
    import json
    import sys
    import time

    sys.stdin.read()
    for chunk in ['{"tool_calls": [{"name": "finish", ',
                  '"arguments": {"result": "ok"}}]}',
                  "\\nNow let me hallucinate the tool output."]:
        print(json.dumps({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": chunk}}),
              flush=True)
    time.sleep(60)
    print(json.dumps({"type": "result", "result": "too late", "usage": {}}),
          flush=True)
"""

_CODEX_REPORTS_NULL_OUTPUT_TOKENS = """
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "hi"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": None, "cached_input_tokens": None,
                                "output_tokens": None}}), flush=True)
"""

_CLAUDE_REPORTS_NULL_OUTPUT_TOKENS = """
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "hi"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "hi",
                      "usage": {"input_tokens": None, "output_tokens": None,
                                "cache_read_input_tokens": None,
                                "cache_creation_input_tokens": None}}), flush=True)
"""


def finish(result: str) -> str:
    """Finish the task and return its result.

    Args:
        result: The final result.

    Returns:
        The result, unchanged.
    """
    return result


class TestI5ToolCallJsonIsNeverShownToTheUser:
    """Neither adapter may stream raw ``tool_calls`` JSON to the printer."""

    def test_codex_strips_tool_call_json_from_the_token_stream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user sees the prose, never the machine-readable block."""
        install_cli(tmp_path, monkeypatch, "codex", _CODEX_EMITS_A_TOOL_CALL)
        tokens: list[str] = []
        model = CodexModel("codex/default", token_callback=tokens.append)
        model.initialize("list the files")

        function_calls, _content, _response = model.generate_and_process_with_tools(
            {"finish": finish}
        )

        assert [call["name"] for call in function_calls] == ["finish"]
        assert "tool_calls" not in "".join(tokens), tokens
        assert "Listing now." in "".join(tokens)

    def test_claude_code_strips_tool_call_json_from_the_token_stream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user sees the prose, never the machine-readable block."""
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EMITS_A_TOOL_CALL)
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("list the files")

        function_calls, _content, _response = model.generate_and_process_with_tools(
            {"finish": finish}
        )

        assert [call["name"] for call in function_calls] == ["finish"]
        assert "tool_calls" not in "".join(tokens), tokens
        assert "Listing now." in "".join(tokens)


_CLAUDE_EMITS_ORDINARY_JSON = """
    import json
    import sys

    sys.stdin.read()
    payload = json.dumps({"note": 'he said "hi" and {braces}'})
    for chunk in ["Result: ", payload, " done"]:
        print(json.dumps({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": chunk}}),
              flush=True)
    print(json.dumps({"type": "result", "result": "ignored", "usage": {}}),
          flush=True)
"""

_CLAUDE_STOPS_MID_JSON = """
    import json
    import sys

    sys.stdin.read()
    for chunk in ["Partial: ", '{"answer": 1']:
        print(json.dumps({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": chunk}}),
              flush=True)
"""

_CLAUDE_STOPS_MID_TOOL_CALL = """
    import json
    import sys

    sys.stdin.read()
    for chunk in ["Calling: ", '{"tool_calls": [{"name": "finish"']:
        print(json.dumps({"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": chunk}}),
              flush=True)
"""


class TestI5FilterEdgeCases:
    """The JSON filter must only ever hide genuine tool-call blocks."""

    def test_ordinary_json_in_the_answer_is_still_shown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON object that is not a tool call must reach the user intact.

        The object contains an escaped quote and a brace inside a string,
        so a naive brace counter would mis-detect where it ends.
        """
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EMITS_ORDINARY_JSON)
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("show me some json")

        function_calls, _content, _response = model.generate_and_process_with_tools(
            {"finish": finish}
        )

        payload = json.dumps({"note": 'he said "hi" and {braces}'})
        assert function_calls == []
        assert "".join(tokens) == f"Result: {payload} done"

    def test_unterminated_ordinary_json_is_flushed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text buffered when the turn ends must not be swallowed."""
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_STOPS_MID_JSON)
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("start some json")

        model.generate_and_process_with_tools({"finish": finish})

        assert "".join(tokens) == 'Partial: {"answer": 1'

    def test_unterminated_tool_call_fragment_is_flushed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a parse-validated tool call may be dropped at end of turn.

        This fragment is cut off before any object closes, so nothing in
        it validates as a tool call and it is released like any other
        buffered text.  The old substring test swallowed it — and with it
        every ordinary fragment that merely mentioned ``tool_calls`` (see
        ``test_tool_call_filter_fence_and_flush.py``, which also pins the
        drop of a fragment whose truncation *does* validate).
        """
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_STOPS_MID_TOOL_CALL)
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("call a tool")

        model.generate_and_process_with_tools({"finish": finish})

        assert "".join(tokens) == 'Calling: {"tool_calls": [{"name": "finish"'


class TestC6EarlyStopDrain:
    """Early stop must keep usage accounting without draining the CLI."""

    def test_early_stop_still_reports_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage is aggregated from ``message_delta`` events (issue #34).

        The queued terminal ``result`` event (sentinel 999999 counts) must
        NOT be consumed: draining the now-agentic CLI past the stop would
        let it keep executing native tools.
        """
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EMITS_A_TOOL_CALL)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
        model.initialize("list the files")

        function_calls, _content, response = model.generate_and_process_with_tools(
            {"finish": finish}
        )

        assert [call["name"] for call in function_calls] == ["finish"]
        assert response["usage"]["output_tokens"] == 5

    def test_a_cli_that_overruns_the_deadline_still_yields_its_tool_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drain that hits the deadline costs accounting, not the step.

        The tool call is already parsed when the drain starts, so
        turning the CLI's own overrun into a failed step throws away
        completed work and re-spends a whole CLI run.
        """
        install_cli(
            tmp_path,
            monkeypatch,
            "claude",
            _CLAUDE_OVERRUNS_THE_DEADLINE_AFTER_A_TOOL_CALL,
        )
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": 3})
        model.initialize("list the files")

        function_calls, _content, _response = model.generate_and_process_with_tools(
            {"finish": finish}
        )

        assert [call["name"] for call in function_calls] == ["finish"]
        assert function_calls[0]["arguments"] == {"result": "ok"}


class TestI7NullUsageFields:
    """A JSON ``null`` usage field must still produce integer counts."""

    def test_codex_null_usage_fields_yield_integers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``"output_tokens": null`` must not become ``None``."""
        install_cli(tmp_path, monkeypatch, "codex", _CODEX_REPORTS_NULL_OUTPUT_TOKENS)
        model = CodexModel("codex/default")
        model.initialize("hi")

        _content, response = model.generate()
        counts = model.extract_input_output_token_counts_from_response(response)

        assert all(isinstance(count, int) for count in counts), counts
        assert sum(counts) == 0

    def test_claude_code_null_usage_fields_yield_integers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``"output_tokens": null`` must not become ``None``."""
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_REPORTS_NULL_OUTPUT_TOKENS)
        model = ClaudeCodeModel("cc/opus")
        model.initialize("hi")

        _content, response = model.generate()
        counts = model.extract_input_output_token_counts_from_response(response)

        assert all(isinstance(count, int) for count in counts), counts
        assert sum(counts) == 0

    def test_a_null_usage_object_is_tolerated(self) -> None:
        """``"usage": null`` must not raise either."""
        codex = CodexModel("codex/default")
        claude = ClaudeCodeModel("cc/opus")

        assert codex.extract_input_output_token_counts_from_response(
            {"usage": None}
        ) == (0, 0, 0, 0)
        assert claude.extract_input_output_token_counts_from_response(
            {"usage": None}
        ) == (0, 0, 0, 0, 0)


class _AnthropicResponse:
    """The only part of an Anthropic message the extractor reads."""

    def __init__(self, usage: Usage) -> None:
        """Store the usage object.

        Args:
            usage: A real ``anthropic.types.Usage`` model.
        """
        self.usage = usage


class TestR5SharedCacheCreationParser:
    """Both adapters must split cache-creation tokens identically."""

    @staticmethod
    def _both(usage_kwargs: dict[str, Any], usage_json: dict[str, Any]) -> None:
        """Assert the two adapters agree on the same wire content."""
        anthropic_counts = AnthropicModel(
            "claude-opus-4-6", api_key="not-used-token-accounting-is-offline"
        ).extract_input_output_token_counts_from_response(
            _AnthropicResponse(Usage(**usage_kwargs))
        )
        cli_counts = ClaudeCodeModel(
            "cc/opus"
        ).extract_input_output_token_counts_from_response({"usage": usage_json})
        assert anthropic_counts == cli_counts

    def test_split_cache_creation_matches(self) -> None:
        """A TTL-split ``cache_creation`` gives the same 5m/1h pair."""
        self._both(
            {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 14,
                "cache_creation": CacheCreation(
                    ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=9
                ),
            },
            {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 14,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 5,
                    "ephemeral_1h_input_tokens": 9,
                },
            },
        )

    def test_aggregate_cache_creation_matches(self) -> None:
        """Without a TTL split both bill the aggregate as one-hour."""
        self._both(
            {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 14,
            },
            {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 14,
            },
        )

    def test_absent_cache_fields_match(self) -> None:
        """A usage object without any cache fields yields zeros on both."""
        self._both(
            {"input_tokens": 11, "output_tokens": 22},
            {"input_tokens": 11, "output_tokens": 22},
        )


def test_stand_in_scripts_are_valid_python() -> None:
    """Guard against a stand-in CLI that silently fails to start."""
    for body in (
        _CODEX_EMITS_A_TOOL_CALL,
        _CLAUDE_EMITS_A_TOOL_CALL,
        _CLAUDE_OVERRUNS_THE_DEADLINE_AFTER_A_TOOL_CALL,
        _CODEX_REPORTS_NULL_OUTPUT_TOKENS,
        _CLAUDE_REPORTS_NULL_OUTPUT_TOKENS,
        _CLAUDE_EMITS_ORDINARY_JSON,
        _CLAUDE_STOPS_MID_JSON,
        _CLAUDE_STOPS_MID_TOOL_CALL,
    ):
        compile(textwrap.dedent(body), "<stand-in>", "exec")
