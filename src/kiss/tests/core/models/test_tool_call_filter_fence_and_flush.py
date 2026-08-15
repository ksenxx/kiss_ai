# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the tool-call stream filter's display contract.

``_ToolCallFilteredStream`` keeps the raw ``{"tool_calls": ...}`` JSON out
of the chat panel while streaming everything else.  Two corners of that
contract are pinned here:

1. A tool call wrapped in a markdown code fence must take its now-empty
   fence wrapper with it — the panel showing a bare ```` ```json ```` /
   ```` ``` ```` pair is the same noise as the JSON itself.  The wrapper
   is suppressed **only** when it wrapped a parse-validated, suppressed
   tool call; a fence around anything else streams untouched.
2. At end of turn the filter may drop a buffered fragment only when it
   parse-validates as a tool call.  Malformed JSON and ordinary text that
   merely contains the substring ``tool_calls`` are never swallowed.

Every test runs a REAL stand-in ``claude`` / ``codex`` executable installed
on ``PATH``: real subprocesses, real streams, no mocks, patches or doubles.
The stand-ins emit the text in controlled chunks so the fence opener, the
JSON object, and the closing fence each get split across streaming-chunk
boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_TOOL_CALL_JSON = (
    '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
)


def _claude_script(chunks: list[str]) -> str:
    """Build a stand-in ``claude`` that streams *chunks* as text deltas.

    Args:
        chunks: The text pieces, one ``content_block_delta`` event each.

    Returns:
        Python source for the stand-in, without a shebang.
    """
    chunk_json = json.dumps(chunks)
    return f"""
    import json
    import sys

    sys.stdin.read()
    chunks = json.loads({chunk_json!r})
    for chunk in chunks:
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "text_delta",
                                      "text": chunk}}}}),
              flush=True)
    print(json.dumps({{"type": "result", "result": "".join(chunks),
                       "usage": {{"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 0}}}}), flush=True)
    """


def _codex_script(text: str) -> str:
    """Build a stand-in ``codex`` that answers with *text* in one message.

    Args:
        text: The full agent-message text.

    Returns:
        Python source for the stand-in, without a shebang.
    """
    text_json = json.dumps(text)
    return f"""
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({{"type": "item.completed",
                       "item": {{"type": "agent_message",
                                 "text": json.loads({text_json!r})}}}}),
          flush=True)
    print(json.dumps({{"type": "turn.completed",
                       "usage": {{"input_tokens": 10,
                                  "cached_input_tokens": 0,
                                  "output_tokens": 5}}}}), flush=True)
    """


def _run_bash(command: str) -> str:
    """Pretend to run a shell command.

    Args:
        command: The command the model asked for.

    Returns:
        A fixed listing.
    """
    return f"ran {command}"


def _stream_claude_tool_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunks: list[str]
) -> tuple[list[dict[str, object]], str]:
    """Run a claude tool turn over *chunks* and capture the streamed panel text.

    Args:
        tmp_path: The test's temporary directory, which becomes ``PATH``.
        monkeypatch: The fixture used by :func:`install_cli`.
        chunks: The text deltas the stand-in CLI streams.

    Returns:
        The parsed tool calls and the concatenated token stream.
    """
    install_cli(tmp_path, monkeypatch, "claude", _claude_script(chunks))
    tokens: list[str] = []
    model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
    model.initialize("list the files")
    calls, _content, _response = model.generate_and_process_with_tools(
        {"Bash": _run_bash}
    )
    return calls, "".join(tokens)


class TestAFencedToolCallTakesItsFenceWithIt:
    """Suppressing the JSON must not leave an empty ``` wrapper behind."""

    def test_single_chunk_fenced_tool_call_leaves_no_fence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prose around the fence streams; the fence and JSON do not."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"Here is the plan:\n```json\n{_TOOL_CALL_JSON}\n```\nAll done."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Here is the plan:" in streamed
        assert "All done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_fence_opener_split_across_chunks_is_suppressed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An opener arriving as ``` `` ``` + ``` `json ``` still vanishes."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            ["Before\n``", "`json\n", _TOOL_CALL_JSON],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Before" in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_object_and_closing_fence_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool call and its closing fence split mid-JSON still vanish."""
        head, tail = _TOOL_CALL_JSON[:20], _TOOL_CALL_JSON[20:]
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            ["```json\n" + head, tail, "\n``", "`\n"],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_codex_fenced_tool_call_leaves_no_fence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other CLI adapter shares the filter and the fence contract."""
        install_cli(
            tmp_path, monkeypatch, "codex",
            _codex_script(f"Working:\n```json\n{_TOOL_CALL_JSON}\n```\nOK."),
        )
        tokens: list[str] = []
        model = CodexModel("codex/default", token_callback=tokens.append)
        model.initialize("list the files")
        calls, _content, _response = model.generate_and_process_with_tools(
            {"Bash": _run_bash}
        )
        streamed = "".join(tokens)
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Working:" in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed


class TestOnlyValidatedToolCallsAreSuppressed:
    """Anything that does not parse as a tool call streams untouched."""

    def test_fenced_malformed_balanced_json_keeps_its_fence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fence around invalid JSON is content, not a tool-call wrapper."""
        bogus = '{"tool_calls": oops}'
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"Look:\n```json\n{bogus}\n```\nend."],
        )
        assert calls == []
        assert f"Look:\n```json\n{bogus}\n```\nend." in streamed

    def test_malformed_balanced_json_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Balanced braces that fail to parse are ordinary text."""
        bogus = 'Data: {"tool_calls": nope} end.'
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [bogus]
        )
        assert calls == []
        assert bogus in streamed

    def test_truncated_ordinary_json_mentioning_tool_calls_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A turn ending mid-object must not swallow it for a substring."""
        fragment = 'Note {"docs": "how tool_calls works", "page": 1'
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [fragment]
        )
        assert calls == []
        assert fragment in streamed

    def test_prose_with_inline_backticks_still_streams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Holding back possible fence openers must not eat inline code."""
        prose = "Run `ls -la` and then ``echo hi`` please."
        _calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [prose]
        )
        assert prose in streamed

    def test_trailing_backticks_at_turn_end_are_released(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A held possible-opener is emitted when the turn simply ends."""
        prose = "The markdown fence marker is ``"
        _calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [prose]
        )
        assert prose in streamed


class TestWhitespaceBetweenOpenerAndBraceIsWrapper:
    """Blank lines, CRLF, and pre-info-string spaces are all wrapper.

    The batch stripper this filter replaced accepted arbitrary whitespace
    between the fence opener and the ``{`` of the tool call, so all of
    these forms are within the original display contract and must vanish
    together with the suppressed JSON.
    """

    def test_blank_line_between_opener_and_object_single_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank line inside the wrapper leaves no empty fence behind."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"Plan:\n```json\n\n{_TOOL_CALL_JSON}\n```\nDone."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "Done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_blank_line_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The opener, blank line, and object may arrive in pieces."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            ["Plan:\n```json\n", "\n", _TOOL_CALL_JSON, "\n```\nDone."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "Done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_crlf_line_endings_single_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CRLF-terminated wrapper is recognized and suppressed whole."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"Plan:\r\n```json\r\n{_TOOL_CALL_JSON}\r\n```\r\nDone."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "Done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_crlf_opener_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk boundary between the CR and the LF still suppresses."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            ["Plan:\r\n```json\r", "\n" + _TOOL_CALL_JSON, "\r\n```\r\nDone."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "Done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_space_before_info_string_single_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``` json (space before the info string) is a valid opener."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"Plan:\n``` json\n{_TOOL_CALL_JSON}\n```\nDone."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "Done." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_space_before_info_string_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The spaced info string may itself straddle a chunk boundary."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            ["Plan:\n``` js", "on\n", _TOOL_CALL_JSON, "\n```"],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Plan:" in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_codex_crlf_fenced_tool_call_leaves_no_fence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other CLI adapter shares the whitespace-tolerant contract."""
        install_cli(
            tmp_path, monkeypatch, "codex",
            _codex_script(
                f"Working:\r\n```json\r\n\r\n{_TOOL_CALL_JSON}\r\n```\r\nOK."
            ),
        )
        tokens: list[str] = []
        model = CodexModel("codex/default", token_callback=tokens.append)
        model.initialize("list the files")
        calls, _content, _response = model.generate_and_process_with_tools(
            {"Bash": _run_bash}
        )
        streamed = "".join(tokens)
        assert [call["name"] for call in calls] == ["Bash"]
        assert "Working:" in streamed
        assert "OK." in streamed
        assert "`" not in streamed
        assert "tool_calls" not in streamed

    def test_blank_line_fence_around_non_tool_call_streams_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wider opener must not suppress a fence around plain JSON."""
        text = 'See:\n```json\n\n{"config": true}\n```\nend.'
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [text]
        )
        assert calls == []
        assert text in streamed

    def test_turn_ending_in_opener_and_blank_lines_is_released(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A held opener plus trailing blank lines is flushed, not eaten."""
        prose = "The block starts with:\n```json\n\n"
        _calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [prose]
        )
        assert prose in streamed


class TestTheClosingFenceEaterStaysInItsLane:
    """Only the ``` wrapper of a suppressed call is consumed, ever."""

    def test_plain_text_after_a_suppressed_fenced_call_streams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no closing fence follows, nothing is eaten in its place."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"```json\n{_TOOL_CALL_JSON}\nNo fence, plain text."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "No fence, plain text." in streamed
        assert "`" not in streamed

    def test_closing_fence_with_trailing_spaces_is_consumed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spaces between the closing ``` and its newline are wrapper too."""
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch,
            [f"```json\n{_TOOL_CALL_JSON}\n```  \nAfter."],
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "After." in streamed
        assert "`" not in streamed

    def test_flush_drops_a_fragment_holding_a_validated_tool_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unclosed outer object hiding a real tool call is suppressed.

        The framework parses that inner call out of the full content and
        executes it, so showing its JSON would break the display contract
        exactly like a well-formed block.
        """
        fragment = f'pre {{"outer": {_TOOL_CALL_JSON}'
        calls, streamed = _stream_claude_tool_turn(
            tmp_path, monkeypatch, [fragment]
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "pre " in streamed
        assert "tool_calls" not in streamed
