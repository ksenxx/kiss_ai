# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Claude Code stream parser (audit 2026-10-18, slice B3).

F1: ``_parse_stream_events`` used to rescan the whole accumulated text at
every text delta containing ``}`` once ``tool_calls`` had been seen, so a
large streamed tool call (a ``Write`` of a file with braces) cost minutes
of CPU after the CLI was done.  The parser now tracks brace depth across
deltas and scans only when a top-level object closes.

F2: a ``_pre_result_content`` copy of the untruncated text was parsed for
tool calls as well, so a ``tool_calls`` block the parser had cut off (one
after a non-tool-calls object) was executed although the recorded
assistant content showed only the first block.  Only the content the
parser returns is parsed now.

The F2 test runs the real ``ClaudeCodeModel.generate_and_process_with_tools``
against a fake ``claude`` on PATH; it is marked ``live_cli`` only to opt
out of the CLI-locator stub of this directory and never spawns the real
CLI.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import _parse_text_based_tool_calls
from kiss.tests.conftest import install_cli_script


def _stream_event(event: dict[str, Any]) -> str:
    return json.dumps({"type": "stream_event", "event": event})


def _text_delta_lines(text: str, chunk: int) -> list[str]:
    """The stream-json lines of one assistant message streaming *text* in *chunk*-sized deltas."""
    lines = [
        _stream_event({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
        _stream_event({"type": "content_block_start", "content_block": {"type": "text"}}),
    ]
    for i in range(0, len(text), chunk):
        lines.append(
            _stream_event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": text[i : i + chunk]},
                }
            )
        )
    lines.append(_stream_event({"type": "content_block_stop"}))
    lines.append(_stream_event({"type": "message_stop"}))
    return lines


def _write_call_text(kilobytes: int) -> str:
    """A ``Write`` tool call whose file content is *kilobytes* of brace-heavy JavaScript."""
    line = "function f(x) { if (x) { return {a: 1, b: [1,2,3]}; } else { return {}; } }\n"
    body = line * (kilobytes * 1024 // len(line))
    return json.dumps(
        {
            "tool_calls": [
                {"name": "Write", "arguments": {"file_path": "/tmp/x.js", "content": body}}
            ]
        }
    )


def _recorded(lines: list[str], consumed: list[str]) -> Iterator[str]:
    """Yield *lines*, appending each to *consumed* as the parser pulls it."""
    for line in lines:
        consumed.append(line)
        yield line


def _model() -> ClaudeCodeModel:
    return ClaudeCodeModel("cc/opus", model_config={}, token_callback=_drop_token)


def _drop_token(token: str) -> None:
    pass


class TestF1LinearTimeToolCallDetection:
    """A large streamed tool call is detected in linear time."""

    def test_64kb_write_call_parses_within_seconds(self) -> None:
        text = _write_call_text(64)
        lines = _text_delta_lines(text, 16) + [json.dumps({"type": "result", "result": text})]
        m = _model()
        started = time.perf_counter()
        content, _result = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        elapsed = time.perf_counter() - started
        # The old parser needed about 30 s here (7.9 s for 32 KB).
        assert elapsed < 3.0, f"parsing took {elapsed:.1f}s"
        assert m._stopped_for_tool_calls is True
        assert content == text
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1
        assert (
            calls[0]["arguments"]["content"]
            == json.loads(text)["tool_calls"][0]["arguments"]["content"]
        )

    def test_detection_survives_deltas_splitting_the_key_and_the_close(self) -> None:
        """``tool_calls`` and the closing braces may straddle delta boundaries."""
        text = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        for chunk in (1, 2, 3, 5, 7, 11):
            m = _model()
            # A model that keeps going after the block (hallucinated output),
            # then one more event the parser must never pull.
            lines = _text_delta_lines(text, chunk)[:-2] + [
                _stream_event(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "\n(no output)"},
                    }
                ),
                "never-read",
            ]
            consumed: list[str] = []
            content, _ = m._parse_stream_events(_recorded(lines, consumed), stop_on_tool_calls=True)
            assert content == text, chunk
            assert m._stopped_for_tool_calls is True
            assert consumed[-1] != "never-read", chunk

    def test_braces_and_quotes_inside_strings_do_not_close_the_block(self) -> None:
        text = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": 'echo "}" \\" } { {'}}]}
        )
        m = _model()
        lines = _text_delta_lines(text, 4) + [json.dumps({"type": "result", "result": text})]
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        assert content == text
        assert m._stopped_for_tool_calls is True
        assert _parse_text_based_tool_calls(content)[0]["arguments"] == {
            "command": 'echo "}" \\" } { {'
        }

    def test_consecutive_blocks_are_all_kept(self) -> None:
        first = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        second = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'
        text = first + "\n" + second + "\n(no output)"
        m = _model()
        lines = _text_delta_lines(text, 6)
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        assert content == first + "\n" + second
        assert [c["arguments"]["command"] for c in _parse_text_based_tool_calls(content)] == [
            "ls",
            "pwd",
        ]

    def test_prose_turn_without_tool_calls_is_untouched(self) -> None:
        text = "Here is a snippet: `{a: {b: 1}}` and that's all."
        m = _model()
        lines = _text_delta_lines(text, 5) + [json.dumps({"type": "result", "result": text})]
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        assert content == text
        assert m._stopped_for_tool_calls is False


_FIRST = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
_SECOND = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'


class TestUnmatchedProseBraceBeforeTheBlock:
    """Review R2/I2: a stray ``{`` in prose must not disable detection.

    The original full-text scanner retried from the next ``{`` when an
    earlier candidate never closed, so prose such as "the literal ``{``"
    followed by a complete ``tool_calls`` block still stopped the stream
    at that block and cut off the hallucinated output after it.  The
    incremental tracker has to recover the same way.
    """

    @pytest.mark.parametrize(
        "prefix",
        [
            "The literal opening brace is `{`.\n",
            'Pass {"cmd" and then run:\n',
            'A brace { and a lone quote " here.\n',
            "Nested prose {x: {y}} braces, then:\n",
            "Doubled {{ brace, then:\n",
        ],
    )
    def test_first_block_stops_the_stream_and_cuts_later_calls(self, prefix: str) -> None:
        text = prefix + _FIRST + "\n(no output)\n" + _SECOND
        for chunk in (1, 4, 16):
            m = _model()
            lines = _text_delta_lines(text, chunk)[:-2] + ["never-read"]
            consumed: list[str] = []
            content, _ = m._parse_stream_events(_recorded(lines, consumed), stop_on_tool_calls=True)
            assert content == prefix + _FIRST, (prefix, chunk)
            assert m._stopped_for_tool_calls is True, (prefix, chunk)
            assert consumed[-1] != "never-read", (prefix, chunk)
            assert [c["arguments"]["command"] for c in _parse_text_based_tool_calls(content)] == [
                "ls"
            ]

    def test_end_of_turn_truncation_after_a_prose_brace(self) -> None:
        """A stream that ends right after the block is still truncated to it."""
        prefix = "The literal opening brace is `{`.\n"
        text = prefix + _FIRST + "\n(no output)\n" + _SECOND
        m = _model()
        lines = _text_delta_lines(text, 4) + [json.dumps({"type": "result", "result": text})]
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        assert content == prefix + _FIRST
        assert m._stopped_for_tool_calls is True

    def test_recovery_keeps_large_payloads_linear(self) -> None:
        """A prose brace that is not rejected up front must not bring the rescans back."""
        text = 'Pass {"cmd" and then run:\n' + _write_call_text(64)
        m = _model()
        lines = _text_delta_lines(text, 16) + [json.dumps({"type": "result", "result": text})]
        started = time.perf_counter()
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        elapsed = time.perf_counter() - started
        assert elapsed < 3.0, f"parsing took {elapsed:.1f}s"
        assert content == text
        assert m._stopped_for_tool_calls is True

    def test_brace_heavy_json_arguments_stay_linear(self) -> None:
        """Objects nested inside the call (not in strings) close at depth > 0 thousands of times."""
        rows = [{"id": i, "pos": {"x": i, "y": {"z": [i, {"w": i}]}}} for i in range(1500)]
        text = json.dumps({"tool_calls": [{"name": "Write", "arguments": {"rows": rows}}]})
        assert len(text) > 64 * 1024
        m = _model()
        lines = _text_delta_lines(text, 16) + [json.dumps({"type": "result", "result": text})]
        started = time.perf_counter()
        content, _ = m._parse_stream_events(iter(lines), stop_on_tool_calls=True)
        elapsed = time.perf_counter() - started
        assert elapsed < 3.0, f"parsing took {elapsed:.1f}s"
        assert content == text
        assert m._stopped_for_tool_calls is True
        assert _parse_text_based_tool_calls(content)[0]["arguments"]["rows"] == rows


# A Python program, not a shell script: Windows cannot run a shebang
# file, so ``install_cli_script`` wraps it in the ``claude.cmd`` shim
# ``shutil.which`` finds there (a bare ``claude`` left the real,
# logged-out CLI first on PATH).
_FAKE_CLAUDE = """#!/usr/bin/env python3
import sys
sys.stdin.read()
sys.stdout.write(open({events!r}, encoding="utf-8").read())
"""


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a ``claude`` on PATH that answers with the events in the returned file."""
    events = tmp_path / "events.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "claude", _FAKE_CLAUDE.format(events=str(events)))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return events


def _bash(command: str) -> str:
    """Run a shell command."""
    return "ok"


@pytest.mark.live_cli
class TestF2OnlyTheReturnedContentIsParsed:
    """A tool_calls block the parser cut off is not executed."""

    def test_block_after_a_non_tool_calls_object_is_not_executed(self, fake_claude: Path) -> None:
        first = '{"tool_calls": [{"name": "_bash", "arguments": {"command": "ls"}}]}'
        cut_off = '{"tool_calls": [{"name": "_bash", "arguments": {"command": "rm -rf build"}}]}'
        text = first + ' {"note": "x"} ' + cut_off
        lines = _text_delta_lines(text, 8) + [json.dumps({"type": "result", "result": text})]
        fake_claude.write_text("\n".join(lines) + "\n")

        m = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
        m.initialize("test")
        function_calls, content, _ = m.generate_and_process_with_tools({"_bash": _bash})

        assert content == first
        assert [c["arguments"] for c in function_calls] == [{"command": "ls"}]
        assert m._stopped_for_tool_calls is True
