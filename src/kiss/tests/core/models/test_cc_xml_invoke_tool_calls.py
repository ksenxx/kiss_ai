# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claude's native ``<invoke>`` call syntax on the Claude Code text tool protocol.

A ``cc/claude-opus-5-5`` turn answered the JSON ``tool_calls`` instructions
with Claude's native syntax as plain text (``<invoke name="Bash">
<parameter name="command">...``), one invoke after another with imagined
results between them.  Nothing parsed it: the stream ran for seven minutes
and 140k characters, no tool ran, and the panel showed the raw XML.  Now
the first block that opens a line is the call, the CLI is stopped right
after it, its string arguments are typed by the tool's signature, and the
panel does not show it.

The tests run ``ClaudeCodeModel.generate_and_process_with_tools`` against
a fake ``claude`` on PATH (marked ``live_cli`` only to opt out of this
directory's CLI-locator stub; the real CLI is never spawned).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import _parse_text_based_tool_calls
from kiss.tests.conftest import install_cli_script

_FAKE_CLAUDE = """#!/usr/bin/env python3
import sys
sys.stdin.read()
sys.stdout.write(open({events!r}, encoding="utf-8").read())
"""

# The opening of the recorded turn, followed by the kind of imagined
# continuation that came after it.
_RECORDED = (
    '\n<invoke name="Bash">\n'
    '<parameter name="command">grep -rln "discover" --include=*.py src | head -30'
    "</parameter>\n"
    '<parameter name="description">Locate discover SEA source</parameter>\n'
    "</invoke>\nThe search found discover.py.\n\n"
    '<invoke name="Read">\n'
    '<parameter name="file_path">/home/ksen/kiss/src/kiss/agents/sea/discover/discover.py'
    "</parameter>\n</invoke>\n\n"
    '<invoke name="finish">\n<parameter name="success">false</parameter>\n</invoke>\n'
)


def Bash(command: str, description: str) -> str:  # noqa: N802 - the tool's name
    """Run a shell command."""
    return "ok"


def Read(file_path: str, max_lines: int = 2000) -> str:  # noqa: N802 - the tool's name
    """Read a file."""
    return "ok"


def type_text(element_id: int, text: str, press_enter: bool = False) -> str:
    """Type text into an element."""
    return "ok"


def finish(success: bool, summary: str = "", options: dict[str, Any] | None = None) -> str:
    """Finish the task."""
    return "ok"


def pick(value: str | int, count: int | None = None) -> str:
    """Pick a value."""
    return "ok"


_TOOLS: dict[str, Callable[..., Any]] = {
    "Bash": Bash,
    "Read": Read,
    "type_text": type_text,
    "finish": finish,
    "pick": pick,
}


def _stream_event(event: dict[str, Any]) -> str:
    return json.dumps({"type": "stream_event", "event": event})


def _delta_lines(text: str, chunk: int) -> list[str]:
    """The stream-json lines of one assistant message streaming *text* in *chunk*-sized deltas."""
    lines = [_stream_event({"type": "content_block_start", "content_block": {"type": "text"}})]
    for i in range(0, len(text), chunk):
        delta = {"type": "text_delta", "text": text[i : i + chunk]}
        lines.append(_stream_event({"type": "content_block_delta", "delta": delta}))
    lines.append(_stream_event({"type": "content_block_stop"}))
    return lines + [json.dumps({"type": "result", "result": text})]


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a ``claude`` on PATH that answers with the events in the returned file."""
    events = tmp_path / "events.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "claude", _FAKE_CLAUDE.format(events=str(events)))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return events


def _turn(events: Path, lines: list[str]) -> tuple[list[dict[str, Any]], str, str, ClaudeCodeModel]:
    """Run one tool-bearing turn; return the calls, the content, the panel text and the model."""
    events.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shown: list[str] = []
    m = ClaudeCodeModel("cc/opus", model_config={"timeout": 20}, token_callback=shown.append)
    m.initialize("test")
    calls, content, _ = m.generate_and_process_with_tools(_TOOLS)
    return calls, content, "".join(shown), m


@pytest.mark.live_cli
class TestInvokeTurns:
    """End-to-end turns whose answer uses the ``<invoke>`` syntax."""

    @pytest.mark.parametrize("chunk", [1, 7, 4096])
    def test_recorded_turn_runs_only_the_first_invoke(self, fake_claude: Path, chunk: int) -> None:
        calls, content, shown, m = _turn(
            fake_claude, _delta_lines("Let me look.\n" + _RECORDED, chunk)
        )

        assert [(c["name"], c["arguments"]) for c in calls] == [
            (
                "Bash",
                {
                    "command": 'grep -rln "discover" --include=*.py src | head -30',
                    "description": "Locate discover SEA source",
                },
            )
        ]
        assert m._stopped_for_tool_calls is True
        assert content.endswith("</invoke>") and content.count("<invoke") == 1
        assert "<invoke" not in shown and "parameter" not in shown
        assert "discover.py" not in shown
        assert shown.startswith("Let me look.\n")

    @pytest.mark.parametrize("chunk", [1, 4096])
    def test_invoke_before_a_json_call_is_the_call(self, fake_claude: Path, chunk: int) -> None:
        finish_call = {"tool_calls": [{"name": "finish", "arguments": {"success": True}}]}
        invoke = '<invoke name="Bash"><parameter name="command">ls</parameter></invoke>'
        text = invoke + "\nInvented result\n" + json.dumps(finish_call)
        calls, content, shown, _ = _turn(fake_claude, _delta_lines(text, chunk))
        assert [(c["name"], c["arguments"]) for c in calls] == [("Bash", {"command": "ls"})]
        assert content == invoke
        assert shown == ""

        message = {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
        calls, content, _, _ = _turn(fake_claude, [json.dumps(message)])
        assert [c["name"] for c in calls] == ["Bash"] and content == invoke

    @pytest.mark.parametrize("chunk", [1, 7])
    def test_json_call_inside_an_invoke_argument_is_data(
        self, fake_claude: Path, chunk: int
    ) -> None:
        quoted = json.dumps({"tool_calls": [{"name": "finish", "arguments": {"success": True}}]})
        text = (
            '<invoke name="type_text"><parameter name="element_id">12</parameter>'
            f'<parameter name="text">{quoted}</parameter></invoke>\nTyped it.'
        )
        calls, content, shown, _ = _turn(fake_claude, _delta_lines(text, chunk))
        assert [(c["name"], c["arguments"]) for c in calls] == [
            ("type_text", {"element_id": 12, "text": quoted})
        ]
        assert content == text[: text.index("\n")]
        assert shown == ""

    def test_json_call_before_an_invoke_is_the_call(self, fake_claude: Path) -> None:
        call = json.dumps({"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]})
        text = '{"note": 1}\n' + call + "\n" + _RECORDED
        message = {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
        calls, content, _, _ = _turn(fake_claude, [json.dumps(message)])
        assert [c["arguments"] for c in calls] == [{"command": "ls"}]
        assert content == '{"note": 1}\n' + call

    @pytest.mark.parametrize("prefix", ['{"tool_calls": []}', '{"tool_calls": [{"name": 7}]}'])
    def test_json_envelope_without_a_call_does_not_preempt_the_invoke(
        self, fake_claude: Path, prefix: str
    ) -> None:
        """A ``tool_calls`` list holding no call is text: the invoke after it is the call.

        The JSON precedence used to count the presence of the list, so an
        empty envelope (or one whose entry has no string ``name``) silenced
        the native call that followed it, both in the batch parser and in
        the streaming early stop that cut the turn at the envelope.
        """
        invoke = '<invoke name="Bash"><parameter name="command">echo audit</parameter></invoke>'
        text = prefix + "\n" + invoke + "\nInvented result\n"
        expected = [("Bash", {"command": "echo audit"})]

        calls, content, shown, m = _turn(fake_claude, _delta_lines(text, 1))
        assert [(c["name"], c["arguments"]) for c in calls] == expected
        assert content == prefix + "\n" + invoke
        assert m._stopped_for_tool_calls is True
        assert "<invoke" not in shown and "Invented" not in shown

        message = {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
        calls, content, _, m = _turn(fake_claude, [json.dumps(message)])
        assert [(c["name"], c["arguments"]) for c in calls] == expected
        assert content == prefix + "\n" + invoke
        assert m._stopped_for_tool_calls is True

    def test_invoke_after_an_unfinished_json_prefix_is_hidden(self, fake_claude: Path) -> None:
        text = (
            '{"tool_calls":\n<invoke name="Bash"><parameter name="command">ls</parameter></invoke>'
        )
        calls, _, shown, _ = _turn(fake_claude, _delta_lines(text, 1))
        assert [c["arguments"] for c in calls] == [{"command": "ls"}]
        assert "<invoke" not in shown

    def test_arguments_are_typed_by_the_signature(self, fake_claude: Path) -> None:
        text = (
            '<invoke name="finish">\n<parameter name="success">false</parameter>\n'
            '<parameter name="summary">123</parameter>\n'
            '<parameter name="options">{"a": [1, 2]}</parameter>\n</invoke>'
        )
        calls, _, _, m = _turn(fake_claude, _delta_lines(text, 5))
        assert calls[0]["arguments"] == {
            "success": False,
            "summary": "123",
            "options": {"a": [1, 2]},
        }
        # The typed call is still the invoke of the turn's text: the next
        # prompt shows it once, not again as a ``[Tool Call]`` line.
        m.add_function_results_to_conversation_and_return([("finish", {"result": "ok"})])
        assert "[Tool Call]" not in m._build_prompt()

        text = (
            '<invoke name="type_text">\n<parameter name="element_id">12</parameter>\n'
            '<parameter name="text">42</parameter>\n'
            '<parameter name="press_enter">not json</parameter>\n</invoke>'
        )
        calls, _, _, _ = _turn(fake_claude, _delta_lines(text, 5))
        assert calls[0]["arguments"] == {"element_id": 12, "text": "42", "press_enter": "not json"}

        text = (
            '<invoke name="pick">\n<parameter name="value">null</parameter>\n'
            '<parameter name="count">3</parameter>\n</invoke>'
        )
        calls, _, _, _ = _turn(fake_claude, _delta_lines(text, 5))
        assert calls[0]["arguments"] == {"value": "null", "count": 3}

    def test_unknown_tool_keeps_text_arguments(self, fake_claude: Path) -> None:
        text = '<invoke name="nope">\n<parameter name="n">5</parameter>\n</invoke>'
        calls, _, _, _ = _turn(fake_claude, _delta_lines(text, 5))
        assert [(c["name"], c["arguments"]) for c in calls] == [("nope", {"n": "5"})]

    def test_whole_message_event_stops_at_the_first_invoke(self, fake_claude: Path) -> None:
        message = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": _RECORDED}]},
        }
        calls, content, _, m = _turn(fake_claude, [json.dumps(message)])
        assert [c["name"] for c in calls] == ["Bash"]
        assert content == _RECORDED[: _RECORDED.index("</invoke>") + len("</invoke>")]
        assert m._stopped_for_tool_calls is True

    def test_json_call_wins_and_its_quoted_invoke_is_data(self, fake_claude: Path) -> None:
        quoted = '<invoke name="Bash"><parameter name="command">rm -rf /</parameter></invoke>'
        call = {
            "tool_calls": [{"name": "Bash", "arguments": {"command": "ls", "description": quoted}}]
        }
        text = json.dumps(call) + "\nThe listing shows:\n" + _RECORDED
        calls, content, shown, _ = _turn(fake_claude, _delta_lines(text, 6))
        assert [c["arguments"] for c in calls] == [{"command": "ls", "description": quoted}]
        assert content == json.dumps(call)
        assert "tool_calls" not in shown

    def test_prose_mentioning_the_syntax_is_shown_and_not_run(self, fake_claude: Path) -> None:
        text = (
            'Claude writes calls as <invoke name="X">...</invoke> blocks.\n'
            '<invoke name="Bash" id="1"></invoke>\n'
            "An unfinished one:\n"
            '<invoke name="Bash">\n<parameter name="command">ls'
        )
        calls, content, shown, m = _turn(fake_claude, _delta_lines(text, 3))
        assert calls == []
        assert content == text and shown == text
        assert m._stopped_for_tool_calls is False

    def test_trailing_partial_opener_is_shown_at_the_end(self, fake_claude: Path) -> None:
        text = "Done. See <inv"
        calls, _, shown, _ = _turn(fake_claude, _delta_lines(text, 4))
        assert calls == [] and shown == text

    def test_a_loop_of_unclosed_blocks_is_read_once(self, fake_claude: Path) -> None:
        """A repetition loop of blocks that never close costs one scan, not one per opener.

        Every ``<invoke name="`` and ``<parameter name="`` searched the rest
        of the text for a closer that was not there: 8,000 units took 18 s
        when the turn ended, and as long in the panel filter once the block
        closed.
        """
        loop = '<invoke name="Bash">\n<parameter name="command">ls\n' * 16_000
        started = time.monotonic()
        calls, content, shown, _ = _turn(fake_claude, _delta_lines(loop, 4096))
        assert calls == [] and content == loop and shown == loop
        calls, _, shown, _ = _turn(fake_claude, _delta_lines(loop + "</invoke>", 4096))
        assert [(c["name"], c["arguments"]) for c in calls] == [("Bash", {})] and shown == ""
        assert time.monotonic() - started < 20


def test_parser_reads_only_line_opening_invokes() -> None:
    assert _parse_text_based_tool_calls('see <invoke name="Bash"></invoke>') == []
    calls = _parse_text_based_tool_calls('  <invoke name="Bash"></invoke>')
    assert [(c["name"], c["arguments"]) for c in calls] == [("Bash", {})]
