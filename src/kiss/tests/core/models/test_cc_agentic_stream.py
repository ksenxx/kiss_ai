# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests for agentic Claude Code stream handling.

The ``claude`` CLI now runs agentically (native tools enabled,
``--dangerously-skip-permissions``), mirroring ``CodexModel``.  These tests
pin the stream-parsing contract for native tool activity:

- ``tool_use`` blocks in assistant events stream as thinking (``$ command``
  for Bash, ``Name({...})`` otherwise), deduplicated by tool-use id across
  the growing assistant-message snapshots the CLI re-sends.
- ``tool_result`` payloads carried by ``user`` events stream as thinking.
- Text accumulates across the multiple assistant messages of an agentic
  run, and the terminal ``result`` event stays authoritative.

Event shapes below match real ``claude --print --output-format stream-json
--verbose --include-partial-messages`` output (v2.1.229).
"""

import json

from kiss.core.models.claude_code_model import ClaudeCodeModel, _tool_result_text


class _Recorder:
    """Collects (kind, text) pairs, tracking thinking start/end boundaries."""

    def __init__(self) -> None:
        self.tokens: list[tuple[str, str]] = []
        self.thinking_events: list[bool] = []
        self._in_thinking = False

    def token_cb(self, text: str) -> None:
        self.tokens.append(("thinking" if self._in_thinking else "text", text))

    def thinking_cb(self, is_start: bool) -> None:
        self._in_thinking = is_start
        self.thinking_events.append(is_start)

    def thinking_tokens(self) -> list[str]:
        return [t for kind, t in self.tokens if kind == "thinking"]

    def text_tokens(self) -> list[str]:
        return [t for kind, t in self.tokens if kind == "text"]


def _make_model(rec: _Recorder) -> ClaudeCodeModel:
    m = ClaudeCodeModel(
        "cc/haiku", token_callback=rec.token_cb, thinking_callback=rec.thinking_cb
    )
    m.initialize("test")
    return m


def _parse(m: ClaudeCodeModel, events: list[dict]) -> tuple[str, dict]:
    return m._parse_stream_events(iter(json.dumps(e) for e in events))


class TestToolUseAsThinking:
    """Native tool invocations stream as thinking lines."""

    def test_bash_tool_use_renders_as_shell_line(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Bash",
                 "input": {"command": "cat note.txt"}},
            ]}},
            {"type": "result", "result": "done", "usage": {}},
        ]
        content, _ = _parse(m, events)
        assert rec.thinking_tokens() == ["$ cat note.txt\n"]
        assert content == "done"

    def test_non_bash_tool_use_renders_name_and_input(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Read",
                 "input": {"file_path": "/tmp/x"}},
            ]}},
            {"type": "result", "result": "ok", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ['Read({"file_path": "/tmp/x"})\n']

    def test_tool_use_deduplicated_across_snapshots(self) -> None:
        """The CLI re-sends assistant snapshots; the same tool_use id emits once."""
        rec = _Recorder()
        m = _make_model(rec)
        block = {"type": "tool_use", "id": "tu_1", "name": "Bash",
                 "input": {"command": "ls"}}
        events = [
            {"type": "assistant", "message": {"id": "m1", "content": [block]}},
            {"type": "assistant", "message": {"id": "m1", "content": [block]}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ["$ ls\n"]

    def test_tool_use_emitted_even_after_partial_content_blocks(self) -> None:
        """Text/thinking dedup for partial messages must not swallow tool_use."""
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "plan"}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "tool_use", "id": "tu_9", "name": "Bash",
                 "input": {"command": "pwd"}},
            ]}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        # "plan" streams once (via deltas); the tool_use still surfaces.
        assert rec.thinking_tokens() == ["plan", "$ pwd\n"]

    def test_tool_use_without_name_is_silent(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "tool_use", "id": "tu_1", "input": {}},
            ]}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == []


class TestToolResultAsThinking:
    """Native tool outputs (user events) stream as thinking."""

    def test_string_tool_result(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": "probe", "is_error": False},
            ]}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ["probe"]

    def test_block_list_tool_result(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": [{"type": "text", "text": "line1"},
                             {"type": "text", "text": "line2"}]},
            ]}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ["line1\nline2"]

    def test_empty_tool_result_is_silent(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": ""},
            ]}},
            {"type": "user", "message": {"role": "user", "content": "plain string"}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == []

    def test_tool_result_does_not_touch_content(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "text", "text": "Checking."}]}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "out"}]}},
            {"type": "assistant", "message": {"id": "m2", "content": [
                {"type": "text", "text": " Done."}]}},
        ]
        content, _ = _parse(m, events)
        assert content == "Checking. Done."
        assert rec.text_tokens() == ["Checking.", " Done."]
        assert rec.thinking_tokens() == ["out"]


class TestAgenticEndToEndStream:
    """A realistic agentic run: think → tool_use → tool_result → answer."""

    def test_full_agentic_turn(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events: list[dict] = [
            {"type": "system", "subtype": "init", "tools": ["Bash"]},
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "thinking", "thinking": "Need to read the file."}]}},
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Bash",
                 "input": {"command": "cat note.txt"}}]}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": "probe", "is_error": False}]}},
            {"type": "assistant", "message": {"id": "m2", "content": [
                {"type": "text", "text": "The file contains: probe"}]}},
            {"type": "result", "result": "The file contains: probe",
             "usage": {"input_tokens": 18, "output_tokens": 174,
                       "cache_read_input_tokens": 22925}},
        ]
        content, result_json = _parse(m, events)
        assert content == "The file contains: probe"
        assert rec.thinking_tokens() == [
            "Need to read the file.", "$ cat note.txt\n", "probe",
        ]
        assert rec.text_tokens() == ["The file contains: probe"]
        # Thinking blocks are balanced (every start has an end).
        assert rec.thinking_events.count(True) == rec.thinking_events.count(False)
        assert result_json["usage"]["output_tokens"] == 174


class TestPartialOnlyToolUse:
    """tool_use blocks arriving only via partial events must still surface."""

    def test_partial_tool_use_reassembled_and_emitted(self) -> None:
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "tu_7",
                                  "name": "Bash", "input": {}}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta",
                          "partial_json": '{"command"'}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta",
                          "partial_json": ': "pwd"}'}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ["$ pwd\n"]

    def test_partial_and_snapshot_tool_use_deduplicated(self) -> None:
        """A snapshot arriving before content_block_stop must not double-emit."""
        rec = _Recorder()
        m = _make_model(rec)
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "tu_8",
                                  "name": "Bash", "input": {}}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta",
                          "partial_json": '{"command": "ls"}'}}},
            {"type": "assistant", "message": {"id": "m1", "content": [
                {"type": "tool_use", "id": "tu_8", "name": "Bash",
                 "input": {"command": "ls"}}]}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {}},
        ]
        _parse(m, events)
        assert rec.thinking_tokens() == ["$ ls\n"]


class TestNoDrainAfterKissToolCallStop:
    """Once a KISS tool_calls block ends the turn, nothing more is consumed."""

    def test_events_after_stop_are_not_consumed(self) -> None:
        """Native tool activity queued after the stop must never surface, and
        usage comes from the message_delta events seen before the stop."""
        rec = _Recorder()
        m = _make_model(rec)
        tc = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        events = [
            {"type": "stream_event", "event": {
                "type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 11, "output_tokens": 7}}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tc}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " rambling on"}}},
            # Everything below must be ignored — the turn already ended.
            {"type": "assistant", "message": {"id": "m9", "content": [
                {"type": "tool_use", "id": "tu_x", "name": "Bash",
                 "input": {"command": "rm -rf /tmp/x"}}]}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_x",
                 "content": "post-stop output"}]}},
            {"type": "result", "result": "SHOULD NOT BE USED",
             "usage": {"input_tokens": 999999, "output_tokens": 999999}},
        ]
        content, result_json = m._parse_stream_events(
            iter(json.dumps(e) for e in events), stop_on_tool_calls=True
        )
        assert content == tc
        assert m._stopped_for_tool_calls is True
        assert result_json == {"usage": {"input_tokens": 11, "output_tokens": 7}}
        assert "rm -rf" not in "".join(rec.thinking_tokens())
        assert "post-stop output" not in "".join(rec.thinking_tokens())


class TestKissToolsPromptNote:
    """The not-native clarification must reach the CLI and never leak."""

    def test_note_reaches_prompt_and_is_restored(self) -> None:
        """During a tool-bearing turn the prompt sent on stdin carries the
        warning that KISS tools are not native tools (appended to the task
        after ``CLI_SYSTEM_PROMPT_HEADER``, never as a ``--system-prompt``
        argument); the model_config is restored afterwards."""
        import pathlib
        import subprocess
        import tempfile
        from typing import Any

        captured_args: list[list[str]] = []
        # The prompt writer works at the file-descriptor level
        # (fileno()/os.write), so capture stdin in a real file.
        stdin_capture = tempfile.NamedTemporaryFile(delete=False)
        stdin_capture.close()
        stream_data = json.dumps(
            {"type": "result", "result": "ok", "usage": {}}
        ) + "\n"

        class _FakeStdout:
            def __init__(self, data: str) -> None:
                self._lines = data.splitlines(keepends=True)
                self._pos = 0

            def __iter__(self) -> "_FakeStdout":
                return self

            def __next__(self) -> str:
                if self._pos >= len(self._lines):
                    raise StopIteration
                line = self._lines[self._pos]
                self._pos += 1
                return line

            def read(self) -> str:
                return ""

            def close(self) -> None:
                pass

        class FakePopen:
            def __init__(self, args: list[str], *a: Any, **kw: Any) -> None:
                captured_args.append(list(args))
                self.returncode = 0
                self.stdin = open(stdin_capture.name, "wb")
                self.stdout = _FakeStdout(stream_data)
                self.stderr = _FakeStdout("")

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def poll(self) -> int | None:
                # None ("still running") so the prompt writer does not
                # bail out before writing the prompt.
                return None

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        m = ClaudeCodeModel("cc/haiku")
        m.initialize("hi")
        original_popen = subprocess.Popen
        subprocess.Popen = FakePopen  # type: ignore[assignment,misc]
        try:
            m.generate_and_process_with_tools({"finish": lambda result: result})
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        assert captured_args, "CLI was never invoked"
        args = captured_args[0]
        assert "--system-prompt" not in args
        prompt = pathlib.Path(stdin_capture.name).read_text()
        assert "# You new system prompt follows:" in prompt
        assert "NOT part of your native tool set" in prompt
        assert "finish" in prompt
        # Restored: the note must not leak into subsequent plain turns.
        assert "system_instruction" not in m.model_config


class TestToolResultTextHelper:
    """Unit tests for the _tool_result_text flattener."""

    def test_string_passthrough(self) -> None:
        assert _tool_result_text("hello") == "hello"

    def test_list_of_text_blocks(self) -> None:
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _tool_result_text(blocks) == "a\nb"

    def test_non_text_blocks_skipped(self) -> None:
        blocks = [{"type": "image", "source": {}}, {"type": "text", "text": "x"}]
        assert _tool_result_text(blocks) == "x"

    def test_none_and_other_types(self) -> None:
        assert _tool_result_text(None) == ""
        assert _tool_result_text(42) == ""
        assert _tool_result_text([]) == ""
