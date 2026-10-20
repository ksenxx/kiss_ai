# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the tool-call JSON scanner is linear, the filter keeps streaming.

Two ways a cc/codex turn used to stop showing anything in the chat panel:

1. ``_iter_balanced_json_objects`` restarted its forward scan at the next
   ``{`` whenever a candidate never closed, so a 100 KB reply with one
   unclosed quote or one unmatched brace cost ``O(braces x length)`` of
   pure-Python scanning after the child had already exited.
2. ``_ToolCallFilteredStream`` held every token after a ``{`` until the
   object closed, so one brace that never closes (``print("{")``) parked
   the rest of the turn until the child exited and the filter flushed.

The filter now releases the held text the moment it stops being the
prefix of a JSON value, and the scanner pairs every brace in one pass.
The streaming tests run a REAL stand-in ``claude`` executable (real
subprocess, real stream, no mocks or doubles) and the in-process tests
use a real :class:`ClaudeCodeModel` as the filtered adapter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import (
    _iter_balanced_json_objects,
    _JsonPrefix,
    _parse_text_based_tool_calls,
    _ToolCallFilteredStream,
)
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_TOOL_CALL_JSON = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
_PWD_CALL_JSON = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'
_LINEAR_BUDGET_SECONDS = 0.5


def _commands(content: str) -> list[str]:
    """Return the ``command`` argument of every tool call parsed from *content*."""
    return [call["arguments"]["command"] for call in _parse_text_based_tool_calls(content)]


class TestTheScannerIsLinear:
    """Pathological-but-realistic inputs finish in well under a second."""

    def test_unclosed_string_before_many_braces(self) -> None:
        content = '{ "oops ' + ("x" * 50 + "{ ") * 2000
        started = time.monotonic()
        assert _parse_text_based_tool_calls(content) == []
        assert time.monotonic() - started < _LINEAR_BUDGET_SECONDS

    def test_many_unmatched_open_braces(self) -> None:
        content = "{ " * 2000 + "x" * 100000
        started = time.monotonic()
        assert _parse_text_based_tool_calls(content) == []
        assert time.monotonic() - started < _LINEAR_BUDGET_SECONDS

    def test_odd_quote_per_paragraph_with_braces(self) -> None:
        """A stray quote per paragraph used to force one rescan per paragraph."""
        content = 'He said "hello {world. ' * 1000 + _TOOL_CALL_JSON
        started = time.monotonic()
        assert _commands(content) == ["ls"]
        assert time.monotonic() - started < _LINEAR_BUDGET_SECONDS

    def test_nested_prefixes_are_parsed_once_each_and_fast(self) -> None:
        content = '{"k": ' * 2000 + "x" + "}" * 2000
        started = time.monotonic()
        assert _iter_balanced_json_objects(content) == []
        assert time.monotonic() - started < _LINEAR_BUDGET_SECONDS

    def test_balanced_malformed_nesting_is_decoded_once(self) -> None:
        """Every inner candidate fails where the outer one did, so none is retried.

        16,000 nested candidates whose outer decode fails on the innermost
        ``x`` used to cost one decode of the remaining suffix each (about
        nine seconds); a later call must still be found.
        """
        content = '{"k": ' * 16000 + "x" + "}" * 16000 + " " + _TOOL_CALL_JSON
        started = time.monotonic()
        assert _commands(content) == ["ls"]
        assert time.monotonic() - started < _LINEAR_BUDGET_SECONDS

    def test_nesting_beyond_the_parser_is_skipped_whole(self) -> None:
        """One recursion-limit walk, not one per inner candidate."""
        content = '{"k": ' * 70000 + "1" + "}" * 70000 + " " + _TOOL_CALL_JSON
        started = time.monotonic()
        assert _commands(content) == ["ls"]
        assert time.monotonic() - started < 2.0


class TestTheScannerKeepsItsResults:
    """The one-pass scanner reports what the restarting scanner reported."""

    @pytest.mark.parametrize(
        "prefix",
        [
            "The literal opening brace is `{`.\n",
            'Pass {"cmd" and then run:\n',
            'A brace { and a lone quote " here.\n',
            "Nested prose {x: {y}} braces, then:\n",
            "Doubled {{ brace, then:\n",
            'Say "{" then "}" then:\n',
        ],
    )
    def test_objects_after_stray_prose_braces_and_quotes_are_found(self, prefix: str) -> None:
        content = prefix + _TOOL_CALL_JSON + "\n(no output)\n" + _PWD_CALL_JSON
        assert _commands(content) == ["ls", "pwd"]

    def test_object_inside_an_unclosed_string_is_found(self) -> None:
        content = '{ "oops {"a": 1}'
        assert _iter_balanced_json_objects(content) == [(8, 16, {"a": 1})]

    def test_a_parsed_object_swallows_the_objects_inside_it(self) -> None:
        content = 'x {"a": {"b": 1}} y {"c": 2}'
        assert _iter_balanced_json_objects(content) == [
            (2, 17, {"a": {"b": 1}}),
            (20, 28, {"c": 2}),
        ]

    def test_an_object_inside_a_failed_outer_candidate_is_found(self) -> None:
        content = '{x {"a": 1} y}'
        assert _iter_balanced_json_objects(content) == [(3, 11, {"a": 1})]

    def test_braces_inside_strings_do_not_pair(self) -> None:
        content = '{"a": "}", "b": "{"} tail'
        assert _iter_balanced_json_objects(content) == [(0, 20, {"a": "}", "b": "{"})]

    def test_raw_newlines_inside_values_still_parse(self) -> None:
        content = '{"tool_calls": [{"name": "finish", "arguments": {"s": "a\nb"}}]}'
        parsed = _parse_text_based_tool_calls(content)
        assert [call["arguments"]["s"] for call in parsed] == ["a\nb"]

    def test_a_backslash_in_prose_does_not_hide_the_object_after_it(self) -> None:
        """A prose backslash is not a JSON escape; the brace after it is a candidate."""
        assert _iter_balanced_json_objects('\\{"a": 1} \\{"b": 2\\}') == [(1, 9, {"a": 1})]
        assert _iter_balanced_json_objects('\\{ then {"a": 1}') == [(8, 16, {"a": 1})]
        assert _iter_balanced_json_objects('C:\\dir\\ {"a": "\\\\"} \\') == [(8, 19, {"a": "\\"})]

    def test_a_call_right_after_a_prose_backslash_is_executed_and_not_shown(self) -> None:
        """The batch parser and the stream filter agree on the same text."""
        content = "\\" + _TOOL_CALL_JSON
        assert _commands(content) == ["ls"]
        during, after = _stream([content])
        assert "".join(during) + "".join(after) == "\\"

    def test_unmatched_closers_are_ignored(self) -> None:
        assert _iter_balanced_json_objects('} } {"a": 1} ]') == [(4, 12, {"a": 1})]

    def test_valid_objects_inside_a_failed_candidate_are_still_found(self) -> None:
        """Skipping the doomed inner candidates must not skip the sound ones.

        The outer object fails at ``x``; everything the decoder consumed
        before that is valid, and an object that starts inside one of the
        outer's strings is a different text altogether.
        """
        assert _iter_balanced_json_objects('{"a": {"b": 1}, "c": x}') == [(6, 14, {"b": 1})]
        assert _iter_balanced_json_objects('{"a": "x{"k": 1}"}') == [(8, 16, {"k": 1})]
        content = '{"k": {"k": x}, "m": {"z": 1}, y} {"w": 2}'
        assert _iter_balanced_json_objects(content) == [(21, 29, {"z": 1}), (34, 42, {"w": 2})]
        content = '{"a": [{"b": x}], "c": [{"d": 1}, {"e": {"f": 2}}], "g": y}'
        assert _iter_balanced_json_objects(content) == [
            (24, 32, {"d": 1}),
            (34, 49, {"e": {"f": 2}}),
        ]

    def test_a_failure_beyond_a_skipped_region_starts_a_fresh_one(self) -> None:
        content = '{"a": {"b": x}} {"c": {"d": {"e": 1}}, "f": y} {"g": 3}'
        assert _iter_balanced_json_objects(content) == [
            (22, 37, {"d": {"e": 1}}),
            (47, 55, {"g": 3}),
        ]


def _filtered(tokens: list[str]) -> tuple[ClaudeCodeModel, _ToolCallFilteredStream]:
    """Build a real adapter whose token stream *tokens* records, wrapped by the filter."""
    model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
    return model, _ToolCallFilteredStream(model)


def _stream(chunks: list[str]) -> tuple[list[str], list[str]]:
    """Feed *chunks* through the filter and report what came out before and after exit.

    Returns:
        The tokens emitted while the turn was open and the tokens emitted
        by the end-of-turn flush.
    """
    tokens: list[str] = []
    model, stream = _filtered(tokens)
    with stream:
        feed = model.token_callback
        assert feed is not None
        for chunk in chunks:
            feed(chunk)
        during = list(tokens)
    return during, tokens[len(during) :]


class TestTheFilterReleasesProseAsSoonAsItIsProse:
    """Held text streams the moment a character rules out a JSON object."""

    def test_quoted_brace_in_prose_streams_line_by_line(self) -> None:
        chunks = ['In Python print("{") opens nothing.\n'] + [
            f"More prose line {i}.\n" for i in range(5)
        ]
        during, after = _stream(chunks)
        assert "".join(during) == "".join(chunks)
        assert after == []

    @pytest.mark.parametrize(
        "prose",
        [
            "Set {x: 1} in the config.",
            "A C block { foo(); } compiles.",
            "Doubled {{ braces }} in a template.",
            'Pass {"cmd" and then run it.',
            'Data: {"tool_calls": nope} end.',
            'Half {"a" "b"} broken.',
            'Colon {"a" 1} missing.',
            'Comma {"a": 1 "b": 2} missing.',
            'Bad close {"a": [1, 2}.',
            'Scalar {"a": 1x} junk.',
            'Literal {"a": truth} junk.',
            "Empty close {} then {] and [}.",
            'Odd value {"a": ?} here.',
            "Trailing {",
        ],
    )
    def test_non_json_after_a_brace_streams_before_the_turn_ends(self, prose: str) -> None:
        during, after = _stream([prose + "\nnext line\n"])
        assert "".join(during) + "".join(after) == prose + "\nnext line\n"
        assert "next line" in "".join(during)

    @pytest.mark.parametrize(
        "obj",
        [
            '{"a": "say \\"hi\\" and {", "b": "\\\\"}',
            '{"a": [], "b": [1, {"c": null}, [true, false]], "d": {}}',
            '{"n": -1.5e+3, "m": 0}',
        ],
    )
    def test_a_plain_json_object_is_held_whole_and_emitted_whole(self, obj: str) -> None:
        during, after = _stream(["See ", obj[:7], obj[7:], " ok"])
        assert during == ["See ", obj, " ok"]
        assert after == []

    def test_feeding_past_the_closed_object_is_refused(self) -> None:
        prefix = _JsonPrefix()
        assert all(prefix.feed(char) for char in '{"a": 1}')
        assert prefix.open == []
        assert prefix.feed(",") is False
        assert prefix.feed("}") is False

    def test_prose_split_across_chunks_streams_as_it_arrives(self) -> None:
        tokens: list[str] = []
        model, stream = _filtered(tokens)
        with stream:
            feed = model.token_callback
            assert feed is not None
            feed("Prose {")
            assert "".join(tokens) == "Prose "
            feed("x")
            assert "".join(tokens) == "Prose {x"
            feed(': 1} and more {"a": 1')
            assert "".join(tokens) == "Prose {x: 1} and more "
            feed("} end")
        assert "".join(tokens) == 'Prose {x: 1} and more {"a": 1} end'

    def test_the_turn_can_end_inside_prose_held_as_json(self) -> None:
        during, after = _stream(['Note {"docs": "how tool_calls works", "page": 1'])
        assert during == ["Note "]
        assert after == ['{"docs": "how tool_calls works", "page": 1']

    def test_a_tool_call_object_is_still_held_and_dropped(self) -> None:
        during, after = _stream(["Listing:\n", _TOOL_CALL_JSON[:20], _TOOL_CALL_JSON[20:], "\n"])
        assert during == ["Listing:\n", "\n"]
        assert after == []

    def test_a_fenced_tool_call_still_takes_its_fence(self) -> None:
        during, after = _stream(["```json\n", _TOOL_CALL_JSON, "\n```\n", "done"])
        assert "".join(during) + "".join(after) == "done"

    def test_a_tool_call_with_raw_newlines_in_a_value_is_still_held(self) -> None:
        call = json.dumps(
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "- a\n- b"}}]}
        ).replace("\\n", "\n")
        during, after = _stream(["Done.\n", call])
        assert during == ["Done.\n"]
        assert after == []

    def test_held_text_proven_prose_that_hides_a_real_call_is_dropped(self) -> None:
        """Same rule as the end-of-turn flush: a validated call never shows."""
        during, _after = _stream([f'{{"outer": {_TOOL_CALL_JSON}, oops\n'])
        assert "tool_calls" not in "".join(during)
        assert "oops" in "".join(during)

    def test_many_false_candidates_in_one_chunk_do_not_recurse(self) -> None:
        chunk = "{x" * 5000 + "\n"
        during, after = _stream([chunk])
        assert "".join(during) + "".join(after) == chunk

    @pytest.mark.parametrize(
        "arguments",
        [
            '{"x": NaN}',
            '{"x": Infinity, "y": -Infinity}',
            '{"a\nb": 1}',
            '{"files": {"a.txt": "one\ntwo", "b\tc": {"k\ry": null}}}',
        ],
    )
    def test_what_the_decoder_accepts_as_a_call_is_never_shown(self, arguments: str) -> None:
        """The prefix check must be no stricter than ``json.loads(strict=False)``.

        Non-finite numbers and raw control characters inside argument
        keys parse as a tool call, so the filter must hold them too; a
        released prefix cannot be taken back.
        """
        call = '{"tool_calls": [{"name": "f", "arguments": ' + arguments + "}]}"
        assert [c["name"] for c in _parse_text_based_tool_calls(call)] == ["f"]
        during, after = _stream(["Go.\n", call[:30], call[30:], "\n"])
        assert during == ["Go.\n", "\n"]
        assert after == []

    @pytest.mark.parametrize(
        "prose",
        [
            'print("{") opens nothing.\nnext line\n',
            'The list {"tool_calls": [{"name\n": 1}]} is odd.\nnext line\n',
        ],
    )
    def test_a_raw_newline_in_an_envelope_key_still_releases_prose(self, prose: str) -> None:
        """Keys above ``arguments`` are identifiers; a newline there proves prose."""
        during, after = _stream([prose])
        assert "".join(during) + "".join(after) == prose
        assert "next line" in "".join(during)


def _claude_script(lines: list[str], pause: float) -> str:
    """Build a stand-in ``claude`` that streams *lines* with *pause* seconds between them."""
    return f"""
    import json
    import sys
    import time

    sys.stdin.read()
    for line in json.loads({json.dumps(lines)!r}):
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "text_delta", "text": line}}}}),
              flush=True)
        time.sleep({pause!r})
    print(json.dumps({{"type": "result", "result": "".join(json.loads({json.dumps(lines)!r})),
                       "usage": {{"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 0}}}}), flush=True)
    """


def _run_bash(command: str) -> str:
    """Pretend to run a shell command."""
    return f"ran {command}"


class TestTheRealCliStreamKeepsFlowing:
    """Lines after a quoted brace reach the token callback while the child still runs."""

    def test_lines_after_a_quoted_brace_arrive_before_the_child_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pause = 0.15
        lines = ['In Python print("{") opens nothing.\n'] + [
            f"Line {i} of the answer.\n" for i in range(10)
        ]
        install_cli(tmp_path, monkeypatch, "claude", _claude_script(lines, pause))
        arrivals: list[tuple[float, str]] = []

        def record(token: str) -> None:
            arrivals.append((time.monotonic(), token))

        model = ClaudeCodeModel("cc/opus", token_callback=record)
        model.initialize("explain braces")
        calls, content, _response = model.generate_and_process_with_tools({"Bash": _run_bash})
        child_done = time.monotonic()

        assert calls == []
        assert content == "".join(lines)
        assert "".join(token for _at, token in arrivals) == "".join(lines)
        first_at = next(at for at, token in arrivals if "opens nothing" in token)
        last_at = next(at for at, token in arrivals if "Line 9" in token)
        # Ten pauses separate the first line from the last; a filter that
        # parked the turn would deliver both in the flush, microseconds apart.
        assert last_at - first_at > 5 * pause
        assert child_done - first_at > 5 * pause

    def test_a_turn_with_malformed_nesting_and_a_nan_call_ends_quickly_and_hides_the_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole turn, scanner included, stays within budget; the call never shows.

        20,000 balanced-but-malformed nested braces used to cost the batch
        scanner over ten seconds after the child had exited, and a ``NaN``
        argument used to be released to the panel as raw JSON although the
        parser returned it as a call.
        """
        junk = '{"k": ' * 20000 + "x" + "}" * 20000
        call = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls", "n": NaN}}]}'
        lines = ["Prose with a backslash \\ first.\n", junk + "\n", "Then:\n", call, "\n"]
        install_cli(tmp_path, monkeypatch, "claude", _claude_script(lines, 0.0))
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("do it")
        started = time.monotonic()
        calls, content, _response = model.generate_and_process_with_tools({"Bash": _run_bash})
        elapsed = time.monotonic() - started

        assert [c["name"] for c in calls] == ["Bash"]
        assert calls[0]["arguments"]["command"] == "ls"
        assert content.startswith("".join(lines[:3]))
        assert "tool_calls" not in "".join(tokens)
        assert "".join(tokens).startswith("".join(lines[:3]))
        assert elapsed < 5.0
