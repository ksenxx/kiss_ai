# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a conversation handed to a CLI model keeps every turn.

``set_model`` hands the running conversation to the new model as it is.
The Responses-API adapter stores Responses items: its ``function_call`` /
``function_call_output`` items have no ``role``, so the CLI models' prompt
builder raised ``KeyError`` on every turn after a switch to Claude Code or
Codex, and its ``input_text`` / ``output_text`` parts flattened to empty
text.  The Chat-Completions and Gemini adapters keep each call only in the
assistant message's ``tool_calls``, which the transcript dropped, so the
CLI read tool results with no call before them.

The prompt is built by the real adapters (no CLI process is started).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.context_compaction import (
    KEEP_RECENT_TOOL_RESULTS,
    apply_compaction,
    dropped_chars,
    plan_compaction,
)
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_mcp_bridge import ToolBridge
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import CLITextModel, _parse_text_based_tool_calls, render_tool_call

_RESPONSES_CONVERSATION: list[dict[str, Any]] = [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix a.py"}]},
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Reading it."}],
    },
    {
        "type": "function_call",
        "call_id": "c1",
        "name": "Read",
        "arguments": '{"file_path": "a.py"}',
    },
    {"type": "function_call_output", "call_id": "c1", "output": "print(1)"},
]


def _chat_call(name: str, arguments: Any) -> dict[str, Any]:
    """Return a chat ``tool_calls`` entry as the API adapters store it."""
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture(params=[ClaudeCodeModel, CodexModel], ids=["claude-code", "codex"])
def cli_model(request: pytest.FixtureRequest) -> CLITextModel:
    """One instance of each CLI-backed adapter."""
    name = "codex/default" if request.param is CodexModel else "cc/opus"
    model: CLITextModel = request.param(name)
    return model


def test_responses_items_render_as_a_transcript(cli_model: CLITextModel) -> None:
    cli_model.conversation = json.loads(json.dumps(_RESPONSES_CONVERSATION))
    assert cli_model._build_prompt() == (
        "[User]: Fix a.py\n\n"
        "[Assistant]: Reading it.\n"
        '[Tool Call] Read({"file_path": "a.py"})\n\n'
        "[Tool Result]: print(1)"
    )


def test_a_single_responses_message_is_the_task_text(cli_model: CLITextModel) -> None:
    cli_model.conversation = json.loads(json.dumps(_RESPONSES_CONVERSATION[:2]))
    assert cli_model._build_prompt() == "Fix a.py"


def test_chat_and_gemini_tool_calls_are_rendered(cli_model: CLITextModel) -> None:
    cli_model.conversation = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_chat_call("Bash", '{"command": "ls"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        {
            "role": "assistant",
            "content": "Now read it.",
            "tool_calls": [_chat_call("Read", {"p": 1})],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    ]
    assert cli_model._build_prompt() == (
        "[User]: List files\n\n"
        '[Assistant]: [Tool Call] Bash({"command": "ls"})\n\n'
        "[Tool Result]: a.py\n\n"
        "[Assistant]: Now read it.\n"
        '[Tool Call] Read({"p": 1})\n\n'
        "[Tool Result]: print(1)"
    )


def test_calls_the_text_already_records_are_not_repeated(cli_model: CLITextModel) -> None:
    """Also a call the text wrote with ``null`` arguments: ``null`` is a value, not a failure."""
    text_call = (
        'Listing.\n{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}, '
        '{"name": "sync", "arguments": null}]}'
    )
    served_call = "```python\nRead(file_path='a.py')\n```"
    cli_model.conversation = [{"role": "user", "content": "Go"}]
    cli_model.add_message_to_conversation("assistant", text_call)
    cli_model._replace_last_assistant_with_tool_calls(
        text_call, _parse_text_based_tool_calls(text_call)
    )
    cli_model.conversation += [
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        {
            "role": "assistant",
            "content": served_call,
            "tool_calls": [_chat_call("Read", '{"file_path": "a.py"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    ]
    assert cli_model._build_prompt() == (
        "[User]: Go\n\n"
        f"[Assistant]: {text_call}\n\n"
        "[Tool Result]: a.py\n\n"
        f"[Assistant]: {served_call}\n\n"
        "[Tool Result]: print(1)"
    )


def test_a_call_the_text_only_names_is_still_rendered(cli_model: CLITextModel) -> None:
    """Prose naming the tool, or a same-name call with other arguments, records no call.

    ``false`` is not ``0``: the two are different arguments of a call.  A
    call as Codex renders one is told by its literals (the string
    ``'false'`` is not ``False``), and a call chained to it is other code.
    """
    text_call = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
    flag_call = '{"tool_calls": [{"name": "flag", "arguments": {"value": false}}]}'
    flag_quote = render_tool_call("flag", {"value": "false"})
    chained = "```python\nBash(command='ls')(command='cat a.py')\n```"
    cli_model.conversation = [
        {"role": "user", "content": "Read the report"},
        {
            "role": "assistant",
            "content": "I will use Bash(...) to inspect the report.",
            "tool_calls": [_chat_call("Bash", '{"command": "cat report.txt"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "20% growth"},
        {
            "role": "assistant",
            "content": text_call,
            "tool_calls": [
                _chat_call("Bash", '{"command": "ls"}'),
                _chat_call("Bash", '{"command": "cat a.py"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        {
            "role": "assistant",
            "content": flag_call,
            "tool_calls": [_chat_call("flag", '{"value": 0}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "set"},
        {
            "role": "assistant",
            "content": flag_quote,
            "tool_calls": [_chat_call("flag", '{"value": false}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "unset"},
        {
            "role": "assistant",
            "content": chained,
            "tool_calls": [_chat_call("Bash", '{"command": "cat a.py"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    ]
    assert cli_model._build_prompt() == (
        "[User]: Read the report\n\n"
        "[Assistant]: I will use Bash(...) to inspect the report.\n"
        '[Tool Call] Bash({"command": "cat report.txt"})\n\n'
        "[Tool Result]: 20% growth\n\n"
        f"[Assistant]: {text_call}\n"
        '[Tool Call] Bash({"command": "cat a.py"})\n\n'
        "[Tool Result]: a.py\n\n"
        f"[Assistant]: {flag_call}\n"
        '[Tool Call] flag({"value": 0})\n\n'
        "[Tool Result]: set\n\n"
        f"[Assistant]: {flag_quote}\n"
        '[Tool Call] flag({"value": false})\n\n'
        "[Tool Result]: unset\n\n"
        f"[Assistant]: {chained}\n"
        '[Tool Call] Bash({"command": "cat a.py"})\n\n'
        "[Tool Result]: print(1)"
    )


_LONG = "x = 1\n" * 200
"""An argument past the 500 characters a compaction leaves alone."""

_LS_TURN = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
"""A turn to age the turns before it out of the window a compaction keeps."""


def _record_own_turn(model: CLITextModel, text: str) -> None:
    """Record *text* as a tool-bearing turn of *model* itself, each call answered ``ok``."""
    calls = _parse_text_based_tool_calls(text)
    model.add_message_to_conversation("assistant", text)
    model._replace_last_assistant_with_tool_calls(text, calls)
    model.add_function_results_to_conversation_and_return(
        [(call["name"], {"result": "ok"}) for call in calls]
    )


def _answer_ok(call: dict[str, Any]) -> tuple[str, str]:
    """Execute *call* for the tool bridge: every tool answers ``ok``."""
    return call["name"], "ok"


def _served_turn(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Return the messages Codex records for a call its tool bridge served."""
    codex = CodexModel("codex/default")
    codex.conversation = []
    with ToolBridge([], _answer_ok) as bridge:
        bridge._server.serve_call({"name": name, "arguments": arguments})
        codex._replay_served_calls(bridge)
    return codex.conversation


def test_a_compacted_argument_is_stubbed_in_the_text_the_cli_reads(cli_model: CLITextModel) -> None:
    """A compaction stubs an old call's argument where the CLI model reads it: the turn's text.

    The stub replaced the ``tool_calls`` record only.  The turn's own text
    (the JSON block, the ``<invoke>``, Codex's rendered call) kept the whole
    argument and no longer matched the record, so the stubbed call was
    listed a second time: the compaction made the prompt longer.
    """
    json_call = {"name": "Write", "arguments": {"file_path": "a.py", "content": _LONG}}
    # One line: its JSON spelling is its raw text, but an <invoke> takes the stub raw.
    invoke = (
        '<invoke name="Write">\n<parameter name="file_path">b.py</parameter>\n'
        f'<parameter name="content">{"y" * 600}</parameter>\n</invoke>'
    )
    served = {"file_path": "c.py", "content": _LONG}
    # Two calls of one turn that a stub cannot tell apart: same size, same opening.
    twins = ["z" * 120 + tail * 480 for tail in "ab"]
    twin_calls = [{"name": "Bash", "arguments": {"command": twin}} for twin in twins]
    cli_model.conversation = [{"role": "user", "content": "Go"}]
    _record_own_turn(cli_model, "Writing.\n" + json.dumps({"tool_calls": [json_call]}))
    _record_own_turn(cli_model, invoke)
    _record_own_turn(cli_model, json.dumps({"tool_calls": twin_calls}))
    cli_model.conversation += _served_turn("Write", served)
    # One more than the kept window: the last result is still unseen by the model.
    for _ in range(KEEP_RECENT_TOOL_RESULTS + 1):
        _record_own_turn(cli_model, _LS_TURN)
    before = cli_model._build_prompt()
    plan = plan_compaction(cli_model.conversation)
    assert [replacement.kind for replacement in plan] == ["argument"] * 5
    apply_compaction(plan)
    after = cli_model._build_prompt()
    assert "[Tool Call]" not in after
    assert after.count("[compacted argument") == 5
    # A stub's newline is two characters in a JSON string or a ``repr``: one each is allowed.
    assert len(before) - len(after) >= dropped_chars(plan) - 5
    stubs = [
        value
        for message in cli_model.conversation
        for call in message.get("tool_calls") or []
        for value in json.loads(call["function"]["arguments"]).values()
        if value.startswith("[compacted argument")
    ]
    # Each stub is in the text in the spelling of its turn, so the model can copy it back.
    in_json, in_invoke, first_twin, second_twin, in_rendering = stubs
    assert json.dumps(in_json)[1:-1] in after and in_invoke in after and repr(in_rendering) in after
    assert json.dumps(first_twin)[1:-1] in after and json.dumps(second_twin)[1:-1] in after
    assert not any(twin in after for twin in twins)


def test_a_stub_takes_the_place_of_its_argument_and_of_nothing_else(
    cli_model: CLITextModel,
) -> None:
    """The stub goes where the argument is, in every repeat of its call, not where its text is.

    The text of a compacted argument is also quoted in the turn's prose
    and opens a path, an argument that is kept; and the turn wrote its
    block twice, which is one call.  The first place the text was found
    took the stub: the prose, then the path, never the argument.
    """
    text = "segment/" * 75
    read = {"name": "Read", "arguments": {"file_path": text + "a.py"}}
    write = {"name": "Write", "arguments": {"file_path": "path.txt", "content": text}}
    block = json.dumps({"tool_calls": [read, write]}, indent=1)
    cli_model.conversation = [{"role": "user", "content": "Go"}]
    _record_own_turn(cli_model, f"I write {text}\n{block}\n{block}")
    for _ in range(KEEP_RECENT_TOOL_RESULTS + 1):
        _record_own_turn(cli_model, _LS_TURN)
    assert apply_compaction(plan_compaction(cli_model.conversation)) > 0
    turn = cli_model._build_prompt().split("\n\n[Tool Result]")[0]
    assert turn.startswith(f"[User]: Go\n\n[Assistant]: I write {text}\n")
    assert "[Tool Call]" not in turn
    # The prose and the path of each block keep the text; no ``content`` does.
    assert turn.count(text) == turn.count(text + "a.py") + 1 == 3
    stored = [
        json.loads(call["function"]["arguments"])
        for call in cli_model.conversation[1]["tool_calls"]
    ]
    assert [call["arguments"] for call in _parse_text_based_tool_calls(turn)] == stored
    assert stored[1]["content"].startswith("[compacted argument")


@pytest.mark.parametrize("rendered", [False, True], ids=["json-block", "rendered-call"])
def test_a_quoted_twin_of_a_compacted_call_is_not_the_call(
    cli_model: CLITextModel, rendered: bool
) -> None:
    """A call is the text's by the id its parse gave it, never by a text its stub fits.

    An API model quotes one ``Write`` and makes another natively: same
    path, contents of one size and one opening, which is all a stub says
    of the text it replaced.  Once a compaction stubbed the record, the
    quote was taken for the call: it lost its content to the stub, and
    the call its line.
    """
    quoted, made = ({"file_path": "a.py", "content": "x" * 120 + tail * 480} for tail in "ab")
    quote = (
        render_tool_call("Write", quoted)
        if rendered
        else json.dumps({"tool_calls": [{"name": "Write", "arguments": quoted}]})
    )
    cli_model.conversation = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": quote,
            "tool_calls": [_chat_call("Write", json.dumps(made))],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    for _ in range(KEEP_RECENT_TOOL_RESULTS + 1):
        _record_own_turn(cli_model, _LS_TURN)
    assert apply_compaction(plan_compaction(cli_model.conversation)) > 0
    stubbed = cli_model.conversation[1]["tool_calls"][0]["function"]["arguments"]
    assert "[compacted argument" in stubbed
    assert cli_model._build_prompt().startswith(
        f"[User]: Go\n\n[Assistant]: {quote}\n[Tool Call] Write({stubbed})\n\n[Tool Result]: ok\n\n"
    )


def test_a_digest_two_calls_of_a_turn_share_tells_neither(cli_model: CLITextModel) -> None:
    """An id tells its call by a digest of 48 bits, which these two calls of one turn share.

    Only the second is long enough for a compaction.  Its stub went to
    the first call, the first one of that digest, and the prompt lost the
    first command.  Such a turn is rendered like one whose ids carry no
    digest: the text as written, the stubbed call on a line of its own.
    """
    commands = ["echo 7837417", "echo " + "x" * 600 + "7998327"]
    text = json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": c}} for c in commands]}
    )
    cli_model.conversation = [{"role": "user", "content": "Go"}]
    _record_own_turn(cli_model, text)
    calls = cli_model.conversation[1]["tool_calls"]
    assert len({call["id"][-12:] for call in calls}) == 1
    for _ in range(KEEP_RECENT_TOOL_RESULTS + 1):
        _record_own_turn(cli_model, _LS_TURN)
    assert apply_compaction(plan_compaction(cli_model.conversation)) > 0
    stubbed = calls[1]["function"]["arguments"]
    assert "[compacted argument" in stubbed
    assert cli_model._build_prompt().startswith(
        f"[User]: Go\n\n[Assistant]: {text}\n[Tool Call] Bash({stubbed})\n\n"
        "[Tool Result]: ok\n\n[Tool Result]: ok\n\n"
    )
