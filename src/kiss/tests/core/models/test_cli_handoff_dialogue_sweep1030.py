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

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import CLITextModel

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
    text_call = 'Listing.\n{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
    served_call = "```python\nRead(file_path='a.py')\n```"
    cli_model.conversation = [{"role": "user", "content": "Go"}]
    cli_model.add_message_to_conversation("assistant", text_call)
    cli_model._replace_last_assistant_with_tool_calls(
        text_call, [{"id": "call_1", "name": "Bash", "arguments": {"command": "ls"}}]
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
